# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch
import torch.nn as nn
from params_proto.neo_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    estimator_hidden_dims = [256, 128]
    actor_hidden_dims = [512, 256, 64]
    critic_hidden_dims = [512, 256, 64]
    estimator_contact_dim = 4
    activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid


class Estimator(nn.Module):
    def __init__(self, num_obs, output_dim, hidden_dims, contact_dim, activation_name):
        super().__init__()
        self.output_dim = output_dim
        self.contact_dim = contact_dim
        self.state_dim = output_dim - contact_dim
        self.net = build_mlp(num_obs, hidden_dims, output_dim, activation_name)

    def forward(self, observations):
        raw_output = self.net(observations)
        state = raw_output[..., :self.state_dim]
        contact_probability = torch.sigmoid(raw_output[..., self.state_dim:])
        return torch.cat((state, contact_probability), dim=-1)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super().__init__()

        del num_obs_history

        self.num_obs = num_obs
        self.num_estimator_outputs = num_privileged_obs
        self.num_actions = num_actions
        self.estimator_contact_dim = AC_Args.estimator_contact_dim
        self.estimator_state_dim = self.num_estimator_outputs - self.estimator_contact_dim

        self.estimator = Estimator(
            num_obs=num_obs,
            output_dim=self.num_estimator_outputs,
            hidden_dims=AC_Args.estimator_hidden_dims,
            contact_dim=self.estimator_contact_dim,
            activation_name=AC_Args.activation,
        )

        actor_input_dim = num_obs + self.num_estimator_outputs
        self.actor_body = build_mlp(actor_input_dim, AC_Args.actor_hidden_dims, num_actions, AC_Args.activation)
        self.critic_body = build_mlp(actor_input_dim, AC_Args.critic_hidden_dims, 1, AC_Args.activation)

        print(f"Estimator MLP: {self.estimator}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def policy_parameters(self):
        return list(self.actor_body.parameters()) + list(self.critic_body.parameters()) + [self.std]

    def estimate(self, observations):
        return self.estimator(observations)

    def _estimated_state_for_policy(self, observations):
        return self.estimate(observations).detach()

    def _actor_input_from_estimator(self, observations):
        estimated_state = self._estimated_state_for_policy(observations)
        return torch.cat((observations, estimated_state), dim=-1)

    def _actor_input_from_target(self, observations, estimator_target):
        return torch.cat((observations, estimator_target.detach()), dim=-1)

    def update_distribution(self, observations, privileged_observations=None):
        del privileged_observations
        mean = self.actor_body(self._actor_input_from_estimator(observations))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, privileged_observations=None, **kwargs):
        del privileged_observations, kwargs
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs"], ob["privileged_obs"], policy_info)

    def act_inference(self, ob, policy_info={}):
        privileged_obs = ob.get("privileged_obs")
        if privileged_obs is not None:
            policy_info["estimator_targets"] = privileged_obs.detach().cpu().numpy()
        return self.act_student(ob["obs"], ob.get("obs_history"), policy_info)

    def act_student(self, observations, observation_history=None, policy_info={}):
        del observation_history
        estimated_state = self._estimated_state_for_policy(observations)
        actions_mean = self.actor_body(torch.cat((observations, estimated_state), dim=-1))
        policy_info["estimator_outputs"] = estimated_state.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observations, privileged_info, policy_info={}):
        if privileged_info is None:
            return self.act_student(observations, None, policy_info)
        actions_mean = self.actor_body(self._actor_input_from_target(observations, privileged_info))
        policy_info["estimator_targets"] = privileged_info.detach().cpu().numpy()
        return actions_mean

    def evaluate(self, critic_observations, privileged_observations=None, **kwargs):
        del privileged_observations, kwargs
        value = self.critic_body(self._actor_input_from_estimator(critic_observations))
        return value


def build_mlp(input_dim, hidden_dims, output_dim, activation_name):
    layers = [nn.Linear(input_dim, hidden_dims[0]), get_activation(activation_name)]
    for layer_index in range(len(hidden_dims)):
        if layer_index == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[layer_index], output_dim))
        else:
            layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
            layers.append(get_activation(activation_name))
    return nn.Sequential(*layers)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
