import unittest
from pathlib import Path

import torch

from mini_gym.envs.base.paper_b_observation import (
    ACTION_DIM,
    ACTOR_INPUT_DIM,
    ESTIMATOR_TARGET_DIM,
    OBS_DIM,
)
from mini_gym_learn.ppo.actor_critic import ActorCritic


ROOT = Path(__file__).resolve().parents[1]


class PaperBPpoSourceTest(unittest.TestCase):
    def read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_actor_critic_uses_explicit_estimator_architecture(self):
        source = self.read("mini_gym_learn/ppo/actor_critic.py")

        self.assertIn("class Estimator", source)
        self.assertIn("estimator_hidden_dims = [256, 128]", source)
        self.assertIn("actor_hidden_dims = [512, 256, 64]", source)
        self.assertIn("critic_hidden_dims = [512, 256, 64]", source)
        self.assertNotIn("adaptation_module", source)
        self.assertNotIn("env_factor_encoder", source)

    def test_ppo_trains_estimator_from_privileged_targets(self):
        source = self.read("mini_gym_learn/ppo/ppo.py")

        self.assertIn("estimator_optimizer", source)
        self.assertIn("num_estimator_substeps", source)
        self.assertIn("F.mse_loss(state_pred, state_target)", source)
        self.assertIn("F.binary_cross_entropy(contact_pred, contact_target)", source)
        self.assertNotIn("adaptation_module", source)
        self.assertNotIn("env_factor_encoder", source)

    def test_runner_exports_estimator_and_body_jit(self):
        source = self.read("mini_gym_learn/ppo/__init__.py")

        self.assertIn("estimator_latest.jit", source)
        self.assertIn("body_latest.jit", source)
        self.assertNotIn("adaptation_module_latest.jit", source)

    def test_play_scripts_load_estimator_and_body_jit(self):
        for relative_path in ("scripts/play.py", "scripts/play_zero_probe.py"):
            source = self.read(relative_path)

            self.assertIn("estimator_latest.jit", source)
            self.assertIn("body_latest.jit", source)
            self.assertIn("torch.jit.load", source)
            self.assertNotIn("ac_weights_last.pt", source)
            self.assertNotIn("ActorCritic(", source)

    def test_train_script_exposes_headless_and_logs_jit_names(self):
        source = self.read("scripts/train.py")

        self.assertIn("def train_mc(headless=True", source)
        self.assertIn("--headless", source)
        self.assertIn("--show", source)
        self.assertIn("estimator_latest.jit", source)
        self.assertIn("body_latest.jit", source)

    def test_train_script_exposes_2070_low_memory_profile(self):
        source = self.read("scripts/train.py")

        self.assertIn("def apply_2070_profile", source)
        self.assertIn('parser.add_argument("-2070", "--rtx2070"', source)
        self.assertIn("Cfg.env.num_envs = 32", source)
        self.assertIn("RunnerArgs.num_steps_per_env = 16", source)
        self.assertIn("Cfg.terrain.num_cols = 32", source)
        self.assertIn("Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 20", source)
        self.assertIn("Cfg.sim.physx.default_buffer_size_multiplier = 5", source)
        self.assertIn('low_memory_profile="2070" if profile_2070 else "default"', source)

    def test_actor_critic_paper_b_runtime_shapes(self):
        model = ActorCritic(
            num_obs=OBS_DIM,
            num_privileged_obs=ESTIMATOR_TARGET_DIM,
            num_obs_history=OBS_DIM,
            num_actions=ACTION_DIM,
        )
        obs = torch.randn(3, OBS_DIM)

        estimator_output = model.estimate(obs)
        action = model.act_student(obs)
        value = model.evaluate(obs, torch.randn(3, ESTIMATOR_TARGET_DIM))

        self.assertEqual(tuple(estimator_output.shape), (3, ESTIMATOR_TARGET_DIM))
        self.assertEqual(tuple(action.shape), (3, ACTION_DIM))
        self.assertEqual(tuple(value.shape), (3, 1))
        self.assertEqual(model.actor_body[0].in_features, ACTOR_INPUT_DIM)
        self.assertEqual(model.critic_body[0].in_features, ACTOR_INPUT_DIM)
        self.assertTrue(torch.all(estimator_output[:, -4:] >= 0.0))
        self.assertTrue(torch.all(estimator_output[:, -4:] <= 1.0))

    def test_actor_body_uses_estimator_output_not_privileged_obs(self):
        model = ActorCritic(
            num_obs=OBS_DIM,
            num_privileged_obs=ESTIMATOR_TARGET_DIM,
            num_obs_history=OBS_DIM,
            num_actions=ACTION_DIM,
        )
        obs = torch.randn(2, OBS_DIM)

        model.update_distribution(obs, torch.zeros(2, ESTIMATOR_TARGET_DIM))
        mean_with_zero_privileged = model.action_mean.detach().clone()
        model.update_distribution(obs, torch.ones(2, ESTIMATOR_TARGET_DIM))
        mean_with_one_privileged = model.action_mean.detach().clone()

        torch.testing.assert_close(mean_with_zero_privileged, mean_with_one_privileged)

    def test_estimator_and_body_are_torchscript_exportable(self):
        model = ActorCritic(
            num_obs=OBS_DIM,
            num_privileged_obs=ESTIMATOR_TARGET_DIM,
            num_obs_history=OBS_DIM,
            num_actions=ACTION_DIM,
        ).cpu()
        obs = torch.zeros(1, OBS_DIM)

        estimator = torch.jit.script(model.estimator)
        body = torch.jit.script(model.actor_body)
        estimator_output = estimator(obs)
        action = body(torch.cat((obs, estimator_output), dim=-1))

        self.assertEqual(tuple(estimator_output.shape), (1, ESTIMATOR_TARGET_DIM))
        self.assertEqual(tuple(action.shape), (1, ACTION_DIM))


if __name__ == "__main__":
    unittest.main()
