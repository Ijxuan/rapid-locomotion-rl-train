import unittest

import torch
import torch.nn.functional as F

from mini_gym_learn.eval_metrics.metrics import adaptation_loss, auxiliary_rewards, estimator_loss


class FakeActorCritic:
    estimator = object()
    estimator_state_dim = 7

    def __init__(self, prediction):
        self.prediction = prediction

    def estimate(self, observations):
        del observations
        return self.prediction


class FakeEnv:
    def __init__(self):
        self.reward_names = ["tracking_lin_vel", "feet_slip"]
        self.reward_scales = {
            "tracking_lin_vel": 3.0,
            "feet_slip": -0.08,
        }
        self.reward_functions = [
            lambda: torch.ones(2),
            lambda: torch.full((2,), 2.0),
        ]


class PaperBEvalMetricsTest(unittest.TestCase):
    def test_estimator_loss_matches_paper_b_supervision_terms(self):
        prediction = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.8, 0.8, 0.2, 0.9, 0.1],
                [0.0, 1.0, 2.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.4, 0.7, 0.3],
            ],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [
                [1.5, 2.5, 3.5, 0.0, 0.1, 0.2, 0.3, 1.0, 0.0, 1.0, 0.0],
                [0.5, 1.5, 2.5, 0.4, 0.5, 0.6, 0.7, 1.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        actor_critic = FakeActorCritic(prediction)
        obs = {
            "obs": torch.zeros(2, 142),
            "privileged_obs": target,
        }

        loss = estimator_loss(None, actor_critic, obs)

        expected_state_loss = torch.mean((prediction[:, :7] - target[:, :7]) ** 2, dim=1)
        expected_contact_loss = F.binary_cross_entropy(
            prediction[:, 7:], target[:, 7:], reduction="none").mean(dim=1)
        torch.testing.assert_close(loss, expected_state_loss + expected_contact_loss)
        torch.testing.assert_close(adaptation_loss(None, actor_critic, obs), loss)

    def test_auxiliary_rewards_returns_all_terms(self):
        rewards = auxiliary_rewards(FakeEnv(), None, None)

        self.assertEqual(set(rewards), {"tracking_lin_vel", "feet_slip"})
        torch.testing.assert_close(rewards["tracking_lin_vel"], torch.full((2,), 3.0))
        torch.testing.assert_close(rewards["feet_slip"], torch.full((2,), -0.16))


if __name__ == "__main__":
    unittest.main()
