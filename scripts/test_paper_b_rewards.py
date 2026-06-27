import unittest

import numpy as np

from mini_gym.envs.base.paper_b_rewards import (
    PAPER_B_NEGATIVE_REWARDS,
    PAPER_B_POSITIVE_REWARDS,
    paper_b_airtime_piecewise,
    paper_b_total_reward,
)


class PaperBRewardHelperTest(unittest.TestCase):
    def test_total_reward_uses_positive_times_exponentiated_negative(self):
        positive = np.array([6.0, 3.0], dtype=np.float32)
        negative = np.array([0.0, -2.0], dtype=np.float32)

        total = paper_b_total_reward(positive, negative, exponential_scale=0.2)

        np.testing.assert_allclose(total, positive * np.exp(0.2 * negative))
        self.assertEqual(total.shape, positive.shape)

    def test_airtime_piecewise_shape_and_stance_branch(self):
        feet_air_time = np.array([[0.12, 0.12, 0.12, 0.12]], dtype=np.float32)
        first_contact = np.array([[False, False, False, False]])
        contact_filter = np.array([[True, True, True, True]])
        command_norm = np.array([0.0], dtype=np.float32)

        reward = paper_b_airtime_piecewise(
            feet_air_time,
            first_contact,
            contact_filter,
            command_norm,
            dt=0.01,
        )

        np.testing.assert_allclose(reward, np.full((1, 4), 0.01, dtype=np.float32))
        self.assertEqual(reward.shape, feet_air_time.shape)

    def test_airtime_piecewise_shape_and_swing_branch(self):
        feet_air_time = np.array([[0.1, 0.35, 0.6, 0.8]], dtype=np.float32)
        first_contact = np.array([[True, True, True, True]])
        contact_filter = np.array([[True, True, True, True]])
        command_norm = np.array([0.5], dtype=np.float32)

        reward = paper_b_airtime_piecewise(
            feet_air_time,
            first_contact,
            contact_filter,
            command_norm,
            airtime_clip=0.3,
            airtime_max=0.25,
            airtime_cap=0.2,
            dt=0.01,
        )

        np.testing.assert_allclose(reward, np.array([[0.0, 0.05, 0.2, 0.2]], dtype=np.float32), atol=1e-6)
        self.assertEqual(reward.shape, feet_air_time.shape)

    def test_reward_term_groups_match_paper_b_plan(self):
        self.assertEqual(
            PAPER_B_POSITIVE_REWARDS,
            ("tracking_lin_vel", "tracking_ang_vel", "feet_air_time"),
        )
        self.assertIn("action_smoothness_2", PAPER_B_NEGATIVE_REWARDS)
        self.assertIn("base_motion", PAPER_B_NEGATIVE_REWARDS)


if __name__ == "__main__":
    unittest.main()
