import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mini_gym.deploy.rapid_locomotion_policy import (
    ACTION_DIM,
    ACTION_SCALE,
    COMMAND_SCALE,
    DEFAULT_Q_POLICY,
    HIP_SCALE_REDUCTION,
    OBS_DIM,
    ObservationHistory,
    action_to_target_q,
    build_observation,
    policy_to_robot_order,
    robot_to_policy_order,
)
from scripts.rl_lcm_policy import resolve_checkpoint


class RapidLocomotionDeployTest(unittest.TestCase):
    def test_zero_action_targets_default_pose(self):
        target = action_to_target_q(np.zeros(ACTION_DIM, dtype=np.float32))
        np.testing.assert_allclose(target, DEFAULT_Q_POLICY)

    def test_hip_action_uses_reduced_scale(self):
        action = np.ones(ACTION_DIM, dtype=np.float32)
        target = action_to_target_q(action)
        expected = DEFAULT_Q_POLICY + ACTION_SCALE
        expected[[0, 3, 6, 9]] = (
            DEFAULT_Q_POLICY[[0, 3, 6, 9]] + ACTION_SCALE * HIP_SCALE_REDUCTION
        )
        np.testing.assert_allclose(target, expected)

    def test_observation_layout_and_scales(self):
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        command = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        q = DEFAULT_Q_POLICY + 0.1
        qd = np.ones(ACTION_DIM, dtype=np.float32) * 2.0
        last_action = np.arange(ACTION_DIM, dtype=np.float32)

        obs = build_observation(gravity, command, q, qd, last_action)

        self.assertEqual(obs.shape, (OBS_DIM,))
        np.testing.assert_allclose(obs[0:3], gravity)
        np.testing.assert_allclose(obs[3:6], command * COMMAND_SCALE)
        np.testing.assert_allclose(obs[6:18], np.ones(ACTION_DIM) * 0.1, atol=1e-6)
        np.testing.assert_allclose(obs[18:30], np.ones(ACTION_DIM) * 0.1, atol=1e-6)
        np.testing.assert_allclose(obs[30:42], last_action)

    def test_history_shifts_left_and_appends_latest_obs(self):
        history = ObservationHistory(length=3, obs_dim=2)
        np.testing.assert_allclose(history.buffer, np.zeros(6))

        history.update([1.0, 2.0])
        np.testing.assert_allclose(history.buffer, [0, 0, 0, 0, 1, 2])
        history.update([3.0, 4.0])
        np.testing.assert_allclose(history.buffer, [0, 0, 1, 2, 3, 4])
        history.update([5.0, 6.0])
        np.testing.assert_allclose(history.buffer, [1, 2, 3, 4, 5, 6])

    def test_policy_robot_mapping_matches_training_dof_order(self):
        values = np.arange(ACTION_DIM, dtype=np.float32)
        expected_policy = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.float32)
        expected_robot = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.float32)
        np.testing.assert_allclose(robot_to_policy_order(values), expected_policy)
        np.testing.assert_allclose(policy_to_robot_order(values), expected_robot)

    def test_resolve_checkpoint_accepts_run_directory(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "train" / "201852.132488" / "checkpoints"
            checkpoint.mkdir(parents=True)
            adaptation = checkpoint / "adaptation_module_latest.jit"
            body = checkpoint / "body_latest.jit"
            adaptation.write_text("adaptation")
            body.write_text("body")

            resolved_adaptation, resolved_body = resolve_checkpoint(Path(tmp))

            self.assertEqual(resolved_adaptation, adaptation)
            self.assertEqual(resolved_body, body)


if __name__ == "__main__":
    unittest.main()
