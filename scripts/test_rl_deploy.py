import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mini_gym.deploy.rapid_locomotion_policy import (
    ACTION_DIM,
    ACTION_SCALE,
    ACTOR_INPUT_DIM,
    DEFAULT_Q_POLICY,
    ESTIMATOR_OUTPUT_DIM,
    OBS_DIM,
    POLICY_DOF_NAMES,
    POLICY_LEG_ORDER,
    ROBOT_LEG_ORDER,
    ObservationHistory,
    action_to_target_q,
    build_observation,
    policy_foot_positions_to_robot_order,
    policy_to_robot_order,
    robot_foot_positions_to_policy_order,
    robot_to_policy_order,
)
from scripts.rl_lcm_policy import resolve_checkpoint


class RapidLocomotionDeployTest(unittest.TestCase):
    def test_deploy_policy_order_matches_mini_cheetah_training_config(self):
        expected_dof_names = (
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        )
        expected_default_q = np.array(
            [0.1, -0.8, 1.62, -0.1, -0.8, 1.62, 0.1, -0.8, 1.62, -0.1, -0.8, 1.62],
            dtype=np.float32,
        )

        self.assertEqual(POLICY_LEG_ORDER, ("FL", "FR", "RL", "RR"))
        self.assertEqual(ROBOT_LEG_ORDER, ("FR", "FL", "RR", "RL"))
        self.assertEqual(POLICY_DOF_NAMES, expected_dof_names)
        np.testing.assert_allclose(DEFAULT_Q_POLICY, expected_default_q)

    def test_zero_action_targets_default_pose(self):
        target = action_to_target_q(np.zeros(ACTION_DIM, dtype=np.float32))
        np.testing.assert_allclose(target, DEFAULT_Q_POLICY)

    def test_action_maps_to_point_one_joint_delta_without_hip_reduction(self):
        action = np.ones(ACTION_DIM, dtype=np.float32)
        target = action_to_target_q(action)
        expected = DEFAULT_Q_POLICY + ACTION_SCALE
        np.testing.assert_allclose(target, expected)

    def test_observation_layout(self):
        base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        base_ang_vel = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        q = DEFAULT_Q_POLICY + 0.1
        qd = np.ones(ACTION_DIM, dtype=np.float32) * 2.0
        previous_q_des = np.tile(DEFAULT_Q_POLICY, 2)
        q_hist = np.ones(ACTION_DIM * 3, dtype=np.float32)
        qd_hist = np.ones(ACTION_DIM * 3, dtype=np.float32) * 2.0
        foot_positions = np.ones(12, dtype=np.float32) * 0.3
        command = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        obs = build_observation(base_quat, base_ang_vel, q, qd, previous_q_des, q_hist, qd_hist, foot_positions, command)

        self.assertEqual(OBS_DIM, 142)
        self.assertEqual(ESTIMATOR_OUTPUT_DIM, 11)
        self.assertEqual(ACTOR_INPUT_DIM, 153)
        self.assertEqual(obs.shape, (OBS_DIM,))
        np.testing.assert_allclose(obs[0:4], base_quat)
        np.testing.assert_allclose(obs[4:7], base_ang_vel)
        np.testing.assert_allclose(obs[7:19], q)
        np.testing.assert_allclose(obs[19:31], qd)
        np.testing.assert_allclose(obs[31:55], previous_q_des)
        np.testing.assert_allclose(obs[127:139], foot_positions)
        np.testing.assert_allclose(obs[139:142], command)

    def test_observation_history_tracks_paper_b_joint_buffers(self):
        history = ObservationHistory()
        history.update_joint_state(DEFAULT_Q_POLICY + 0.1, np.ones(ACTION_DIM))
        history.update_desired_joint_positions(DEFAULT_Q_POLICY + 0.2)

        np.testing.assert_allclose(
            history.joint_position_error_history[:ACTION_DIM], np.ones(ACTION_DIM) * 0.1, atol=1e-6)
        np.testing.assert_allclose(history.joint_velocity_history[:ACTION_DIM], np.ones(ACTION_DIM))
        np.testing.assert_allclose(history.previous_desired_joint_positions[:ACTION_DIM], DEFAULT_Q_POLICY + 0.2)

    def test_observation_history_uses_paper_b_sparse_joint_delays(self):
        history = ObservationHistory()

        for step in range(7):
            q = DEFAULT_Q_POLICY + float(step)
            qd = np.ones(ACTION_DIM, dtype=np.float32) * float(step)
            history.update_joint_state(q, qd)

        expected_errors = np.concatenate(
            [
                np.ones(ACTION_DIM, dtype=np.float32) * 4.0,
                np.ones(ACTION_DIM, dtype=np.float32) * 2.0,
                np.ones(ACTION_DIM, dtype=np.float32) * 0.0,
            ]
        )
        expected_velocities = expected_errors.copy()

        np.testing.assert_allclose(history.joint_position_error_history, expected_errors)
        np.testing.assert_allclose(history.joint_velocity_history, expected_velocities)

    def test_policy_robot_mapping_matches_training_dof_order(self):
        values = np.arange(ACTION_DIM, dtype=np.float32)
        expected_policy = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.float32)
        expected_robot = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.float32)
        np.testing.assert_allclose(robot_to_policy_order(values), expected_policy)
        np.testing.assert_allclose(policy_to_robot_order(values), expected_robot)

    def test_foot_position_mapping_matches_training_foot_order(self):
        values = np.arange(12, dtype=np.float32)
        expected_policy = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.float32)

        np.testing.assert_allclose(robot_foot_positions_to_policy_order(values), expected_policy)
        np.testing.assert_allclose(policy_foot_positions_to_robot_order(values), expected_policy)

    def test_lcm_policy_source_uses_robot_policy_order_conversions(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts/rl_lcm_policy.py").read_text(encoding="utf-8")

        self.assertIn("robot_to_policy_order(msg.q)", source)
        self.assertIn("robot_to_policy_order(msg.qd)", source)
        self.assertIn("robot_foot_positions_to_policy_order", source)
        self.assertIn("policy_to_robot_order(action_policy)", source)
        self.assertIn("policy_to_robot_order(target_q_policy)", source)

    def test_resolve_checkpoint_accepts_run_directory(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "train" / "201852.132488" / "checkpoints"
            checkpoint.mkdir(parents=True)
            estimator = checkpoint / "estimator_latest.jit"
            body = checkpoint / "body_latest.jit"
            estimator.write_text("estimator")
            body.write_text("body")

            resolved_estimator, resolved_body = resolve_checkpoint(Path(tmp))

            self.assertEqual(resolved_estimator, estimator)
            self.assertEqual(resolved_body, body)


if __name__ == "__main__":
    unittest.main()
