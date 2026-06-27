import unittest
from types import SimpleNamespace

import numpy as np

from mini_gym.envs.base.paper_b_defaults import apply_paper_b_mini_cheetah_defaults
from mini_gym.envs.base.paper_b_observation import (
    ACTION_DIM,
    ACTOR_INPUT_DIM,
    ESTIMATOR_OUTPUT_DIM,
    ESTIMATOR_TARGET_DIM,
    ESTIMATOR_TARGET_SLICES,
    OBSERVATION_SLICES,
    OBS_DIM,
    PAPER_B_ACTION_SCALE,
    PAPER_B_KD,
    PAPER_B_KP,
    action_to_desired_joint_positions,
    estimator_target_components,
    layout_width,
    observation_components,
)


def fake_cfg():
    return SimpleNamespace(
        env=SimpleNamespace(),
        control=SimpleNamespace(),
        sim=SimpleNamespace(dt=0.005),
        commands=SimpleNamespace(),
        rewards=SimpleNamespace(scales=SimpleNamespace()),
        domain_rand=SimpleNamespace(),
        init_state=SimpleNamespace(),
    )


class PaperBLayoutTest(unittest.TestCase):
    def test_observation_and_estimator_dimensions(self):
        self.assertEqual(OBS_DIM, 142)
        self.assertEqual(ESTIMATOR_TARGET_DIM, 11)
        self.assertEqual(ESTIMATOR_OUTPUT_DIM, 11)
        self.assertEqual(ACTOR_INPUT_DIM, 153)
        self.assertEqual(layout_width(OBSERVATION_SLICES), OBS_DIM)
        self.assertEqual(layout_width(ESTIMATOR_TARGET_SLICES), ESTIMATOR_TARGET_DIM)

    def test_key_observation_slices(self):
        self.assertEqual(OBSERVATION_SLICES["base_quat"], slice(0, 4))
        self.assertEqual(OBSERVATION_SLICES["base_ang_vel"], slice(4, 7))
        self.assertEqual(OBSERVATION_SLICES["dof_pos"], slice(7, 19))
        self.assertEqual(OBSERVATION_SLICES["dof_vel"], slice(19, 31))
        self.assertEqual(OBSERVATION_SLICES["previous_desired_joint_positions"], slice(31, 55))
        self.assertEqual(OBSERVATION_SLICES["joint_position_error_history"], slice(55, 91))
        self.assertEqual(OBSERVATION_SLICES["joint_velocity_history"], slice(91, 127))
        self.assertEqual(OBSERVATION_SLICES["foot_positions_body"], slice(127, 139))
        self.assertEqual(OBSERVATION_SLICES["commands"], slice(139, 142))

    def test_action_maps_to_nominal_plus_point_one_delta(self):
        nominal = np.arange(ACTION_DIM, dtype=np.float32)
        action = np.ones(ACTION_DIM, dtype=np.float32)
        desired = action_to_desired_joint_positions(action, nominal)
        np.testing.assert_allclose(desired, nominal + PAPER_B_ACTION_SCALE)

    def test_observation_components_concatenate_to_paper_b_layout(self):
        parts = observation_components(
            np.full((2, 4), 1.0, dtype=np.float32),
            np.full((2, 3), 2.0, dtype=np.float32),
            np.full((2, 12), 3.0, dtype=np.float32),
            np.full((2, 12), 4.0, dtype=np.float32),
            np.full((2, 24), 5.0, dtype=np.float32),
            np.full((2, 36), 6.0, dtype=np.float32),
            np.full((2, 36), 7.0, dtype=np.float32),
            np.full((2, 12), 8.0, dtype=np.float32),
            np.full((2, 3), 9.0, dtype=np.float32),
        )

        obs = np.concatenate(parts, axis=-1)

        self.assertEqual(obs.shape, (2, OBS_DIM))
        np.testing.assert_allclose(obs[:, OBSERVATION_SLICES["base_quat"]], 1.0)
        np.testing.assert_allclose(obs[:, OBSERVATION_SLICES["previous_desired_joint_positions"]], 5.0)
        np.testing.assert_allclose(obs[:, OBSERVATION_SLICES["joint_position_error_history"]], 6.0)
        np.testing.assert_allclose(obs[:, OBSERVATION_SLICES["foot_positions_body"]], 8.0)
        np.testing.assert_allclose(obs[:, OBSERVATION_SLICES["commands"]], 9.0)

    def test_estimator_target_components_concatenate_to_expected_layout(self):
        parts = estimator_target_components(
            np.full((2, 3), 1.0, dtype=np.float32),
            np.full((2, 4), 2.0, dtype=np.float32),
            np.full((2, 4), 3.0, dtype=np.float32),
        )

        target = np.concatenate(parts, axis=-1)

        self.assertEqual(target.shape, (2, ESTIMATOR_TARGET_DIM))
        np.testing.assert_allclose(target[:, ESTIMATOR_TARGET_SLICES["base_lin_vel"]], 1.0)
        np.testing.assert_allclose(target[:, ESTIMATOR_TARGET_SLICES["foot_height"]], 2.0)
        np.testing.assert_allclose(target[:, ESTIMATOR_TARGET_SLICES["contact_probability"]], 3.0)

    def test_common_defaults_apply_paper_b_training_shape(self):
        cfg = fake_cfg()
        apply_paper_b_mini_cheetah_defaults(cfg)

        self.assertEqual(cfg.env.num_envs, 800)
        self.assertEqual(cfg.env.num_observations, OBS_DIM)
        self.assertEqual(cfg.env.num_privileged_obs, ESTIMATOR_TARGET_DIM)
        self.assertEqual(cfg.env.estimator_output_dim, ESTIMATOR_OUTPUT_DIM)
        self.assertEqual(cfg.env.num_observation_history, 1)
        self.assertTrue(cfg.env.use_paper_b_observation)

        self.assertEqual(cfg.control.action_scale, PAPER_B_ACTION_SCALE)
        self.assertEqual(cfg.control.hip_scale_reduction, 1.0)
        self.assertEqual(cfg.control.stiffness, {"joint": PAPER_B_KP})
        self.assertEqual(cfg.control.damping, {"joint": PAPER_B_KD})
        self.assertAlmostEqual(cfg.control.decimation * cfg.sim.dt, 0.01)

        self.assertEqual(cfg.commands.num_commands, 3)
        self.assertEqual(cfg.commands.paper_b_vx_initial, [-0.5, 1.0])
        self.assertEqual(cfg.commands.paper_b_vx_final, [-1.75, 3.5])
        self.assertEqual(cfg.commands.zero_command_probability, 0.1)

        self.assertTrue(cfg.rewards.use_paper_b_reward)
        self.assertFalse(cfg.rewards.only_positive_rewards)
        self.assertEqual(cfg.rewards.scales.tracking_lin_vel, 3.0)
        self.assertEqual(cfg.rewards.scales.feet_clearance, -15.0)
        self.assertEqual(cfg.rewards.paper_b_termination_penalty, -10.0)

        self.assertEqual(cfg.domain_rand.friction_range, [0.4, 1.0])
        self.assertEqual(cfg.domain_rand.motor_friction_haa_hfe_range, [0.0, 0.3])
        self.assertEqual(cfg.domain_rand.motor_friction_kfe_range, [0.1, 0.7])


if __name__ == "__main__":
    unittest.main()
