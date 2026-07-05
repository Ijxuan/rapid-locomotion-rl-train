"""Default settings for the Ji et al. RA-L 2022 reproduction task."""

from __future__ import annotations

from .paper_b_observation import (
    ESTIMATOR_OUTPUT_DIM,
    ESTIMATOR_TARGET_DIM,
    OBS_DIM,
    PAPER_B_ACTION_SCALE,
    PAPER_B_KD,
    PAPER_B_KP,
    PAPER_B_POLICY_DT,
)


def apply_paper_b_common_defaults(cfg):
    """Apply Paper B defaults to a Cfg-like object."""
    cfg.env.num_observations = OBS_DIM
    cfg.env.num_privileged_obs = ESTIMATOR_TARGET_DIM
    cfg.env.num_observation_history = 1
    cfg.env.estimator_output_dim = ESTIMATOR_OUTPUT_DIM
    cfg.env.use_paper_b_observation = True

    cfg.control.action_scale = PAPER_B_ACTION_SCALE
    cfg.control.hip_scale_reduction = 1.0
    cfg.control.stiffness = {"joint": PAPER_B_KP}
    cfg.control.damping = {"joint": PAPER_B_KD}
    cfg.control.decimation = 4
    cfg.sim.dt = PAPER_B_POLICY_DT / cfg.control.decimation

    cfg.commands.num_commands = 3
    cfg.commands.heading_command = False
    cfg.commands.command_curriculum = True
    cfg.commands.paper_b_command_curriculum = True
    cfg.commands.lin_vel_x = [-0.5, 1.0]
    cfg.commands.lin_vel_y = [-1.0, 1.0]
    cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    cfg.commands.paper_b_vx_initial = [-0.5, 1.0]
    cfg.commands.paper_b_vx_final = [-1.75, 3.5]
    cfg.commands.paper_b_vx_curriculum_k = 0.002
    cfg.commands.paper_b_vx_curriculum_midpoint = 1000
    cfg.commands.zero_command_probability = 0.1

    cfg.rewards.use_paper_b_reward = True
    cfg.rewards.only_positive_rewards = False
    cfg.rewards.paper_b_reward_exponential_scale = 0.02
    cfg.rewards.paper_b_reward_gate_floor = 0.05
    cfg.rewards.paper_b_desired_foot_height = 0.09
    cfg.rewards.paper_b_termination_penalty = -10.0
    cfg.rewards.paper_b_stance_command_threshold = 0.1
    cfg.rewards.paper_b_airtime_clip = 0.3
    cfg.rewards.paper_b_airtime_max = 0.25
    cfg.rewards.paper_b_airtime_cap = 0.2
    cfg.rewards.moving_stand_still_command_threshold = 0.3
    cfg.rewards.moving_stand_still_velocity_threshold = 0.15
    for reward_name in (
        "lin_vel_z", "ang_vel_xy", "base_height", "collision", "feet_stumble", "action_rate", "stand_still",
        "dof_pos_limits", "dof_vel_limits", "torque_limits", "tracking_lin_vel_lat", "tracking_lin_vel_long",
        "feet_contact_forces",
    ):
        if hasattr(cfg.rewards.scales, reward_name):
            setattr(cfg.rewards.scales, reward_name, 0.0)
    cfg.rewards.scales.tracking_lin_vel = 3.0
    cfg.rewards.scales.tracking_ang_vel = 3.0
    cfg.rewards.scales.feet_air_time = 0.3
    cfg.rewards.scales.feet_slip = -0.08
    cfg.rewards.scales.feet_clearance = -15.0
    cfg.rewards.scales.orientation = -3.0
    cfg.rewards.scales.torques = -6e-4
    cfg.rewards.scales.dof_pos = -0.75
    cfg.rewards.scales.dof_vel = -6e-4
    cfg.rewards.scales.dof_acc = -0.02
    cfg.rewards.scales.action_smoothness_1 = -2.5
    cfg.rewards.scales.action_smoothness_2 = -1.2
    cfg.rewards.scales.base_motion = -1.5
    cfg.rewards.scales.moving_stand_still = -10.0
    cfg.rewards.scales.termination = -10.0
    cfg.normalization.clip_actions = 1.0

    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [0.4, 1.0]
    cfg.domain_rand.randomize_motor_friction = True
    cfg.domain_rand.motor_friction_haa_hfe_range = [0.0, 0.3]
    cfg.domain_rand.motor_friction_kfe_range = [0.1, 0.7]
    cfg.domain_rand.randomize_pd_gains = True
    cfg.domain_rand.Kp_noise_range = [-2.0, 2.0]
    cfg.domain_rand.Kd_noise_range = [-0.1, 0.1]
    cfg.domain_rand.randomize_foot_radius = False
    cfg.domain_rand.foot_radius_range = [0.006, 0.010]
    cfg.domain_rand.foot_position_noise_range = [[-0.010, 0.010], [-0.005, 0.005], [-0.020, 0.020]]
    cfg.domain_rand.obs_noise_dof_pos = [-0.05, 0.05]
    cfg.domain_rand.obs_noise_dof_vel = [-0.5, 0.5]
    cfg.domain_rand.obs_noise_base_quat = [-0.03, 0.03]
    cfg.domain_rand.obs_noise_foot_pos = [-0.03, 0.03]
    cfg.domain_rand.obs_noise_base_ang_vel = [-0.1, 0.1]

    cfg.init_state.paper_b_randomize_initial_state = True
    cfg.init_state.paper_b_reuse_previous_state_probability = 0.25
    cfg.init_state.paper_b_reuse_previous_state_min_steps = 100
    cfg.init_state.noise_quat = [-0.2, 0.2]
    cfg.init_state.noise_dof_pos = [-0.2, 0.2]
    cfg.init_state.noise_dof_vel = [-2.5, 2.5]
    cfg.init_state.noise_lin_vel_x = [-1.0, 1.0]
    cfg.init_state.noise_lin_vel_yz = [-0.5, 0.5]
    cfg.init_state.noise_ang_vel = [-0.7, 0.7]

    if hasattr(cfg, "asset"):
        cfg.asset.terminate_after_contacts_on = ["base", "trunk"]
        cfg.asset.foot_name = "calf"
        cfg.asset.collapse_fixed_joints = False


def apply_paper_b_mini_cheetah_defaults(cfg):
    apply_paper_b_common_defaults(cfg)
    cfg.env.num_envs = 800
    cfg.rewards.base_height_target = 0.30
