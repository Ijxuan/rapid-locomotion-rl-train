from typing import Union

from params_proto.neo_proto import Meta

from mini_gym.envs.base.paper_b_defaults import apply_paper_b_mini_cheetah_defaults
from mini_gym.envs.base.legged_robot_config import Cfg


def config_mini_cheetah(Cnfg: Union[Cfg, Meta]):
    apply_paper_b_mini_cheetah_defaults(Cnfg)

    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.3]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': -0.8,  # [rad]
        'RL_thigh_joint': -0.8,  # [rad]
        'FR_thigh_joint': -0.8,  # [rad]
        'RR_thigh_joint': -0.8,  # [rad]

        'FL_calf_joint': 1.62,  # [rad]
        'RL_calf_joint': 1.62,  # [rad]
        'FR_calf_joint': 1.62,  # [rad]
        'RR_calf_joint': 1.62  # [rad]
    }

    _ = Cnfg.control
    _.control_type = 'P'
    _.stiffness = {'joint': 17., 'calf_joint': 34.}  # [N*m/rad]
    _.damping = {'joint': 0.4, 'calf_joint': 0.8}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.1
    _.hip_scale_reduction = 1.0
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["calf"]
    _.terminate_after_contacts_on = ["base", "trunk"]
    _.collapse_fixed_joints = False
    _.self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
    _.flip_visual_attachments = False
    _.fix_base_link = False

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.30

    _ = Cnfg.rewards.scales
    _.torques = -6e-4
    _.collision = -1.0
    _.dof_pos_limits = 0.0
    _.orientation = -3.0
    _.base_height = 0.0

    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 50

    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False

    _ = Cnfg.env
    _.num_observations = 142
    _.num_privileged_obs = 11
    _.estimator_output_dim = 11
    _.num_observation_history = 1
    _.use_paper_b_observation = True
    _.observe_vel = False
    _.num_envs = 800

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = True
    _.paper_b_command_curriculum = True
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.num_commands = 3
    _.lin_vel_x = [-0.5, 1.0]
    _.lin_vel_y = [-1.0, 1.0]
    _.ang_vel_yaw = [-1.0, 1.0]
    _.paper_b_vx_initial = [-0.5, 1.0]
    _.paper_b_vx_final = [-1.75, 3.5]
    _.zero_command_probability = 0.1

    _ = Cnfg.domain_rand
    _.randomize_base_mass = False
    _.added_mass_range = [-1, 3]
    _.push_robots = False
    _.max_push_vel_xy = 0.5
    _.randomize_friction = True
    _.friction_range = [0.4, 1.0]
    _.randomize_restitution = False
    _.restitution_range = [0.0, 1.0]
    _.restitution = 0.5  # default terrain restitution
    _.randomize_com_displacement = False
    _.com_displacement_range = [-0.1, 0.1]
    _.randomize_motor_strength = False
    _.motor_strength_range = [0.9, 1.1]
    _.randomize_Kp_factor = False
    _.Kp_factor_range = [0.8, 1.3]
    _.randomize_Kd_factor = False
    _.Kd_factor_range = [0.5, 1.5]
    _.randomize_motor_friction = True
    _.motor_friction_haa_hfe_range = [0.0, 0.3]
    _.motor_friction_kfe_range = [0.1, 0.7]
    _.randomize_pd_gains = True
    _.Kp_noise_range = [-2.0, 2.0]
    _.Kd_noise_range = [-0.1, 0.1]
    _.randomize_foot_radius = False
    _.foot_radius_range = [0.006, 0.010]
    _.rand_interval_s = 6
