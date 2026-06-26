from typing import Union

from params_proto.neo_proto import Meta

from mini_gym.envs.base.legged_robot_config import Cfg


def config_mini_cheetah(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.32]  # x,y,z [m]
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
    _.stiffness = {'joint': 20.}  # [N*m/rad]
    _.damping = {'joint': 0.5}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf'
    _.foot_name = "calf"
    _.penalize_contacts_on = ["base", "trunk", "hip", "thigh"]
    _.terminate_after_contacts_on = ["base", "thigh"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    _.flip_visual_attachments = False
    _.fix_base_link = False

    _ = Cnfg.rewards
    # 奖励总和允许为负；低速调参建议保持 False，否则负惩罚会被裁掉，坏动作差异不明显。
    _.only_positive_rewards = False
    _.soft_dof_pos_limit = 0.9  # 关节位置软限位比例，越小越早触发 dof_pos_limits 惩罚。
    _.base_height_target = 0.30  # base_height 奖励的目标机身高度，单位 m。
    _.use_terminal_body_height = True  # 机身过低时直接终止 episode，用于防止趴地/膝行。
    _.terminal_body_height = 0.20  # 机身高度低于该值时终止，单位 m。
    _.stand_still_lin_threshold = 0.03  # |vxy_cmd| 小于该值进入 stand 静止模式。
    _.stand_still_yaw_threshold = 0.05  # |wz_cmd| 小于该值才允许进入 stand 静止模式。
    _.low_speed_threshold = 0.20  # |vxy_cmd| 小于该值视为低速四脚支撑区。
    _.walk_min_speed = 0.20  # 走步态区间下限，含该速度。
    _.walk_max_speed = 0.50  # 走步态区间上限，不含该速度。
    _.run_min_speed = 0.50  # 跑步态预留区间下限；当前命令采样不主动训练高速跑。
    _.crawl_lin_threshold = 0.20  # 旧 crawl 逻辑阈值，当前与 low_speed_threshold 对齐。
    _.crawl_base_height_target = 0.30  # 低速爬行时希望保持的最低机身高度。
    _.still_contact_force_threshold = 1.0  # 判断足端/小腿接触的接触力阈值，单位 N。
    _.use_virtual_foot_contact = True  # 用 calf 刚体加虚拟 toe 偏移估计有效足端接触。
    _.virtual_foot_offset = 0.19  # 虚拟 toe 相对 calf 局部坐标的向下偏移，单位 m。
    _.valid_foot_height_threshold = 0.035  # toe 低于该高度且 calf 有力，才算合法足端接触。
    _.illegal_calf_contact_time = 0.15  # 小腿非法接触持续超过该时间后终止，单位 s。
    _.crawl_min_contacts = 4  # 低速区最少支撑脚数量；设 4 表示四脚尽量贴地。
    _.crawl_swing_height_target = 0.025  # 低速区摆动脚目标离地高度，单位 m。
    _.crawl_max_swing_height = 0.05  # 低速区摆动脚超过该高度会额外惩罚，单位 m。
    _.crawl_max_air_time = 0.15  # 低速区单脚离地超过该时间会惩罚，单位 s。
    _.feet_air_time_target = 0.15  # 高速 gait 足端空中时间奖励目标；当前 feet_air_time scale 为 0。
    _.gait_swing_balance_ema_alpha = 0.02  # 高速摆动占空比 EMA 更新系数；当前 gait_swing_balance scale 为 0。
    _.walk_step_duration = 0.35  # 走步态每条腿的相位时长，完整 FR/RL/FL/RR 周期为 4 倍。
    _.walk_swing_sequence = ["FR", "RL", "FL", "RR"]  # 走步态期望摆动腿顺序；当前只记录诊断，不惩罚顺序。
    _.walk_swing_height_target = 0.03  # walk 目标摆动脚离地高度，单位 m。
    _.walk_swing_height_max = 0.06  # walk 摆动脚超过该高度会额外惩罚，单位 m。

    _ = Cnfg.rewards.scales
    # 下面大多数 scale 会在 legged_robot.py 中再乘以控制周期 dt；标注“每事件”的项已在函数内抵消 dt。
    # 正数是奖励，负数是惩罚，0 表示关闭。
    _.survival = 1.0  # 存活奖励；提高可鼓励活满 episode，但过高会压过步态质量项。
    _.termination = -20.0  # 非 timeout 终止惩罚；实际单次惩罚约为 scale * dt。
    _.tracking_lin_vel = 2.0  # 线速度跟踪奖励；提高会更追速度，但可能牺牲接触稳定性。
    _.tracking_ang_vel = 0.5  # yaw 角速度跟踪奖励；当前 wz 命令固定 0，主要抑制自转。
    _.lin_vel_z = -2.0  # 惩罚机身 z 方向速度，降低上下跳动。
    _.ang_vel_xy = -0.1  # 惩罚 roll/pitch 角速度，降低机身晃动。
    _.torques = -0.0002  # 惩罚关节力矩，数值过大会导致动作无力。
    _.dof_acc = -2.5e-7  # 惩罚关节加速度，抑制高频抖动。
    _.dof_pos_limits = -10.0  # 惩罚接近/超过关节限位，防止极端关节姿态。
    _.orientation = -2.0  # 惩罚 roll/pitch 倾斜，增强机身水平稳定。
    _.base_height = -10.0  # 惩罚机身高度偏离 base_height_target。
    _.collision = -5.0  # 惩罚 base/trunk/hip/thigh 等非足端碰撞。
    _.feet_air_time = 0.0  # 足端空中时间奖励；低速训练建议关闭，避免原地抬腿。
    _.gait_swing_balance = 0.0  # 高速 gait 摆动均衡项；当前走步阶段关闭。
    _.action_rate = -0.03  # 惩罚一阶动作变化，越大动作越平滑但响应越慢。
    _.action_smoothness_2 = -0.03  # 惩罚二阶动作变化，抑制动作加速度和突变。
    _.stand_still = -2.0  # stand 模式惩罚偏离默认关节角，增强 0 速站立。
    _.feet_contact_still = -2.0  # stand 模式惩罚缺少足端接触，鼓励四脚着地。
    _.still_base_vel = -2.0  # stand 模式惩罚 base 平移和 yaw 速度，减少漂移。
    _.still_dof_vel = -0.1  # stand 模式惩罚关节速度，减少站立抖腿。
    _.still_action = -0.1  # stand 模式惩罚非零 action，鼓励回默认站姿。
    _.crawl_feet_contact = 0.0  # 旧低速缺接触惩罚；当前用 crawl_min_contacts 代替。
    _.crawl_min_contacts = -2.0  # 低速区接触脚少于 crawl_min_contacts 时惩罚。
    _.crawl_foot_slip = -0.05  # 低速区接触脚 xy 滑移惩罚；过大可能诱导抬脚躲惩罚。
    _.crawl_base_vel = 0.0  # 旧低速额外速度惩罚；当前依赖 tracking_lin_vel。
    _.crawl_action_rate = 0.0  # 旧低速额外动作变化惩罚；当前依赖通用 action_rate。
    _.crawl_dof_vel = -0.02  # 低速区关节速度惩罚，抑制低速乱动。
    _.crawl_action = -0.02  # 低速区 action 幅值惩罚，动作更保守。
    _.crawl_base_height = 0.0  # 低速区低机身惩罚；当前依赖 base_height 和终止高度。
    _.crawl_swing_clearance = -10.0  # 低速区摆动脚高度惩罚，避免 0.1/0.15 明显抬腿。
    _.crawl_excess_air_time = -1.0  # 低速区离地时间过长惩罚。
    _.walk_exact_three_contacts = -4.0  # walk 区惩罚接触脚数量偏离 3，鼓励单脚摆动。
    _.walk_sequence_contact = 0.0  # walk 区换腿顺序惩罚；当前关闭，只保留顺序诊断曲线。
    _.walk_swing_clearance = -10.0  # walk 区摆动脚高度惩罚，目标接近 walk_swing_height_target。
    _.walk_stance_slip = -0.05  # walk 区三条支撑腿滑移惩罚。
    _.walk_swing_progress = 0.5  # walk 区目标摆动腿沿命令方向前摆奖励。
    _.walk_repeat_last_leg = -2.0  # walk 区连续两次抬同一条腿的每事件惩罚，防止固定抬一条腿。
    _.walk_repeat_recent_leg = -0.5  # walk 区当前抬腿与上上次相同的每事件小惩罚，鼓励短期换腿。
    _.walk_swing_leg_coverage = 1.0  # walk 区最近 4 次抬腿覆盖 4 条腿的每事件奖励，不规定具体顺序。
    _.illegal_calf_contact = -10.0  # 非法小腿接触惩罚，配合 illegal_calf_contact_time 终止。

    _ = Cnfg.terrain
    _.mesh_type = 'plane'
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = False
    _.border_size = 50

    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False

    _ = Cnfg.env
    _.num_observations = 42
    _.observe_vel = False
    _.num_envs = 4000

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 4.0
    _.command_curriculum = False
    _.use_low_speed_command_sampler = True
    _.zero_command_probability = 0.20
    _.low_command_probability = 0.20
    _.walk_command_probability = 0.60
    _.low_command_values = [0.05, 0.10, 0.15]
    _.low_command_jitter = 0.01
    _.walk_command_range = [0.20, 0.50]
    _.crawl_command_probability = 0.0
    _.crawl_command_values = [0.1, 0.2]
    _.crawl_command_jitter = 0.02
    _.gait_command_range = [0.50, 0.70]
    _.lin_vel_deadband = 0.03
    _.yaw_vel_deadband = 0.05
    _.gait_lin_threshold = 0.50
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [0.0, 0.50]
    _.lin_vel_y = [0.0, 0.0]
    _.ang_vel_yaw = [0.0, 0.0]
    _.limit_vel_x = [0.0, 0.50]
    _.limit_vel_y = [0.0, 0.0]
    _.limit_vel_yaw = [0.0, 0.0]

    _ = Cnfg.domain_rand
    _.randomize_base_mass = True
    _.added_mass_range = [-0.5, 0.5]
    _.push_robots = False
    _.max_push_vel_xy = 0.5
    _.randomize_friction = True
    _.friction_range = [0.5, 1.25]
    _.randomize_restitution = False
    _.restitution_range = [0.0, 0.0]
    _.restitution = 0.0
    _.randomize_com_displacement = True
    _.com_displacement_range = [-0.02, 0.02]
    _.randomize_motor_strength = True
    _.motor_strength_range = [0.95, 1.05]
    _.randomize_Kp_factor = False
    _.Kp_factor_range = [0.8, 1.3]
    _.randomize_Kd_factor = False
    _.Kd_factor_range = [0.5, 1.5]
    _.rand_interval_s = 6

    Cnfg.noise.add_noise = False
