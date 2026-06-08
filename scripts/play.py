import isaacgym

assert isaacgym
import torch
import numpy as np

from mini_gym.envs import *
from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm


def quat_xyzw_to_yaw(quat_xyzw):
    # Isaac Gym 的 root state 四元数顺序是 x, y, z, w，这里只提取机身 yaw 角。
    x, y, z, w = quat_xyzw
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def load_env(headless=False):
    # prepare environment
    config_mini_cheetah(Cfg)

    from ml_logger import logger

    print(logger.glob("*"))
    print(logger.prefix)

    params = logger.load_pkl('parameters.pkl')

    if 'kwargs' in params[0]:
        deps = params[0]['kwargs']

        from mini_gym_learn.ppo.ppo import PPO_Args
        from mini_gym_learn.ppo.actor_critic import AC_Args
        from mini_gym_learn.ppo import RunnerArgs

        AC_Args._update(deps)
        PPO_Args._update(deps)
        RunnerArgs._update(deps)
        Cfg.terrain._update(deps)
        Cfg.commands._update(deps)
        Cfg.normalization._update(deps)
        Cfg.env._update(deps)
        Cfg.domain_rand._update(deps)
        Cfg.rewards._update(deps)
        Cfg.reward_scales._update(deps)
        Cfg.perception._update(deps)
        Cfg.domain_rand._update(deps)
        Cfg.control._update(deps)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 18
    Cfg.sim.physx.default_buffer_size_multiplier = 1

    from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from mini_gym_learn.ppo.actor_critic import ActorCritic

    actor_critic = ActorCritic(
        num_obs=Cfg.env.num_observations,
        num_privileged_obs=Cfg.env.num_privileged_obs,
        num_obs_history=Cfg.env.num_observations * \
                        Cfg.env.num_observation_history,
        num_actions=Cfg.env.num_actions)

    print(logger.prefix)
    print(logger.glob("*"))
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    policy = actor_critic.act_inference

    return env, policy


def play_mc(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from mini_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    print(recent_runs)

    logger.configure(Path(recent_runs[-1]).resolve())

    env, policy = load_env(headless=headless)

    num_eval_steps = 500
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    yaw_angles = np.zeros(num_eval_steps)

    base_env = env.env
    # 只画右前和右后两个 abad/hip 横摆髋关节，避免 12 个关节混在一起看不清。
    # FR_hip_joint: 正数为右前腿内收，负数为右前腿外摆。
    # RR_hip_joint: 正数为右后腿内收，负数为右后腿外摆。
    right_hip_joint_names = ["FR_hip_joint", "RR_hip_joint"]
    right_hip_joint_indices = [
        base_env.dof_names.index(name) for name in right_hip_joint_names
    ]
    right_hip_joint_positions = np.zeros((num_eval_steps, len(right_hip_joint_names)))

    obs = env.reset()

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        yaw_angles[i] = quat_xyzw_to_yaw(base_env.root_states[0, 3:7].cpu().numpy())
        right_hip_joint_positions[i] = (
            base_env.dof_pos[0, right_hip_joint_indices].cpu().numpy()
        )

    # 画图时把 yaw 展开后转成角度制，避免跨过 +/-pi 时曲线突然跳变。
    yaw_degrees = np.rad2deg(np.unwrap(yaw_angles))
    right_hip_joint_degrees = np.rad2deg(right_hip_joint_positions)

    # 绘制前向速度、机身 yaw，以及右前/右后两个髋关节角度。
    from matplotlib import pyplot as plt
    time_axis = np.linspace(0, num_eval_steps * env.dt, num_eval_steps)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(time_axis, measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(time_axis, target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(time_axis, yaw_degrees, color="tab:green", linestyle="-", label="Body yaw")
    axs[1].legend()
    axs[1].set_title("Body Yaw Angle")
    axs[1].set_ylabel("Yaw (deg)")

    right_hip_colors = ["tab:blue", "tab:orange"]
    for joint_i, joint_name in enumerate(right_hip_joint_names):
        axs[2].plot(
            time_axis,
            right_hip_joint_degrees[:, joint_i],
            color=right_hip_colors[joint_i],
            linestyle="-",
            label=joint_name,
        )
    axs[2].legend()
    axs[2].set_title("Right Hip Joint Angles")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Joint Angle (deg)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_mc(headless=False)
