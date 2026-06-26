import isaacgym

assert isaacgym
import torch
import numpy as np

from mini_gym.envs import *
from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm


def load_env(headless=False):
    # 准备 Mini Cheetah 环境配置；后面会再用训练 run 里的参数覆盖一次。
    config_mini_cheetah(Cfg)

    from ml_logger import logger

    print(logger.glob("*"))
    print(logger.prefix)

    # 从当前 logger 指向的 run 读取训练时保存的参数，保证回放时网络结构、观测维度等与训练一致。
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

    # 回放/评估时关闭域随机化，避免把策略问题和随机扰动混在一起判断。
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

    # 关键可调项：禁止评估过程中自动重采样 command。
    # 这里设成 1000s，并且主循环里每一步都会重新写入固定命令，确保测试的是指定 vx。
    Cfg.commands.resampling_time = 1000.0

    # 评估只跑 1 个环境，便于渲染和读取第 0 个机器人的诊断量。
    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    
    Cfg.viewer.pos = [1.5, -1.5, 1.0]
    Cfg.viewer.lookat = [0.0, 0.0, 0.3]
    # 单环境评估不需要训练时的大接触缓冲区；显存不足时可以继续调小。
    Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 18
    Cfg.sim.physx.default_buffer_size_multiplier = 1

    from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # 加载策略网络；play.py 当前读取的是最新 run 下的 ac_weights_last.pt，不是 .jit 导出文件。
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

    # 默认选择 runs/rapid-locomotion 下修改时间最新的 run。
    # 如果要评估指定 run，可以把 recent_runs[-1] 替换成具体路径。
    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    print(recent_runs)

    logger.configure(Path(recent_runs[-1]).resolve())

    env, policy = load_env(headless=headless)

    # 关键可调项：低速评估命令列表，单位 m/s。
    # 当前用于分别检查 0 速站立、0.1/0.2 准静态低速、0.3 起步态切换。
    x_vel_commands = [0.0, 0.1, 0.2, 0.3]

    # 关键可调项：每个速度连续测试 20s；想快速看效果可以改成 5.0 或 10.0。
    num_eval_steps = int(20.0 / env.dt)

    # 记录每个速度下的前向速度、有效足端接触数，以及 0 速时的 12 个关节位置。
    measured_x_vels = np.zeros((len(x_vel_commands), num_eval_steps))
    contact_counts = np.zeros_like(measured_x_vels)
    zero_command_joint_positions = np.zeros((num_eval_steps, 12))

    # 打印一行量化指标，方便和视频/渲染观察相互校验。
    print("cmd  vx_mae  support_ratio  illegal_calf  max_toe_z  min_base_z  dof_vel_rms")
    for command_index, x_vel_cmd in enumerate(x_vel_commands):
        obs = env.reset()

        # reset 后立即写入本轮固定命令，并重新计算观测，避免第一步仍使用环境默认 command。
        env.commands[:, :3] = torch.tensor([x_vel_cmd, 0.0, 0.0], device=env.device)
        env.env.compute_observations()

        # 清空历史观测，避免上一组速度命令残留到下一组测试。
        env.obs_history[:] = 0.
        obs = env.get_observations()

        illegal_contact_steps = 0
        maximum_toe_height = -np.inf
        minimum_base_height = np.inf
        dof_velocity_sq_sum = 0.
        for i in tqdm(range(num_eval_steps), desc=f"vx={x_vel_cmd:.1f}"):
            # 每一步都重写固定命令，防止环境内部 reset/resample 逻辑改变测试命令。
            env.commands[:, :3] = torch.tensor([x_vel_cmd, 0.0, 0.0], device=env.device)
            with torch.no_grad():
                actions = policy(obs)
            obs, rew, done, info = env.step(actions)

            # 这些诊断量依赖当前低速奖励重构中的虚拟足端/非法小腿接触逻辑。
            measured_x_vels[command_index, i] = env.base_lin_vel[0, 0].item()
            contact_counts[command_index, i] = env.valid_foot_contacts[0].float().sum().item()
            illegal_contact_steps += int(torch.any(env.illegal_calf_contacts[0]).item())
            maximum_toe_height = max(maximum_toe_height, env.virtual_foot_clearances[0].max().item())
            minimum_base_height = min(minimum_base_height, env.root_states[0, 2].item())
            dof_velocity_sq_sum += torch.mean(torch.square(env.dof_vel[0])).item()
            if command_index == 0:
                zero_command_joint_positions[i] = env.dof_pos[0, :].cpu()

        # 0 速要求四脚支撑；0.1/0.2/0.3 允许最多一条腿摆动，所以用三脚支撑作为最低验收线。
        required_contacts = 4 if x_vel_cmd == 0.0 else 3
        support_ratio = np.mean(contact_counts[command_index] >= required_contacts)

        # vx_mae 越小表示速度跟踪越准；但低速阶段还要同时看 support_ratio 和 illegal_calf。
        vx_mae = np.mean(np.abs(measured_x_vels[command_index] - x_vel_cmd))
        illegal_contact_ratio = illegal_contact_steps / num_eval_steps
        dof_velocity_rms = np.sqrt(dof_velocity_sq_sum / num_eval_steps)
        print(f"{x_vel_cmd:3.1f}  {vx_mae:6.3f}  {support_ratio:13.3f}  "
              f"{illegal_contact_ratio:12.3f}  {maximum_toe_height:10.3f}  "
              f"{minimum_base_height:10.3f}  {dof_velocity_rms:11.3f}")

    from matplotlib import pyplot as plt
    time_axis = np.linspace(0, num_eval_steps * env.dt, num_eval_steps)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    # 图 1：实际 vx 与目标 vx；检查是否漂移、跟不上或过冲。
    for command_index, x_vel_cmd in enumerate(x_vel_commands):
        axs[0].plot(time_axis, measured_x_vels[command_index], label=f"Measured {x_vel_cmd:.1f}")
        axs[0].plot(time_axis, np.full(num_eval_steps, x_vel_cmd), linestyle="--", linewidth=1)
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    # 图 2：有效足端接触数量；0 速理想值接近 4，低速爬行至少应长期保持 3。
    for command_index, x_vel_cmd in enumerate(x_vel_commands):
        axs[1].plot(time_axis, contact_counts[command_index], label=f"{x_vel_cmd:.1f} m/s")
    axs[1].legend()
    axs[1].set_title("Valid Foot Contacts")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Contact Count")

    # 图 3：只画 0 速时的关节角；用于判断是否还存在周期性抬腿/抖腿。
    axs[2].plot(time_axis, zero_command_joint_positions, linestyle="-")
    axs[2].set_title("Zero-command Joint Positions")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_mc(headless=False)
