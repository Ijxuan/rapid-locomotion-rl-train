import isaacgym

assert isaacgym

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_gym import MINI_GYM_ROOT_DIR
from mini_gym.envs import *  # noqa: F401,F403
from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper
from scripts.rl_lcm_policy import resolve_checkpoint
from tqdm import tqdm

ESTIMATOR_JIT_NAME = "estimator_latest.jit"
BODY_JIT_NAME = "body_latest.jit"
DEFAULT_X_VEL_COMMANDS = [1.0, 2.0, -0.5]
DEFAULT_YAW_VEL_COMMANDS = [1.0]
SECONDS_PER_COMMAND = 1.5

# Set these to floats to test playback-only PD gains, e.g. 20.0 / 0.6.
# None keeps the gains saved by the training run.
EVAL_KP_OVERRIDE = None
EVAL_KD_OVERRIDE = None


def quat_xyzw_to_yaw(quat_xyzw):
    x, y, z, w = quat_xyzw
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def is_plain_value(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_plain_value(v) for v in value)
    return False


def update_proto_group(group, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, dict) and hasattr(group, key):
            target = getattr(group, key)
            if isinstance(target, dict):
                target.update({k: v for k, v in value.items() if is_plain_value(v)})
            else:
                update_proto_group(target, value)
        elif is_plain_value(value):
            setattr(group, key, value)


def apply_saved_parameters(params) -> None:
    from mini_gym_learn.ppo import RunnerArgs
    from mini_gym_learn.ppo.actor_critic import AC_Args
    from mini_gym_learn.ppo.ppo import PPO_Args

    if isinstance(params, dict):
        for name, values in params.get("Cfg", {}).items():
            if hasattr(Cfg, name) and isinstance(values, dict):
                update_proto_group(getattr(Cfg, name), values)
        AC_Args._update(params.get("AC_Args", {}))
        PPO_Args._update(params.get("PPO_Args", {}))
        RunnerArgs._update(params.get("RunnerArgs", {}))
        return

    if isinstance(params, (list, tuple)):
        for item in params:
            if isinstance(item, dict) and any(key in item for key in ("Cfg", "AC_Args", "PPO_Args", "RunnerArgs")):
                apply_saved_parameters(item)
                return

    if isinstance(params, (list, tuple)) and params and "kwargs" in params[0]:
        deps = params[0]["kwargs"]
        AC_Args._update(deps)
        PPO_Args._update(deps)
        RunnerArgs._update(deps)
        for group_name in ("terrain", "commands", "normalization", "env", "domain_rand", "rewards", "control"):
            getattr(Cfg, group_name)._update(deps)
        return

    raise RuntimeError(f"unsupported parameters.pkl format: {type(params)}")


def load_saved_parameters(run_dir: Path):
    params_path = run_dir / "parameters.pkl"
    with params_path.open("rb") as f:
        return pickle.load(f)


def disable_eval_randomization() -> None:
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_motor_friction = False
    Cfg.domain_rand.randomize_pd_gains = False
    Cfg.domain_rand.randomize_foot_radius = False
    Cfg.noise.add_noise = False
    Cfg.commands.resampling_time = 1000.0


def apply_eval_pd_overrides() -> None:
    if EVAL_KP_OVERRIDE is not None:
        Cfg.control.stiffness = {"joint": float(EVAL_KP_OVERRIDE)}
    if EVAL_KD_OVERRIDE is not None:
        Cfg.control.damping = {"joint": float(EVAL_KD_OVERRIDE)}


def load_jit_policy(run_dir: Path, device: str):
    estimator_path, body_path = resolve_checkpoint(run_dir / "checkpoints")
    estimator = torch.jit.load(str(estimator_path), map_location=device)
    body = torch.jit.load(str(body_path), map_location=device)
    estimator.eval()
    body.eval()

    def policy(obs_dict):
        obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
        obs = obs.to(device)
        with torch.no_grad():
            estimator_output = estimator(obs)
            return body(torch.cat((obs, estimator_output), dim=-1))

    print(f"loaded Paper B JIT policy ({ESTIMATOR_JIT_NAME}, {BODY_JIT_NAME}) from {estimator_path.parent}")
    return policy


def load_env(run_dir: Path, headless=False):
    config_mini_cheetah(Cfg)

    params = load_saved_parameters(run_dir)
    apply_saved_parameters(params)
    disable_eval_randomization()
    apply_eval_pd_overrides()

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 18
    Cfg.sim.physx.default_buffer_size_multiplier = 1

    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)
    policy = load_jit_policy(run_dir, env.device)
    print(f"play control gains: stiffness={Cfg.control.stiffness}, damping={Cfg.control.damping}")
    return env, policy


def latest_run_dir() -> Path:
    runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError("no runs found under runs/rapid-locomotion")
    return Path(runs[-1]).resolve()


def set_eval_command_and_observation(env, base_env, command):
    command_tensor = torch.tensor(command, dtype=torch.float, device=base_env.device).view(1, 3)
    base_env.commands[:, :3] = command_tensor
    base_env.compute_observations()
    return {"obs": base_env.obs_buf.clone(), "privileged_obs": base_env.privileged_obs_buf, "obs_history": env.obs_history}


def sync_eval_history(env, base_env):
    env.obs_history[:, :] = 0.0
    env.obs_history = torch.cat((env.obs_history[:, base_env.num_obs:], base_env.obs_buf), dim=-1)


def sync_eval_base_kinematics(base_env):
    base_env.base_quat[:] = base_env.root_states[:, 3:7]
    base_env.base_lin_vel[:] = quat_rotate_inverse(base_env.base_quat, base_env.root_states[:, 7:10])
    base_env.base_ang_vel[:] = quat_rotate_inverse(base_env.base_quat, base_env.root_states[:, 10:13])
    base_env.projected_gravity[:] = quat_rotate_inverse(base_env.base_quat, base_env.gravity_vec)


def set_eval_default_stance(env, base_env):
    env_ids = torch.arange(base_env.num_envs, dtype=torch.long, device=base_env.device)
    actor_indices = torch.tensor([
        base_env.gym.get_actor_index(base_env.envs[i], base_env.actor_handles[i], gymapi.DOMAIN_SIM)
        for i in range(base_env.num_envs)
    ], dtype=torch.int32, device=base_env.device)
    actor_indices_long = actor_indices.to(dtype=torch.long)

    base_state = base_env.base_init_state.clone().view(1, -1).repeat(base_env.num_envs, 1)
    base_state[:, :3] += base_env.env_origins[env_ids]
    base_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=base_env.device)
    base_state[:, 7:13] = 0.0
    dof_pos = base_env.default_dof_pos.repeat(base_env.num_envs, 1)

    def apply_state_tensors():
        base_env.root_states[actor_indices_long] = base_state
        base_env.dof_pos[env_ids] = dof_pos
        base_env.dof_vel[env_ids] = 0.0
        base_env.gym.set_actor_root_state_tensor_indexed(
            base_env.sim,
            gymtorch.unwrap_tensor(base_env.root_states),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )
        base_env.gym.set_dof_state_tensor_indexed(
            base_env.sim,
            gymtorch.unwrap_tensor(base_env.dof_state),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )

    apply_state_tensors()
    base_env.gym.simulate(base_env.sim)
    base_env.gym.fetch_results(base_env.sim, True)
    base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
    base_env.gym.refresh_dof_state_tensor(base_env.sim)
    base_env.gym.refresh_rigid_body_state_tensor(base_env.sim)
    base_env.gym.refresh_net_contact_force_tensor(base_env.sim)
    apply_state_tensors()
    base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
    base_env.gym.refresh_dof_state_tensor(base_env.sim)
    base_env.gym.refresh_rigid_body_state_tensor(base_env.sim)
    base_env.gym.refresh_net_contact_force_tensor(base_env.sim)

    base_env.actions[env_ids] = 0.0
    base_env.last_last_actions[env_ids] = 0.0
    base_env.last_actions[env_ids] = 0.0
    base_env.last_dof_vel[env_ids] = 0.0
    base_env.last_root_vel[env_ids] = 0.0
    base_env.feet_air_time[env_ids] = 0.0
    base_env.last_contacts[env_ids] = False
    base_env.episode_length_buf[env_ids] = 0
    base_env.reset_buf[env_ids] = 0
    base_env.time_out_buf[env_ids] = False
    if hasattr(base_env, "joint_pos_target"):
        base_env.joint_pos_target[env_ids] = base_env.default_dof_pos
    if hasattr(base_env, "_reset_paper_b_history_buffers"):
        base_env._reset_paper_b_history_buffers(env_ids)

    sync_eval_base_kinematics(base_env)
    base_env.compute_observations()
    sync_eval_history(env, base_env)


def reset_eval_to_default_stance(env, base_env, command):
    env.reset()
    set_eval_default_stance(env, base_env)
    return set_eval_command_and_observation(env, base_env, command)


def play_mc(headless=True, x_vel_commands=None, yaw_vel_commands=None, seconds_per_command=SECONDS_PER_COMMAND):
    from ml_logger import logger

    run_dir = latest_run_dir()
    print(run_dir)

    logger.configure(run_dir)
    env, policy = load_env(run_dir, headless=headless)
    base_env = env.env

    x_vel_commands = x_vel_commands if x_vel_commands is not None else DEFAULT_X_VEL_COMMANDS
    yaw_vel_commands = yaw_vel_commands if yaw_vel_commands is not None else DEFAULT_YAW_VEL_COMMANDS
    command_specs = [
        (float(x_vel_cmd), float(yaw_vel_cmd))
        for yaw_vel_cmd in yaw_vel_commands
        for x_vel_cmd in x_vel_commands
    ]
    num_eval_steps = max(1, int(seconds_per_command / base_env.dt))

    measured_x_vels = np.zeros((len(command_specs), num_eval_steps))
    measured_y_vels = np.zeros_like(measured_x_vels)
    measured_yaw_rates = np.zeros_like(measured_x_vels)
    target_x_vels = np.zeros_like(measured_x_vels)
    target_yaw_rates = np.zeros_like(measured_x_vels)
    yaw_angles = np.zeros_like(measured_x_vels)
    base_heights = np.zeros_like(measured_x_vels)
    action_rms = np.zeros_like(measured_x_vels)
    reset_counts = np.zeros(len(command_specs), dtype=int)

    right_hip_joint_names = ["FR_hip_joint", "RR_hip_joint"]
    right_hip_joint_indices = [base_env.dof_names.index(name) for name in right_hip_joint_names]
    right_hip_joint_positions = np.zeros((len(command_specs), num_eval_steps, len(right_hip_joint_names)))

    print("cmd_vx  cmd_yaw  mean_vx  vx_mae  mean_yaw  yaw_mae  min_base_z  yaw_end_deg  resets  action_rms")
    for command_index, (x_vel_cmd, yaw_vel_cmd) in enumerate(command_specs):
        command = (x_vel_cmd, 0.0, yaw_vel_cmd)
        obs = reset_eval_to_default_stance(env, base_env, command)
        target_x_vels[command_index, :] = x_vel_cmd
        target_yaw_rates[command_index, :] = yaw_vel_cmd
        reset_next_step = False

        for i in tqdm(range(num_eval_steps), desc=f"vx={x_vel_cmd:.2f}, yaw={yaw_vel_cmd:.2f}"):
            if reset_next_step:
                obs = reset_eval_to_default_stance(env, base_env, command)
                reset_next_step = False
            else:
                obs = set_eval_command_and_observation(env, base_env, command)
            actions = policy(obs)
            obs, rew, done, info = env.step(actions)

            measured_x_vels[command_index, i] = base_env.base_lin_vel[0, 0].item()
            measured_y_vels[command_index, i] = base_env.base_lin_vel[0, 1].item()
            measured_yaw_rates[command_index, i] = base_env.base_ang_vel[0, 2].item()
            yaw_angles[command_index, i] = quat_xyzw_to_yaw(base_env.root_states[0, 3:7].cpu().numpy())
            base_heights[command_index, i] = base_env.root_states[0, 2].item()
            action_rms[command_index, i] = torch.sqrt(torch.mean(torch.square(actions[0]))).item()
            right_hip_joint_positions[command_index, i] = base_env.dof_pos[0, right_hip_joint_indices].cpu().numpy()
            done_now = bool(done[0].item())
            reset_counts[command_index] += int(done_now)
            reset_next_step = done_now

        yaw_degrees = np.rad2deg(np.unwrap(yaw_angles[command_index]))
        vx_mae = np.mean(np.abs(measured_x_vels[command_index] - x_vel_cmd))
        yaw_mae = np.mean(np.abs(measured_yaw_rates[command_index] - yaw_vel_cmd))
        print(
            f"{x_vel_cmd:6.2f}  {yaw_vel_cmd:7.2f}  "
            f"{np.mean(measured_x_vels[command_index]):7.3f}  {vx_mae:6.3f}  "
            f"{np.mean(measured_yaw_rates[command_index]):8.3f}  {yaw_mae:7.3f}  "
            f"{np.min(base_heights[command_index]):10.3f}  {yaw_degrees[-1]:11.2f}  "
            f"{reset_counts[command_index]:6d}  {np.mean(action_rms[command_index]):10.3f}"
        )

    yaw_degrees = np.rad2deg(np.unwrap(yaw_angles, axis=1))
    right_hip_joint_degrees = np.rad2deg(right_hip_joint_positions)

    from matplotlib import pyplot as plt

    time_axis = np.linspace(0, num_eval_steps * base_env.dt, num_eval_steps)
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    command_labels = [f"vx {x_vel_cmd:.1f}, yaw {yaw_vel_cmd:.1f}" for x_vel_cmd, yaw_vel_cmd in command_specs]
    for command_index, label in enumerate(command_labels):
        line, = axs[0].plot(time_axis, measured_x_vels[command_index], linestyle="-", label=label)
        axs[0].plot(time_axis, target_x_vels[command_index], color=line.get_color(), linestyle="--", linewidth=0.8)
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_ylabel("Velocity (m/s)")

    for command_index, label in enumerate(command_labels):
        line, = axs[1].plot(time_axis, measured_yaw_rates[command_index], linestyle="-", label=label)
        axs[1].plot(time_axis, target_yaw_rates[command_index], color=line.get_color(), linestyle="--", linewidth=0.8)
    axs[1].legend()
    axs[1].set_title("Yaw Angular Velocity")
    axs[1].set_ylabel("Yaw Rate (rad/s)")

    for command_index, label in enumerate(command_labels):
        axs[2].plot(time_axis, yaw_degrees[command_index], linestyle="-", label=label)
    axs[2].legend()
    axs[2].set_title("Body Yaw Angle")
    axs[2].set_ylabel("Yaw (deg)")

    right_hip_colors = ["tab:blue", "tab:orange"]
    for joint_i, joint_name in enumerate(right_hip_joint_names):
        axs[3].plot(time_axis, right_hip_joint_degrees[0, :, joint_i],
                    color=right_hip_colors[joint_i], linestyle="-", label=f"{joint_name} @ {command_labels[0]}")
    axs[3].legend()
    axs[3].set_title("Right Hip Joint Angles, First Command")
    axs[3].set_ylabel("Joint Angle (deg)")

    for command_index, label in enumerate(command_labels):
        axs[4].plot(time_axis, base_heights[command_index], linestyle="-", label=label)
    axs[4].legend()
    axs[4].set_title("Base Height")
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Height (m)")

    plt.tight_layout()
    plt.show()


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without opening Isaac Gym viewer")
    parser.add_argument("--x-vel-commands", default=",".join(str(v) for v in DEFAULT_X_VEL_COMMANDS),
                        help="comma-separated forward velocity commands, e.g. 0,0.5,1,2")
    parser.add_argument("--yaw-vel-commands", default=",".join(str(v) for v in DEFAULT_YAW_VEL_COMMANDS),
                        help="comma-separated yaw angular velocity commands in rad/s, e.g. -1,-0.5,0,0.5,1")
    parser.add_argument("--seconds-per-command", type=float, default=SECONDS_PER_COMMAND)
    parser.add_argument("--kp", type=float, default=None, help="playback-only joint Kp override")
    parser.add_argument("--kd", type=float, default=None, help="playback-only joint Kd override")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.kp is not None:
        EVAL_KP_OVERRIDE = args.kp
    if args.kd is not None:
        EVAL_KD_OVERRIDE = args.kd
    play_mc(
        headless=args.headless,
        x_vel_commands=parse_float_list(args.x_vel_commands),
        yaw_vel_commands=parse_float_list(args.yaw_vel_commands),
        seconds_per_command=args.seconds_per_command,
    )
