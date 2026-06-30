import isaacgym

assert isaacgym

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

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

    if isinstance(params, (list, tuple)) and params and "kwargs" in params[0]:
        deps = params[0]["kwargs"]
        AC_Args._update(deps)
        PPO_Args._update(deps)
        RunnerArgs._update(deps)
        for group_name in ("terrain", "commands", "normalization", "env", "domain_rand", "rewards", "control"):
            getattr(Cfg, group_name)._update(deps)
        return

    raise RuntimeError(f"unsupported parameters.pkl format: {type(params)}")


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

    from ml_logger import logger

    params = logger.load_pkl("parameters.pkl")
    apply_saved_parameters(params)
    disable_eval_randomization()

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
    return env, policy


def latest_run_dir() -> Path:
    runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError("no runs found under runs/rapid-locomotion")
    return Path(runs[-1]).resolve()


def play_mc(headless=True):
    from ml_logger import logger

    run_dir = latest_run_dir()
    print(run_dir)

    logger.configure(run_dir)
    env, policy = load_env(run_dir, headless=headless)
    base_env = env.env

    num_eval_steps = 500
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    yaw_angles = np.zeros(num_eval_steps)

    right_hip_joint_names = ["FR_hip_joint", "RR_hip_joint"]
    right_hip_joint_indices = [base_env.dof_names.index(name) for name in right_hip_joint_names]
    right_hip_joint_positions = np.zeros((num_eval_steps, len(right_hip_joint_names)))

    obs = env.reset()
    base_env.commands[:, 0] = x_vel_cmd
    base_env.commands[:, 1] = y_vel_cmd
    base_env.commands[:, 2] = yaw_vel_cmd
    base_env.compute_observations()
    env.obs_history[:, :] = 0.0
    obs = {"obs": base_env.obs_buf.clone(), "privileged_obs": base_env.privileged_obs_buf, "obs_history": env.obs_history}

    for i in tqdm(range(num_eval_steps)):
        base_env.commands[:, 0] = x_vel_cmd
        base_env.commands[:, 1] = y_vel_cmd
        base_env.commands[:, 2] = yaw_vel_cmd
        actions = policy(obs)
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = base_env.base_lin_vel[0, 0]
        yaw_angles[i] = quat_xyzw_to_yaw(base_env.root_states[0, 3:7].cpu().numpy())
        right_hip_joint_positions[i] = base_env.dof_pos[0, right_hip_joint_indices].cpu().numpy()

    yaw_degrees = np.rad2deg(np.unwrap(yaw_angles))
    right_hip_joint_degrees = np.rad2deg(right_hip_joint_positions)

    from matplotlib import pyplot as plt

    time_axis = np.linspace(0, num_eval_steps * base_env.dt, num_eval_steps)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(time_axis, measured_x_vels, color="black", linestyle="-", label="Measured")
    axs[0].plot(time_axis, target_x_vels, color="black", linestyle="--", label="Desired")
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


if __name__ == "__main__":
    play_mc(headless=False)
