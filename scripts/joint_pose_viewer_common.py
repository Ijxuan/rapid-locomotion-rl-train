#!/usr/bin/env python3
"""Shared Isaac Gym pose viewer helpers for Mini Cheetah joint checks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import isaacgym

assert isaacgym

from isaacgym import gymapi, gymtorch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_gym.envs import *  # noqa: F401,F403
from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
from mini_gym.utils.torch_utils import quat_rotate_inverse


def npfmt(x, precision: int = 4) -> str:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return np.array2string(arr, precision=precision, suppress_small=True)


def disable_randomization() -> None:
    for name in (
        "push_robots",
        "randomize_friction",
        "randomize_gravity",
        "randomize_restitution",
        "randomize_motor_offset",
        "randomize_motor_strength",
        "randomize_friction_indep",
        "randomize_ground_friction",
        "randomize_base_mass",
        "randomize_Kd_factor",
        "randomize_Kp_factor",
        "randomize_motor_friction",
        "randomize_pd_gains",
        "randomize_foot_radius",
        "randomize_joint_friction",
        "randomize_com_displacement",
    ):
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, False)
    Cfg.noise.add_noise = False


def configure_single_env() -> None:
    config_mini_cheetah(Cfg)
    # Pose viewers are asset checks; keep them usable while temporarily hiding
    # one foot link for visual comparison.
    Cfg.env.use_paper_b_observation = False
    disable_randomization()
    Cfg.env.num_envs = 1
    Cfg.env.num_recording_envs = 1
    Cfg.env.record_video = False
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 1
    Cfg.terrain.border_size = 0
    Cfg.terrain.curriculum = False
    Cfg.sim.physx.max_gpu_contact_pairs = 2**18
    Cfg.sim.physx.default_buffer_size_multiplier = 1


def make_env(sim_device: str, headless: bool):
    configure_single_env()
    return VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)


def set_static_pose(base_env, dof_pos: torch.Tensor, base_height: float | None = None) -> None:
    env_ids = torch.tensor([0], dtype=torch.long, device=base_env.device)
    env_handle = base_env.envs[0]
    actor_handle = base_env.actor_handles[0]
    actor_index = base_env.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
    base_state = base_env.base_init_state.clone().view(1, -1)
    if base_height is not None:
        base_state[:, 2] = float(base_height)
    base_state[:, :3] += base_env.env_origins[env_ids]
    base_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=base_env.device)
    base_state[:, 7:13] = 0.0

    dof_pos = dof_pos.to(base_env.device)
    actor_indices = torch.tensor([actor_index], dtype=torch.int32, device=base_env.device)

    def apply_state_tensors() -> None:
        base_env.root_states[actor_index] = base_state[0]
        base_env.dof_pos[env_ids] = dof_pos
        base_env.dof_vel[env_ids] = 0.0
        base_env.gym.set_actor_root_state_tensor_indexed(
            base_env.sim,
            gymtorch.unwrap_tensor(base_env.root_states),
            gymtorch.unwrap_tensor(actor_indices),
            1,
        )
        base_env.gym.set_dof_state_tensor_indexed(
            base_env.sim,
            gymtorch.unwrap_tensor(base_env.dof_state),
            gymtorch.unwrap_tensor(actor_indices),
            1,
        )

    apply_state_tensors()
    base_env.gym.simulate(base_env.sim)
    base_env.gym.fetch_results(base_env.sim, True)
    base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
    base_env.gym.refresh_dof_state_tensor(base_env.sim)
    base_env.gym.refresh_rigid_body_state_tensor(base_env.sim)
    apply_state_tensors()
    base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
    base_env.gym.refresh_dof_state_tensor(base_env.sim)
    base_env.gym.refresh_rigid_body_state_tensor(base_env.sim)
    base_env.commands[:, :] = 0.0
    base_env.actions[:, :] = 0.0
    base_env.last_last_actions[:, :] = 0.0
    base_env.last_actions[:, :] = 0.0
    base_env.last_dof_vel[:, :] = 0.0
    base_env.last_root_vel[:, :] = 0.0

    if hasattr(base_env, "_reset_paper_b_history_buffers"):
        base_env._reset_paper_b_history_buffers(env_ids)

    base_env.base_quat[:] = base_env.root_states[:, 3:7]
    base_env.base_lin_vel[:] = quat_rotate_inverse(base_env.base_quat, base_env.root_states[:, 7:10])
    base_env.base_ang_vel[:] = quat_rotate_inverse(base_env.base_quat, base_env.root_states[:, 10:13])
    base_env.projected_gravity[:] = quat_rotate_inverse(base_env.base_quat, base_env.gravity_vec)
    base_env.compute_observations()


def print_pose(base_env, pose_name: str) -> None:
    print(f"pose                            = {pose_name}")
    print(f"dof_names                       = {base_env.dof_names}")
    print(f"default_dof_pos rad             = {npfmt(base_env.default_dof_pos)}")
    print(f"default_dof_pos deg             = {npfmt(base_env.default_dof_pos * (180.0 / np.pi))}")
    print(f"current_dof_pos rad             = {npfmt(base_env.dof_pos[0])}")
    print(f"current_dof_pos deg             = {npfmt(base_env.dof_pos[0] * (180.0 / np.pi))}")
    print(f"current_minus_default rad       = {npfmt(base_env.dof_pos[0] - base_env.default_dof_pos[0])}")
    print(f"root_state                      = {npfmt(base_env.root_states[0])}")
    print(f"commands                        = {npfmt(base_env.commands[0])}")


def hold_viewer(base_env, hold_seconds: float) -> None:
    if base_env.headless:
        return

    if hold_seconds > 0:
        end_time = time.time() + hold_seconds
        while time.time() < end_time:
            base_env.render_gui(sync_frame_time=True)
            time.sleep(1.0 / 60.0)
        return

    while True:
        base_env.render_gui(sync_frame_time=True)
        time.sleep(1.0 / 60.0)


def run_pose_viewer(pose_name: str, zero_joints: bool) -> None:
    parser = argparse.ArgumentParser(description=f"View Mini Cheetah {pose_name} in Isaac Gym.")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", help="print pose values without opening the viewer")
    parser.add_argument("--base-height", type=float, default=None, help="override base z height before display")
    parser.add_argument("--hold-seconds", type=float, default=0.0, help="viewer hold time; 0 means until closed")
    args = parser.parse_args()

    base_env = make_env(sim_device=args.sim_device, headless=args.headless)
    try:
        if zero_joints:
            dof_pos = torch.zeros_like(base_env.default_dof_pos)
        else:
            dof_pos = base_env.default_dof_pos.clone()
        set_static_pose(base_env, dof_pos, base_height=args.base_height)
        print_pose(base_env, pose_name)
        hold_viewer(base_env, args.hold_seconds)
    finally:
        base_env.close()
