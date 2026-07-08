#!/usr/bin/env python3
"""Compare Mini Cheetah standing height under three PD gain settings."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import isaacgym

assert isaacgym

from isaacgym import gymapi, gymtorch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_gym.envs import *  # noqa: F401,F403
from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
from mini_gym.utils.torch_utils import quat_rotate_inverse
from scripts.joint_pose_viewer_common import disable_randomization


DEFAULT_GAINS = "8.5,0.2;17.0,0.4;34.0,0.8"
DEFAULT_LABELS = ("low", "current", "high")


def parse_gains(value: str) -> List[Tuple[float, float]]:
    gains: List[Tuple[float, float]] = []
    for item in value.split(";"):
        fields = [field.strip() for field in item.split(",")]
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "gains must look like '8.5,0.2;17,0.4;34,0.8'"
            )
        kp, kd = (float(fields[0]), float(fields[1]))
        if kp <= 0.0 or kd < 0.0:
            raise argparse.ArgumentTypeError("kp must be > 0 and kd must be >= 0")
        gains.append((kp, kd))
    if len(gains) != 3:
        raise argparse.ArgumentTypeError("exactly three kp,kd pairs are required")
    return gains


def configure_env(headless: bool) -> None:
    config_mini_cheetah(Cfg)
    disable_randomization()
    Cfg.env.num_envs = 3
    Cfg.env.num_recording_envs = 0
    Cfg.env.record_video = False
    Cfg.env.use_paper_b_observation = False
    Cfg.env.episode_length_s = 100
    Cfg.commands.paper_b_command_curriculum = False
    Cfg.commands.command_curriculum = False
    Cfg.commands.lin_vel_x = [0.0, 0.0]
    Cfg.commands.lin_vel_y = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw = [0.0, 0.0]
    Cfg.asset.terminate_after_contacts_on = []
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 3
    Cfg.terrain.border_size = 0
    Cfg.terrain.curriculum = False
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.sim.physx.max_gpu_contact_pairs = 2**18
    Cfg.sim.physx.default_buffer_size_multiplier = 1
    if not headless:
        Cfg.viewer.pos = [4.0, -5.0, 2.0]
        Cfg.viewer.lookat = [4.0, 0.0, 0.35]


def make_env(sim_device: str, headless: bool) -> VelocityTrackingEasyEnv:
    configure_env(headless=headless)
    return VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)


def actor_indices(base_env) -> torch.Tensor:
    indices = []
    for env_handle, actor_handle in zip(base_env.envs, base_env.actor_handles):
        indices.append(base_env.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM))
    return torch.tensor(indices, dtype=torch.int32, device=base_env.device)


def set_default_standing_pose(base_env, base_height: float) -> None:
    env_ids = torch.arange(base_env.num_envs, dtype=torch.long, device=base_env.device)
    indices_i32 = actor_indices(base_env)
    indices_long = indices_i32.long()

    root_state = base_env.base_init_state.clone().view(1, -1).repeat(base_env.num_envs, 1)
    root_state[:, 2] = float(base_height)
    root_state[:, :3] += base_env.env_origins[env_ids]
    root_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=base_env.device)
    root_state[:, 7:13] = 0.0

    base_env.root_states[indices_long] = root_state
    base_env.dof_pos[env_ids] = base_env.default_dof_pos.expand(base_env.num_envs, -1)
    base_env.dof_vel[env_ids] = 0.0
    base_env.gym.set_actor_root_state_tensor_indexed(
        base_env.sim,
        gymtorch.unwrap_tensor(base_env.root_states),
        gymtorch.unwrap_tensor(indices_i32),
        len(indices_i32),
    )
    base_env.gym.set_dof_state_tensor_indexed(
        base_env.sim,
        gymtorch.unwrap_tensor(base_env.dof_state),
        gymtorch.unwrap_tensor(indices_i32),
        len(indices_i32),
    )
    base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
    base_env.gym.refresh_dof_state_tensor(base_env.sim)
    base_env.gym.refresh_rigid_body_state_tensor(base_env.sim)

    for tensor_name in (
        "commands",
        "actions",
        "last_actions",
        "last_last_actions",
        "last_dof_vel",
        "last_root_vel",
    ):
        getattr(base_env, tensor_name)[:] = 0.0
    base_env.episode_length_buf[:] = 0
    base_env.reset_buf[:] = 0
    base_env.time_out_buf[:] = False

    if hasattr(base_env, "_reset_paper_b_history_buffers"):
        base_env._reset_paper_b_history_buffers(env_ids)

    base_env.base_quat[:] = base_env.root_states[:, 3:7]
    base_env.base_lin_vel[:] = quat_rotate_inverse(base_env.base_quat, base_env.root_states[:, 7:10])
    base_env.base_ang_vel[:] = quat_rotate_inverse(base_env.base_quat, base_env.root_states[:, 10:13])
    base_env.projected_gravity[:] = quat_rotate_inverse(base_env.base_quat, base_env.gravity_vec)
    base_env.compute_observations()


def apply_pd_gains(base_env, gains: Sequence[Tuple[float, float]]) -> None:
    base_env.Kp_factors[:] = 1.0
    base_env.Kd_factors[:] = 1.0
    base_env.Kp_additive[:] = 0.0
    base_env.Kd_additive[:] = 0.0

    p_den = torch.clamp(base_env.p_gains, min=1.0e-6)
    d_den = torch.clamp(base_env.d_gains, min=1.0e-6)
    for env_id, (kp, kd) in enumerate(gains):
        base_env.Kp_factors[env_id, :] = float(kp) / p_den
        base_env.Kd_factors[env_id, :] = float(kd) / d_den


def labels_for_gains(gains: Sequence[Tuple[float, float]]) -> Tuple[str, str, str]:
    default_gains = parse_gains(DEFAULT_GAINS)
    if len(gains) == len(default_gains) and all(
        abs(kp - default_kp) < 1.0e-6 and abs(kd - default_kd) < 1.0e-6
        for (kp, kd), (default_kp, default_kd) in zip(gains, default_gains)
    ):
        return DEFAULT_LABELS
    return ("env0", "env1", "env2")


def settle(base_env, seconds: float) -> None:
    steps = max(1, int(round(seconds / base_env.dt)))
    actions = torch.zeros(base_env.num_envs, base_env.num_actions, device=base_env.device)
    for _ in range(steps):
        base_env.step(actions)


def contact_count(base_env, names: Sequence[str]) -> torch.Tensor:
    body_indices = [
        idx for idx, body_name in enumerate(base_env.body_names)
        if any(name in body_name for name in names)
    ]
    if not body_indices:
        return torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
    indices = torch.tensor(body_indices, dtype=torch.long, device=base_env.device)
    forces = torch.norm(base_env.contact_forces[:, indices, :], dim=-1)
    return torch.sum(forces > 0.1, dim=1)


def print_report(
    base_env,
    labels: Sequence[str],
    requested_gains: Sequence[Tuple[float, float]] | None = None,
) -> None:
    rigid_body_state = base_env.rigid_body_state.view(base_env.num_envs, base_env.num_bodies, 13)
    foot_center_z = rigid_body_state[:, base_env.feet_indices, 2].mean(dim=1)
    foot_force_z = base_env.contact_forces[:, base_env.feet_indices, 2].sum(dim=1)
    calf_contacts = contact_count(base_env, ("calf",))
    base_z = base_env.root_states[:, 2] - base_env.env_origins[:, 2]
    actual_kp = base_env.p_gains.unsqueeze(0) * base_env.Kp_factors + base_env.Kp_additive
    actual_kd = base_env.d_gains.unsqueeze(0) * base_env.Kd_factors + base_env.Kd_additive

    print("env  label    kp_mean  kd_mean  base_z   foot_z   foot_fz  calf_contacts")
    for env_id in range(base_env.num_envs):
        label = labels[env_id]
        print(
            f"{env_id:<4d}{label:<9s}"
            f"{actual_kp[env_id].mean().item():>8.3f}"
            f"{actual_kd[env_id].mean().item():>9.3f}"
            f"{base_z[env_id].item():>8.4f}"
            f"{foot_center_z[env_id].item():>9.4f}"
            f"{foot_force_z[env_id].item():>9.3f}"
            f"{int(calf_contacts[env_id].item()):>15d}"
        )
        if requested_gains is not None:
            kp, kd = requested_gains[env_id]
            if abs(actual_kp[env_id].mean().item() - kp) > 1.0e-4:
                print(f"warning: env {env_id} requested kp={kp}, applied mean differs")
            if abs(actual_kd[env_id].mean().item() - kd) > 1.0e-4:
                print(f"warning: env {env_id} requested kd={kd}, applied mean differs")


def print_joint_errors(base_env, labels: Sequence[str]) -> None:
    target = getattr(base_env, "joint_pos_target", base_env.default_dof_pos.expand(base_env.num_envs, -1))
    error_deg = (target - base_env.dof_pos) * (180.0 / torch.pi)

    print()
    print("joint error deg = target_angle - current_angle")
    header = f"{'joint':<18s}" + "".join(f"{label:>12s}" for label in labels)
    print(header)
    for joint_id, joint_name in enumerate(base_env.dof_names):
        row = f"{joint_name:<18s}" + "".join(
            f"{error_deg[env_id, joint_id].item():>12.4f}"
            for env_id in range(base_env.num_envs)
        )
        print(row)

    max_abs = torch.max(torch.abs(error_deg), dim=1).values
    print("max_abs_error_deg " + "".join(f"{value.item():>12.4f}" for value in max_abs))


def hold_viewer(base_env, seconds: float) -> None:
    if base_env.headless:
        return
    actions = torch.zeros(base_env.num_envs, base_env.num_actions, device=base_env.device)
    if seconds <= 0.0:
        while True:
            base_env.step(actions)
            time.sleep(1.0 / 60.0)
    end_time = time.time() + seconds
    while time.time() < end_time:
        base_env.step(actions)
        time.sleep(1.0 / 60.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", help="print only, without opening Isaac Gym viewer")
    parser.add_argument("--base-height", type=float, default=0.30, help="initial root z for all three robots")
    parser.add_argument("--settle-seconds", type=float, default=3.0, help="simulation time before comparing height")
    parser.add_argument("--hold-seconds", type=float, default=0.0, help="viewer hold time; 0 means until closed")
    parser.add_argument("--gains", type=parse_gains, default=parse_gains(DEFAULT_GAINS),
                        help="three 'kp,kd' pairs separated by ';'")
    parser.add_argument("--current-gains", action="store_true",
                        help="use the configured current kp/kd for all three robots")
    parser.add_argument("--print-joint-errors", action="store_true",
                        help="print target-current joint angle errors in degrees")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_env = make_env(sim_device=args.sim_device, headless=args.headless)
    try:
        labels = ("current0", "current1", "current2") if args.current_gains else labels_for_gains(args.gains)
        set_default_standing_pose(base_env, args.base_height)
        if not args.current_gains:
            apply_pd_gains(base_env, args.gains)
        settle(base_env, args.settle_seconds)
        print_report(base_env, labels, None if args.current_gains else args.gains)
        if args.print_joint_errors:
            print_joint_errors(base_env, labels)
        hold_viewer(base_env, args.hold_seconds)
    finally:
        base_env.close()


if __name__ == "__main__":
    main()
