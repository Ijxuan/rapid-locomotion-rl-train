#!/usr/bin/env python3
"""Probe the rapid-locomotion policy at level/default/zero-command state."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import isaacgym

assert isaacgym

import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse

from mini_gym.envs import *  # noqa: F401,F403
from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper
from mini_gym_learn.ppo import RunnerArgs
from mini_gym_learn.ppo.actor_critic import AC_Args, ActorCritic
from mini_gym_learn.ppo.ppo import PPO_Args


def resolve_run_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "parameters.pkl").exists():
        return path

    runs = sorted(path.glob("train/*"), key=lambda p: p.stat().st_mtime)
    runs = [p for p in runs if (p / "parameters.pkl").exists()]
    if runs:
        return runs[-1]

    raise FileNotFoundError(f"could not find parameters.pkl under {path}")


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


def apply_saved_parameters(run_dir: Path) -> None:
    params_path = run_dir / "parameters.pkl"
    with params_path.open("rb") as f:
        params = pickle.load(f)

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
        Cfg.terrain._update(deps)
        Cfg.commands._update(deps)
        Cfg.normalization._update(deps)
        Cfg.env._update(deps)
        Cfg.domain_rand._update(deps)
        Cfg.rewards._update(deps)
        if hasattr(Cfg, "reward_scales"):
            Cfg.reward_scales._update(deps)
        if hasattr(Cfg, "perception"):
            Cfg.perception._update(deps)
        Cfg.control._update(deps)
        return

    raise RuntimeError(f"unsupported parameters.pkl format: {type(params)}")


def disable_eval_randomization() -> None:
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
    Cfg.noise.add_noise = False


def load_env_and_policy(run_dir: Path, headless: bool, sim_device: str):
    config_mini_cheetah(Cfg)
    apply_saved_parameters(run_dir)
    disable_eval_randomization()

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 1
    Cfg.terrain.border_size = 0
    Cfg.sim.physx.max_gpu_contact_pairs = 2**18
    Cfg.sim.physx.default_buffer_size_multiplier = 1

    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    actor_critic = ActorCritic(
        num_obs=Cfg.env.num_observations,
        num_privileged_obs=Cfg.env.num_privileged_obs,
        num_obs_history=Cfg.env.num_observations * Cfg.env.num_observation_history,
        num_actions=Cfg.env.num_actions,
    )
    weights = torch.load(run_dir / "checkpoints" / "ac_weights_last.pt", map_location=env.device)
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    actor_critic.eval()

    return env, actor_critic


def hip_outward_sign(joint_name: str, current_value: float) -> float:
    if joint_name.startswith(("FL_", "RL_")):
        return 1.0
    if joint_name.startswith(("FR_", "RR_")):
        return -1.0
    if abs(current_value) > 1e-6:
        return 1.0 if current_value > 0.0 else -1.0
    return 1.0


def set_default_hip_outward(base_env, hip_out_deg: float | None) -> list[tuple[str, float]]:
    if hip_out_deg is None:
        return []

    hip_out_rad = abs(float(hip_out_deg)) * np.pi / 180.0
    changed = []
    for i, joint_name in enumerate(base_env.dof_names):
        is_hip_joint = "hip" in joint_name and "thigh" not in joint_name
        if not is_hip_joint and i % 3 != 0:
            continue

        current_value = float(base_env.default_dof_pos[0, i])
        value = hip_outward_sign(joint_name, current_value) * hip_out_rad
        base_env.default_dof_pos[:, i] = value
        changed.append((joint_name, value))
    return changed


def force_level_default_zero_command(env: HistoryWrapper):
    base_env = env.env
    env_ids = torch.tensor([0], dtype=torch.long, device=base_env.device)

    dof_pos = base_env.default_dof_pos[env_ids].clone()
    base_state = base_env.base_init_state.clone().view(1, -1)
    base_state[:, :3] += base_env.env_origins[env_ids]
    base_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=base_env.device)
    base_state[:, 7:13] = 0.0
    base_env.set_idx_pose(env_ids, dof_pos, base_state)

    base_env.commands[:, :] = 0.0
    base_env.actions[:, :] = 0.0
    base_env.last_actions[:, :] = 0.0
    base_env.last_dof_vel[:, :] = 0.0
    base_env.last_root_vel[:, :] = 0.0

    base_env.base_quat[:] = base_env.root_states[:, 3:7]
    base_env.base_lin_vel[:] = quat_rotate_inverse(base_env.base_quat, base_env.root_states[:, 7:10])
    base_env.base_ang_vel[:] = quat_rotate_inverse(base_env.base_quat, base_env.root_states[:, 10:13])
    base_env.projected_gravity[:] = quat_rotate_inverse(base_env.base_quat, base_env.gravity_vec)
    base_env.compute_observations()
    env.obs_history[:, :] = 0.0

    return base_env.obs_buf.clone()


def hold_viewer(env: HistoryWrapper, seconds: float) -> None:
    if seconds <= 0.0:
        return

    base_env = env.env
    end_time = time.time() + seconds
    while time.time() < end_time:
        base_env.render_gui(sync_frame_time=True)
        time.sleep(1.0 / 60.0)


def action_to_target_q(base_env, action: torch.Tensor) -> torch.Tensor:
    scaled = action[:, :12] * Cfg.control.action_scale
    scaled[:, [0, 3, 6, 9]] *= Cfg.control.hip_scale_reduction
    return scaled + base_env.default_dof_pos


def npfmt(x, precision: int = 4) -> str:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return np.array2string(arr, precision=precision, suppress_small=True)


def rad_to_deg(x: torch.Tensor) -> torch.Tensor:
    return x * (180.0 / np.pi)


def print_joint_angle_table(
    name: str,
    obs: torch.Tensor,
    action: torch.Tensor,
    target_q: torch.Tensor,
    base_env,
) -> None:
    obs0 = obs[0]
    default_q = base_env.default_dof_pos[0]
    input_q = default_q + obs0[6:18] / Cfg.normalization.obs_scales.dof_pos
    output_q = target_q[0]
    delta_q = output_q - default_q

    print(f"\n===== {name}: policy joint angles =====")
    print("order = IsaacGym policy/training DOF order")
    print(
        "joint                 "
        "input_rad input_deg | action  delta_deg | output_rad output_deg"
    )
    for i, joint_name in enumerate(base_env.dof_names):
        print(
            f"{joint_name:<21}"
            f"{float(input_q[i]):>9.4f} {float(rad_to_deg(input_q[i])):>9.2f} | "
            f"{float(action[0, i]):>6.3f} {float(rad_to_deg(delta_q[i])):>9.2f} | "
            f"{float(output_q[i]):>10.4f} {float(rad_to_deg(output_q[i])):>10.2f}"
        )


def print_probe(
    name: str,
    obs: torch.Tensor,
    history: torch.Tensor,
    actor_critic,
    base_env,
    angles_only: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        latent = actor_critic.adaptation_module(history)
        action = actor_critic.actor_body(torch.cat((obs, latent), dim=-1))
    target_q = action_to_target_q(base_env, action)

    if angles_only:
        print_joint_angle_table(name, obs, action, target_q, base_env)
        return action

    obs0 = obs[0]
    print(f"\n===== {name} =====")
    print(f"obs.shape={tuple(obs.shape)}, history.shape={tuple(history.shape)}")
    print(f"projected_gravity obs[0:3]     = {npfmt(obs0[0:3])}")
    print(f"command_scaled obs[3:6]        = {npfmt(obs0[3:6])}")
    print(f"q_minus_default obs[6:18]      = {npfmt(obs0[6:18])}")
    print(f"qd_scaled obs[18:30]           = {npfmt(obs0[18:30])}")
    print(f"last_action obs[30:42]         = {npfmt(obs0[30:42])}")
    print(f"latent                         = {npfmt(latent[0])}")
    print(f"action                         = {npfmt(action[0])}")
    print(f"action_norm                    = {float(torch.norm(action[0])):.6f}")
    print(f"target_q                       = {npfmt(target_q[0])}")
    print(f"target_q - default_q           = {npfmt((target_q - base_env.default_dof_pos)[0])}")
    return action


def compare_jit(run_dir: Path, obs: torch.Tensor, history: torch.Tensor, actor_action: torch.Tensor) -> None:
    adaptation = torch.jit.load(str(run_dir / "checkpoints" / "adaptation_module_latest.jit"), map_location=obs.device)
    body = torch.jit.load(str(run_dir / "checkpoints" / "body_latest.jit"), map_location=obs.device)
    adaptation.eval()
    body.eval()
    with torch.no_grad():
        latent = adaptation(history)
        jit_action = body(torch.cat((obs, latent), dim=-1))
    print("\n===== JIT comparison on repeat_history =====")
    print(f"jit_action                     = {npfmt(jit_action[0])}")
    print(f"max_abs_diff(actor, jit)       = {float(torch.max(torch.abs(actor_action - jit_action))):.9f}")


def parse_print_steps(value: str) -> set[int] | None:
    text = value.strip().lower()
    if text in {"", "all"}:
        return None
    steps: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        step = int(part)
        if step <= 0:
            raise ValueError("--print-steps uses 1-based positive step numbers")
        steps.add(step)
    return steps


def rollout_zero_command(
    env: HistoryWrapper,
    actor_critic,
    steps: int,
    print_steps: set[int] | None,
    angles_only: bool = False,
) -> None:
    if steps <= 0:
        return

    base_env = env.env
    obs = {"obs": base_env.obs_buf.clone(), "privileged_obs": None, "obs_history": env.obs_history.clone()}
    print("\n===== zero-command rollout =====")
    for i in range(steps):
        step_number = i + 1
        base_env.commands[:, :] = 0.0
        with torch.no_grad():
            action = actor_critic.act_student(obs["obs"], obs["obs_history"])
        if print_steps is None or step_number in print_steps:
            if angles_only:
                target_q = action_to_target_q(base_env, action)
                print_joint_angle_table(
                    f"rollout step {step_number}",
                    obs["obs"],
                    action,
                    target_q,
                    base_env,
                )
            else:
                print(
                    f"step={step_number:03d} action_norm={float(torch.norm(action[0])):.6f} "
                    f"base_lin_vel={npfmt(base_env.base_lin_vel[0], 3)} "
                    f"dof_pos={npfmt(base_env.dof_pos[0], 3)}"
                )
        obs, rew, done, info = env.step(action)
        base_env.commands[:, :] = 0.0
        if bool(done[0]):
            print(f"done=True at step {step_number}, rew={float(rew[0]):.6f}")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/rapid-locomotion/example", help="run dir or run group dir")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--show", action="store_true", help="open Isaac Gym viewer")
    parser.add_argument("--steps", type=int, default=0, help="optional zero-command rollout steps")
    parser.add_argument("--print-steps", default="all", help="1-based rollout steps to print, e.g. 1,15,30")
    parser.add_argument("--angles-only", action="store_true", help="only print policy input/output joint angles")
    parser.add_argument("--skip-initial-probe", action="store_true", help="only run rollout; skip reset/repeat probes")
    parser.add_argument("--hip-out-deg", type=float, default=None, help="set default hip/abad outward angle in degrees")
    parser.add_argument("--hold-seconds", type=float, default=0.0, help="keep Isaac Gym viewer open after rollout")
    parser.set_defaults(compare_jit=True)
    parser.add_argument("--no-compare-jit", action="store_false", dest="compare_jit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(Path(args.run))
    env, actor_critic = load_env_and_policy(run_dir, headless=not args.show, sim_device=args.sim_device)
    base_env = env.env

    env.reset()
    hip_pose = set_default_hip_outward(base_env, args.hip_out_deg)
    obs = force_level_default_zero_command(env)
    history_zero = torch.zeros(1, Cfg.env.num_observations * Cfg.env.num_observation_history, device=base_env.device)
    history_repeat = obs.repeat(1, Cfg.env.num_observation_history)

    print(f"run_dir                         = {run_dir}")
    print(f"dof_names                       = {base_env.dof_names}")
    if hip_pose:
        print(
            "hip_out_default_deg             = "
            + ", ".join(f"{name}:{value * 180.0 / np.pi:.2f}" for name, value in hip_pose)
        )
    if not args.angles_only:
        print(f"default_dof_pos                 = {npfmt(base_env.default_dof_pos[0])}")
        print(f"root_state                      = {npfmt(base_env.root_states[0])}")
        print(f"commands                        = {npfmt(base_env.commands[0])}")
        print(f"projected_gravity tensor        = {npfmt(base_env.projected_gravity[0])}")
        print(f"dof_pos                         = {npfmt(base_env.dof_pos[0])}")
        print(f"dof_vel                         = {npfmt(base_env.dof_vel[0])}")

    if not args.skip_initial_probe:
        print_probe(
            "reset_history: obs current, history all zeros",
            obs,
            history_zero,
            actor_critic,
            base_env,
            angles_only=args.angles_only,
        )
        repeat_action = print_probe(
            "repeat_history: same zero-command obs repeated 15 frames",
            obs,
            history_repeat,
            actor_critic,
            base_env,
            angles_only=args.angles_only,
        )
        if args.compare_jit and not args.angles_only:
            compare_jit(run_dir, obs, history_repeat, repeat_action)

    rollout_zero_command(env, actor_critic, args.steps, parse_print_steps(args.print_steps), args.angles_only)
    hold_viewer(env, args.hold_seconds)


if __name__ == "__main__":
    main()
