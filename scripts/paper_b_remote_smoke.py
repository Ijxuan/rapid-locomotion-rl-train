#!/usr/bin/env python3
"""Isaac Gym smoke checks for the Paper B Mini Cheetah branch.

Import order matters for Isaac Gym: keep isaacgym before torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import isaacgym

assert isaacgym

import torch

from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper
from mini_gym_learn.ppo import Runner, RunnerArgs
from mini_gym_learn.ppo.actor_critic import ActorCritic
from mini_gym_learn.ppo.ppo import PPO_Args
from scripts.rl_lcm_policy import load_and_validate_checkpoint


def configure_cfg(num_envs: int) -> None:
    config_mini_cheetah(Cfg)
    Cfg.env.num_envs = num_envs
    Cfg.env.record_video = False
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 1
    Cfg.terrain.border_size = 0
    Cfg.sim.physx.max_gpu_contact_pairs = 2**18
    Cfg.sim.physx.default_buffer_size_multiplier = 1


def make_env(num_envs: int, sim_device: str):
    configure_cfg(num_envs)
    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=True, cfg=Cfg)
    wrapped = HistoryWrapper(env)
    return env, wrapped


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values")


def check_env(num_envs: int, sim_device: str):
    base_env, env = make_env(num_envs, sim_device)
    obs = env.reset()
    actor_obs = obs["obs"]
    privileged_obs = obs["privileged_obs"]

    print("obs shape:", tuple(actor_obs.shape))
    print("privileged shape:", tuple(privileged_obs.shape))
    print("feet indices:", base_env.feet_indices.detach().cpu().tolist())
    print("termination indices:", base_env.termination_contact_indices.detach().cpu().tolist())

    assert tuple(actor_obs.shape) == (num_envs, 142)
    assert tuple(privileged_obs.shape) == (num_envs, 11)
    assert_finite("obs", actor_obs)
    assert_finite("privileged_obs", privileged_obs)
    return base_env, env


def check_steps(num_envs: int, sim_device: str, steps: int):
    base_env, env = check_env(num_envs, sim_device)
    actions = torch.zeros(num_envs, Cfg.env.num_actions, device=base_env.device)
    for step in range(steps):
        obs, rew, done, info = env.step(actions)
        assert_finite(f"step {step} obs", obs["obs"])
        assert_finite(f"step {step} reward", rew)
        if "privileged_obs" not in obs:
            raise AssertionError("wrapped step did not return privileged_obs")

    old_reset = base_env.reset_buf.clone()
    old_timeout = base_env.time_out_buf.clone()
    base_env.reset_buf[:] = 1
    base_env.time_out_buf[:] = False
    terminal_penalty = base_env._reward_termination() * base_env.cfg.rewards.paper_b_termination_penalty
    base_env.reset_buf[:] = old_reset
    base_env.time_out_buf[:] = old_timeout
    if not torch.allclose(terminal_penalty, torch.full_like(terminal_penalty, -10.0)):
        raise AssertionError(f"terminal penalty mismatch: {terminal_penalty[:4]}")

    print("step smoke OK; reward finite; terminal penalty=-10")
    return base_env, env


def check_command_and_dr(num_envs: int, sim_device: str, resamples: int):
    base_env, env = check_env(num_envs, sim_device)
    del env
    env_ids = torch.arange(num_envs, device=base_env.device)
    zero_count = 0
    total_count = 0
    for _ in range(resamples):
        base_env._resample_commands(env_ids)
        zero_count += int((torch.norm(base_env.commands[:, :3], dim=1) == 0).sum().item())
        total_count += num_envs
    zero_ratio = zero_count / max(1, total_count)

    for name in ("motor_frictions", "Kp_additive", "Kd_additive", "paper_b_foot_radii"):
        assert hasattr(base_env, name), name
        assert_finite(name, getattr(base_env, name))

    print(f"zero command ratio: {zero_ratio:.3f}")
    if not 0.03 <= zero_ratio <= 0.20:
        raise AssertionError(f"zero command ratio outside smoke range: {zero_ratio:.3f}")
    print("DR buffers finite")
    return zero_ratio


def check_ppo(num_envs: int, sim_device: str, iterations: int, steps_per_iter: int):
    _, env = make_env(num_envs, sim_device)
    RunnerArgs.num_steps_per_env = steps_per_iter
    PPO_Args.num_learning_epochs = 1
    PPO_Args.num_mini_batches = 1
    runner = Runner(env, device=sim_device)
    alg = runner.alg

    obs_dict = env.get_observations()
    obs = obs_dict["obs"].to(sim_device)
    privileged_obs = obs_dict["privileged_obs"].to(sim_device)
    obs_history = obs_dict["obs_history"].to(sim_device)

    for iteration in range(iterations):
        with torch.inference_mode():
            for _ in range(steps_per_iter):
                actions = alg.act(obs, privileged_obs, obs_history)
                obs_dict, rewards, dones, infos = env.step(actions)
                obs = obs_dict["obs"].to(sim_device)
                privileged_obs = obs_dict["privileged_obs"].to(sim_device)
                obs_history = obs_dict["obs_history"].to(sim_device)
                alg.process_env_step(rewards.to(sim_device), dones.to(sim_device), infos)
            alg.compute_returns(obs, privileged_obs)
        value_loss, surrogate_loss, estimator_loss = alg.update()
        for name, value in {
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "estimator_loss": estimator_loss,
        }.items():
            if not torch.isfinite(torch.tensor(value)):
                raise AssertionError(f"{name} is not finite: {value}")
        print(
            f"ppo iteration {iteration}: "
            f"value_loss={value_loss:.6f}, surrogate_loss={surrogate_loss:.6f}, estimator_loss={estimator_loss:.6f}"
        )


def check_jit():
    configure_cfg(num_envs=1)
    model = ActorCritic(
        num_obs=Cfg.env.num_observations,
        num_privileged_obs=Cfg.env.num_privileged_obs,
        num_obs_history=Cfg.env.num_observations * Cfg.env.num_observation_history,
        num_actions=Cfg.env.num_actions,
    ).cpu()
    with TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "checkpoints"
        checkpoint.mkdir(parents=True)
        torch.jit.script(model.estimator).save(str(checkpoint / "estimator_latest.jit"))
        torch.jit.script(model.actor_body).save(str(checkpoint / "body_latest.jit"))
        _, _, estimator_path, body_path, estimator_shape, action_shape = load_and_validate_checkpoint(checkpoint, torch)
        print("estimator jit:", estimator_path.name, estimator_shape)
        print("body jit:", body_path.name, action_shape)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=["env", "steps", "command-dr", "ppo", "jit", "all"], default="all")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--ppo-iters", type=int, default=2)
    parser.add_argument("--steps-per-iter", type=int, default=4)
    parser.add_argument("--command-resamples", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.check in {"env", "all"}:
        check_env(args.num_envs, args.sim_device)
    if args.check in {"steps", "all"}:
        check_steps(args.num_envs, args.sim_device, args.steps)
    if args.check in {"command-dr", "all"}:
        check_command_and_dr(max(args.num_envs, 20), args.sim_device, args.command_resamples)
    if args.check in {"ppo", "all"}:
        check_ppo(args.num_envs, args.sim_device, args.ppo_iters, args.steps_per_iter)
    if args.check in {"jit", "all"}:
        check_jit()
    print(f"Paper B smoke check '{args.check}' OK")


if __name__ == "__main__":
    main()
