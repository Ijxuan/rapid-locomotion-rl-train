#!/usr/bin/env python3
"""TorchScript policy node for the rapid-locomotion LCM bridge."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_gym.deploy.rapid_locomotion_policy import (
    ACTION_DIM,
    DEFAULT_Q_POLICY,
    OBS_DIM,
    OBS_HISTORY_DIM,
    ObservationHistory,
    action_to_target_q,
    build_observation,
)


def monotonic_us() -> int:
    return int(time.monotonic() * 1_000_000)


def add_lcm_types_path(path: Path) -> None:
    path = path.resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def add_system_lcm_path_if_needed() -> Path | None:
    if importlib.util.find_spec("lcm") is not None:
        return None

    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidate = Path("/usr/local/lib") / py_version / "site-packages"
    if (candidate / "lcm").exists():
        sys.path.append(str(candidate))
        return candidate
    return None


def resolve_checkpoint(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError("--checkpoint must be a directory")

    adaptation = path / "adaptation_module_latest.jit"
    body = path / "body_latest.jit"
    if adaptation.exists() and body.exists():
        return adaptation, body

    candidates: list[tuple[float, Path, Path, Path]] = []
    for adaptation in path.rglob("adaptation_module_latest.jit"):
        body = adaptation.parent / "body_latest.jit"
        if body.exists():
            candidates.append((adaptation.stat().st_mtime, adaptation.parent, adaptation, body))

    if candidates:
        _, _, adaptation, body = sorted(candidates, key=lambda item: (item[0], str(item[1])))[-1]
        return adaptation, body

    raise FileNotFoundError(
        "could not find adaptation_module_latest.jit and body_latest.jit under " + str(path)
    )


def load_and_validate_checkpoint(checkpoint: Path, torch):
    adaptation_path, body_path = resolve_checkpoint(checkpoint)
    adaptation = torch.jit.load(str(adaptation_path), map_location="cpu")
    body = torch.jit.load(str(body_path), map_location="cpu")
    adaptation.eval()
    body.eval()

    with torch.no_grad():
        latent = adaptation(torch.zeros(1, OBS_HISTORY_DIM, dtype=torch.float32))
        action = body(torch.zeros(1, OBS_DIM + int(latent.shape[1]), dtype=torch.float32))
    if int(latent.shape[1]) != 18:
        raise RuntimeError(f"expected latent dim 18, got {tuple(latent.shape)}")
    if tuple(action.shape) != (1, ACTION_DIM):
        raise RuntimeError(f"expected action shape (1, 12), got {tuple(action.shape)}")

    return adaptation, body, adaptation_path, body_path, tuple(latent.shape), tuple(action.shape)


class RapidRLPolicyNode:
    def __init__(self, args: argparse.Namespace) -> None:
        lcm_path = add_system_lcm_path_if_needed()

        import lcm
        import torch
        from rl_policy_cmd_lcmt import rl_policy_cmd_lcmt
        from rl_robot_state_lcmt import rl_robot_state_lcmt

        self.lcm = lcm.LCM(args.lcm_url) if args.lcm_url else lcm.LCM()
        self.torch = torch
        self.rl_policy_cmd_lcmt = rl_policy_cmd_lcmt
        self.rl_robot_state_lcmt = rl_robot_state_lcmt
        self.command_channel = args.command_channel
        self.state_channel = args.state_channel
        self.log_interval_s = max(0.1, float(args.log_interval))
        self.zero_action = bool(args.zero_action)
        self.log_vectors = bool(args.log_vectors)
        self.history = ObservationHistory()
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_target_q = DEFAULT_Q_POLICY.copy()
        self.sequence = 0
        self.received_count = 0
        self.published_count = 0
        self.start_time_s = time.monotonic()
        self.last_log_time_s = self.start_time_s
        self.last_state_time_s: float | None = None
        self.last_state_sequence: int | None = None
        self.last_state_latency_ms: float | None = None
        self.last_inference_time_ms: float | None = None
        self.last_command = np.zeros(3, dtype=np.float32)

        (
            self.adaptation,
            self.body,
            adaptation_path,
            body_path,
            latent_shape,
            action_shape,
        ) = load_and_validate_checkpoint(Path(args.checkpoint), torch)

        self.lcm.subscribe(args.state_channel, self.handle_state)
        print(f"[rl_lcm_policy] loaded checkpoint: {adaptation_path.parent}")
        print(f"[rl_lcm_policy] warmup latent={latent_shape}, action={action_shape}")
        if lcm_path is not None:
            print(f"[rl_lcm_policy] using LCM Python module path: {lcm_path}")
        print(f"[rl_lcm_policy] listening on {args.state_channel}, publishing {args.command_channel}")
        print(
            f"[rl_lcm_policy] 中文状态日志: 每 {self.log_interval_s:.1f} 秒打印一次收包/发包/推理状态",
            flush=True,
        )
        if self.zero_action:
            print("[rl_lcm_policy] 诊断模式: 不运行策略，固定发布 zero action/default target_q", flush=True)
        if self.log_vectors:
            print("[rl_lcm_policy] 诊断模式: 每次状态日志附带 action 和 target_q 数组", flush=True)

    def handle_state(self, channel: str, data: bytes) -> None:
        del channel
        msg = self.rl_robot_state_lcmt.decode(data)
        self.received_count += 1
        self.last_state_time_s = time.monotonic()
        self.last_state_latency_ms = max(0.0, (monotonic_us() - int(msg.timestamp_us)) / 1000.0)
        self.last_state_sequence = int(msg.sequence)
        self.last_command = np.asarray(msg.command, dtype=np.float32)
        obs = build_observation(
            msg.projected_gravity,
            msg.command,
            msg.q,
            msg.qd,
            self.last_action,
        )
        obs_history = self.history.update(obs)

        if self.zero_action:
            inference_time_ms = 0.0
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            target_q = DEFAULT_Q_POLICY.copy()
        else:
            start = time.perf_counter()
            with self.torch.no_grad():
                hist_t = self.torch.from_numpy(obs_history).view(1, OBS_HISTORY_DIM)
                obs_t = self.torch.from_numpy(obs).view(1, OBS_DIM)
                latent = self.adaptation(hist_t)
                action_t = self.body(self.torch.cat((obs_t, latent), dim=1))
            inference_time_ms = (time.perf_counter() - start) * 1000.0

            action = action_t.detach().cpu().numpy().reshape(ACTION_DIM).astype(np.float32)
            target_q = action_to_target_q(action)
        self.last_action = action.copy()
        self.last_target_q = target_q.copy()

        cmd = self.rl_policy_cmd_lcmt()
        cmd.timestamp_us = monotonic_us()
        cmd.sequence = self.sequence
        cmd.state_sequence = msg.sequence
        cmd.status = 1
        cmd.inference_time_ms = float(inference_time_ms)
        cmd.action = [float(x) for x in action]
        cmd.target_q = [float(x) for x in target_q]
        self.lcm.publish(self.command_channel, cmd.encode())
        self.sequence += 1
        self.published_count += 1
        self.last_inference_time_ms = float(inference_time_ms)

    def log_status_if_due(self) -> None:
        now = time.monotonic()
        if now - self.last_log_time_s < self.log_interval_s:
            return

        uptime_s = now - self.start_time_s
        self.last_log_time_s = now
        if self.last_state_time_s is None:
            print(
                f"[rl_lcm_policy] 状态: 运行 {uptime_s:.1f}s, 还没有收到 "
                f"{self.state_channel}；已发布 {self.published_count} 条策略命令",
                flush=True,
            )
            return

        age_ms = (now - self.last_state_time_s) * 1000.0
        latency = -1.0 if self.last_state_latency_ms is None else self.last_state_latency_ms
        inference = -1.0 if self.last_inference_time_ms is None else self.last_inference_time_ms
        command = ", ".join(f"{x:.2f}" for x in self.last_command)
        action_norm = float(np.linalg.norm(self.last_action))
        print(
            f"[rl_lcm_policy] 状态: 运行 {uptime_s:.1f}s, 已收 {self.received_count} 条状态, "
            f"已发 {self.published_count} 条策略, 最近 state_seq={self.last_state_sequence}, "
            f"距上次状态 {age_ms:.0f}ms, 链路 {latency:.3f}ms, 推理 {inference:.3f}ms, "
            f"cmd=[{command}], action_norm={action_norm:.3f}",
            flush=True,
        )
        if self.log_vectors:
            action_text = np.array2string(self.last_action, precision=3, suppress_small=True)
            target_text = np.array2string(self.last_target_q, precision=3, suppress_small=True)
            print(f"[rl_lcm_policy] action={action_text}", flush=True)
            print(f"[rl_lcm_policy] target_q={target_text}", flush=True)

    def run(self) -> None:
        while True:
            self.lcm.handle_timeout(100)
            self.log_status_if_due()


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[2]
    default_types = workspace_root / "Cheetah-Software-RL" / "lcm-types" / "python"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="directory containing adaptation/body JIT files")
    parser.add_argument("--lcm-types-dir", default=str(default_types), help="generated Python LCM type directory")
    parser.add_argument("--lcm-url", default="", help="optional LCM URL; defaults to lcm.LCM()")
    parser.add_argument("--state-channel", default="rl_robot_state")
    parser.add_argument("--command-channel", default="rl_policy_cmd")
    parser.add_argument("--validate-only", action="store_true", help="load checkpoint and exit without LCM")
    parser.add_argument("--log-interval", type=float, default=1.0, help="status log interval in seconds")
    parser.add_argument("--zero-action", action="store_true", help="diagnostic mode: publish default targets")
    parser.add_argument("--log-vectors", action="store_true", help="print latest action and target_q in status logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        import torch

        _, _, adaptation_path, body_path, latent_shape, action_shape = load_and_validate_checkpoint(
            Path(args.checkpoint), torch
        )
        print(f"[rl_lcm_policy] checkpoint: {adaptation_path.parent}")
        print(f"[rl_lcm_policy] adaptation: {adaptation_path.name}")
        print(f"[rl_lcm_policy] body: {body_path.name}")
        print(f"[rl_lcm_policy] latent={latent_shape}, action={action_shape}")
        return

    add_lcm_types_path(Path(args.lcm_types_dir))
    node = RapidRLPolicyNode(args)
    node.run()


if __name__ == "__main__":
    main()
