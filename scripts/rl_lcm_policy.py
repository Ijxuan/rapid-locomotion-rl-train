#!/usr/bin/env python3
"""TorchScript Paper B policy node for the Mini Cheetah LCM bridge."""

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
    ACTOR_INPUT_DIM,
    DEFAULT_Q_POLICY,
    ESTIMATOR_OUTPUT_DIM,
    OBS_DIM,
    ObservationHistory,
    action_to_target_q,
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

    estimator = path / "estimator_latest.jit"
    body = path / "body_latest.jit"
    if estimator.exists() and body.exists():
        return estimator, body

    candidates: list[tuple[float, Path, Path, Path]] = []
    for estimator in path.rglob("estimator_latest.jit"):
        body = estimator.parent / "body_latest.jit"
        if body.exists():
            candidates.append((estimator.stat().st_mtime, estimator.parent, estimator, body))

    if candidates:
        _, _, estimator, body = sorted(candidates, key=lambda item: (item[0], str(item[1])))[-1]
        return estimator, body

    raise FileNotFoundError("could not find estimator_latest.jit and body_latest.jit under " + str(path))


def load_and_validate_checkpoint(checkpoint: Path, torch):
    estimator_path, body_path = resolve_checkpoint(checkpoint)
    estimator = torch.jit.load(str(estimator_path), map_location="cpu")
    body = torch.jit.load(str(body_path), map_location="cpu")
    estimator.eval()
    body.eval()

    with torch.no_grad():
        estimator_output = estimator(torch.zeros(1, OBS_DIM, dtype=torch.float32))
        action = body(torch.zeros(1, ACTOR_INPUT_DIM, dtype=torch.float32))
    if int(estimator_output.shape[1]) != ESTIMATOR_OUTPUT_DIM:
        raise RuntimeError(f"expected estimator dim 11, got {tuple(estimator_output.shape)}")
    if tuple(action.shape) != (1, ACTION_DIM):
        raise RuntimeError(f"expected action shape (1, 12), got {tuple(action.shape)}")

    return estimator, body, estimator_path, body_path, tuple(estimator_output.shape), tuple(action.shape)


def _msg_vector(msg, name: str, fallback, size: int) -> np.ndarray:
    return np.asarray(getattr(msg, name, fallback), dtype=np.float32).reshape(size)


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
            self.estimator,
            self.body,
            estimator_path,
            body_path,
            estimator_shape,
            action_shape,
        ) = load_and_validate_checkpoint(Path(args.checkpoint), torch)

        self.lcm.subscribe(args.state_channel, self.handle_state)
        print(f"[rl_lcm_policy] loaded checkpoint: {estimator_path.parent}")
        print(f"[rl_lcm_policy] estimator={estimator_shape}, body_action={action_shape}")
        if lcm_path is not None:
            print(f"[rl_lcm_policy] using LCM Python module path: {lcm_path}")
        print(f"[rl_lcm_policy] listening on {args.state_channel}, publishing {args.command_channel}")
        print(f"[rl_lcm_policy] status log every {self.log_interval_s:.1f}s", flush=True)
        if self.zero_action:
            print("[rl_lcm_policy] diagnostic mode: publish zero action/default target_q", flush=True)
        if self.log_vectors:
            print("[rl_lcm_policy] diagnostic mode: include action and target_q vectors", flush=True)

    def handle_state(self, channel: str, data: bytes) -> None:
        del channel
        msg = self.rl_robot_state_lcmt.decode(data)
        self.received_count += 1
        self.last_state_time_s = time.monotonic()
        self.last_state_latency_ms = max(0.0, (monotonic_us() - int(msg.timestamp_us)) / 1000.0)
        self.last_state_sequence = int(msg.sequence)
        self.last_command = np.asarray(msg.command, dtype=np.float32)

        obs = self.history.build(
            _msg_vector(msg, "base_quat", [0.0, 0.0, 0.0, 1.0], 4),
            _msg_vector(msg, "base_ang_vel", [0.0, 0.0, 0.0], 3),
            msg.q,
            msg.qd,
            _msg_vector(msg, "foot_positions_body", np.zeros(12, dtype=np.float32), 12),
            msg.command,
        )

        if self.zero_action:
            inference_time_ms = 0.0
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            target_q = DEFAULT_Q_POLICY.copy()
        else:
            start = time.perf_counter()
            with self.torch.no_grad():
                obs_t = self.torch.from_numpy(obs).view(1, OBS_DIM)
                estimator_output = self.estimator(obs_t)
                action_t = self.body(self.torch.cat((obs_t, estimator_output), dim=1))
            inference_time_ms = (time.perf_counter() - start) * 1000.0
            action = action_t.detach().cpu().numpy().reshape(ACTION_DIM).astype(np.float32)
            target_q = action_to_target_q(action)

        self.history.update_desired_joint_positions(target_q)
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
                f"[rl_lcm_policy] running {uptime_s:.1f}s, no messages yet on "
                f"{self.state_channel}; published {self.published_count} commands",
                flush=True,
            )
            return

        age_ms = (now - self.last_state_time_s) * 1000.0
        latency = -1.0 if self.last_state_latency_ms is None else self.last_state_latency_ms
        inference = -1.0 if self.last_inference_time_ms is None else self.last_inference_time_ms
        command = ", ".join(f"{x:.2f}" for x in self.last_command)
        action_norm = float(np.linalg.norm(self.last_action))
        print(
            f"[rl_lcm_policy] running {uptime_s:.1f}s, received {self.received_count} states, "
            f"published {self.published_count} commands, last state_seq={self.last_state_sequence}, "
            f"state_age={age_ms:.0f}ms, latency={latency:.3f}ms, inference={inference:.3f}ms, "
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
    parser.add_argument("--checkpoint", required=True, help="directory containing estimator/body JIT files")
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

        _, _, estimator_path, body_path, estimator_shape, action_shape = load_and_validate_checkpoint(
            Path(args.checkpoint), torch
        )
        print(f"[rl_lcm_policy] checkpoint: {estimator_path.parent}")
        print(f"[rl_lcm_policy] estimator: {estimator_path.name}")
        print(f"[rl_lcm_policy] body: {body_path.name}")
        print(f"[rl_lcm_policy] estimator={estimator_shape}, action={action_shape}")
        return

    add_lcm_types_path(Path(args.lcm_types_dir))
    node = RapidRLPolicyNode(args)
    node.run()


if __name__ == "__main__":
    main()
