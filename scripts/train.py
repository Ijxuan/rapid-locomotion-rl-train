import isaacgym

assert isaacgym


def _reward_title_zh(reward_key):
  """Convert internal reward metric names to readable Chinese chart titles."""
  explicit_map = {
    "total": "总奖励",
    "termination": "终止惩罚",
    "tracking_lin_vel": "线速度跟踪",
    "tracking_ang_vel": "角速度跟踪",
    "tracking_lin_vel_lat": "横向线速度跟踪",
    "tracking_lin_vel_long": "纵向线速度跟踪",
    "lin_vel_z": "竖直线速度惩罚",
    "ang_vel_xy": "横摆角速度惩罚",
    "orientation": "姿态惩罚",
    "torques": "力矩惩罚",
    "dof_vel": "关节速度惩罚",
    "dof_acc": "关节加速度惩罚",
    "dof_pos": "关节位置惩罚",
    "base_height": "机体高度惩罚",
    "base_motion": "机体抖动惩罚",
    "feet_air_time": "足端腾空奖励",
    "feet_stumble": "足端绊倒惩罚",
    "feet_slip": "足端打滑惩罚",
    "feet_clearance": "足端抬脚高度误差惩罚",
    "collision": "碰撞惩罚",
    "action_rate": "动作变化率惩罚",
    "action_smoothness_1": "动作平滑惩罚1",
    "action_smoothness_2": "动作平滑惩罚2",
    "stand_still": "静止惩罚",
    "moving_stand_still": "移动指令下速度缺口/反向惩罚",
    "command_area": "指令覆盖面积",
    "terrain_level": "地形等级",
    "max_command_yaw": "最大偏航指令",
    "paper_b_positive_reward": "Paper-B正奖励合计",
    "paper_b_negative_reward": "Paper-B门控负项合计",
    "paper_b_gate": "Paper-B指数门控",
    "paper_b_nontermination_reward": "Paper-B非终止奖励",
    "mean_cmd_vx": "平均目标前向速度",
    "mean_base_vx": "平均实际前向速度",
    "mean_abs_vx_error": "平均前向速度绝对误差",
    "mean_cmd_vy": "平均目标横向速度",
    "mean_base_vy": "平均实际横向速度",
    "mean_abs_vy_error": "平均横向速度绝对误差",
    "mean_cmd_yaw": "平均目标偏航角速度",
    "mean_base_yaw_rate": "平均实际偏航角速度",
    "mean_abs_yaw_error": "平均偏航角速度绝对误差",
    "moving_cmd_fraction": "移动前向指令占比",
    "moving_standstill_fraction": "移动指令下速度不足/反向占比",
  }
  if reward_key in explicit_map:
    return explicit_map[reward_key]
  return f"奖励项/{reward_key}"


def _reward_note_zh(reward_key):
  """Return optimization hint for each metric: bigger/smaller/target."""
  bigger_is_better = {
    "total",
    "tracking_lin_vel",
    "tracking_ang_vel",
    "tracking_lin_vel_lat",
    "tracking_lin_vel_long",
    "feet_air_time",
    "terrain_level",
    "max_command_yaw",
    "paper_b_positive_reward",
    "paper_b_gate",
    "paper_b_nontermination_reward",
  }
  zero_is_best = {
    "termination",
    "lin_vel_z",
    "ang_vel_xy",
    "orientation",
    "torques",
    "dof_vel",
    "dof_acc",
    "dof_pos",
    "base_height",
    "base_motion",
    "feet_stumble",
    "feet_slip",
    "feet_clearance",
    "collision",
    "action_rate",
    "action_smoothness_1",
    "action_smoothness_2",
    "stand_still",
    "moving_stand_still",
    "paper_b_negative_reward",
  }
  smaller_is_better = {
    "mean_abs_vx_error",
    "mean_abs_vy_error",
    "mean_abs_yaw_error",
    "moving_standstill_fraction",
  }
  informational = {
    "mean_cmd_vx",
    "mean_base_vx",
    "mean_cmd_vy",
    "mean_base_vy",
    "mean_cmd_yaw",
    "mean_base_yaw_rate",
    "moving_cmd_fraction",
  }
  if reward_key in bigger_is_better:
    return "备注：越大越好"
  if reward_key in zero_is_best:
    return "备注：越接近0越好"
  if reward_key == "command_area":
    return "备注：Paper-B路径下仅作兼容显示，不用于判断课程扩展"
  if reward_key in smaller_is_better:
    return "备注：越小越好，0最好"
  if reward_key in informational:
    return "备注：诊断指标，用于对比目标和实际"
  return "备注：目标是稳定收敛"


DASHBOARD_METRIC_ORDER = (
  "rew_total",
  "rew_paper_b_nontermination_reward",
  "rew_termination",
  "rew_paper_b_gate",
  "rew_paper_b_positive_reward",
  "rew_paper_b_negative_reward",
  "rew_tracking_lin_vel",
  "mean_cmd_vx",
  "mean_base_vx",
  "mean_abs_vx_error",
  "rew_tracking_ang_vel",
  "mean_cmd_yaw",
  "mean_base_yaw_rate",
  "mean_abs_yaw_error",
  "mean_cmd_vy",
  "mean_base_vy",
  "mean_abs_vy_error",
  "moving_cmd_fraction",
  "moving_standstill_fraction",
  "rew_moving_stand_still",
  "rew_feet_air_time",
  "rew_feet_clearance",
  "rew_feet_slip",
  "rew_orientation",
  "rew_base_motion",
  "rew_dof_pos",
  "rew_dof_vel",
  "rew_dof_acc",
  "rew_torques",
  "rew_action_smoothness_1",
  "rew_action_smoothness_2",
  "command_area",
  "terrain_level",
  "max_command_yaw",
)


UNBOUNDED_NEGATIVE_REWARD_NAMES = {
  "paper_b_negative_reward",
  "feet_clearance",
  "feet_slip",
  "dof_vel",
  "dof_acc",
  "base_motion",
  "lin_vel_z",
  "ang_vel_xy",
  "base_height",
  "collision",
  "feet_stumble",
  "action_rate",
  "stand_still",
}


def _metric_name(metric_key):
  return metric_key[len("rew_"):] if metric_key.startswith("rew_") else metric_key


def _yaml_quote(text):
  return '"' + str(text).replace('"', '\\"') + '"'


def _as_float(value, default=None):
  try:
    if hasattr(value, "item"):
      value = value.item()
    return float(value)
  except (TypeError, ValueError):
    return default


def _as_float_list(values):
  if values is None:
    return []
  if hasattr(values, "detach"):
    values = values.detach().cpu()
  if hasattr(values, "numpy"):
    values = values.numpy()
  if hasattr(values, "tolist"):
    values = values.tolist()
  if isinstance(values, (int, float)):
    return [float(values)]
  result = []
  try:
    iterator = iter(values)
  except TypeError:
    return result
  for value in iterator:
    if isinstance(value, (list, tuple)):
      result.extend(_as_float_list(value))
    else:
      converted = _as_float(value)
      if converted is not None:
        result.append(converted)
  return result


def _nested_get(obj, path, default=None):
  current = obj
  for part in path.split("."):
    if current is None:
      return default
    if isinstance(current, dict):
      current = current.get(part, default)
    else:
      current = getattr(current, part, default)
  return current


def _fmt_number(value):
  if value == float("inf"):
    return "inf"
  if value == float("-inf"):
    return "-inf"
  value = float(value)
  if abs(value) < 1e-12:
    return "0"
  if abs(value - round(value)) < 1e-9:
    return str(int(round(value)))
  text = f"{value:.4f}".rstrip("0").rstrip(".")
  return text if text != "-0" else "0"


def _range_closed(low, high):
  return f"[{_fmt_number(low)}, {_fmt_number(high)}]"


def _negative_range_from_raw_max(scale, raw_max_per_step, episode_steps):
  scale = _as_float(scale, 0.0)
  raw_max_per_step = max(0.0, _as_float(raw_max_per_step, 0.0))
  if scale >= 0.0:
    return _range_closed(0.0, scale * raw_max_per_step * episode_steps)
  return _range_closed(scale * raw_max_per_step * episode_steps, 0.0) + "，0最好"


def _command_range(cfg, name, default):
  if name == "lin_vel_x" and _nested_get(cfg, "commands.paper_b_command_curriculum", False):
    value = _nested_get(cfg, "commands.paper_b_vx_final", default)
  else:
    value = _nested_get(cfg, f"command_ranges.{name}", None)
    if value is None:
      value = _nested_get(cfg, f"commands.{name}", default)
  values = _as_float_list(value)
  return values[:2] if len(values) >= 2 else list(default)


def _dashboard_context(env=None):
  cfg = getattr(env, "cfg", None)
  reward_scales = getattr(env, "reward_scales", None)
  if reward_scales is None:
    reward_scales = _nested_get(cfg, "rewards.scales", {})
  if not isinstance(reward_scales, dict):
    reward_scales = vars(reward_scales)

  dt = _as_float(getattr(env, "dt", None), None)
  if dt is None:
    sim_dt = _as_float(_nested_get(cfg, "sim.dt", None), 0.005)
    decimation = _as_float(_nested_get(cfg, "control.decimation", None), 1.0)
    dt = sim_dt * decimation
  episode_steps = _as_float(getattr(env, "max_episode_length", None), None)
  if episode_steps is None:
    episode_steps = _as_float(_nested_get(cfg, "env.max_episode_length", None), None)
  if episode_steps is None:
    episode_length_s = _as_float(_nested_get(cfg, "env.episode_length_s", None), 20.0)
    episode_steps = episode_length_s / dt if dt else 2000.0

  foot_count = len(getattr(env, "feet_indices", [])) if env is not None and hasattr(env, "feet_indices") else 4
  foot_count = foot_count or 4
  return {
    "cfg": cfg,
    "reward_scales": reward_scales,
    "dt": dt,
    "episode_steps": episode_steps,
    "foot_count": foot_count,
    "action_clip": _as_float(_nested_get(cfg, "normalization.clip_actions", None), 1.0),
    "torque_limits": _as_float_list(getattr(env, "torque_limits", None)),
    "dof_pos_limits": _as_float_list(getattr(env, "dof_pos_limits", None)),
    "default_dof_pos": _as_float_list(getattr(env, "default_dof_pos", None)),
  }


def _reward_scale(context, name, default=0.0):
  return _as_float(context["reward_scales"].get(name, default), default)


def _paper_b_positive_reward_max(context):
  cfg = context["cfg"]
  foot_airtime_cap = _as_float(_nested_get(cfg, "rewards.paper_b_airtime_cap", None), 0.2)
  per_step = max(0.0, _reward_scale(context, "tracking_lin_vel"))
  per_step += max(0.0, _reward_scale(context, "tracking_ang_vel"))
  per_step += max(0.0, _reward_scale(context, "feet_air_time")) * context["foot_count"] * foot_airtime_cap
  return per_step * context["episode_steps"]


def _paper_b_direct_reward_bounds(context):
  return float("-inf"), 0.0


def _torque_penalty_range(context):
  torque_limits = context["torque_limits"]
  if not torque_limits:
    return "(-inf, 0]，0最好"
  raw_max = sum(limit * limit for limit in torque_limits)
  return _negative_range_from_raw_max(_reward_scale(context, "torques"), raw_max, context["episode_steps"])


def _dof_pos_penalty_range(context):
  limits = context["dof_pos_limits"]
  defaults = context["default_dof_pos"]
  if len(limits) < 2 or not defaults:
    return "(-inf, 0]，0最好"
  pairs = list(zip(limits[0::2], limits[1::2]))
  defaults = defaults[-len(pairs):]
  raw_max = 0.0
  for (low, high), default in zip(pairs, defaults):
    raw_max += max(abs(low - default), abs(high - default)) ** 2
  return _negative_range_from_raw_max(_reward_scale(context, "dof_pos"), raw_max, context["episode_steps"])


def _dashboard_range_text(metric_key, env=None):
  context = _dashboard_context(env)
  cfg = context["cfg"]
  name = _metric_name(metric_key)
  episode_steps = context["episode_steps"]

  if not metric_key.startswith("rew_"):
    if name in {"moving_cmd_fraction", "moving_standstill_fraction", "command_area"}:
      return "[0, 1]"
    if name == "terrain_level":
      num_rows = max(1.0, _as_float(_nested_get(cfg, "terrain.num_rows", None), 1.0))
      return _range_closed(0.0, num_rows - 1.0)
    if name == "max_command_yaw":
      yaw_limit = _as_float(_nested_get(cfg, "commands.max_yaw_curriculum", None), 1.0)
      return _range_closed(0.0, yaw_limit)
    if name == "mean_cmd_vx":
      low, high = _command_range(cfg, "lin_vel_x", [-1.0, 1.0])
      return _range_closed(low, high)
    if name == "mean_cmd_vy":
      low, high = _command_range(cfg, "lin_vel_y", [-1.0, 1.0])
      return _range_closed(low, high)
    if name == "mean_cmd_yaw":
      low, high = _command_range(cfg, "ang_vel_yaw", [-1.0, 1.0])
      return _range_closed(low, high)
    if name in {"mean_abs_vx_error", "mean_abs_vy_error", "mean_abs_yaw_error"}:
      return "[0, inf)，0最好"
    if name in {"mean_base_vx", "mean_base_vy", "mean_base_yaw_rate"}:
      return "(-inf, inf)"
    return "未标定"

  if name == "paper_b_gate":
    gate_floor = _as_float(_nested_get(cfg, "rewards.paper_b_reward_gate_floor", None), 0.05)
    return _range_closed(gate_floor, 1.0)
  if name in {"tracking_lin_vel", "tracking_ang_vel", "tracking_lin_vel_lat", "tracking_lin_vel_long"}:
    return _range_closed(0.0, max(0.0, _reward_scale(context, name)) * episode_steps)
  if name == "feet_air_time":
    raw_max = context["foot_count"] * _as_float(_nested_get(cfg, "rewards.paper_b_airtime_cap", None), 0.2)
    return _range_closed(0.0, max(0.0, _reward_scale(context, name)) * raw_max * episode_steps)
  if name == "moving_stand_still":
    return "(-inf, 0]，0最好"
  if name == "termination":
    penalty = _as_float(_nested_get(cfg, "rewards.paper_b_termination_penalty", None), None)
    if penalty is None:
      penalty = _reward_scale(context, "termination")
    return _range_closed(min(0.0, penalty), max(0.0, penalty)) + "，0最好"
  if name == "paper_b_positive_reward":
    return _range_closed(0.0, _paper_b_positive_reward_max(context))
  if name == "paper_b_nontermination_reward":
    direct_min, direct_max = _paper_b_direct_reward_bounds(context)
    return _range_closed(direct_min, _paper_b_positive_reward_max(context) + direct_max)
  if name == "total":
    penalty = _as_float(_nested_get(cfg, "rewards.paper_b_termination_penalty", None), _reward_scale(context, "termination"))
    direct_min, direct_max = _paper_b_direct_reward_bounds(context)
    return _range_closed(direct_min + min(0.0, penalty), _paper_b_positive_reward_max(context) + direct_max)
  if name == "orientation":
    return _negative_range_from_raw_max(_reward_scale(context, name), 3.141592653589793 ** 2, episode_steps)
  if name == "torques":
    return _torque_penalty_range(context)
  if name == "dof_pos":
    return _dof_pos_penalty_range(context)
  if name == "action_smoothness_1":
    raw_max = 12.0 * (2.0 * context["action_clip"]) ** 2
    return _negative_range_from_raw_max(_reward_scale(context, name), raw_max, episode_steps)
  if name == "action_smoothness_2":
    raw_max = 12.0 * (4.0 * context["action_clip"]) ** 2
    return _negative_range_from_raw_max(_reward_scale(context, name), raw_max, episode_steps)
  if name in UNBOUNDED_NEGATIVE_REWARD_NAMES:
    return "(-inf, 0]，0最好"
  return "未标定"


def _auxiliary_dashboard_metric_keys(env=None):
  cfg = getattr(env, "cfg", None)
  keys = []
  if _nested_get(cfg, "commands.command_curriculum", False):
    keys.append("command_area")
  if _nested_get(cfg, "terrain.curriculum", False):
    keys.append("terrain_level")
  if _nested_get(cfg, "commands.yaw_command_curriculum", False):
    keys.append("max_command_yaw")
  return keys


def _ordered_dashboard_metric_keys(reward_metric_keys, diagnostic_metric_keys=None, env=None):
  available = set(reward_metric_keys or [])
  available.update(diagnostic_metric_keys or [])
  available.update(_auxiliary_dashboard_metric_keys(env))

  ordered = []
  for key in DASHBOARD_METRIC_ORDER:
    if key in available and key not in ordered:
      ordered.append(key)

  unknown = sorted(key for key in available if key not in ordered)
  return ordered + unknown


def _chart_lines_for_metric(metric_key, env=None):
  name = _metric_name(metric_key)
  note = _reward_note_zh(name)
  range_text = _dashboard_range_text(metric_key, env)
  title = f"训练/{_reward_title_zh(name)} | 范围: {range_text}\\n{note}"
  description = f"{note}；理论范围: {range_text}"
  return [
    f"- title: {_yaml_quote(title)}",
    f"  description: {_yaml_quote(description)}",
    f"  yKey: train/episode/{metric_key}/mean",
    "  xKey: iterations",
  ]


def _render_dashboard_charts(reward_metric_keys, diagnostic_metric_keys=None, env=None):
  chart_lines = ["charts:"]
  for key in _ordered_dashboard_metric_keys(reward_metric_keys, diagnostic_metric_keys, env):
    chart_lines.extend(_chart_lines_for_metric(key, env))
  chart_lines.extend([
    "- title: 训练视频",
    "  type: video",
    "  glob: \"videos/*.mp4\"",
  ])
  return "\n".join(chart_lines) + "\n"


def _write_dashboard_charts(logger, reward_metric_keys, diagnostic_metric_keys=None, env=None):
  logger.log_text(_render_dashboard_charts(reward_metric_keys, diagnostic_metric_keys, env), filename=".charts.yml")


def apply_2070_profile(Cfg, RunnerArgs):
    """Reduce Isaac Gym and rollout buffers for smaller GPUs."""
    Cfg.env.num_envs = 32
    Cfg.env.record_video = False
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 4
    Cfg.terrain.border_size = 0
    Cfg.terrain.curriculum = False
    Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 20
    Cfg.sim.physx.default_buffer_size_multiplier = 5
    RunnerArgs.num_steps_per_env = 16
    RunnerArgs.save_video_interval = 0
    return {
        "num_envs": Cfg.env.num_envs,
        "num_steps_per_env": RunnerArgs.num_steps_per_env,
        "terrain_num_rows": Cfg.terrain.num_rows,
        "terrain_num_cols": Cfg.terrain.num_cols,
        "terrain_border_size": Cfg.terrain.border_size,
        "max_gpu_contact_pairs": Cfg.sim.physx.max_gpu_contact_pairs,
        "default_buffer_size_multiplier": Cfg.sim.physx.default_buffer_size_multiplier,
        "record_video": Cfg.env.record_video,
    }


def train_mc(headless=True, sim_device="cuda:0", num_learning_iterations=4000, profile_2070=False):
    import torch

    from mini_gym.envs.base.legged_robot_config import Cfg
    from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
    from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from mini_gym_learn.ppo import Runner
    from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from mini_gym_learn.ppo.actor_critic import AC_Args
    from mini_gym_learn.ppo.ppo import PPO_Args
    from mini_gym_learn.ppo import RunnerArgs

    config_mini_cheetah(Cfg)
    profile_settings = apply_2070_profile(Cfg, RunnerArgs) if profile_2070 else {}

    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)

    reward_metric_keys = [f"rew_{k}" for k in env.episode_sums.keys()]
    diagnostic_metric_keys = list(getattr(env, "episode_metric_sums", {}).keys())
    _write_dashboard_charts(logger, reward_metric_keys, diagnostic_metric_keys, env)

    # log the experiment parameters
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg),
                      PaperB=dict(
                          observation_dim=Cfg.env.num_observations,
                          estimator_dim=Cfg.env.estimator_output_dim,
                          actor_input_dim=Cfg.env.num_observations + Cfg.env.estimator_output_dim,
                          estimator_jit="estimator_latest.jit",
                          body_jit="body_latest.jit",
                          headless=headless,
                          sim_device=sim_device,
                          num_learning_iterations=num_learning_iterations,
                          low_memory_profile="2070" if profile_2070 else "default",
                          low_memory_profile_settings=profile_settings,
                      ))

    env = HistoryWrapper(env)
    runner = Runner(env, device=sim_device)
    runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True, eval_freq=100)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train the Paper B Mini Cheetah reproduction policy.")
    parser.set_defaults(headless=True)
    parser.add_argument("--headless", action="store_true", dest="headless", help="run without the Isaac Gym viewer")
    parser.add_argument("--show", action="store_false", dest="headless", help="open the Isaac Gym viewer")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("-2070", "--rtx2070", action="store_true", dest="profile_2070",
                        help="use a lower-memory training profile for RTX 2070/low-VRAM GPUs")
    return parser.parse_args()


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from mini_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'rapid-locomotion/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )

    args = parse_args()
    train_mc(headless=args.headless, sim_device=args.sim_device, num_learning_iterations=args.iterations,
             profile_2070=args.profile_2070)
