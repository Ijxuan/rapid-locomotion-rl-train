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
    "feet_clearance": "足端抬脚奖励",
    "collision": "碰撞惩罚",
    "action_rate": "动作变化率惩罚",
    "action_smoothness_1": "动作平滑惩罚1",
    "action_smoothness_2": "动作平滑惩罚2",
    "stand_still": "静止惩罚",
    "command_area": "指令覆盖面积",
    "terrain_level": "地形等级",
    "max_command_yaw": "最大偏航指令",
    "paper_b_positive_reward": "Paper-B正奖励合计",
    "paper_b_negative_reward": "Paper-B负项合计",
    "paper_b_gate": "Paper-B指数门控",
    "paper_b_nontermination_reward": "Paper-B非终止奖励",
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
    "feet_clearance",
    "command_area",
    "terrain_level",
    "max_command_yaw",
    "paper_b_positive_reward",
    "paper_b_gate",
    "paper_b_nontermination_reward",
  }
  smaller_is_better = {
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
    "collision",
    "action_rate",
    "action_smoothness_1",
    "action_smoothness_2",
    "stand_still",
    "paper_b_negative_reward",
  }
  if reward_key in bigger_is_better:
    return "备注：越大越好"
  if reward_key in smaller_is_better:
    return "备注：越小越好（更接近0更好）"
  return "备注：目标是稳定收敛"


def _write_dashboard_charts(logger, reward_metric_keys):
  total_note = _reward_note_zh("total")
  chart_lines = [
    "charts:",
    f"- title: \"训练/总奖励\\n{total_note}\"",
    f"  description: {total_note}",
    "  yKey: train/episode/rew_total/mean",
    "  xKey: iterations",
  ]

  # Keep the default command curriculum chart if available.
  command_note = _reward_note_zh("command_area")
  chart_lines.extend([
    f"- title: \"训练/指令覆盖面积\\n{command_note}\"",
    f"  description: {command_note}",
    "  yKey: train/episode/command_area/mean",
    "  xKey: iterations",
  ])

  for key in sorted(reward_metric_keys):
    if key == "rew_total":
      continue
    reward_name = key[len("rew_"):] if key.startswith("rew_") else key
    note = _reward_note_zh(reward_name)
    chart_lines.extend([
      f"- title: \"训练/{_reward_title_zh(reward_name)}\\n{note}\"",
      f"  description: {note}",
      f"  yKey: train/episode/{key}/mean",
      "  xKey: iterations",
    ])

  chart_lines.extend([
    "- title: 训练视频",
    "  type: video",
    "  glob: \"videos/*.mp4\"",
  ])

  logger.log_text("\n".join(chart_lines) + "\n", filename=".charts.yml")


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
    _write_dashboard_charts(logger, reward_metric_keys)

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
