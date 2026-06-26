from pathlib import Path
import re


TOTAL_LEARNING_ITERATIONS = 4500
RESUME_RUN = "runs/rapid-locomotion/2026-06-24/train/020136.400895"
def latest_checkpoint(run_path):
    checkpoints = sorted(Path(run_path).glob("checkpoints/ac_weights_*.pt"))
    numbered_checkpoints = []
    for checkpoint in checkpoints:
        match = re.fullmatch(r"ac_weights_(\d{6})\.pt", checkpoint.name)
        if match:
            numbered_checkpoints.append((int(match.group(1)), checkpoint))

    if not numbered_checkpoints:
        raise FileNotFoundError(f"No numbered checkpoints found under {Path(run_path) / 'checkpoints'}")

    return max(numbered_checkpoints, key=lambda item: item[0])


def train_mc(headless=True, resume_run=None, total_learning_iterations=TOTAL_LEARNING_ITERATIONS):

    import isaacgym
    assert isaacgym
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
    RunnerArgs.max_iterations = total_learning_iterations

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    # log the experiment parameters
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg))

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}")

    start_iteration = 0
    if resume_run is not None:
        checkpoint_iteration, checkpoint_path = latest_checkpoint(resume_run)
        weights = torch.load(checkpoint_path, map_location=runner.device)
        runner.alg.actor_critic.load_state_dict(weights)
        start_iteration = checkpoint_iteration + 1
        runner.current_learning_iteration = start_iteration
        runner.tot_timesteps = start_iteration * runner.num_steps_per_env * runner.env.num_envs
        print(f"Resuming from {checkpoint_path} at iteration {start_iteration}")

    remaining_iterations = max(0, total_learning_iterations - start_iteration)
    if remaining_iterations == 0:
        print(f"Run already reached target iteration {total_learning_iterations}. Nothing to train.")
        return

    runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from ml_logger import logger
    from mini_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    resume_run = Path(MINI_GYM_ROOT_DIR, RESUME_RUN).resolve() if RESUME_RUN else None
    if resume_run is not None:
        logger.configure(resume_run)
    else:
        logger.configure(logger.utcnow(f'rapid-locomotion/%Y-%m-%d/{stem}/%H%M%S.%f'),
                         root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    charts_path = resume_run / ".charts.yml" if resume_run is not None else None
    if charts_path is None or not charts_path.exists():
        logger.log_text("""
                charts: 
                - title: 总奖励
                  yKey: train/episode/rew_total/mean
                  xKey: iterations
                - title: 终止率
                  yKey: train/episode/termination_rate/mean
                  xKey: iterations
                - title: 终止原因：base/thigh接触
                  yKey: train/episode/termination_contact_body_rate/mean
                  xKey: iterations
                - title: 终止原因：机身高度过低
                  yKey: train/episode/termination_body_height_rate/mean
                  xKey: iterations
                - title: 终止原因：小腿非法接触
                  yKey: train/episode/termination_illegal_calf_rate/mean
                  xKey: iterations
                - title: 平均回合时长
                  yKey: train/episode/episode_length_s/mean
                  xKey: iterations
                - title: 速度跟踪奖励
                  yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - title: 站立接触脚数量
                  yKey: train/episode/diag_stand_contact_count/mean
                  xKey: iterations
                - title: 低速接触脚数量
                  yKey: train/episode/diag_crawl_contact_count/mean
                  xKey: iterations
                - title: 走步态接触脚数量
                  yKey: train/episode/diag_walk_contact_count/mean
                  xKey: iterations
                - title: 走步态正好三脚支撑比例
                  yKey: train/episode/diag_walk_exact3_rate/mean
                  xKey: iterations
                - title: 走步态换腿顺序匹配率
                  yKey: train/episode/diag_walk_sequence_match/mean
                  xKey: iterations
                - title: 走步态摆动腿前进量
                  yKey: train/episode/diag_walk_swing_progress/mean
                  xKey: iterations
                - title: 走步态连续重复抬腿率
                  yKey: train/episode/diag_walk_repeat_last_leg_rate/mean
                  xKey: iterations
                - title: 走步态近期重复抬腿率
                  yKey: train/episode/diag_walk_repeat_recent_leg_rate/mean
                  xKey: iterations
                - title: 走步态四腿覆盖率
                  yKey: train/episode/diag_walk_swing_leg_coverage_rate/mean
                  xKey: iterations
                - title: 非法小腿接触率
                  yKey: train/episode/diag_illegal_calf_contact_rate/mean
                  xKey: iterations
                - title: 走步态三脚惩罚
                  yKey: train/episode/rew_walk_exact_three_contacts/mean
                  xKey: iterations
                - title: 走步态摆动高度惩罚
                  yKey: train/episode/rew_walk_swing_clearance/mean
                  xKey: iterations
                - title: 走步态支撑脚滑移惩罚
                  yKey: train/episode/rew_walk_stance_slip/mean
                  xKey: iterations
                - title: 走步态连续重复抬腿惩罚
                  yKey: train/episode/rew_walk_repeat_last_leg/mean
                  xKey: iterations
                - title: 走步态近期重复抬腿惩罚
                  yKey: train/episode/rew_walk_repeat_recent_leg/mean
                  xKey: iterations
                - title: 走步态四腿覆盖奖励
                  yKey: train/episode/rew_walk_swing_leg_coverage/mean
                  xKey: iterations
                - title: PPO价值损失
                  yKey: mean_value_loss/mean
                  xKey: iterations
                - title: PPO策略损失
                  yKey: mean_surrogate_loss/mean
                  xKey: iterations
                - title: 每轮训练耗时
                  yKey: time_iter/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                """, filename=".charts.yml", dedent=True)

    # Set headless=False only for interactive debugging; recorded videos remain enabled.
    train_mc(headless=True, resume_run=resume_run)
    # train_mc(headless=True, resume_run=None)
