import isaacgym

assert isaacgym


def train_mc(headless=True, sim_device="cuda:0", num_learning_iterations=4000):
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

    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)

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
    return parser.parse_args()


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from mini_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'rapid-locomotion/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/command_area/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                """, filename=".charts.yml", dedent=True)

    args = parse_args()
    train_mc(headless=args.headless, sim_device=args.sim_device, num_learning_iterations=args.iterations)
