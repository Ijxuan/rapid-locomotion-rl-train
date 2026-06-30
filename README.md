# Paper B Mini Cheetah 复现分支

本分支 `paper-b-reproduction` 基于原始 rapid-locomotion 训练框架和 Mini Cheetah 资产，改造成 Ji et al. 的论文 B：**Concurrent Training of a Control Policy and a State Estimator**（RA-L 2022）复现配置。

当前分支的主要变化：

* 观测改为论文 B 的 142 维 layout。
* estimator 监督目标和输出为 11 维：`base_lin_vel(3) + foot_height(4) + contact_probability(4)`。
* actor 输入为 `obs(142) + estimator_output(11) = 153` 维，输出 12 维 action。
* actor / critic MLP 为 `[512, 256, 64]`，estimator MLP 为 `[256, 128]`。
* 训练导出两个 TorchScript 文件：`estimator_latest.jit` 和 `body_latest.jit`。
* reward 聚合使用论文 B 公式：`r_total = r_pos * exp(0.2 * r_neg)`。
* Mini Cheetah 默认配置为 800 env、100 Hz policy、`action_scale=0.1`、`Kp=17`、`Kd=0.4`。
* command curriculum、初始状态随机化、25% previous-final-state reset、ground friction、observation noise、motor dry friction、PD additive noise、sphere-foot radius 随机化均按论文 B 路径接入。

## 关键文件

训练入口：

```text
scripts/train.py
```

远端 Isaac Gym smoke：

```text
scripts/paper_b_remote_smoke.py
```

播放 / 探针：

```text
scripts/play.py
scripts/play_zero_probe.py
```

部署侧 observation 和 I/O 辅助：

```text
mini_gym/deploy/rapid_locomotion_policy.py
scripts/rl_lcm_policy.py
```

Paper B 相关核心实现：

```text
mini_gym/envs/base/paper_b_observation.py
mini_gym/envs/base/paper_b_rewards.py
mini_gym/envs/base/paper_b_commands.py
mini_gym/envs/base/paper_b_assets.py
mini_gym/envs/base/paper_b_defaults.py
mini_gym/envs/base/legged_robot.py
mini_gym_learn/ppo/actor_critic.py
mini_gym_learn/ppo/ppo.py
mini_gym_learn/ppo/rollout_storage.py
mini_gym_learn/ppo/__init__.py
```

## 环境准备

进入仓库后先激活测试 / 训练环境：

```bash
cd /home/xjtx/rl/rapid-locomotion-rl-main
conda activate gym4
```

如果当前 shell 找不到 `conda activate`，可先加载 conda：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate gym4
```

本机可以运行不依赖 Isaac Gym 的 Python / Torch 检查。真正创建环境、reset、step、PPO smoke、正式训练都需要安装 Isaac Gym 并使用可用 NVIDIA GPU。

## 本机非 Isaac 检查

这些命令不会创建 Isaac Gym 环境，适合在本机先检查代码结构、shape、TorchScript 导出和部署 I/O。

```bash
python -m unittest \
  scripts.test_paper_b_layout \
  scripts.test_paper_b_rewards \
  scripts.test_paper_b_ppo_source \
  scripts.test_paper_b_command_asset \
  scripts.test_paper_b_eval_metrics \
  scripts.test_rl_deploy
```

静态语法检查：

```bash
python -m py_compile \
  mini_gym/envs/base/paper_b_observation.py \
  mini_gym/envs/base/paper_b_rewards.py \
  mini_gym/envs/base/paper_b_commands.py \
  mini_gym/envs/base/paper_b_assets.py \
  mini_gym/envs/base/legged_robot.py \
  mini_gym/envs/mini_cheetah/velocity_tracking/velocity_tracking_easy_env.py \
  mini_gym_learn/ppo/actor_critic.py \
  mini_gym_learn/ppo/ppo.py \
  mini_gym_learn/ppo/rollout_storage.py \
  mini_gym_learn/ppo/__init__.py \
  mini_gym_learn/eval_metrics/metrics.py \
  mini_gym/deploy/rapid_locomotion_policy.py \
  scripts/train.py \
  scripts/play.py \
  scripts/play_zero_probe.py \
  scripts/paper_b_remote_smoke.py \
  scripts/rl_lcm_policy.py
```

## 远端 Isaac Gym Smoke

在安装好 Isaac Gym 的机器上拉取当前分支：

```bash
git checkout paper-b-reproduction
git pull
conda activate gym4
```

最小环境检查：创建指定数量的 env，`reset()` 后确认 observation shape 为 142、privileged target shape 为 11，并检查 `*_foot` body 和 terminal contact 配置。

```bash
python scripts/paper_b_remote_smoke.py --check env --num-envs 8 --sim-device cuda:0
```

PPO smoke：跑少量 rollout 和 update，确认 value loss、surrogate loss、estimator loss 有限。

```bash
python scripts/paper_b_remote_smoke.py --check ppo --num-envs 8 --ppo-iters 2 --steps-per-iter 4 --sim-device cuda:0
```

step smoke：跑 5 step，确认 reward finite、reset 正常、terminal penalty 为 `-10`。

```bash
python scripts/paper_b_remote_smoke.py --check steps --num-envs 8 --steps 5 --sim-device cuda:0
```

command / DR smoke：确认 10% zero-command 采样比例大致正常，DR buffer 有限。

```bash
python scripts/paper_b_remote_smoke.py --check command-dr --num-envs 20 --command-resamples 20 --sim-device cuda:0
```

JIT smoke：确认 `estimator_latest.jit` 和 `body_latest.jit` 的导出、查找、加载和 play loader 路径正确。

```bash
python scripts/paper_b_remote_smoke.py --check jit --sim-device cuda:0
```

一次性跑完整 smoke：

```bash
python scripts/paper_b_remote_smoke.py --check all --num-envs 8 --sim-device cuda:0
```

## 怎么训练

正式训练入口是：

```bash
python scripts/train.py --headless --sim-device cuda:0 --iterations 4000
```

这条命令会使用 `config_mini_cheetah(Cfg)` 中的 Paper B 默认配置：

* `Cfg.env.num_envs = 800`
* `Cfg.env.num_observations = 142`
* `Cfg.env.num_privileged_obs = 11`
* `Cfg.control.action_scale = 0.1`
* `Cfg.control.stiffness = {"joint": 17.0}`
* `Cfg.control.damping = {"joint": 0.4}`
* policy dt 为 0.01 s，即 100 Hz

如果要打开 Isaac Gym viewer：

```bash
python scripts/train.py --show --sim-device cuda:0 --iterations 4000
```

如果只是快速确认训练链路，不想完整训练 4000 iteration：

```bash
python scripts/train.py --headless --sim-device cuda:0 --iterations 10
```

低显存机器可以加 `-2070`，默认配置不会变，只有传这个选项时才会降低并行环境数量、terrain 尺寸和 PhysX GPU buffer：

```bash
python scripts/train.py -2070 --headless --sim-device cuda:0 --iterations 4000
```

快速检查低显存训练链路：

```bash
python scripts/train.py -2070 --headless --sim-device cuda:0 --iterations 10
```

`-2070` profile 当前会使用：

* `Cfg.env.num_envs = 128`
* `RunnerArgs.num_steps_per_env = 16`
* `Cfg.terrain.num_rows = 1`
* `Cfg.terrain.num_cols = 1`
* `Cfg.terrain.border_size = 0`
* `Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 20`
* `Cfg.sim.physx.default_buffer_size_multiplier = 5`
* `Cfg.env.record_video = False`

RTX 4070 Ti 建议先用默认 800 env 跑正式训练。RTX 2050 / RTX 2070 这类显存较小的机器优先用 `-2070` 或只跑 smoke test。

训练日志会写到：

```text
runs/rapid-locomotion/<日期>/train/<时间戳>/
```

关键输出在：

```text
runs/rapid-locomotion/<日期>/train/<时间戳>/checkpoints/
```

其中：

```text
ac_weights_*.pt           # PyTorch 权重 checkpoint
ac_weights_last.pt        # 最新 PyTorch 权重副本
estimator_latest.jit      # estimator TorchScript
body_latest.jit           # actor body TorchScript
```

部署和播放路径默认使用 `estimator_latest.jit` + `body_latest.jit`，不再使用旧的 `adaptation_module_latest.jit`。

## 训练曲线

训练时 `ml_logger` 会记录 reward、value loss、surrogate loss、estimator loss 等指标。若要看 dashboard，可在 `runs` 目录上层启动：

```bash
python -m ml_dash.app
python -m ml_dash.server .
```

浏览器默认访问：

```text
http://localhost:3001
```

常用 profile：

```text
Username: runs
API: http://localhost:8081
Access Token: 留空
```

## 播放和探针

播放最近一次训练结果：

```bash
python scripts/play.py
```

headless 播放逻辑默认会加载最新 run 下的：

```text
checkpoints/estimator_latest.jit
checkpoints/body_latest.jit
```

零命令探针，用于看 0 command 下 observation、estimator 输出、action 和目标关节角：

```bash
python scripts/play_zero_probe.py --run runs/rapid-locomotion/example --headless
```

带 viewer：

```bash
python scripts/play_zero_probe.py --run runs/rapid-locomotion/example --show
```

只打印关节角：

```bash
python scripts/play_zero_probe.py --run runs/rapid-locomotion/example --angles-only
```

## 部署侧 JIT 校验

只校验 checkpoint 目录下是否能找到并加载 `estimator_latest.jit` 和 `body_latest.jit`：

```bash
python scripts/rl_lcm_policy.py \
  --checkpoint runs/rapid-locomotion/<日期>/train/<时间戳>/checkpoints \
  --validate-only
```

实际 LCM 节点会从 robot state 构造 142 维 observation，先跑 estimator，再把 `obs + estimator_output` 输入 body，最终发布 12 维 action 和目标关节位置。

## 原项目来源

本仓库最初来自 **Rapid Locomotion via Reinforcement Learning**：

```bibtex
@inproceedings{margolisyang2022rapid,
  title={Rapid Locomotion via Reinforcement Learning},
  author={Margolis, Gabriel and Yang, Ge and Paigwar,
          Kartik and Chen, Tao and Agrawal, Pulkit},
  booktitle={Robotics: Science and Systems},
  year={2022}
}
```

原项目基于：

* legged gym / Rudin et al.
* Isaac Gym / NVIDIA
* rsl_rl / Rudin et al.
* ml_logger / jaynes

保留的第三方代码仍遵循仓库中的原始 license 文件。
