# Lunar Lander RL Course Project

这个仓库用于课程大作业：用强化学习完成 `LunarLander-v3` 基础任务，并在同一套实验框架下尝试变体任务。

## 仓库结构

```text
.
├── configs/                 # 每个实验一个 JSON 配置
├── lunar_lander_rl/          # 可复用 Python 包
│   ├── envs/                 # Lunar Lander 环境与变体
│   │   ├── common.py         # 少量通用绘图、坐标和窗口工具
│   │   ├── base.py           # 基础环境包装
│   │   ├── moving_pad.py     # 移动平台变体
│   │   ├── obstacle.py       # 障碍物变体
│   │   └── waypoint.py       # 轨迹点变体
│   ├── experiments.py        # 训练、评估、模型加载
│   └── config.py             # 配置读写工具
├── scripts/                  # 命令行入口
│   ├── check_envs.py         # 检查自定义环境是否符合 Gymnasium/SB3 接口
│   ├── render_env.py         # 渲染随机/启发式策略 GIF
│   ├── train.py              # 按配置训练
│   └── evaluate.py           # 加载模型并评估
├── outputs/                  # 训练模型、指标、GIF 等产物
├── reports/                  # 实验报告、图表、PPT 素材
├── PROJECT_INTRO.md          # 项目介绍与课程计划
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt
```

检查自定义环境：

```bash
python scripts/check_envs.py
```

训练基础 PPO：

```bash
python scripts/train.py --config configs/base_ppo.json --output-dir outputs/base_ppo
```

训练 1M 步基础 PPO：

```bash
python scripts/train.py --config configs/base_ppo_1m.json --output-dir outputs/base_ppo_1m
```

评估已训练模型：

```bash
python scripts/evaluate.py \
  --algorithm ppo \
  --model outputs/base_ppo/ppo_BaseLunarLander-v0.zip \
  --env BaseLunarLander-v0 \
  --vec-normalize outputs/base_ppo/vec_normalize.pkl
```

查看训练过程 TensorBoard：

```bash
tensorboard --logdir outputs/base_ppo_1m/tensorboard
```

生成训练曲线 PNG：

```bash
python scripts/plot_training.py \
  --monitor outputs/base_ppo_1m/monitor/train.monitor.csv \
  --output reports/base_ppo_1m_training.png
```

渲染自定义环境 GIF：

```bash
python scripts/render_env.py --env all --output outputs/obstacle_demo.gif
```

## 已注册的任务

| 环境 ID | 说明 |
|---|---|
| `LunarLander-v3` | Gymnasium 原始基础任务 |
| `BaseLunarLander-v0` | 本项目基础任务包装环境，直接包装原始 `LunarLander-v3` |
| `MovingPadLunarLander-v0` | 着陆平台水平移动 |
| `ObstacleLunarLander-v0` | 加入障碍物坐标和碰撞惩罚 |
| `WaypointLunarLander-v0` | 需要依次经过轨迹点后着陆 |

## 扩展方式

新增实验优先复制 `configs/base_ppo.json`，修改 `env_id`、`algorithm`、`total_timesteps` 和 `model_kwargs`。

新增任务时，优先在 `lunar_lander_rl/envs/` 下新建单独文件，例如 `wind.py` 或 `fuel_limit.py`。每个环境类都直接继承 `gym.Env`，并在自己的 `__init__` 中通过 `gym.make("LunarLander-v3", ...)` 包装原始环境，再独立修改观测、奖励、终止条件和绘制逻辑。最后在 `lunar_lander_rl/envs/__init__.py` 的 `register_custom_lunar_envs()` 中注册新的环境 ID。

当前训练入口使用 Stable-Baselines3 提供最小可运行闭环。后续如果要加入自写 Q-Learning、DQN、Actor-Critic 或 PPO，建议放在 `lunar_lander_rl/algorithms/`，并复用 `configs/`、`outputs/` 和 `scripts/evaluate.py` 的约定。
