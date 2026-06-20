# 拓展 2：障碍物避障 — 实现要点

## 一、目标

在 `LunarLander-v3` 基础上加入圆形障碍物，智能体需要在降落过程中主动避开并最终安全着陆。任务从"姿态/轨迹控制"升级为"控制 + 安全避障"。

## 二、整体策略

- 在**独立新目录 `obstacle_lander/`** 中实现，import 复用 `lunar_lander_rl` 的算法，不重写训练循环。
- 对基础项目做**最小改动**：给每个 `train()` 增加可选 `env_factory` 参数（默认 `None`，向后兼容）。
- 训练入口 `obstacle_lander/train.py` 负责构造带 wrapper 的环境并传给 `train()`。

## 三、环境设计

### 3.1 障碍物表示与放置

- 每个障碍物 `(cx, cy, radius)`，默认 3 个固定障碍物，`radius=0.12`。
- 放置原则：`y∈[0.3, 1.2]`（介于起点 `y≈1.4` 与着陆平台 `y=0` 之间），避开平台正上方 `|x|<0.3 & y<0.4`，保证"绕开后可达平台"。
- 支持 `--random-obstacles`：每个 episode 重新采样（与平台、与已放置障碍物两两去重），测试泛化。

### 3.2 状态空间

- 原 8 维 + 6 维（3 个障碍物相对飞船的 `(rel_x, rel_y)`）= **14 维**。
- `Box` 相对坐标上下界取 **±3.0**（最坏 `|lander_x − cx| ≤ 2.3`，留余量）。

### 3.3 碰撞检测与奖励

对每个障碍物计算 `dist`，按区间处理：

| 区间 | 处理 |
|---|---|
| `dist < radius`（碰撞） | `reward -= 100`，**`terminated=True`** |
| `radius ≤ dist < 2*radius`（警告区） | `reward -= 0.5 * (1 − dist/(2*radius))`（线性递减） |
| `dist ≥ 2*radius`（安全） | 不处理 |

**关键语义**：碰撞是真正的失败终止态，必须用 `terminated=True`，**不能**用 `truncated=True`（后者在 Gymnasium 中指"外部截断、MDP 本可继续"，会让 bootstrap 含义混乱）。

## 四、`ObstacleLanderEnv` wrapper 实现要点

继承 `gym.Wrapper`，关键点：

- 构造时扩展 `observation_space`：拼接基础 `low/high` 与 `±3.0` 的扩展段。
- `obstacles` 默认参数用 `None` + 内部赋值（**避免可变默认参数陷阱**），存一份 `_default_obstacles` 用于退化。
- `_extend_obs(obs)`：取 `obs[0], obs[1]` 计算每个障碍物相对偏移，拼接返回 14 维。
- `_obstacle_bonus(x, y)`：返回 `(reward 增量, 是否碰撞)`；命中碰撞立即返回。
- `reset(*, seed, options)`：随机模式下重新采样障碍物，再调 `self.env.reset(...)`，**用 `self.env`**（不是 `self.base_env`，gym.Wrapper 标准 API）。
- `step(action)`：调底层 step → 拿到 `obs[0], obs[1]` → 计算 bonus → 碰撞则 `terminated=True` 并写 `info["collision"]=True` → 返回扩展观测。
- `_sample_obstacles()`：拒绝采样 50 次内取够 N 个，失败则退化为默认障碍物，保证 `reset` 不卡死。

## 五、基础项目最小改动

- `dqn.py / ppo.py / actor_critic.py`：`train(cfg)` → `train(cfg, env_factory=None)`；内部 `env = env_factory(cfg.seed) if env_factory else make_env(seed=cfg.seed)`。每个文件 ~3 行，默认参数保证不破坏现状。
- `q_learning.py` **不改**：拓展 2 不评估 Q-Learning（见第六节）。
- `evaluate.py` **不改**：拓展 2 自带评估器（见第七节）。
- `common.py` 可选：抽出 `_make_base_env` 供 `make_env` 和本 wrapper 共用，约 5 行。

## 六、Q-Learning 的特殊处理

`q_learning.py` 写死了 8 维的 `OBS_LOW / OBS_HIGH / N_BINS`。直接扩到 14 维会触发维度灾难（即使每维只 4 个 bin，状态数也乘以 `4^6 = 4096`）。

处理方式：**拓展 2 不评估 Q-Learning**，在报告中把它作为"维度灾难 → 深度 RL 必要性"的反面证据。（备选：只用前 8 维离散化、忽略障碍物维度，但表格方法在随机障碍物下基本学不动，价值不大。）

## 七、评估与 GIF（`obstacle_lander/evaluate.py`）

**不复用** `common.evaluate_policy` —— 它内部直接 `gym.make("LunarLander-v3")`，维度是 8，加载 14 维策略会崩。

本拓展独立实现：
- 用 `ObstacleLanderEnv` 包装的 `render_mode="rgb_array"` 环境跑 N 个 episode。
- 输出 `mean/std/min/max return` 与 **`collision_rate`**（避障专项指标，对应 `info["collision"]` 频率）。
- 用 `imageio.mimsave` 把单 episode 帧存成 GIF。
- **诚实声明**：Gymnasium 默认渲染器画不出我们自定义的障碍物圆，GIF 中看不到障碍物；如需可见，需后续用 Pillow/opencv 在每帧叠加圆形（增量工作，第一阶段不做）。

## 八、文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `obstacle_lander/__init__.py` | 新增 | 空包 |
| `obstacle_lander/env.py` | 新增 | `ObstacleLanderEnv` wrapper |
| `obstacle_lander/train.py` | 新增 | 训练入口，调用各算法 `train(..., env_factory=...)` |
| `obstacle_lander/evaluate.py` | 新增 | 障碍物环境评估 + GIF |
| `obstacle_lander/outputs/` | 新增 | 训练产物 |
| `lunar_lander_rl/dqn.py` | 微调 | `train()` 加 `env_factory` 参数 |
| `lunar_lander_rl/ppo.py` | 微调 | 同上 |
| `lunar_lander_rl/actor_critic.py` | 微调 | 同上 |

## 九、运行示例

```
python -m obstacle_lander.train --algo dqn --episodes 400
python -m obstacle_lander.train --algo ppo --random-obstacles
python -m obstacle_lander.evaluate --algo dqn \
    --model-dir obstacle_lander/outputs/dqn \
    --episodes 10 --gif obstacle_lander/outputs/dqn_obstacle.gif
```

## 十、冒烟测试（长训练前的最小检查）

- `reset` 返回 `obs` 长度 = 14；`observation_space.shape[0] = 14`。
- 把飞船放到障碍物中心附近，确认 `reward` 大幅为负、`terminated=True`。
- `--episodes 2` 能跑完两个完整 episode。
- 基础项目 `python -m lunar_lander_rl.dqn --episodes 2` 仍正常（验证 `env_factory=None` 路径未坏）。

## 十一、预期效果

- DQN / PPO / Actor-Critic 都能学到避障策略，但**收敛更慢、最终 return 略低**（避障 shaping 必然引入负奖励）。
- 算法差异更明显：**PPO** 在 14 维 + 随机障碍物下稳定性最好；**DQN** 固定障碍物学好、随机障碍物泛化差；**Actor-Critic** 方差大；**Q-Learning** 因维度灾难基本不可学。
- 新增 `collision_rate` 指标可直接对比各算法的**避障安全性**，而不只看 return。
