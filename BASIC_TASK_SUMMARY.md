# LunarLander 基础任务整理

本文档整理仓库中基础 `LunarLander-v3` 任务已经实现的代码、算法设计、实验配置和已有结果。对应代码主要位于 `lunar_lander_rl/`，结果主要位于 `outputs/baselines_*`、`outputs/dqn`、`outputs/ppo`、`outputs/actor_critic` 和 `outputs/compare`。

## 1. 任务定义

基础任务使用 Gymnasium 的 `LunarLander-v3` 环境。智能体需要控制登月飞船在二维物理环境中安全降落到中间着陆平台，目标是在尽量少消耗燃料的前提下保持姿态稳定、降低速度并完成着陆。

通常以评估平均回报是否达到 `200+` 作为 solved 参考线。本仓库中的结果主要用于课程大作业中的算法比较，当前大部分实验是单 seed、短预算或中等预算结果，因此更适合说明算法趋势，而不是作为严格统计结论。

## 2. 环境描述

- 环境 ID：`LunarLander-v3`
- 动作空间：离散 4 动作
- 最大步数：训练与评估默认 `1000`
- 依赖：`gymnasium`、`box2d`、`numpy`、`torch`
- 环境入口：`lunar_lander_rl/common.py` 中的 `make_env`

动作含义如下：

| 动作编号 | 含义 |
|---:|---|
| 0 | 不喷火 |
| 1 | 左侧发动机 |
| 2 | 主发动机 |
| 3 | 右侧发动机 |

原始观测为 8 维连续/二值混合向量：

| 维度 | 含义 |
|---:|---|
| 0 | 水平位置 `x` |
| 1 | 垂直位置 `y` |
| 2 | 水平速度 `vx` |
| 3 | 垂直速度 `vy` |
| 4 | 飞船角度 `angle` |
| 5 | 角速度 `angular_v` |
| 6 | 左支架是否接触地面 |
| 7 | 右支架是否接触地面 |

评估指标由 `evaluate_policy` 统一输出：

| 指标 | 含义 |
|---|---|
| `mean_return` | 固定测试回合平均回报 |
| `std_return` | 测试回报标准差 |
| `min_return` / `max_return` | 测试回合最小/最大回报 |
| `history.csv` | 训练每回合回报、最好回报、探索率等日志 |

## 3. 已有代码结构

| 文件 | 内容 |
|---|---|
| `lunar_lander_rl/common.py` | 环境创建、随机种子、MLP、评估、保存日志 |
| `lunar_lander_rl/q_learning.py` | 表格 Q-Learning，连续状态离散化 |
| `lunar_lander_rl/dqn.py` | DQN，经验回放、目标网络、epsilon-greedy |
| `lunar_lander_rl/actor_critic.py` | 朴素 Actor-Critic，episode-level 更新 |
| `lunar_lander_rl/ppo.py` | PPO，GAE、clipped objective、mini-batch 更新 |
| `lunar_lander_rl/run_baselines.py` | 统一批处理入口，支持 smoke/course/formal profile |
| `lunar_lander_rl/evaluate.py` | 加载模型并评估/可视化 |

## 4. 算法设计与公式

### 4.1 Q-Learning

Q-Learning 是表格值函数方法。由于原环境观测是连续状态，本仓库先将 8 维观测离散化：

- 前 6 维连续变量各分为 8 个 bin。
- 两个支架接触状态各分为 2 个 bin。
- Q 表形状为 `8^6 * 2^2 * 4` 个动作值。

动作选择使用 epsilon-greedy：

```text
a_t = random action, with probability epsilon
a_t = argmax_a Q(s_t, a), otherwise
```

Q 值更新公式：

```text
Q(s_t, a_t) <- Q(s_t, a_t) + alpha [r_t + gamma max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
```

默认关键配置：

| 参数 | 默认值 |
|---|---:|
| episodes | 400 |
| learning rate `alpha` | 0.08 |
| discount `gamma` | 0.99 |
| epsilon | `1.0 -> 0.05` |
| epsilon decay | 0.995 |

### 4.2 DQN

DQN 用神经网络近似动作价值函数 `Q_theta(s,a)`，避免粗离散化造成的信息损失。网络结构为：

```text
Linear(obs_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, action_dim)
```

核心机制：

- replay buffer 存储 `(s, a, r, s', done)`，随机采样 batch 降低样本相关性。
- target network `Q_{theta^-}` 每隔固定步数同步，降低 bootstrap 目标震荡。
- epsilon-greedy 进行探索。
- 损失函数使用 Huber loss。

TD 目标与损失：

```text
y = r + gamma (1 - done) max_{a'} Q_{theta^-}(s', a')
L(theta) = Huber(Q_theta(s,a), y)
```

默认关键配置：

| 参数 | 默认值 |
|---|---:|
| episodes | 400 |
| hidden dim | 128 |
| learning rate | 1e-3 |
| gamma | 0.99 |
| replay size | 100000 |
| batch size | 64 |
| warmup steps | 1000 |
| target update | 1000 |
| epsilon decay steps | 30000 |

### 4.3 Actor-Critic

Actor-Critic 同时学习策略和值函数，网络共享两层 MLP，然后分出 actor head 和 critic head：

```text
shared MLP -> actor logits pi_theta(a|s)
           -> critic value V_phi(s)
```

策略采样：

```text
a_t ~ Categorical(logits_theta(s_t))
```

折扣回报：

```text
G_t = sum_{k=t}^{T} gamma^{k-t} r_k
```

优势估计与损失：

```text
A_t = G_t - V_phi(s_t)
L_policy = - mean(log pi_theta(a_t|s_t) A_t)
L_value = MSE(V_phi(s_t), G_t)
L = L_policy + c_v L_value - c_e H(pi_theta)
```

默认关键配置：

| 参数 | 默认值 |
|---|---:|
| episodes | 600 |
| hidden dim | 128 |
| learning rate | 1e-3 |
| gamma | 0.99 |
| value coef | 0.5 |
| entropy coef | 0.01 |

### 4.4 PPO

PPO 是策略优化方法，本仓库实现了离散动作 PPO + GAE。网络结构为两层 Tanh MLP，共享 body 后分别输出 actor logits 和 critic value。

GAE 递推：

```text
delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma lambda A_{t+1}
```

PPO clipped objective：

```text
rho_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
L_clip = - mean(min(rho_t A_t, clip(rho_t, 1-epsilon, 1+epsilon) A_t))
```

总损失：

```text
L = L_clip + c_v MSE(V(s_t), R_t) - c_e H(pi_theta)
```

默认关键配置：

| 参数 | 默认值 |
|---|---:|
| updates | 200 |
| rollout steps | 1024 |
| hidden dim | 128 |
| learning rate | 3e-4 |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| clip coef | 0.2 |
| update epochs | 4 |
| minibatch size | 256 |

## 5. 实验配置

统一入口：

```bash
python -m lunar_lander_rl.run_baselines --profile smoke
python -m lunar_lander_rl.run_baselines --profile course --output-dir outputs/baselines_course
python -m lunar_lander_rl.run_baselines --profile formal --algorithms q_learning,dqn --output-dir outputs/baselines_formal_q_dqn
```

profile 含义：

| profile | 用途 | 配置概要 |
|---|---|---|
| smoke | 验证代码链路是否可运行 | 极少训练回合，不用于性能比较 |
| course | 课程预算下四算法同配置比较 | Q/DQN/AC 120 episodes，PPO 40 updates x 512 steps |
| formal | 默认更长训练配置 | 默认 Q/DQN 400 episodes，PPO 200 updates，AC 600 episodes |

## 6. 基础任务已有结果

### 6.1 Course profile 四算法对比

输出目录：`outputs/baselines_course`

| 方法 | 训练配置 | 测试回合 | 平均回报 | 标准差 | 结论 |
|---|---|---:|---:|---:|---|
| Q-Learning | 120 episodes | 5 | -290.97 | 125.63 | 粗离散化难以泛化 |
| DQN | 120 episodes | 5 | -117.89 | 16.88 | 短预算下最好且更稳定 |
| PPO | 40 updates x 512 steps | 5 | -838.45 | 405.44 | 当前超参下未形成有效策略 |
| Actor-Critic | 120 episodes | 5 | -751.89 | 581.14 | 高方差明显 |

结论：在统一短预算下，DQN 明显优于其余三种方法。Q-Learning 受状态离散化限制，Actor-Critic 和 PPO 在当前实现/超参下训练不稳定。

### 6.2 Formal probe：Q-Learning 与 DQN 更长训练

输出目录：`outputs/baselines_formal_q_dqn`

| 方法 | 训练配置 | 测试回合 | 平均回报 | 标准差 | 观察 |
|---|---|---:|---:|---:|---|
| Q-Learning | 400 episodes | 5 | -131.02 | 138.80 | 有改善，但测试波动仍大 |
| DQN | 400 episodes | 5 | 118.19 | 134.83 | 接近 solved，但未稳定达到 200+ |

DQN 训练日志中后期多次出现正回报和 `200+` 单回合，说明已经学习到姿态控制、水平速度控制和接近地面减速等关键行为；评估均值仍低于 solved 标准，主要受训练预算、单 seed、按单回合最高回报保存 checkpoint 等因素影响。

### 6.3 已有单独长训结果

| 方法 | 输出目录 | 训练规模 | 测试回合 | 平均回报 | 标准差 | 观察 |
|---|---|---:|---:|---:|---:|---|
| PPO | `outputs/ppo` | 200 updates x 1024 steps | 5 | -531.31 | 134.87 | 当前 PPO 实现/超参未学到稳定降落 |
| Actor-Critic | `outputs/actor_critic` | 600 episodes | 5 | -569.20 | 102.57 | episode-level 更新方差仍较大 |

### 6.4 可视化结果

已有基础任务 GIF：

| 文件 | 说明 |
|---|---|
| `outputs/dqn_lunar_lander.gif` | DQN 基础着陆演示 |
| `outputs/test_dqn_lunar_lander.gif` | DQN 测试演示 |

在报告中可引用：

```markdown
![DQN LunarLander 着陆效果](outputs/dqn_lunar_lander.gif)
```

## 7. 效果分析

1. Q-Learning 的主要问题是状态离散化。LunarLander 的连续观测对速度、角度、角速度很敏感，粗网格会丢失控制细节；加密网格又导致状态数指数增长，采样不足。
2. DQN 在当前仓库中最适合基础任务。环境动作本身是离散的，DQN 可以直接学习每个动作的 Q 值，同时保留连续状态输入。
3. Actor-Critic 当前实现是较朴素的 episode-level 更新，优势估计来自单条轨迹，容易被坠毁等高方差回报带偏。
4. PPO 理论上更稳定，但当前实现仍需调参和更严谨的 checkpoint 选择。可优先尝试 reward normalization、独立 eval mean 保存 best model、多 seed、学习率和 entropy 系数搜索。

