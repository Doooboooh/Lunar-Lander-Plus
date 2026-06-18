# LunarLander 强化学习实验报告

本文档汇总基础 `LunarLander-v3` 任务和“先按轨迹绕行再降落”拓展任务的实验配置、结果与分析。基础任务已完成 Q-Learning、DQN、PPO、Actor-Critic 四种方法的统一训练尝试，并新增批处理入口用于复现实验。

## 1. 基础任务

### 1.1 环境与评价指标

- 环境：`LunarLander-v3`，离散动作空间 4 个动作。
- 原始观测：8 维，包含位置、速度、角度、角速度、左右支架接触状态。
- 统一评价：训练结束后使用固定测试 seed 评估若干 episode，记录平均回报、标准差、最小/最大回报。
- 训练记录：每个算法输出 `history.csv`，保存每回合 reward、当前最好 reward、epsilon 或 update 等信息。
- 参考标准：Gym/Gymnasium 通常将平均回报 `200+` 视为 solved。本仓库中的实验主要用于课程对比和 insight，不把单 seed 结果当成最终统计结论。

### 1.2 已实现方法

| 方法 | 文件 | 核心配置 | 实验定位 |
|---|---|---|---|
| Q-Learning | `lunar_lander_rl/q_learning.py` | 将 8 维连续状态粗离散化，学习 Q 表 | 传统表格方法基线 |
| DQN | `lunar_lander_rl/dqn.py` | MLP Q 网络、经验回放、目标网络、epsilon-greedy | 离散动作深度值函数基线 |
| Actor-Critic | `lunar_lander_rl/actor_critic.py` | 共享 MLP，每个 episode 后更新策略和值函数 | 朴素策略梯度对照 |
| PPO | `lunar_lander_rl/ppo.py` | clipped objective、GAE、mini-batch 多轮更新 | 稳定策略优化方法原型 |

新增统一批处理入口：

```bash
python -m lunar_lander_rl.run_baselines --profile smoke
python -m lunar_lander_rl.run_baselines --profile course --output-dir outputs/baselines_course
python -m lunar_lander_rl.run_baselines --profile formal --algorithms q_learning,dqn --output-dir outputs/baselines_formal_q_dqn
```

批处理脚本会按算法和 seed 分目录保存模型、`history.csv`、`metrics.json`，并在顶层生成 `summary.json` 与 `summary.md`。

## 2. 基础实验结果

### 2.1 Smoke-test：流程验证

输出目录：`outputs/baselines_smoke`

| 方法 | 训练规模 | 测试回合 | 平均回报 | 标准差 | 结论 |
|---|---:|---:|---:|---:|---|
| Q-Learning | 5 episodes | 2 | -130.96 | 41.32 | 只能证明流程可跑 |
| DQN | 5 episodes | 2 | -254.86 | 88.54 | replay 尚未充分 |
| PPO | 2 updates x 128 steps | 2 | -261.93 | 12.59 | 更新步数过少 |
| Actor-Critic | 5 episodes | 2 | -358.05 | 58.75 | 高方差明显 |

这组结果只作为代码链路验证，不用于判断算法优劣。

### 2.2 Course profile：四算法同预算对比

输出目录：`outputs/baselines_course`

| 方法 | 训练配置 | 测试回合 | 平均回报 | 标准差 | 最小/最大回报 |
|---|---|---:|---:|---:|---:|
| Q-Learning | 120 episodes | 5 | -290.97 | 125.63 | -443.23 / -135.17 |
| DQN | 120 episodes | 5 | -117.89 | 16.88 | -140.29 / -92.48 |
| PPO | 40 updates x 512 steps | 5 | -838.45 | 405.44 | -1618.73 / -497.22 |
| Actor-Critic | 120 episodes | 5 | -751.89 | 581.14 | -1604.35 / -161.18 |

在这个较短但统一的课程预算下，DQN 明显最好。它虽然还没有 solved，但评估回报已经显著高于 Q-Learning，并且测试标准差更小。PPO 和 Actor-Critic 的测试表现很差，说明当前实现与超参在短预算下没有形成稳定策略。

### 2.3 Formal probe：Q-Learning 与 DQN 长一点训练

输出目录：`outputs/baselines_formal_q_dqn`

| 方法 | 训练配置 | 测试回合 | 平均回报 | 标准差 | 训练观察 |
|---|---|---:|---:|---:|---|
| Q-Learning | 400 episodes | 5 | -131.02 | 138.80 | 训练中偶有高分 episode，但测试策略不稳定 |
| DQN | 400 episodes | 5 | 118.19 | 134.83 | 训练后期多次达到正回报，最好训练回合约 298.8 |

DQN 在 400 episodes 后已经接近但尚未达到 solved 标准。其训练日志显示 episode 240 以后多次出现 `200+` 回报，说明它学到了较稳定的姿态控制和减速降落行为；测试均值仍低于 200，主要来自单 seed、训练时长不足和 best checkpoint 按单回合回报保存带来的方差。

另有已有长训结果：

| 方法 | 输出目录 | 训练规模 | 测试回合 | 平均回报 | 标准差 | 观察 |
|---|---|---:|---:|---:|---:|---|
| PPO | `outputs/ppo` | 200 updates x 1024 steps | 5 | -531.31 | 134.87 | 当前选模和超参没有学到稳定降落 |
| Actor-Critic | `outputs/actor_critic` | 600 episodes | 5 | -569.20 | 102.57 | 朴素 episode-level 更新方差仍然很大 |

## 3. 方法分析

### 3.1 Q-Learning 为什么不 work

Q-Learning 需要离散状态表，但 LunarLander 的观测是连续的 8 维状态。当前实现使用粗离散化后，状态表大小为 `8^6 * 2^2 * 4 = 1,048,576` 个动作值。这个规模已经不小，但仍然会丢失速度、角度和角速度的细节；如果继续加密离散网格，状态数又会指数增长，采样远远不够。结果表现为训练中偶尔撞到高分轨迹，但贪心测试策略泛化差。

### 3.2 DQN 为什么相对 work

DQN 仍然利用离散动作空间，但不再把连续观测硬切成表格状态，而是用 MLP 近似 Q 函数。经验回放降低了样本相关性，目标网络降低了 bootstrap 目标的震荡。400 episodes 的实验中，DQN 后期已经能频繁获得正回报，说明它学到了“保持姿态、控制水平漂移、接近地面时减速”的关键控制模式。

当前 DQN 还没有稳定 solved，原因主要是训练规模仍偏小、只跑了一个 seed、checkpoint 按单回合最高分保存会偏向偶然好轨迹。后续可以增加训练回合、按周期 eval mean 保存 best model，并跑 3 个以上 seed。

### 3.3 Actor-Critic 为什么当前不稳定

朴素 Actor-Critic 每个 episode 后才更新一次，优势估计依赖单条轨迹，方差很大。LunarLander 中坠毁、燃料消耗、接触地面等事件会导致回报跨度很大，因此策略梯度容易被少数坏轨迹带偏。course 和已有长训都显示它的测试方差高，说明当前实现更适合作为“策略梯度高方差”的反例，而不是最终强基线。

### 3.4 PPO 为什么理论上适合，但当前结果不佳

PPO 的 clipped objective 和 GAE 理论上能比朴素 Actor-Critic 稳定，尤其适合后续 waypoint 这种多阶段任务。但当前仓库实现仍比较简化，且保存 best model 主要依据训练期 recent return，而不是独立 eval mean；短预算下 policy 可能在 deterministic argmax 评估时表现很差。已有 200 updates 长训仍未成功，说明需要进一步调参，例如 reward normalization、learning rate、entropy 系数、rollout 步数、独立评估选模，以及多 seed 验证。

## 4. 拓展任务：按轨迹绕行后再降落

### 4.1 任务设计

本仓库新增 `lunar_lander_rl/trajectory_env.py`，将原始 LunarLander 包装为 waypoint 任务。飞船需要按顺序接近检查点，完成后再进入降落阶段。

当前支持三种路线：

| 任务名 | 路线含义 | 难度 |
|---|---|---|
| `two_waypoint` | 左右两个高空检查点 | 第一阶段原型，验证能否先飞向目标点 |
| `orbit` | 四个点形成近似环绕路径 | 更接近绕固定轨迹一圈 |
| `figure_eight` | 八个点形成简化 8 字路径 | 更长时序依赖，作为进阶目标 |

包装器保留原始 8 维状态，并额外加入 5 维信息：当前 waypoint 的相对位置、距离、路线完成进度、是否进入降落阶段。奖励由原始 LunarLander 奖励叠加 waypoint shaping：接近目标给奖励，远离目标扣分，进入目标半径给 bonus，未完成路线提前落地给惩罚。

### 4.2 拓展实验入口与指标

```bash
python -m lunar_lander_rl.run_trajectory --quick
python -m lunar_lander_rl.run_trajectory --algorithm dqn --task two_waypoint
python -m lunar_lander_rl.run_trajectory --algorithm ppo --task orbit
python -m lunar_lander_rl.evaluate_trajectory ppo --model-dir outputs/trajectory/two_waypoint/ppo --checkpoint best_policy.pt --task two_waypoint
```

为了系统比较多个拓展子目标，新增批处理入口：

```bash
python -m lunar_lander_rl.run_trajectory_suite --profile probe \
  --tasks single_left,near_two_waypoint,two_waypoint \
  --algorithms dqn,ppo,actor_critic

python -m lunar_lander_rl.run_trajectory_suite --profile course \
  --tasks two_waypoint --algorithms dqn,ppo \
  --output-dir outputs/trajectory_suite_two_waypoint_course

python -m lunar_lander_rl.run_trajectory_suite --profile course \
  --tasks figure_eight --algorithms dqn \
  --output-dir outputs/trajectory_suite_figure_eight_course
```

任意路径/手绘路径接口也已经支持。可以把鼠标采样点保存成 JSON，然后作为 waypoint route 训练：

```bash
python -m http.server 8000
# 浏览器打开 http://localhost:8000/examples/path_drawer.html 画路径并下载 JSON

python -m lunar_lander_rl.run_trajectory \
  --algorithm dqn \
  --waypoints-file examples/drawn_diamond_path.json \
  --route-label drawn_diamond
```

拓展任务额外保存 `trajectory_metrics.json`：

| 指标 | 含义 |
|---|---|
| `mean_waypoints_completed` | 平均每局完成几个检查点 |
| `route_completion_rate` | 完整走完指定路线的比例 |
| `landed_after_route_rate` | 走完路线后再双脚接触地面的比例 |
| `mean_final_target_distance` | episode 结束时离当前目标点的平均距离 |

这些指标比单纯 reward 更能证明“飞船真的先绕行再降落”。

### 4.3 当前拓展结果

#### 4.3.1 Smoke-test：固定路径和自定义路径接口

```bash
python -m lunar_lander_rl.run_trajectory_suite --profile smoke \
  --tasks single_left,two_waypoint \
  --algorithms dqn,ppo \
  --waypoints-file examples/drawn_diamond_path.json \
  --custom-label drawn_diamond \
  --output-dir outputs/trajectory_suite_smoke
```

| 任务 | 方法 | 平均回报 | 平均完成点数 | 路线完成率 | 说明 |
|---|---|---:|---:|---:|---|
| `single_left` | DQN | -376.6 | 0.0 / 1 | 0.00 | smoke，只验证流程 |
| `single_left` | PPO | -196.9 | 0.0 / 1 | 0.00 | smoke，只验证流程 |
| `two_waypoint` | DQN | -376.6 | 0.0 / 2 | 0.00 | smoke，只验证流程 |
| `two_waypoint` | PPO | -196.9 | 0.0 / 2 | 0.00 | smoke，只验证流程 |
| `drawn_diamond` | DQN | -377.0 | 0.0 / 4 | 0.00 | 自定义路径文件可运行 |
| `drawn_diamond` | PPO | -197.1 | 0.0 / 4 | 0.00 | 自定义路径文件可运行 |

另用单次入口补跑了 `drawn_diamond` + Actor-Critic quick，平均回报 -273.1，完成点数 0.0 / 4，路线完成率 0.00。这组结果不用于判断性能，但证明固定路线和“画好路径后作为 waypoint 文件输入”的工程链路已打通。

#### 4.3.2 Probe：从单点到两点的课程难度

```bash
python -m lunar_lander_rl.run_trajectory_suite --profile probe \
  --tasks single_left,near_two_waypoint,two_waypoint \
  --algorithms dqn,ppo,actor_critic \
  --trajectory-eval-episodes 3 \
  --output-dir outputs/trajectory_suite_probe
```

| 任务 | 方法 | 平均回报 | 平均完成点数 | 路线完成率 | 最终目标距离 |
|---|---|---:|---:|---:|---:|
| `single_left` | DQN | -195.81 | 0.33 / 1 | 0.33 | 0.88 |
| `single_left` | PPO | -231.93 | 0.00 / 1 | 0.00 | 1.15 |
| `single_left` | Actor-Critic | -243.01 | 0.00 / 1 | 0.00 | 1.15 |
| `near_two_waypoint` | DQN | -154.97 | 0.67 / 2 | 0.00 | 0.91 |
| `near_two_waypoint` | PPO | -219.14 | 0.33 / 2 | 0.00 | 1.29 |
| `near_two_waypoint` | Actor-Critic | -276.55 | 0.33 / 2 | 0.00 | 1.35 |
| `two_waypoint` | DQN | -192.46 | 0.33 / 2 | 0.00 | 1.10 |
| `two_waypoint` | PPO | -231.93 | 0.00 / 2 | 0.00 | 1.15 |
| `two_waypoint` | Actor-Critic | -254.38 | 0.00 / 2 | 0.00 | 1.13 |

Probe 结果说明：DQN 最先学到“朝 waypoint 飞”的局部行为，能偶尔完成单点，也能在近距离双点任务中平均完成更多 waypoint；PPO 和 Actor-Critic 在这个短预算下更难形成有效的 waypoint-seeking 行为。

#### 4.3.3 Two-waypoint course：第一小目标是否成功

```bash
python -m lunar_lander_rl.run_trajectory_suite --profile course \
  --tasks two_waypoint --algorithms dqn,ppo \
  --trajectory-eval-episodes 5 \
  --output-dir outputs/trajectory_suite_two_waypoint_course
```

| 任务 | 方法 | 平均回报 | 平均完成点数 | 路线完成率 | 完成路线后降落率 | 最终目标距离 |
|---|---|---:|---:|---:|---:|---:|
| `two_waypoint` | DQN | -148.21 | 0.40 / 2 | 0.20 | 0.00 | 0.78 |
| `two_waypoint` | PPO | -370.77 | 0.00 / 2 | 0.00 | 0.00 | 4.85 |

DQN 在 5 次 trajectory evaluation 中有 1 次完整通过两个 waypoint，说明第一小目标出现了弱成功，但不稳定，也没有做到“绕完再稳定降落”。PPO 在当前实现和预算下没有完成路线。

#### 4.3.4 Figure-eight course：能否轻松扩展到 8 字路径

```bash
python -m lunar_lander_rl.run_trajectory_suite --profile course \
  --tasks figure_eight --algorithms dqn \
  --trajectory-eval-episodes 5 \
  --output-dir outputs/trajectory_suite_figure_eight_course
```

| 任务 | 方法 | 平均回报 | 平均完成点数 | 路线完成率 | 最终目标距离 |
|---|---|---:|---:|---:|---:|
| `figure_eight` | DQN | -182.82 | 0.60 / 8 | 0.00 | 1.30 |

两个 waypoint 上的弱成功没有自然迁移到 8 字路径。DQN 仍能偶尔接近前几个 waypoint，但没有学到长路径的稳定阶段切换；这说明“局部目标跟踪”不等于“长时序轨迹规划”。

#### 4.3.5 Relaxed shaping 诊断

另跑了一组 relaxed probe：半径从 0.16 放宽到 0.30，并提高 waypoint bonus、route bonus。结果仍然没有完整路线成功：

| 任务 | 方法 | 平均完成点数 | 路线完成率 |
|---|---|---:|---:|
| `two_waypoint` | DQN | 0.00 / 2 | 0.00 |
| `two_waypoint` | PPO | 0.33 / 2 | 0.00 |
| `figure_eight` | DQN | 0.00 / 8 | 0.00 |
| `figure_eight` | PPO | 0.33 / 8 | 0.00 |

这说明失败不只是 waypoint 半径太严格。短预算下，探索、信用分配、阶段切换和 best checkpoint 选择共同限制了学习。

### 4.4 拓展任务 insight

1. DQN 在当前三阶段任务中最好用，但不是“轻松泛化”。它在基础 LunarLander 和两点 waypoint 上表现较好，是因为动作空间离散，Q 网络可以直接学习每个动作在当前目标相对位置下的价值。但从 2 个 waypoint 扩展到 8 字路径后，任务从局部控制变成长时序阶段任务，早期动作对很久以后的 route bonus 才产生影响，DQN 的 bootstrapping 和 epsilon 探索很难稳定分配这类长期信用。
2. PPO 理论上更适合长路径，因为它直接优化随机策略，能保持一定探索，并且 clipped objective 可以限制策略更新幅度。可是当前实验中 PPO 没有超过 DQN，主要原因可能是实现偏简化、训练预算小、reward 尺度变化大，而且评估时使用 deterministic argmax，可能没有体现随机策略的探索优势。
3. “能完成两个点”不代表能完成任意路径。当前观测只告诉策略当前目标点、进度比例和是否进入降落阶段；它没有显式提供下一段路径的曲率、剩余 waypoint 序列或未来目标。因此策略更像在学“追当前点”，而不是学“沿整条轨迹规划速度和姿态”。
4. 手绘路径接口已经具备工程可行性：只要把鼠标轨迹采样成 waypoint JSON，就可以复用同一个 wrapper 训练。但要让同一个策略泛化到很多手绘路径，需要训练时随机化路径，或加入序列模型/层级策略；只在一条固定路径上训练出的 DQN/PPO 不应期待自动泛化到任意新路径。
5. 下一步最有希望的路线是 curriculum：先单点，再近距离双点，再标准双点，再 orbit/figure-eight；同时把 best checkpoint 从“单回合最高 reward”改成“独立 trajectory eval 的 route completion rate”，否则模型保存可能偏向偶然高分但路线完成差的策略。

## 5. 结论与后续工作

基础任务中，当前最有效的方法是 DQN：它在 120 episodes 已明显优于其他方法，在 400 episodes 时测试均值达到 118.19，训练中已出现 solved 级别单回合。Q-Learning 体现了表格方法在连续高维状态下的局限；Actor-Critic 和 PPO 当前结果强调了策略梯度方法对实现细节、选模方式和训练预算的敏感性。

后续最值得补的是：

1. 基础任务增加 3 个以上 seed，并按独立 eval mean 保存 best checkpoint。
2. 继续训练 DQN，争取稳定达到 `200+` solved 标准并保存对应 GIF。
3. 对 PPO 加 reward/advantage normalization 与更稳的选模策略，再重跑基础任务。
4. 拓展任务中 DQN 已经在 `two_waypoint` 上出现弱成功，但还不能稳定降落；应继续用 curriculum 从 `single_left`、`near_two_waypoint` 逐步推进。
5. PPO 在理论上更适合长时序策略优化，但当前实现没有赢过 DQN；后续应加入 reward normalization、独立 eval 选模、并尝试 stochastic policy evaluation。
6. 对 8 字和手绘路径，当前接口已支持，但策略未成功；需要更强的路径进度编码、课程学习或层级策略，而不是直接期待两点策略无痛泛化。
7. 保存至少一个拓展任务 GIF，用于展示“先绕行再降落”的行为，而不仅是数值指标。
