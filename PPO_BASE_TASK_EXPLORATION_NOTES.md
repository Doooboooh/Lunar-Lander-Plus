# PPO 基础任务探索记录

本文档用于持续记录基础 `LunarLander-v3` 任务上的 PPO 优化过程，包括已经做过的尝试、失败现象、当前判断和下一步优先方向。后续若继续优化基础任务上的 PPO，请优先更新本文档，而不是只把结果散落在 `outputs/` 目录里。

更新时间：`2026-06-23`

## 1. 当前目标

当前阶段的明确门槛是：

1. 先在基础环境 `LunarLander-v3` 上找到一个效果明显优于当前版本的 PPO。
2. 至少达到“稳定降落趋势明显，独立评估平均回报大于 `100`”。
3. 在基础环境门槛未达到之前，不继续把 PPO 作为拓展 waypoint 环境的主力方案。

目前结论：**这个门槛还没有达到。**

## 2. 当前基线问题

仓库原始 PPO 实现和结果：

- 代码：[lunar_lander_rl/ppo.py](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/lunar_lander_rl/ppo.py)
- 原始长训结果目录：`outputs/ppo`
- 旧报告中的记录：平均回报约 `-531.31`

原始版本的主要问题：

1. `best_policy.pt` 主要依据训练过程里的 recent return 保存，而不是依据独立 deterministic evaluation。
2. 网络结构固定为 `2 x 128 + tanh`，搜索空间太窄。
3. 没有观测归一化。
4. 没有系统化的超参搜索入口。

## 3. 已完成的代码改动

### 3.1 PPO 训练器增强

文件：[lunar_lander_rl/ppo.py](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/lunar_lander_rl/ppo.py)

已加入：

- 可配置网络宽度 `hidden_dim`
- 可配置层数 `hidden_layers`
- 可配置激活函数 `activation in {tanh, relu, elu}`
- 可配置 `eval_interval`
- 可配置 `selection_eval_episodes`
- 训练中按独立评估均值保存 `best_policy.pt`
- `RunningMeanStd` 观测归一化
- 归一化统计量保存为 `obs_norm.npz`

### 3.2 搜索脚本

文件：[lunar_lander_rl/run_ppo_search.py](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/lunar_lander_rl/run_ppo_search.py)

作用：

- 支持 `base` 和 `trajectory` 两种模式
- 支持 `tiny / probe / course` 三种预算 profile
- 支持候选配置集合搜索
- 支持通过 `--names` 指定只跑某几组配置
- 支持覆盖 `updates`、`rollout_steps`、`eval_interval`、`selection_eval_episodes`
- 自动输出 `summary.json` 和 `summary.md`

### 3.3 PPO 模型加载链路修正

以下文件已支持在评估/trajectory 评估时一并加载 `obs_norm.npz`：

- [lunar_lander_rl/run_ppo_search.py](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/lunar_lander_rl/run_ppo_search.py)
- [lunar_lander_rl/run_trajectory.py](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/lunar_lander_rl/run_trajectory.py)
- [lunar_lander_rl/run_trajectory_suite.py](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/lunar_lander_rl/run_trajectory_suite.py)

## 4. 已尝试的实验阶段

### 4.1 第一阶段：小预算结构/超参筛选

目标：先看 PPO 对结构和基础超参是否敏感。

代表结果目录：

- [outputs/ppo_search_probe/base/probe/summary.md](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/outputs/ppo_search_probe/base/probe/summary.md)

主要候选：

- `baseline_tanh_128`
- `relu_128_low_entropy`
- `tanh_256_low_lr`
- `elu_256`

结果：

| 排名 | 配置 | 平均回报 |
|---:|---|---:|
| 1 | `relu_128_low_entropy` | `-137.91` |
| 2 | `baseline_tanh_128` | `-200.12` |
| 3 | `tanh_256_low_lr` | `-237.18` |
| 4 | `elu_256` | `-248.29` |

结论：

- PPO 对结构和熵系数是敏感的。
- 小预算下 `relu + lower entropy` 看起来最好。
- 但即使是最好的配置，也离 `100+` 的目标差得很远。

### 4.2 第二阶段：course 预算下复验小预算优胜者

目标：验证“小预算更好”的配置在更长训练下是否仍成立。

代表结果目录：

- [outputs/ppo_search_course/base/course/summary.md](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/outputs/ppo_search_course/base/course/summary.md)

结果：

| 排名 | 配置 | 平均回报 |
|---:|---|---:|
| 1 | `relu_128_low_entropy` | `-153.11` |
| 2 | `baseline_tanh_128` | `-663.50` |

结论：

- `relu_128_low_entropy` 明显优于原始基线。
- 但它仍然远低于 `100+`，甚至没有稳定进入正回报区域。
- 单纯靠“降低 entropy + 换激活函数”不够。

### 4.3 第三阶段：扩大结构搜索面

目标：引入更宽网络、更长 rollout、更保守 clip 等组合。

代表结果目录：

- [outputs/ppo_search_base_focus/base/course/summary.md](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/outputs/ppo_search_base_focus/base/course/summary.md)

结果：

| 排名 | 配置 | 平均回报 |
|---:|---|---:|
| 1 | `tanh_256_low_lr` | `-153.11` |
| 2 | `relu_256_low_lr` | `-340.66` |
| 3 | `relu_256_longer_rollout` | `-365.42` |
| 4 | `relu_128_low_entropy` | `-441.16` |
| 5 | `conservative_clip` | `-653.16` |

结论：

- 一旦训练拉长，`relu` 系列配置普遍更容易崩。
- 相对更稳的是 `tanh_256_low_lr`。
- “更宽 + 更低学习率”比“更激进的 relu 结构”更靠谱。

### 4.4 第四阶段：长训练确认

目标：把最有希望的 `tanh_256` 系列拉长到 `200 updates`，确认是否只是预算不足。

代表结果目录：

- `outputs/ppo_search_base_long/base/course/summary.json`

关键现象：

- `tanh_256_low_lr`
- `tanh_256_very_low_entropy`

这两组在长训练中最好 checkpoint 的独立评估一度能接近 `-100`，但最终 5 episode eval 仍在 `-153.11` 左右，没有突破正回报，更没有达到 `100+`。

结论：

- 问题已经不只是“训练没跑够”。
- 当前 PPO 实现存在更深层的稳定性瓶颈。

### 4.5 第五阶段：加入观测归一化

目标：验证 observation normalization 是否能改善 PPO 的训练早期行为和稳定性。

代表结果目录：

- [outputs/ppo_search_norm_base/base/course/summary.md](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/outputs/ppo_search_norm_base/base/course/summary.md)

结果：

| 排名 | 配置 | 平均回报 |
|---:|---|---:|
| 1 | `tanh_256_low_lr` | `-153.11` |
| 2 | `tanh_256_very_low_entropy` | `-153.11` |

观察：

- 归一化没有把训练搞坏。
- 某些 early eval 变得更健康，早期学习曲线也更像“在学东西”。
- 但最终结果仍未跨过关键门槛。

结论：

- 观测归一化是合理改动，建议保留。
- 但它不是决定性修复。

### 4.6 第六阶段：更保守的 PPO 更新

目标：降低 critic/更新强度，避免“前期学到一点，后期又洗掉”。

代表结果目录：

- [outputs/ppo_search_norm_value/base/course/summary.md](/workspace/volume/doubohan-disk2/proj/Lunar-Lander-Plus/outputs/ppo_search_norm_value/base/course/summary.md)

尝试：

- `tanh_256_low_value`
- `tanh_256_low_value_fewer_epochs`
- `tanh_256_clip01`

结果：

| 排名 | 配置 | 平均回报 |
|---:|---|---:|
| 1 | `tanh_256_low_value_fewer_epochs` | `-271.31` |
| 2 | `tanh_256_low_value` | `-319.85` |
| 3 | `tanh_256_clip01` | `-363.88` |

结论：

- 更保守更新没有改善结果，反而更差。
- 说明当前问题不只是“更新太猛”。

## 5. 已确认的失败方向

下面这些方向已经有比较明确的负面证据，后续除非改动了 PPO 的更底层机制，否则不建议优先重复投入时间：

1. 只靠 `relu + lower entropy` 就想把 PPO 拉到可用水平。
2. 只靠加宽网络或加长 rollout，而不改训练稳定化机制。
3. 只靠降低 `value_coef`、减少 `update_epochs`、减小 `clip_coef`。
4. 只靠把训练时长从 `40/80` update 继续机械拉长。

## 6. 当前最靠谱的基础 PPO 候选

如果必须从目前尝试里选一个“相对最值得继续”的基础 PPO 起点，优先顺序如下：

1. `tanh_256_low_lr`
2. `tanh_256_very_low_entropy`
3. `relu_128_low_entropy`

原因：

- `tanh_256_low_lr` 在更长训练里最稳，虽然没有达到目标。
- `tanh_256_very_low_entropy` 和它表现接近，可以作为同族对照。
- `relu_128_low_entropy` 小预算时效果好，但长训练更容易崩。

## 7. 关键现象总结

这轮实验暴露出的核心现象不是“PPO 完全学不会”，而是：

1. PPO 在训练早期偶尔能学到一点有效控制。
2. 但后续更新经常把已经学到的行为冲掉。
3. 独立评估均值会在训练中出现明显震荡，最好时也大多停留在 `-100` 左右。
4. 当前实现下，PPO 的问题更像“训练稳定性不足”，不是“搜索空间太小”。

## 8. 当前判断

截至 `2026-06-23`，可以给出下面这个判断：

**当前仓库里的 PPO 还不适合直接作为拓展任务主算法。**

在基础环境里都还没有达到：

- 稳定正回报
- 独立评估平均回报 `> 100`
- 明显优于当前 DQN 基线

因此，后续 PPO 工作必须继续留在基础环境，直到先把这个门槛过掉。

## 9. 下一步优先方向

后续最值得尝试的不是继续盲搜同类超参，而是补更标准的 PPO 稳定化机制。建议优先级如下：

1. 多环境并行 rollout
2. learning rate annealing
3. reward/return normalization
4. value loss clipping 或更稳的 critic 训练
5. advantage/return 的更系统统计监控
6. 多 seed 评估，而不是只盯一个 seed

更具体的执行建议：

1. 先在当前 PPO 训练器里加入并行环境。
2. 再加入学习率退火和 return normalization。
3. 以 `tanh_256_low_lr` 为基础配置重新跑基础环境。
4. 目标是先拿到一个独立评估 `mean_return > 100` 的基础 PPO。
5. 只有达到这个门槛后，才继续进入拓展 waypoint 环境探索。

## 10. 推荐复现实验命令

基础环境小预算筛选：

```bash
python -m lunar_lander_rl.run_ppo_search \
  --mode base \
  --profile probe \
  --max-configs 4 \
  --eval-episodes 3 \
  --output-dir outputs/ppo_search_probe
```

基础环境聚焦 `tanh_256_low_lr` 一组：

```bash
python -m lunar_lander_rl.run_ppo_search \
  --mode base \
  --profile course \
  --names tanh_256_low_lr,tanh_256_very_low_entropy \
  --updates 80 \
  --eval-interval 20 \
  --selection-eval-episodes 5 \
  --eval-episodes 5 \
  --output-dir outputs/ppo_search_norm_base
```

当前不建议继续优先运行的方向：

```bash
python -m lunar_lander_rl.run_ppo_search \
  --mode trajectory \
  --task two_waypoint \
  ...
```

原因：基础环境门槛尚未通过。

## 11. 结论

这轮工作最大的价值，不是已经找到了一个足够好的 PPO，而是把“哪些方向不够用”排得比较清楚了。

已经确认的结论是：

- 单纯超参搜索不够。
- 单纯网络结构搜索不够。
- 单纯拉长训练不够。
- 观测归一化值得保留，但不是决定性修复。

后续若继续优化 PPO，请默认从“增强 PPO 实现稳定性”开始，而不是再做一轮同质化参数搜索。

