# 拓展 3：多个飞船顺序降落

负责人：邢书一

把单飞船的 `LunarLander-v3` 任务扩展为**多阶段决策**：一次 episode 控制 N 艘飞船依次降落，
体现"顺序调度 + 飞船间约束 + 任务级奖励"。

## 一、设计思路

不重新训练飞船控制，而是**复用基础实验训练好的单飞船策略**（默认 DQN，也可切换 PPO /
Actor-Critic / Q-Learning），把工作重点放在**调度层**与**任务级奖励**上：

- 每架飞船用单飞船策略完整跑一个子 episode；
- 调度器为每架规划起飞点（避开已降落飞船）；
- 记录每架落点，作为后续飞船的约束；
- 任务级奖励 = 各架原生 reward 之和 + 顺序奖 + 避碰惩罚 + 全局成功 bonus。

## 二、关键发现（报告重点）

复用预训练 DQN 时发现：**单飞船策略是一个"回中器"**。

无论飞船从哪个 x 起飞，落点都收敛到平台中心 ±0.1：

```
15 次从随机起飞点 (-0.6~0.6) 降落，落点 x：
  mean = -0.065, std = 0.027, 90% 落在 [-0.11, -0.03]
```

这导致"让多架飞船在空间上错开落点"在不重训策略的前提下**不可行**。我们验证了三种
诱导偏移落点的方法均失败：

| 方法 | 结果 |
| ---- | ---- |
| 喂观测偏置（obs[0] -= target） | 目标 +0.4 仍落到 +0.00 |
| 给朝目标的初始水平速度 | 落点不变，速度大反而坠毁 |
| PD 横向覆盖动作 | 破坏策略升力，reward 跌至 -568 |

**结论**：低层控制目标（策略回中）与高层调度目标（错开落点）冲突，任何外部干预都会
破坏策略的着陆稳定性。这是复用预训练策略做空间调度的根本限制。

## 三、本模块的实现：时序错开

鉴于上述发现，本模块采用**时序错开（sequential）**模式实现多飞船任务：

- 多架飞船**依次**起飞、降落；
- 前一架完成（落地或失败）后**物理清除**，下一架再进入；
- 用**时序顺序**而非空间分离实现多飞船调度。

这既保证了 demo 可演示，又如实反映了复用单飞船策略的能力边界。代码同时保留了
`spatial`（空间错开）模式接口，供未来用"目标条件策略"（改奖励重训）升级。

## 四、用法

在项目根目录 `Lunar-Lander-Plus/` 下（已激活 `.venv`）：

```bash
# 跑 3 艘飞船、5 个 episode，打印每架状态 + 统计
python -m lunar_lander_rl.multi_lander.demo --num-landers 3 --episodes 5

# 切换单飞船策略
python -m lunar_lander_rl.multi_lander.demo --algorithm ppo --num-landers 4 --episodes 10

# 保存一段多飞船 GIF（rgb_array 录制，不弹窗）
python -m lunar_lander_rl.multi_lander.demo --num-landers 3 --episodes 1 --gif outputs/multi_lander_sequential.gif
```

## 五、评价指标（多飞船任务级）

| 指标 | 说明 |
| ---- | ---- |
| `mean_total_return` | 整个多飞船 episode 的累计 reward 均值 |
| `mean_landed` | 平均平稳落地飞船数（/ 总数） |
| `mean_collided` | 平均碰撞次数 |
| `all_success_rate` | 全部飞船平稳且无碰撞的成功率 |

## 六、文件结构

```
multi_lander/
  config.py     # MultiLanderConfig：任务规模、约束、奖励权重、模式
  policy.py     # SingleLanderPolicy：复用 evaluate.load_policy 的统一 act 接口
  scheduler.py  # LanderScheduler：起飞点 / 目标落点规划（sequential & spatial）
  env.py        # SequentialMultiLanderEnv：多飞船顺序降落环境 + 落点重定位
  demo.py       # 命令行入口：跑 episode + 统计 + GIF
```

## 七、典型结果（DQN，3 艘，5 episode）

```
mean_total_return: ~761
mean_landed: 2.2 / 3
mean_collided: 0.0
all_success_rate: 0.4
```

全部成功的 episode 合计 reward 约 900（3 架 × ~250 原生 + 顺序奖 + 全局 bonus）。
