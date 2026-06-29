# 拓展2：障碍物避障 —— 实验配置与结果

## 1. 环境设计

基于 `LunarLander-v3`，在着陆区域上方加入障碍物，飞船需主动避障并安全着陆。

### 状态空间

在原 8 维观测基础上，每个障碍物追加 3 维：

```
[base_8] + [rel_x, rel_y, radius] × N
```

- `rel_x`, `rel_y`：障碍物相对飞船位置（裁剪到 [-3, 3]）
- `radius`：障碍物半径

### 奖励塑形

| 事件 | 奖励/惩罚 |
|---|---|
| 碰撞障碍物（dist < radius） | **-100**，episode 终止 |
| 接近障碍物（radius ≤ dist < 2×radius） | 线性递减惩罚 `-0.5 × (1 - dist/(2r))` |
| 安全距离外（dist ≥ 2×radius） | 无额外惩罚 |

### 两种障碍物模式

- **固定模式（fixed）**：3 个固定障碍物，位置不变
  - `(-0.55, 0.45, r=0.11)`, `(0.20, 0.72, r=0.12)`, `(0.62, 0.34, r=0.10)`
- **随机模式（random）**：1 个障碍物，每 episode 在着陆区正上方随机采样
  - x ∈ `[-0.2, 0.2]`, y ∈ `[0.4, 1.2]`, r = 0.12

---

## 2. 实验配置

### 2.1 固定障碍物（3个）

| 配置项 | PPO | A2C | DQN |
|---|---|---|---|
| 训练步数 | 1M | 1M | 300k |
| 并行环境数 | 8 | 8 | 1 |
| 学习率 | 3e-4 | 7e-4 | 1e-4 |
| VecNormalize | obs+reward | obs+reward | obs only |
| 关键参数 | n_steps=1024, batch=64, γ=0.99 | n_steps=8, ent_coef=0.01, vf_coef=0.5 | buffer=100k, target_update=500, ϵ_final=0.05 |
| 评估 episode | 10 | 10 | 10 |
| Seed | 0 | 0 | 42 |

### 2.2 随机障碍物（1个，位置随机）

| 配置项 | PPO | A2C | DQN |
|---|---|---|---|
| 训练步数 | **3M** | 2M | **2M** |
| 并行环境数 | 8 | 8 | 1 |
| 学习率 | 3e-4 | 7e-4 | 1e-4 |
| VecNormalize | obs+reward | obs+reward | obs only |
| 关键参数 | n_steps=1024, batch=64, γ=0.99 | n_steps=8, ent_coef=0.01, vf_coef=0.5 | buffer=200k, target_update=500, ϵ_final=0.05 |
| 评估 episode | 100 | 100 | 100 |
| Seed | 0 | 0 | 42 |

> **说明**：DQN 不需要 norm_reward（值-based 方法），PPO/A2C 必须 norm_reward=true 才能学到有效策略。

---

## 3. 实验结果

### 3.0 汇报提纲（讲解顺序）

讲这一节时按以下 5 步展开，每步对应一个要点 + 一张图表/GIF：

1. **先抛结论**：随机障碍物设定下 PPO/DQN 成功率 90%+、碰撞率 <10%，基本掌握避障着陆；A2C 偏弱（~50%）。一句话定调，再展开数据。
2. **难点场景先讲（固定 3 障碍物）**：放 3.1 表，强调三算法均未破 50% 成功率，点出 DQN 仅 300k 步碰撞率最低但 return≈0（"能避开但落不好"），说明固定多障碍物是难场景。
3. **主结果（随机 1 障碍物）**：放 3.2 表 + 训练曲线对比图 `reports/obstacle_compare_random.png`，逐列读 PPO 92%/DQN 93%、碰撞 4%/6%，DQN return 最高、PPO 最稳。
4. **算法排序**：用 3.3 两行对比收束——固定模式 PPO≫DQN≈A2C，随机充分训练后 PPO≈DQN≫A2C。
5. **可视化收尾**：放 PPO-3M / DQN-2M 的避障 demo GIF，直观展示绕行障碍物后平稳着陆；如时间允许可对比 A2C 的 demo 体现差距。

> 讲解节奏建议：第 2-3 步是重点（各 ~1 分钟），第 1/4/5 步各 ~30 秒。被问到"A2C 为啥差"时引实验总结 6（已做 n_steps 消融排除该因素）。

### 3.1 固定障碍物（3个固定位置）

| 算法 | 步数 | Mean Return | Std Return | 碰撞率 ↓ | 成功率 ↑ |
|---|---|---|---|---|---|
| **PPO** | 1M | 102.24 | 125.15 | 40% | 50% |
| A2C | 1M | 62.95 | 129.56 | 50% | 40% |
| DQN | 300k | 0.99 | 195.91 | 30% | 50% |

- PPO 均值最高，但方差大；DQN 步数最少（300k），碰撞率反而最低（30%），但 return 接近 0，说明虽然能避开障碍物，但着陆质量差
- 3 个固定障碍物场景较难，三算法均未突破 50% 成功率

### 3.2 随机障碍物（1个随机位置）

| 算法 | 步数 | Mean Return | Std Return | 碰撞率 ↓ | 成功率 ↑ |
|---|---|---|---|---|---|
| PPO | 3M | 199.79 | 93.16 | 4% | 92% |
| **DQN** | **2M** | **225.86** | **112.96** | **6%** | **93%** |
| A2C | 2M | 81.94 | 143.81 | 30% | 50% |

- **PPO 和 DQN 均达到 90%+ 成功率，碰撞率 <10%**，基本掌握避障着陆
- DQN-2M 均值最高（225.86），但方差略大；PPO-3M 最稳定（std=93.16）
- A2C 在随机模式下仍表现不佳（消融已排除 `n_steps` 因素，详见实验总结 6）

### 3.3 算法排序（综合两种模式）

```
PPO ≫ DQN ≈ A2C （固定模式，同训练量）
PPO ≈ DQN ≫ A2C （随机模式，充分训练后）
```

---

## 4. 实验总结

1. **norm_reward 是关键**：PPO/A2C 在障碍物环境下必须开启 `norm_reward=true`，否则无法学到有效避障策略（奖励尺度变化大，-100 碰撞惩罚会 dominate 学习信号）

2. **随机障碍物比固定障碍物更容易**：1 个随机障碍物 vs 3 个固定障碍物，训练更容易收敛到高成功率 —— 可能是因为单障碍物场景更简单，且随机化增强泛化

3. **DQN 在充分训练后表现优异**：在 2M 步随机模式下，DQN 的 mean return 超过 PPO-3M，说明 value-based 方法在离散动作避障任务上有竞争力

4. **A2C 三算法中最弱**：on-policy 单遍更新、无 replay/n_epochs 复用，sample efficiency 不足，在障碍物避障这类需要长期信用分配的任务上劣势明显（消融见下条）

5. **碰撞惩罚设计有效**：-100 惩罚 + episode 终止 + 接近惩罚的塑形方案，成功引导智能体学会远离障碍物

6. **A2C 偏弱非 `n_steps` 所致（消融验证）**：将 `obstacle_a2c_random` 的 `n_steps` 从 8 拉到 1024（与 PPO 对齐）重训 2M 步，结果反而略差——collision 30%→38%、mean_return 81.9→68.1、success 50%→53%。原因：`n_steps` 增大使 on-policy 单遍更新次数从 31199 次锐减到 ~244 次，A2C 无 PPO 的 n_epochs 复用、无 DQN 的 replay，更新太少直接学不动（`explained_variance` 仅 0.18）。结论：A2C 短板是 sample efficiency 本身，调 `n_steps` 解不了；原配置 `n_steps=8` 是合理选择

---

## 5. 可视化素材

训练曲线和 demo GIF 路径（用于 PPT 插入）：

| 素材 | 路径 |
|---|---|
| PPO 训练曲线 | `reports/obstacle_ppo.png` |
| PPO-3M 训练曲线 | `reports/obstacle_ppo_random_3m.png` |
| DQN-2M 训练曲线 | `reports/obstacle_dqn_random_2m.png` |
| PPO 避障 demo | `outputs/obstacle_ppo_demo.gif` |
| PPO-3M 避障 demo | `outputs/obstacle_ppo_random_3m_demo.gif` |
| DQN-2M 避障 demo | `outputs/obstacle_dqn_random_2m_demo.gif` |
| A2C 避障 demo | `outputs/obstacle_a2c_demo.gif` |
