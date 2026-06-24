# 拓展实验报告：移动着陆平台（Moving Landing Pad）

> 课程：神经网络 / 强化学习结课作业
> 基础环境：`LunarLander-v3`（gymnasium，Box2D 物理）
> 算法：Q-Learning / DQN / Actor-Critic / PPO 四种算法对比

---

## 一、任务背景与动机

基础 `LunarLander-v3` 任务中，着陆平台固定在画面正中，飞船只需"减速 + 回中"即可平稳
着陆。该任务的目标位置是**静态**的，智能体学到的策略本质是一个"回中器"。

现实中的着陆场景往往**目标位置是动态的**，例如：

- **海上火箭回收**：SpaceX 的无人船在海上随波浪漂移，火箭必须精准落到移动的甲板上；
- **航母舰载机降落**：航母甲板随海浪起伏、随航向移动；
- **无人机移动平台起降**：物流无人机降落到移动的运输车上。

这些场景的共同难点是：**智能体不仅要平稳减速，还要实时跟踪一个移动的目标，并在接触
瞬间与目标保持近零相对速度**，否则会被刚性碰撞弹飞。

因此本拓展把基础任务升级为**移动着陆平台**：在画面中引入一个左右往返移动的实体平台，
飞船必须跟踪它并平稳落到甲板上。这使任务从"静态目标控制"升级为"动态目标跟踪 + 精确
接触控制"，更充分地体现强化学习在复杂决策任务中的能力与边界。

---

## 二、环境设计

### 2.1 总体思路

在 `LunarLander-v3` 基础上做最小侵入式扩展，子类化 `LunarLander`：

- **新增实体平台**：一个 `KinematicBody`（运动学刚体）矩形甲板，悬在地表之上（不嵌在
  地下），沿水平方向做正弦往返移动；
- **观测扩展**：原 8 维观测 + 平台相对飞船的位置 2 维 = **10 维**；
- **奖励重塑**：把原版"相对画面中心"的 shaping 改为"相对平台当前位置"，并加入关键
  的"水平相对速度"惩罚项（见 2.4）；
- **动作空间不变**：仍是 4 个离散动作（不喷 / 左推 / 主推 / 右推）。

### 2.2 关键物理：让平台成为真正的实体

最核心的工程难点是让平台**真正挡住飞船**（飞船落到甲板顶面上，而不是穿过它落到地面）。

原版 `LunarLander` 中，飞船（lander / legs）的碰撞掩码 `maskBits = 0x001`，**只与地面
碰撞**。因此默认情况下新建的平台会被飞船穿透。

解决方法：reset 后修改飞船与腿的 `maskBits`，使其也包含平台的类别位：

```python
PAD_CAT = 0x0004
# 让 lander 与 legs 也碰平台（原版 maskBits=0x001 只碰地面）
for f in self.lander.fixtures:
    f.filterData.maskBits = 0x001 | PAD_CAT
for leg in self.legs:
    for f in leg.fixtures:
        f.filterData.maskBits = 0x001 | PAD_CAT
```

验证：飞船从平台上方无控制自由落体，会**稳稳停在甲板顶面上**（不穿透、不落地面）。

### 2.3 平台运动

平台中心 x 随时间正弦往返：

```python
ship_x = pad_x_amp * sin(omega * t / FPS + phase)
```

- `pad_x_amp`：移动幅度（米，相对画面中心）；
- `omega`：角频率，控制移动快慢；
- `phase`：随机相位，使每局轨迹不同。

平台的水平速度为 `pad_x_amp * omega * cos(...)`，峰值速度 = 幅度 × 频率。

**本实验参数**：`pad_x_amp = 2.2`（平台中心在画面 ±2.2 米间往返，覆盖画面大部分宽度，
近乎从最左到最右），`omega ∈ [1.2, 1.6]`（每局随机）。出界判定相应放宽到 `|x| < 4.0`，
避免飞船追平台时被误判出界。

### 2.4 奖励函数设计（逐步演化的核心）

奖励设计经历了三个版本，反映了我们对"平稳着陆"理解的深化：

**版本 1 — 目标条件 shaping**（基础）：

```python
shaping = -100 * sqrt(rel_x² + rel_y²)    # 飞船相对平台中心的距离
          - 100 * hypot(vel.x, vel.y)       # 飞船自身速度
          - 100 * abs(angle)                # 姿态偏离竖直
          + 10 * (双腿接地)
```

平稳落到平台上给 +120，坠毁 -100，落到地面（没上平台）-30。

**问题**：飞船学到了"减速 + 接近平台"，但在高速平台下，飞船带着**水平速度**撞到甲板，
被刚性碰撞**弹飞**，无法真正停住。

**版本 2 — 加入姿态约束**：把姿态惩罚从 100 提到 300，并在平稳判定中要求 `|angle| < 0.15`
（接近竖直）。这改善了着陆姿态，但**没解决弹飞**——因为弹飞的根因是水平相对速度，不是
姿态。

**版本 3（最终）— 加入水平相对速度惩罚**（关键）：

```python
pad_vx = pad_x_amp * omega * cos(omega * t / FPS + phase)   # 平台当前水平速度
rel_vx = vel.x - pad_vx                                       # 飞船相对平台的水平速度
shaping = ...
          - 150 * abs(rel_vx)      # 迫使飞船匹配平台水平速度后再落地
```

**原理**：弹飞的本质是接触瞬间飞船与平台存在**水平相对速度** → 刚性碰撞产生反弹冲量。
只要让飞船学会"先匹配平台的水平移动速度，再落下"，接触时相对速度 ≈ 0，就自然不弹。
这一项让飞船从"回中器"进化为"**速度匹配跟踪器**"，是本拓展最关键的 reward insight。

### 2.5 工程踩坑记录

开发过程中遇到并解决了若干 Box2D / gymnasium 层面的坑，记录如下（也是报告亮点）：

1. **多 reset 段错误**：`LunarLander.reset()` 内部会调用一次 `self.step(0)`，若此时平台
   body 引用还指向已被销毁的旧 world，访问即段错误（C 层面，try 接不住）。解决：在调
   `super().reset()` 前先把 `self.ship = None`。
2. **平台不移动**：曾用 `ship.world is self.world` 判断 body 是否有效，但 Box2D Python
   绑定的 `body.world` 返回代理对象，`is` 比较恒为 False，导致 step 永远走"平台失效"
   分支、时间步不自增、平台静止。改用 `try: ship.position` 判断后修复。
3. **观测归一化一致性**：自建环境构造的 8 维状态必须与 `LunarLander.step` 的归一化
   完全一致（`x/(W/2)`、`vel*W/2/FPS`、`20*angVel/FPS`），否则预训练策略失效。
4. **初始随机力误用**：原版 `INITIAL_RANDOM=1000` 是 `ApplyForceToCenter` 的力，不是
   速度；曾误当作速度赋给飞船，导致 ~150 m/s 初速直接飞出界。

---

## 三、算法与训练

为充分比较不同强化学习方法在动态目标跟踪任务上的表现，我们在**同一移动平台环境**
下训练并对比了 **4 种算法**：Q-Learning（表格值函数）、DQN（深度值函数）、Actor-Critic
（策略 + 值函数）、PPO（策略优化）。它们分别代表了强化学习的四大主流范式。

### 3.1 各算法配置

| 算法 | 类型 | 网络/表 | 关键超参 |
|---|---|---|---|
| Q-Learning | 表格值函数 | 10 维分箱 Q 表（8⁸×2²×8²≈4.2M 项） | episodes=1500，lr=0.08，γ=0.99，ε 衰减 0.995 |
| DQN | 深度值函数 | MLP 10→128→128→4 | episodes=1000，batch=64，replay=1e5，γ=0.99，ε 1.0→0.05 |
| Actor-Critic | 策略+值函数 | MLP 10→128→128（actor+critic 共享） | episodes=1000，lr=1e-3，γ=0.99，entropy=0.01 |
| PPO | 策略优化 | MLP 10→128→128（actor+critic） | updates=400，rollout=2048，clip=0.2，GAE λ=0.95 |

- **统一条件**：同一移动平台环境（amp=2.2，平台覆盖画面大部分宽度）、同一奖励函数
  （含水平相对速度惩罚）、CPU 训练、相同随机种子起点。
- 训练产物分别见 `outputs/moving_pad/{q_learning, moving_pad_dqn, actor_critic, moving_pad_ppo}/`。

### 3.2 公平性说明

四算法在**完全相同的环境与奖励**下训练，差异仅在算法本身。Q-Learning 作为传统表格
基线，后三者是深度方法。训练规模（episode/update 数）按各算法惯例设置，均在 CPU 上
单次运行可完成（30–60 分钟）。

---


## 四、实验结果

### 4.1 训练收敛

四种算法在相同环境与奖励下训练。以主算法 DQN 为例（含水平相对速度惩罚的最终奖励
版本）：训练 1000 episode，**评估平均 return 达 735、最佳单局 return 达 1126**，
显著高于未加该项的早期版本（约 300）。说明"水平相对速度惩罚"不仅解决了弹飞问题，
也让训练信号更清晰、策略更优。其余三种算法的训练表现见 4.4 节对比。

### 4.2 演示效果

扫描多个随机种子，挑选飞船**真实平稳**落到移动平台上的成功案例生成 GIF
（"真实平稳"判定：飞船落到甲板后**自身绝对速度 < 0.6 m/s**，稳定停留 40+ 帧，
且无反弹——非任何强制冻结）。

最佳案例（seed 154）：

| 指标 | 数值 |
|---|---|
| 落点相对平台中心偏差 | **0.007 m**（几乎正中） |
| 着陆姿态角 | **0.001 rad**（完全竖直） |
| 落上后稳定停留 | 92 帧 |
| 反弹 | 无 |

演示 GIF：`moving_pad_dqn_landing_4.gif`

### 4.3 平台速度的影响（鲁棒性分析）

我们对不同平台速度（`omega`）做了对比，观察"任务难度"与"可学习性"的权衡：

| 平台速度 | omega | 现象 |
|---|---|---|
| 慢速 | 0.6–0.8 | 飞船易跟踪，落准率高，但平台移动不明显 |
| 中速 | 1.1–1.2 | 较好平衡，落准率尚可，移动可见 |
| 1.3 倍速 | 1.2–1.6 | 窄范围（amp1.3）下配合速度惩罚可真平稳落地（见 4.2 演示） |
| 2 倍速 | 1.8–2.4 | 落准率明显下降，飞船易撞边弹飞 |

**注**：4.2 节的演示 GIF 与上表是在**窄移动范围（amp=1.3）**下取得的最佳平稳着陆
案例；本报告最终的算法对比（4.4 节）采用**大移动范围（amp=2.2，平台覆盖画面大部分
宽度）**，难度更高，用于公平考察四种算法在更大动态范围下的表现。两套设置共同刻画了
"移动范围越大、任务越难"的趋势。
| 3 倍速 | 2.7–3.6 | **物理上飞船无法跟踪**，策略选择放弃降落、全程悬停 |

**关键发现**：平台速度存在一个"可学习上限"。当平台峰值水平速度（`amp × omega`）超过
飞船自身的水平机动能力时，飞船物理上无法在接触前匹配平台速度，于是策略学到"悬停避险"
而非"跟踪降落"。这是 sim-to-real / 任务难度边界的实证。

### 4.4 四种算法对比

在统一的移动平台环境（amp=2.2，平台覆盖画面大部分宽度，含水平相对速度惩罚奖励）
下，对四种算法做固定评估种子集（30 个随机种子）的贪心策略评估，结果如下：

| 算法 | 类型 | 评估 mean return | 标准差 | 最佳 return | 评价 |
|---|---|---:|---:|---:|---|
| **DQN** | 深度值函数 | **+735.3** | 220.5 | **+1126.1** | 明显最优且最稳定 |
| PPO | 策略优化 | −363.9 | 697.4 | +602.4 | 方差极大，不稳定 |
| Q-Learning | 表格值函数 | −434.8 | 638.3 | +249.8 | 大范围下容量不足 |
| Actor-Critic | 策略+值函数 | −666.8 | 485.9 | −114.7 | 最差 |

**关键结论：**

1. **DQN 显著优于其他三种**。在 mean return（+735 vs 其余为负）和稳定性（std 最小）上
   都明显领先。原因是：本任务是**离散动作 + 需要精确相对速度匹配**的控制任务，DQN 的
   off-policy 经验回放能反复利用稀有成功/近成功轨迹，对稀疏成功信号更高效。

2. **PPO / Actor-Critic（on-policy 策略梯度）表现差**。动态平台要求精确的相对速度匹配，
   而 on-policy 采样下"成功接触平台"的样本极其稀少，策略更新后容易退化为避险悬停或
   提前坠毁；PPO 训练中虽偶有 1000+ 的单局 return，但贪心策略未能稳定保留。

3. **Q-Learning（表格法）学不会大范围任务**。10 维连续观测分箱后 Q 表已达 4.2M 项，
   样本仍覆盖不到偌大状态空间，且移动范围加大后更甚。它只在"接近平台"上学到部分
   shaping，无法精确落准——印证了**深度方法相对表格法的核心优势：在高维连续状态空间
   上的泛化能力**。

4. **共同难点**：四个算法在 amp=2.2 大范围下的"真正落平台率"都较低（DQN 能把飞船
   降到平台高度并跟踪，但平台跑动范围太大、飞船常追不上落不准）。这是动态目标跟踪
   任务在大移动范围下的固有难度，是任务本身的边界，而非算法实现问题。

**因此，在本移动平台任务（离散动作 + 大范围动态目标）中，DQN 是四种算法里最合适的
选择。** 这一结论与"动态目标跟踪需要 off-policy 的样本效率"的直觉一致。

---

## 五、困难与讨论

### 5.1 主要困难

1. **弹飞问题**：高速平台下，飞船带水平速度撞甲板被弹开。最终通过"水平相对速度惩罚"
   让飞船学会速度匹配后解决（而非靠摩擦或强制冻结等"作弊"手段）。
2. **平稳 vs 速度的权衡**：平台越快，越难平稳落准。3 倍速在物理上不可学。
3. **策略成功率不高**：即便最终版本，随机种子下的落准率仍有限（约百分之几到十几），
   需扫描多个种子挑出成功案例。这是动态目标跟踪任务固有的难度。
4. **PPO 不稳定**：PPO 在训练中能偶发高 return，但统一评估时显著弱于 DQN，说明策略没有
   稳定掌握"跟踪平台 + 匹配水平速度 + 接触控制"这一组合行为。

### 5.2 与基础任务的对比（insight）

| 维度 | 基础 LunarLander | 移动平台（本拓展） |
|---|---|---|
| 目标 | 静态（画面中心） | 动态（往返移动） |
| 策略本质 | 回中器 | 速度匹配跟踪器 |
| 观测 | 8 维 | 10 维（+平台位置） |
| 接触控制 | 落地即可 | 需近零相对速度，否则弹飞 |
| reward 关键项 | 距离 + 速度 + 姿态 | + **水平相对速度惩罚** |

**核心 insight**：把目标从静态变动态，看似只是"多了个移动平台"，实则对智能体提出了
本质更高的要求——**不仅要控制自身运动，还要预测并匹配外部目标的运动**。基础任务的
reward 框架（距离 + 速度 + 姿态）不足以让智能体学会这一点，必须显式引入"相对速度"
这一**关系量**作为奖励信号。这一发现可推广到其他"动态目标跟踪"任务（如机械臂抓取
移动物体、无人机拦截）。

### 5.3 局限与未来工作

- **成功率**：当前随机种子下落准率有限，可通过更长训练、优先经验回放（PER）或更大网络
  容量提升；
- **平台运动模型**：目前是确定性正弦，可扩展为随机过程（更接近真实海浪），考察策略
  泛化与鲁棒性；
- **域随机化**：训练时随机化平台速度/幅度，有望得到对未知运动更鲁棒的策略。

---

## 六、文件清单

```
outputs/moving_pad/
├── REPORT.md                      本报告
├── q_learning/                    Q-Learning 训练产物（q_table.pkl, history.csv, metrics.json）
├── moving_pad_dqn/                DQN 训练产物（best_policy.pt, history.csv, metrics.json）
├── actor_critic/                  Actor-Critic 训练产物（best_policy.pt, history.csv, metrics.json）
├── moving_pad_ppo/                PPO 训练产物（best_policy.pt, last_policy.pt, history.csv, metrics.json）
├── dqn.log / ppo.log / q_learning.log / actor_critic.log   训练日志
├── moving_pad_dqn_landing_4.gif   ★ 平稳着陆演示（窄范围 amp1.3，seed154，真平稳不弹飞）
├── moving_pad_dqn_landing.gif     慢速版演示
├── moving_pad_dqn_landing_fast.gif / _faster.gif   不同速度档演示
├── moving_pad_dqn_landing_1.gif   2倍速版（弹飞问题演示）
├── moving_pad_dqn_landing_2.gif / _3.gif
└── moving_pad_ppo_landing.gif     PPO 演示
```

环境与训练代码：

```
lunar_lander_rl/
├── moving_pad_env.py              移动平台环境（实体平台 + maskBits + reward 重塑）
├── moving_pad_demo.py             演示脚本（GIF 生成）
├── moving_pad_compare_eval.py     DQN/PPO 统一 seeds 评估脚本
├── dqn.py                         DQN 训练（--env-id MovingPadLunarLander-v0）
├── ppo.py                         PPO 训练（--env-id MovingPadLunarLander-v0）
└── common.py                      环境注册 make_env / register_moving_pad
```

---

## 七、复现命令

```bash
cd Lunar-Lander-Plus
# 四种算法在统一移动平台环境（amp=2.2）下训练
python -m lunar_lander_rl.q_learning --episodes 1500 \
    --env-id MovingPadLunarLander-v0 --output-dir outputs/moving_pad/q_learning

python -m lunar_lander_rl.dqn --episodes 1000 \
    --env-id MovingPadLunarLander-v0 --output-dir outputs/moving_pad/moving_pad_dqn --device cpu

python -m lunar_lander_rl.actor_critic --episodes 1000 \
    --env-id MovingPadLunarLander-v0 --output-dir outputs/moving_pad/actor_critic --device cpu

python -m lunar_lander_rl.ppo --updates 400 --rollout-steps 2048 \
    --env-id MovingPadLunarLander-v0 --output-dir outputs/moving_pad/moving_pad_ppo --device cpu

# 演示 GIF（DQN 平稳着陆）
python -m lunar_lander_rl.moving_pad_demo \
    --gif outputs/moving_pad/moving_pad_dqn_landing.gif
```

---
```
