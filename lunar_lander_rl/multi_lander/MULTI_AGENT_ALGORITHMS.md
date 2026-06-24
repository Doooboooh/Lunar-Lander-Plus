# 多飞船顺序降落实验路线

项目目标是把单飞船 `LunarLander-v3` 扩展到多个飞船顺序或同时降落。建议按难度从低到高做三组实验，这样报告里能清楚说明“直接扩维能做到什么、为什么需要更经典的多智能体算法”。

## 1. 单智能体基线

先保留原始 `LunarLander-v3`，训练并比较 DQN、PPO、Actor-Critic。

用途：

- 验证基础飞船控制能力。
- 作为多飞船低层控制器或算法对照组。
- 证明 DQN 适合离散动作；PPO/Actor-Critic 更新更稳定，但训练成本通常更高。

已有入口：

```bash
python -m lunar_lander_rl.dqn --episodes 400 --output-dir outputs/baseline/dqn
python -m lunar_lander_rl.ppo --episodes 400 --output-dir outputs/baseline/ppo
python -m lunar_lander_rl.actor_critic --episodes 400 --output-dir outputs/baseline/actor_critic
```

## 2. 不改 DQN 核心，只扩输入输出

这是最适合作为第一版多智能体强化学习尝试的方案。

### 参数共享 DQN

每艘船都使用同一个 DQN 网络。输入从单船 8 维扩展到 23 维：自身状态、目标点、其他两艘船的相对状态。输出仍是单船 4 个动作。每一步会产生 3 条 transition，放入同一个 replay buffer。

优点是训练样本多、动作空间没有爆炸；缺点是每艘船独立选动作，协作只来自共享奖励和观测。

已有入口：

```bash
python -m lunar_lander_rl.multi_lander.multi_agent_dqn --episodes 600 --output-dir outputs/multi_lander/multi_agent_dqn
```

### 集中式联合动作 DQN

新增脚本 `joint_action_dqn.py` 是最直接的“1 个信号变 3 个信号”版本：把三艘船观测拼成 69 维输入，输出 64 个联合动作，对应 `(a0, a1, a2)`。

优点是能显式学习三艘船的联合动作；缺点是动作数按 `4^N` 增长，飞船更多时会很快不可扩展。

运行方式：

```bash
python -m lunar_lander_rl.multi_lander.joint_action_dqn --episodes 600 --output-dir outputs/multi_lander/joint_action_dqn
```

这组实验可以回答：DQN 不改 Bellman 更新、不改 replay buffer、不改 target network，只扩大输入和输出，是否能学到基本协同。

## 3. 如果直接扩维效果不好

推荐按下面顺序升级，而不是一开始就写复杂框架。

### 目标条件 DQN

给低层策略加入 `goal_x`，让单船不只会回到平台中心，而是能降落到指定横向目标。这样顺序降落或错位降落才有可控基础。

已有入口：

```bash
python -m lunar_lander_rl.multi_lander.goal_conditioned_dqn --episodes 600 --output-dir outputs/baseline/goal_dqn
```

### MAPPO

多智能体 PPO 的经典做法是集中式 critic、分散式 actor。actor 给每艘船独立输出动作，critic 看到全局状态。它比 DQN 更适合多智能体里的非平稳问题，也更适合报告中作为“经典多智能体算法”。

适用结论：如果 DQN 扩维不稳定、碰撞多、成功率波动大，MAPPO 是最自然的下一步。

### QMIX / VDN

如果希望保留 DQN 的值函数路线，可以尝试 QMIX 或 VDN。每个智能体学习自己的 Q 值，再用 mixing network 合成团队 Q 值。它比联合动作 DQN 更可扩展，适合协作任务。

适用结论：如果想强调“从 DQN 到多智能体 DQN 家族”的延伸，QMIX 比 MAPPO 更贴近原算法路线。

## 推荐报告结论

建议最终报告采用这个逻辑：

1. 单飞船任务中，DQN/PPO/Actor-Critic 都能学习基础着陆，DQN 是离散动作下的简洁基线。
2. 多飞船任务中，直接扩展 DQN 有两种方式：参数共享 DQN 和集中式联合动作 DQN。
3. 参数共享 DQN 更可扩展；联合动作 DQN 更直观，但动作空间随飞船数指数增长。
4. 如果直接 DQN 效果不够稳定，应升级到目标条件策略、MAPPO 或 QMIX。
5. 对课程项目来说，先展示 DQN 扩维实验，再解释 MAPPO/QMIX 的必要性，是最清晰也最稳妥的技术路线。
