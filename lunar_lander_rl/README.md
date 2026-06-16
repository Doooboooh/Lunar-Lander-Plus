# LunarLander-v3 强化学习 Demo

这个目录用同一个 `LunarLander-v3` 环境演示四种训练方法：

- `q_learning.py`：表格 Q-Learning。由于 LunarLander 的观测是连续值，这里先把状态离散化，作为传统基线。
- `dqn.py`：Deep Q-Network，使用经验回放和目标网络。
- `ppo.py`：Proximal Policy Optimization，使用 clipped policy objective 和 GAE。
- `actor_critic.py`：朴素 Actor-Critic，每个 episode 后做一次策略和值函数更新。

## 安装依赖

在 `proj` 目录运行：

```bash
pip install -r requirements.txt
```

## 单独运行

```bash
python -m lunar_lander_rl.q_learning --episodes 400
python -m lunar_lander_rl.dqn --episodes 400
python -m lunar_lander_rl.ppo --updates 200
python -m lunar_lander_rl.actor_critic --episodes 600
```

每个脚本都会训练、评估，并把结果保存到对应的 `outputs/...` 目录：

- `history.csv`：训练过程中每回合的 reward。
- `metrics.json`：评估均值、方差、最小值、最大值。
- `best_policy.pt` 或 `q_table.pkl`：训练得到的模型或 Q 表。

## 测试和可视化某个策略

训练后可以重新加载模型做测试：

```bash
python -m lunar_lander_rl.evaluate dqn --model-dir outputs/dqn --episodes 10
python -m lunar_lander_rl.evaluate ppo --model-dir outputs/ppo --episodes 10
python -m lunar_lander_rl.evaluate actor_critic --model-dir outputs/actor_critic --episodes 10
python -m lunar_lander_rl.evaluate q_learning --model-dir outputs/q_learning --episodes 10
```

打开可视化窗口：

```bash
python -m lunar_lander_rl.evaluate dqn --model-dir outputs/dqn --render
```

保存一局 DQN 效果为 GIF：

```bash
python -m lunar_lander_rl.evaluate dqn --model-dir outputs/dqn --episodes 1 --gif outputs/dqn_lunar_lander.gif
```

报告中插入：

```markdown
![DQN LunarLander 着陆效果](outputs/dqn_lunar_lander.gif)
```

## 快速测试四种方法

```bash
python -m lunar_lander_rl.run_all --quick
```

`--quick` 只用于检查代码能否跑通，episode 很少，效果不会好。

完整对比：

```bash
python -m lunar_lander_rl.run_all
```

## 预期效果

LunarLander 官方通常把平均回报 `200+` 视为 solved。默认参数是课堂 demo 规模，训练时间和随机种子会明显影响结果。一般趋势是：

```text
Q-Learning      连续状态被粗糙离散化，效果通常最弱
Actor-Critic    能学习，但方差较大
DQN             离散动作控制的经典强基线
PPO             更稳定，通常适合继续扩展成正式实验
```
