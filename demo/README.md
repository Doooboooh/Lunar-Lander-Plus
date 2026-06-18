# OpenAI Gym 极简 Demo

这个目录提供一个最小可运行脚本，演示三个环境：

- `Pendulum-v1`：连续动作空间，用简单 PD 控制让摆靠近竖直方向。
- `LunarLander-v3`：离散动作空间，用简单启发式控制主引擎和侧向引擎。
- `Blackjack-v1`：离散动作空间，20 点以下要牌，否则停牌。

安装依赖：

```bash
pip install -r ../requirements.txt
```

运行全部环境：

```bash
python gym_minimal_demo.py --env all --episodes 3
```

只运行一个环境：

```bash
python gym_minimal_demo.py --env Pendulum-v1
python gym_minimal_demo.py --env LunarLander-v3
python gym_minimal_demo.py --env Blackjack-v1
```

显示图形界面：

```bash
python gym_minimal_demo.py --env LunarLander-v3 --render
```

说明：脚本优先使用维护中的 `gymnasium` 包；如果环境里只有旧版 `gym`，也会尝试兼容旧 API。
