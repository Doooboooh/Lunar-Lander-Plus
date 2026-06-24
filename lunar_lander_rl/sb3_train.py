"""基于 Stable-Baselines3 的移动平台训练脚本（重构版）。

用 SB3 的标准算法接口（DQN / PPO / A2C）训练我们自定义的移动平台环境。
算法的"怎么训练"交给 SB3，我们的核心创新（移动平台环境 + reward 重塑）保留在
moving_pad_env.py 里，通过 gymnasium 标准接口接入 SB3。

用法：
    python -m lunar_lander_rl.sb3_train --algo dqn  --timesteps 200000
    python -m lunar_lander_rl.sb3_train --algo ppo  --timesteps 200000
    python -m lunar_lander_rl.sb3_train --algo a2c  --timesteps 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from .common import register_moving_pad

ENV_ID = register_moving_pad()
ALGOS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}


def make_env(seed: int = 0, monitor_log: str = None):
    """用 Monitor 包装环境，记录每个 episode 的 reward（供画训练曲线）。"""
    env = gym.make(ENV_ID)
    if monitor_log:
        env = Monitor(env, filename=monitor_log)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def train(algo: str, timesteps: int, seed: int = 0, output_dir: str = "outputs/moving_pad/sb3",
          device: str = "cpu", eval_episodes: int = 10):
    assert algo in ALGOS, f"algo 必须是 {list(ALGOS)}"
    cls = ALGOS[algo]
    out = Path(output_dir) / algo
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(seed, monitor_log=str(out / "monitor"))
    print(f"[SB3-{algo}] 环境={ENV_ID} timesteps={timesteps} device={device}")

    # 各算法共用 MlpPolicy（SB3 默认中等网络），学习率/批大小按算法合理默认
    model = cls("MlpPolicy", env, verbose=0, seed=seed, device=device)

    # 训练（SB3 内部封装了经验回放/rollout/更新等全部逻辑）
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(out / "model.zip")

    # 评估：SB3 标准评估接口
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=eval_episodes, deterministic=True)
    print(f"[SB3-{algo}] 训练完成。eval mean_return={mean_r:.1f} std={std_r:.1f}")
    print(f"           模型 -> {out/'model.zip'}")
    env.close()

    # 保存指标
    import json
    (out / "metrics.json").write_text(json.dumps({
        "algorithm": f"SB3-{algo}", "timesteps": timesteps,
        "eval_mean_return": float(mean_r), "eval_std_return": float(std_r),
    }, indent=2), encoding="utf-8")
    return {"mean_return": float(mean_r), "std_return": float(std_r)}


def main():
    p = argparse.ArgumentParser(description="SB3 训练：移动平台 DQN/PPO/A2C")
    p.add_argument("--algo", choices=list(ALGOS), required=True)
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/moving_pad/sb3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-episodes", type=int, default=10)
    args = p.parse_args()
    train(args.algo, args.timesteps, args.seed, args.output_dir, args.device, args.eval_episodes)


if __name__ == "__main__":
    main()
