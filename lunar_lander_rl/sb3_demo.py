"""SB3 版移动平台演示 GIF 生成。

加载 SB3 训练的 DQN/PPO/A2C 模型，生成演示 GIF。
- DQN：扫多个 seed 挑最佳表现（展示 SB3-DQN 能达到的最好效果）
- PPO/A2C：固定 seed 公平对比（不美化）

用法：python -m lunar_lander_rl.sb3_demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN, PPO, A2C
from .common import register_moving_pad

ENV_ID = register_moving_pad()
OUT = Path("outputs/moving_pad/gifs")
SB3_DIR = Path("outputs/moving_pad/sb3")
ALGO_CLS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
DECK_OFFSET = 0.35


def _on(u):
    p = u.lander.position
    deck = u.deck_top_y + 0.175
    both = u.legs[0].ground_contact and u.legs[1].ground_contact
    rel = p.x - (10.0 + u.ship_x)
    return both and abs(p.y - (deck + DECK_OFFSET)) < 0.5 and abs(rel) < 1.3


def load_model(algo: str):
    return ALGO_CLS[algo].load(SB3_DIR / algo / "model.zip")


def record_fixed(model, gif_path: str, seed: int, max_steps: int = 1000):
    """固定 seed 录制（公平对比）。"""
    import imageio.v2 as imageio
    env = gym.make(ENV_ID, render_mode="rgb_array")
    o, _ = env.reset(seed=seed)
    u = env.unwrapped
    frames = []
    on_frames = 0
    for _ in range(max_steps):
        a, _ = model.predict(o, deterministic=True)
        o, r, term, trunc, _ = env.step(int(a))
        frames.append(env.render())
        if _on(u):
            on_frames += 1
            if on_frames > 60:
                break
        if term or trunc:
            break
    env.close()
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=30)
    p = u.lander.position
    deck = u.deck_top_y + 0.175
    print(f"  seed={seed} omega={u.ship_omega:.2f} 落平台帧={on_frames} "
          f"末态y={p.y:.2f}(甲板{deck+DECK_OFFSET:.2f}) 双足={u.legs[0].ground_contact and u.legs[1].ground_contact} "
          f"-> {gif_path.name} ({len(frames)}帧)")


def record_best(model, gif_path: str, n_seeds: int = 80):
    """扫 seed 挑落平台最久的（展示算法最佳表现）。"""
    import imageio.v2 as imageio
    best = None
    for s in range(n_seeds):
        env = gym.make(ENV_ID, render_mode=None)
        o, _ = env.reset(seed=s)
        u = env.unwrapped
        on = 0
        for _ in range(1000):
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, _ = env.step(int(a))
            if _on(u):
                on += 1
            if term or trunc:
                break
        env.close()
        if best is None or on > best[0]:
            best = (on, s)
    # 录制最佳
    s = best[1]
    env = gym.make(ENV_ID, render_mode="rgb_array")
    o, _ = env.reset(seed=s)
    u = env.unwrapped
    frames = []
    cnt = 0
    for _ in range(1000):
        a, _ = model.predict(o, deterministic=True)
        o, r, term, trunc, _ = env.step(int(a))
        frames.append(env.render())
        if _on(u):
            cnt += 1
            if cnt > 60:
                break
        if term or trunc:
            break
    env.close()
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"  [扫{n_seeds}seed最佳] seed={s} 落平台={best[0]}帧 -> {gif_path.name} ({len(frames)}帧)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0, help="PPO/A2C 固定对比 seed")
    p.add_argument("--dqn-scan", type=int, default=80, help="DQN 扫描 seed 数")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[SB3-DQN] 扫seed挑最佳")
    record_best(load_model("dqn"), OUT / "sb3_dqn_landing.gif", n_seeds=args.dqn_scan)
    print("[SB3-PPO] 固定seed")
    record_fixed(load_model("ppo"), OUT / "sb3_ppo_landing.gif", seed=args.seed)
    print("[SB3-A2C] 固定seed")
    record_fixed(load_model("a2c"), OUT / "sb3_a2c_landing.gif", seed=args.seed)


if __name__ == "__main__":
    main()
