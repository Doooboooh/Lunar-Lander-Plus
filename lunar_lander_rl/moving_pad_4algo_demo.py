"""为四种算法各生成一个移动平台演示 GIF。

每算法加载其训练产物，扫多个随机种子，挑"落到平台上停留最久"的案例录制。
产物：outputs/moving_pad/gifs/<algo>_landing.gif

用法：
    python -m lunar_lander_rl.moving_pad_4algo_demo
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from .common import register_moving_pad, mlp
from .q_learning import discretize

ENV_ID = register_moving_pad()
OUT = Path("outputs/moving_pad/gifs")
DECK_OFFSET = 0.35   # 飞船腿高，落甲板时 y ≈ deck_top + 0.35


def _on(u):
    """飞船是否落在甲板上（双足 + 平台高度 + 平台宽度内）。"""
    p = u.lander.position
    deck = u.deck_top_y + 0.175
    both = u.legs[0].ground_contact and u.legs[1].ground_contact
    rel = p.x - (10.0 + u.ship_x)
    return both and abs(p.y - (deck + DECK_OFFSET)) < 0.5 and abs(rel) < 1.3


def _load_dqn():
    dev = torch.device("cpu")
    net = mlp(10, 4, 128).to(dev)
    net.load_state_dict(torch.load("outputs/moving_pad/moving_pad_dqn/best_policy.pt", map_location=dev))
    net.eval()
    def act(o):
        with torch.no_grad():
            t = torch.tensor(o, dtype=torch.float32, device=dev).unsqueeze(0)
            return int(net(t).argmax(1).item())
    return act


def _load_ppo():
    from lunar_lander_rl.ppo import ActorCriticNet
    dev = torch.device("cpu")
    net = ActorCriticNet(10, 4, 128).to(dev)
    net.load_state_dict(torch.load("outputs/moving_pad/moving_pad_ppo/last_policy.pt", map_location=dev))
    net.eval()
    def act(o):
        with torch.no_grad():
            t = torch.tensor(o, dtype=torch.float32, device=dev).unsqueeze(0)
            out = net(t)
            lo = out[0] if isinstance(out, tuple) else out
            return int(lo.argmax(1).item())
    return act


def _load_ac():
    from lunar_lander_rl.actor_critic import ActorCriticNet
    dev = torch.device("cpu")
    net = ActorCriticNet(10, 4, 128).to(dev)
    net.load_state_dict(torch.load("outputs/moving_pad/actor_critic/best_policy.pt", map_location=dev))
    net.eval()
    def act(o):
        with torch.no_grad():
            t = torch.tensor(o, dtype=torch.float32, device=dev).unsqueeze(0)
            out = net(t)
            lo = out[0] if isinstance(out, tuple) else out
            return int(lo.argmax(1).item())
    return act


def _load_ql():
    q = pickle.load(open("outputs/moving_pad/q_learning/q_table.pkl", "rb"))
    def act(o):
        return int(np.argmax(q[discretize(o)]))
    return act


def record(act, gif_path: str, seed: int = 5, max_steps: int = 1000):
    """固定 seed 录制：四算法面对完全相同的平台轨迹（omega/phase 由 seed 决定）。
    不挑种子、不美化，如实展示各算法在同一条平台运动下的真实表现。
    """
    import imageio.v2 as imageio
    env = gym.make(ENV_ID, render_mode="rgb_array")
    o, _ = env.reset(seed=seed)
    u = env.unwrapped
    omega, phase = u.ship_omega, u.ship_phase
    frames = []
    on_frames = 0
    for _ in range(max_steps):
        a = act(o)
        o, r, term, trunc, _ = env.step(a)
        frames.append(env.render())
        if _on(u):
            on_frames += 1
            if on_frames > 60:    # 落上后多录 60 帧看是否停住
                break
        if term or trunc:
            break
    env.close()
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=30)
    # 终态信息
    p = u.lander.position
    deck = u.deck_top_y + 0.175
    print(f"  seed={seed} omega={omega:.2f} phase={phase:.2f} | "
          f"落平台帧={on_frames} 末态y={p.y:.2f}(甲板{deck+DECK_OFFSET:.2f}) "
          f"双足={u.legs[0].ground_contact and u.legs[1].ground_contact} "
          f"-> {gif_path.name} ({len(frames)}帧)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=5,
                   help="固定种子：四算法面对完全相同的平台轨迹（omega/phase 由它决定）")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, loader in [("dqn", _load_dqn), ("ppo", _load_ppo),
                         ("actor_critic", _load_ac), ("q_learning", _load_ql)]:
        print(f"[{name}]")
        act = loader()
        record(act, OUT / f"{name}_landing.gif", seed=args.seed)


if __name__ == "__main__":
    main()
