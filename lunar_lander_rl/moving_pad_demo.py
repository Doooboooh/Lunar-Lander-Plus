"""移动平台变体：生成 DQN / PPO 策略的降落 GIF + 训练曲线对比图。

产物：
  outputs/moving_pad_dqn_landing.gif     DQN 策略在移动平台上的降落过程
  outputs/moving_pad_ppo_landing.gif     PPO 策略在移动平台上的降落过程
  outputs/moving_pad_compare.png         两算法训练曲线对比图

用法：
    python -m lunar_lander_rl.moving_pad_demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from .common import register_moving_pad, mlp
from gymnasium.envs.box2d.lunar_lander import SCALE

ENV_ID = register_moving_pad()
OUTPUTS = Path("outputs")


def _make_env(seed: int = 0):
    return gym.make(ENV_ID, render_mode="rgb_array")


def _load_dqn(model_path: str, device: str = "cpu"):
    net = mlp(10, 4, 128).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    def act(obs):
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return int(net(t).argmax(1).item())
    return act


def _load_ppo(model_path: str, device: str = "cpu"):
    from lunar_lander_rl.ppo import ActorCriticNet
    net = ActorCriticNet(10, 4, 128).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    def act(obs):
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            out = net(t)
            logits = out[0] if isinstance(out, tuple) else out
            return int(logits.argmax(1).item())
    return act


def record_gif(policy_fn, gif_path: str, seed: int = 0, max_steps: int = 1000,
               device: str = "cpu", label: str = "") -> dict:
    import imageio.v2 as imageio
    env = _make_env(seed)
    obs, _ = env.reset(seed=seed)
    frames = []
    total = 0.0
    u = env.unwrapped
    landed = False
    for t in range(max_steps):
        a = policy_fn(obs)
        obs, r, term, trunc, _ = env.step(a)
        total += r
        frames.append(env.render())
        if term or trunc:
            break
    # 判定是否落在平台上
    pos = u.lander.position
    pad_wx, pad_wy = u._pad_center_world()
    rel_x = abs((pos.x - pad_wx) / SCALE)
    both = u.legs[0].ground_contact and u.legs[1].ground_contact
    landed = bool(rel_x < 0.6 and both)
    env.close()
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"[{label}] {len(frames)} 帧, return={total:.1f}, 落在平台上={landed} -> {gif_path}")
    return {"return": total, "landed": landed, "frames": len(frames)}


def plot_compare(dqn_dir: str = "outputs/moving_pad_dqn",
                 ppo_dir: str = "outputs/moving_pad_ppo",
                 out: str = "outputs/moving_pad_compare.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    def load_history(d: str) -> tuple[list, str]:
        p = Path(d) / "history.csv"
        if not p.exists():
            return [], d
        xs, ys = [], []
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                # 兼容 DQN(return)与 PPO(recent_return)两种列名
                key = "return" if "return" in row else "recent_return"
                try:
                    xs.append(int(row.get("episode", row.get("update", len(xs)))))
                    ys.append(float(row[key]))
                except (ValueError, KeyError):
                    continue
        return ys, d

    dqn_y, _ = load_history(dqn_dir)
    ppo_y, _ = load_history(ppo_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    if dqn_y:
        ax.plot(range(len(dqn_y)), dqn_y, alpha=0.25, color="C0")
        # 滑动平均
        if len(dqn_y) >= 10:
            ma = np.convolve(dqn_y, np.ones(20)/20, mode="valid")
            ax.plot(range(19, len(dqn_y)), ma, color="C0", label="DQN (MA20)")
        else:
            ax.plot(range(len(dqn_y)), dqn_y, color="C0", label="DQN")
    if ppo_y:
        ax.plot(range(len(ppo_y)), ppo_y, alpha=0.25, color="C1")
        if len(ppo_y) >= 10:
            ma = np.convolve(ppo_y, np.ones(10)/10, mode="valid")
            ax.plot(range(9, len(ppo_y)), ma, color="C1", label="PPO (MA10)")
        else:
            ax.plot(range(len(ppo_y)), ppo_y, color="C1", label="PPO")
    ax.set_xlabel("Episode (DQN) / Update (PPO)")
    ax.set_ylabel("Return")
    ax.set_title("Moving Landing Pad: DQN vs PPO (training curves)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[compare] 训练曲线对比图 -> {out}")


def main():
    p = argparse.ArgumentParser(description="移动平台 DQN/PPO 演示 + 对比图")
    p.add_argument("--dqn-model", default="outputs/moving_pad_dqn/best_policy.pt")
    p.add_argument("--ppo-model", default="outputs/moving_pad_ppo/last_policy.pt")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-gif", action="store_true")
    args = p.parse_args()

    if not args.skip_gif:
        if Path(args.dqn_model).exists():
            dqn = _load_dqn(args.dqn_model, args.device)
            record_gif(dqn, "outputs/moving_pad_dqn_landing.gif",
                       seed=args.seed, label="DQN", device=args.device)
        else:
            print(f"[skip] DQN 模型不存在: {args.dqn_model}")
        if Path(args.ppo_model).exists():
            ppo = _load_ppo(args.ppo_model, args.device)
            record_gif(ppo, "outputs/moving_pad_ppo_landing.gif",
                       seed=args.seed, label="PPO", device=args.device)
        else:
            print(f"[skip] PPO 模型不存在: {args.ppo_model}")
    plot_compare()


if __name__ == "__main__":
    main()
