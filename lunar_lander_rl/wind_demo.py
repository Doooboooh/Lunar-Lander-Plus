"""横风变体演示：对比基础 DQN 在无风 vs 强风下的着陆表现。

产物：
  outputs/wind/wind_none.gif       无风降落（baseline）
  outputs/wind/wind_strong.gif     强风降落（飞船被吹偏/坠毁）
  outputs/wind/wind_compare.png    各风强下成功率 + 平均 return 对比图

用法：
    python -m lunar_lander_rl.wind_demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import mlp, make_wind_env, WIND_LEVELS


def _load_policy(model_path: str, device: str = "cpu"):
    dev = torch.device(device)
    net = mlp(8, 4, 128).to(dev)
    net.load_state_dict(torch.load(model_path, map_location=dev))
    net.eval()
    def act(o):
        with torch.no_grad():
            t = torch.tensor(o, dtype=torch.float32, device=dev).unsqueeze(0)
            return int(net(t).argmax(1).item())
    return act


def record_gif(policy, level: str, gif_path: str, seed: int = 0, fps: int = 30,
               device: str = "cpu") -> dict:
    import imageio.v2 as imageio
    import gymnasium as gym
    from .common import WIND_LEVELS
    kwargs = WIND_LEVELS.get(level, WIND_LEVELS["medium"])
    if kwargs is None:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
    else:
        env = gym.make("LunarLander-v3", render_mode="rgb_array", **kwargs)
    o, _ = env.reset(seed=seed)
    frames = []
    total = 0.0
    u = env.unwrapped
    for t in range(1000):
        a = policy(o)
        o, r, term, trunc, _ = env.step(a)
        total += r
        frames.append(env.render())
        if term or trunc:
            break
    env.close()
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)
    landed = bool(u.legs[0].ground_contact and u.legs[1].ground_contact)
    rel_x = u.lander.position.x - 10.0
    print(f"[wind/{level}] {len(frames)} 帧 return={total:.1f} 落点={rel_x:+.2f} 双足接地={landed} -> {gif_path}")
    return {"return": total, "landed": landed, "rel_x": rel_x}


def evaluate_all(policy, model_path: str, n_seeds: int = 12, device: str = "cpu") -> dict:
    """各风强下统计成功率/平均 return/平均落点偏移。"""
    stats = {}
    for level in WIND_LEVELS:
        oks, rets, drifts = 0, [], []
        for seed in range(n_seeds):
            env = make_wind_env(level, seed=seed)
            o, _ = env.reset(seed=seed)
            u = env.unwrapped
            tot = 0.0
            for _ in range(1000):
                a = policy(o)
                o, r, term, trunc, _ = env.step(a)
                tot += r
                if term or trunc:
                    break
            p = u.lander.position
            both = u.legs[0].ground_contact and u.legs[1].ground_contact
            sp = float(np.hypot(u.lander.linearVelocity.x, u.lander.linearVelocity.y))
            ok = both and sp < 2 and abs(p.x - 10.0) < 0.6
            oks += ok
            rets.append(tot)
            drifts.append(p.x - 10.0)
            env.close()
        stats[level] = {"success": oks / n_seeds,
                        "mean_return": float(np.mean(rets)),
                        "mean_drift": float(np.mean(drifts))}
        print(f"  {level:7s}: 成功率={oks}/{n_seeds} return={np.mean(rets):7.1f} 落点偏移={np.mean(drifts):+.2f}")
    return stats


def plot_compare(stats: dict, out: str = "outputs/wind/wind_compare.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = list(stats.keys())
    succ = [stats[l]["success"] for l in levels]
    rets = [stats[l]["mean_return"] for l in levels]
    drift = [stats[l]["mean_drift"] for l in levels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = range(len(levels))
    axes[0].bar(x, succ, color="C2")
    axes[0].set_title("Landing success rate")
    axes[0].set_ylabel("success rate")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(levels)
    axes[1].bar(x, rets, color="C0")
    axes[1].set_title("Mean return")
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(levels)
    axes[2].bar(x, drift, color="C1")
    axes[2].set_title("Mean landing x drift (m)")
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_xticks(list(x)); axes[2].set_xticklabels(levels)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Wind disturbance robustness: DQN on LunarLander")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[wind] 对比图 -> {out}")


def main():
    p = argparse.ArgumentParser(description="横风变体演示")
    p.add_argument("--model", default="outputs/baseline/dqn/best_policy.pt")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-gif", action="store_true")
    args = p.parse_args()

    policy = _load_policy(args.model, args.device)
    print("[wind] 各风强统计（基础 DQN，标准 LunarLander 训练）：")
    stats = evaluate_all(policy, args.model, args.n_seeds, args.device)

    if not args.skip_gif:
        # 挑能体现对比的两段：无风 + 强风
        record_gif(policy, "none", "outputs/wind/wind_none.gif", seed=args.seed, device=args.device)
        record_gif(policy, "strong", "outputs/wind/wind_strong.gif", seed=args.seed, device=args.device)
    plot_compare(stats)


if __name__ == "__main__":
    main()
