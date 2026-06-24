"""绘制 SB3 三算法在移动平台上的 episode-reward 训练曲线（基于 SB3 Monitor）。

读取 outputs/moving_pad/sb3/{algo}/monitor.csv，画原始 + 滑动平均，三算法同图对比。
产物：outputs/moving_pad/compare/sb3_training_curves.png

用法：python -m lunar_lander_rl.sb3_curves
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_monitor_rewards(monitor_csv: str) -> np.ndarray:
    """读 SB3 Monitor csv，返回每个 episode 的 reward 数组。Monitor csv 前 2 行是元信息。"""
    p = Path(monitor_csv)
    if not p.exists():
        return np.array([])
    rewards = []
    with open(p, newline="") as f:
        # SB3 monitor: 第1行 #{"..."}, 第2行 列名 r,l,t, 第3行起数据
        lines = f.readlines()
        for line in lines[2:]:   # 跳过前两行元信息
            parts = line.strip().split(",")
            if len(parts) >= 1:
                try:
                    rewards.append(float(parts[0]))   # r = episode reward
                except ValueError:
                    continue
    return np.array(rewards)


def moving_avg(y, w):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def main():
    sb3_dir = Path("outputs/moving_pad/sb3")
    cfg = [("dqn", "DQN", "C0", 20),
           ("ppo", "PPO", "C1", 20),
           ("a2c", "A2C", "C2", 20)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, label, color, w in cfg:
        y = load_monitor_rewards(sb3_dir / algo / "monitor.csv.monitor" if False else sb3_dir / algo / "monitor.csv")
        # SB3 Monitor 文件名是 <filename>.monitor.csv？实际 Monitor(filename="x") 写 x.csv
        if len(y) == 0:
            # 试别的命名
            for cand in ["monitor.csv", "monitor.monitor.csv"]:
                y = load_monitor_rewards(sb3_dir / algo / cand)
                if len(y):
                    break
        if len(y) == 0:
            print(f"  跳过 {label}（无 monitor 数据）")
            continue
        ax.plot(range(len(y)), y, alpha=0.2, color=color)
        ma = moving_avg(y, w)
        ax.plot(range(w - 1, len(y)), ma, color=color, linewidth=2,
                label=f"{label} (MA{w}, best={y.max():.0f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Moving Landing Pad: SB3 training curves (200k timesteps)")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path("outputs/moving_pad/compare/sb3_training_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
