"""绘制四种算法在移动平台上的 episode-reward 训练曲线对比图。

读取 outputs/moving_pad/{moving_pad_dqn, moving_pad_ppo, q_learning, actor_critic}/history.csv，
画原始曲线（淡）+ 滑动平均（深），四算法同图对比。
产物：outputs/moving_pad/compare/4algo_training_curves.png

用法：python -m lunar_lander_rl.moving_pad_4algo_plot
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_returns(d: str) -> tuple[np.ndarray, str]:
    """返回 (returns 数组, x 轴标签)。"""
    p = Path("outputs/moving_pad") / d / "history.csv"
    if not p.exists():
        return np.array([]), d
    xs, ys = [], []
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                xs.append(int(row.get("episode", row.get("update", len(xs)))))
                ys.append(float(row["return"]))
            except (ValueError, KeyError):
                continue
    return np.array(ys), d


def moving_avg(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def main():
    algos = [("moving_pad_dqn", "DQN", "C0", 20),
             ("moving_pad_ppo", "PPO", "C1", 10),
             ("q_learning", "Q-Learning", "C2", 20),
             ("actor_critic", "Actor-Critic", "C3", 20)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for d, label, color, w in algos:
        y, _ = load_returns(d)
        if len(y) == 0:
            print(f"  跳过 {label}（无 history）")
            continue
        ax.plot(range(len(y)), y, alpha=0.2, color=color)
        ma = moving_avg(y, w)
        ax.plot(range(w - 1, len(y)), ma, color=color, linewidth=2,
                label=f"{label} (MA{w}, best={y.max():.0f})")
    ax.set_xlabel("Episode (DQN/Q-Learning/Actor-Critic) / Update (PPO)")
    ax.set_ylabel("Return")
    ax.set_title("Moving Landing Pad: 4-algorithm training curves")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path("outputs/moving_pad/compare/4algo_training_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
