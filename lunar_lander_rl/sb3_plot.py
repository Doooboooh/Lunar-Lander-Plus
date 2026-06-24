"""SB3 三算法在移动平台上的对比柱状图（基于 eval metrics）。

产物：outputs/moving_pad/compare/sb3_eval_compare.png

用法：python -m lunar_lander_rl.sb3_plot
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    sb3_dir = Path("outputs/moving_pad/sb3")
    algos = ["dqn", "ppo", "a2c"]
    means, stds = [], []
    for a in algos:
        m = json.loads((sb3_dir / a / "metrics.json").read_text(encoding="utf-8"))
        means.append(m["eval_mean_return"])
        stds.append(m["eval_std_return"])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["C0", "C1", "C2"]
    bars = ax.bar([a.upper() for a in algos], means, yerr=stds, capsize=8, color=colors, alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_ylabel("Eval mean return (10 episodes)")
    ax.set_title("Moving Landing Pad: SB3 algorithm comparison (200k timesteps)")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (30 if v >= 0 else -60),
                f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path("outputs/moving_pad/compare/sb3_eval_compare.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
