"""Compare PPO/A2C/DQN training curves for one obstacle setting on one figure.

Plots smoothed episode reward vs cumulative environment timesteps, so the three
algorithms (which trained for different total_timesteps) are compared on a
shared, fair x-axis.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (label, color, monitor csv) per setting.
FIXED_RUNS = [
    ("PPO (1M)", "#1f77b4", "outputs/obstacle_ppo/monitor/train.monitor.csv"),
    ("A2C (1M)", "#ff7f0e", "outputs/obstacle_a2c/monitor/train.monitor.csv"),
    ("DQN (300k)", "#2ca02c", "outputs/obstacle_dqn/monitor/train.monitor.csv"),
]
RANDOM_RUNS = [
    ("PPO (3M)", "#1f77b4", "outputs/obstacle_ppo_random_3m/monitor/train.monitor.csv"),
    ("A2C (2M)", "#ff7f0e", "outputs/obstacle_a2c_random/monitor/train.monitor.csv"),
    ("DQN (2M)", "#2ca02c", "outputs/obstacle_dqn_random_2m/monitor/train.monitor.csv"),
]


def moving_average(values, window):
    if window <= 1:
        return values
    result = []
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        result.append(sum(values[start : idx + 1]) / (idx + 1 - start))
    return result


def load_monitor_csv(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                rows.append(line)
    reader = csv.DictReader(rows)
    rewards, lengths = [], []
    for row in reader:
        rewards.append(float(row["r"]))
        lengths.append(int(row["l"]))
    return rewards, lengths


def cumulative_steps(lengths):
    total = 0
    out = []
    for length in lengths:
        total += length
        out.append(total)
    return out


def plot_setting(runs, window, output, title):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, color, csv_path in runs:
        path = Path(csv_path)
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        rewards, lengths = load_monitor_csv(path)
        steps = cumulative_steps(lengths)
        smooth = moving_average(rewards, window)
        ax.plot(steps, smooth, color=color, linewidth=2, label=label)
        # faint raw curve for context
        ax.plot(steps, rewards, color=color, linewidth=0.5, alpha=0.25)

    ax.set_xlabel("Cumulative environment timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved plot to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    plot_setting(
        FIXED_RUNS, args.window,
        "reports/obstacle_compare_fixed.png",
        "Fixed obstacles (3) — PPO / A2C / DQN training reward",
    )
    plot_setting(
        RANDOM_RUNS, args.window,
        "reports/obstacle_compare_random.png",
        "Random obstacle (1) — PPO / A2C / DQN training reward",
    )


if __name__ == "__main__":
    main()
