import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", default="outputs/base_ppo_1m/monitor/train.monitor.csv")
    parser.add_argument("--output", default="reports/base_ppo_1m_training.png")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    rewards, lengths = load_monitor_csv(args.monitor)
    episodes = list(range(1, len(rewards) + 1))
    smooth_rewards = moving_average(rewards, args.window)
    smooth_lengths = moving_average(lengths, args.window)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(episodes, rewards, color="#b8c1cc", linewidth=0.8, label="episode reward")
    axes[0].plot(episodes, smooth_rewards, color="#1f77b4", linewidth=2, label=f"{args.window}-episode average")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(episodes, lengths, color="#d6b0a4", linewidth=0.8, label="episode length")
    axes[1].plot(episodes, smooth_lengths, color="#d62728", linewidth=2, label=f"{args.window}-episode average")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Length")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"saved plot to {output}")


if __name__ == "__main__":
    main()
