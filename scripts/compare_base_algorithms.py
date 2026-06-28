import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunar_lander_rl.config import load_config
from lunar_lander_rl.experiments import train_from_config


DEFAULT_CONFIGS = [
    "configs/base_compare_ppo.json",
    "configs/base_compare_dqn.json",
    "configs/base_compare_a2c.json",
]


def load_metrics(path: Path) -> dict:
    return load_config(path)


def collect_rows(output_root: Path, configs: list[str]) -> list[dict]:
    rows = []
    for config_path in configs:
        config = load_config(config_path)
        name = config["name"]
        metrics_path = output_root / name / "metrics.json"
        metrics = load_metrics(metrics_path)
        rows.append(
            {
                "name": name,
                "algorithm": metrics["algorithm"].upper(),
                "env_id": metrics["env_id"],
                "seed": metrics["seed"],
                "total_timesteps": metrics["total_timesteps"],
                "eval_episodes": metrics["eval_episodes"],
                "mean_reward": metrics["mean_reward"],
                "std_reward": metrics["std_reward"],
                "model_path": metrics["model_path"],
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Base LunarLander Algorithm Comparison",
        "",
        "| Algorithm | Timesteps | Eval episodes | Mean reward | Std reward |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["mean_reward"], reverse=True):
        lines.append(
            "| {algorithm} | {total_timesteps} | {eval_episodes} | {mean_reward:.2f} | {std_reward:.2f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "All runs use `BaseLunarLander-v0` with the same seed and evaluation protocol.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda item: item["algorithm"])
    labels = [row["algorithm"] for row in rows]
    means = [row["mean_reward"] for row in rows]
    stds = [row["std_reward"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4e79a7", "#59a14f", "#f28e2b"]
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors[: len(labels)], edgecolor="#2f3640")
    ax.axhline(200, color="#2ca02c", linestyle="--", linewidth=1.4, label="Solved threshold")
    ax.axhline(0, color="#2f3640", linewidth=0.8)
    ax.set_title("PPO vs DQN vs A2C on BaseLunarLander-v0")
    ax.set_ylabel("Mean evaluation reward")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-root", default="outputs/base_algorithm_comparison")
    parser.add_argument("--report-dir", default="reports/base_algorithm_comparison")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)

    if not args.skip_train:
        for config_path in args.configs:
            config = load_config(config_path)
            train_from_config(config, output_root / config["name"])

    rows = collect_rows(output_root, args.configs)
    write_csv(rows, report_dir / "metrics.csv")
    write_markdown(rows, report_dir / "summary.md")
    write_plot(rows, report_dir / "mean_reward.png")

    for row in sorted(rows, key=lambda item: item["mean_reward"], reverse=True):
        print(
            "{algorithm}: mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}".format(
                **row
            )
        )
    print(f"saved comparison reports to {report_dir}")


if __name__ == "__main__":
    main()
