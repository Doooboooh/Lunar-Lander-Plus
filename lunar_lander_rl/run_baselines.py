from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

from .actor_critic import ActorCriticConfig, train as train_actor_critic
from .common import ensure_dir, save_json
from .dqn import DQNConfig, train as train_dqn
from .ppo import PPOConfig, train as train_ppo
from .q_learning import QLearningConfig, train as train_q_learning


ALGORITHMS = ("q_learning", "dqn", "ppo", "actor_critic")


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_algorithm_list(value: str) -> list[str]:
    if value == "all":
        return list(ALGORITHMS)
    algorithms = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(algorithms) - set(ALGORITHMS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown algorithm(s): {', '.join(unknown)}")
    return algorithms


def build_config(algorithm: str, profile: str, seed: int, output_dir: Path, device: str):
    if profile == "smoke":
        if algorithm == "q_learning":
            return QLearningConfig(episodes=5, eval_episodes=2, seed=seed, output_dir=str(output_dir))
        if algorithm == "dqn":
            return DQNConfig(
                episodes=5,
                warmup_steps=64,
                batch_size=32,
                eval_episodes=2,
                seed=seed,
                device=device,
                output_dir=str(output_dir),
            )
        if algorithm == "ppo":
            return PPOConfig(
                updates=2,
                rollout_steps=128,
                minibatch_size=64,
                eval_episodes=2,
                seed=seed,
                device=device,
                output_dir=str(output_dir),
            )
        return ActorCriticConfig(episodes=5, eval_episodes=2, seed=seed, device=device, output_dir=str(output_dir))

    if profile == "course":
        if algorithm == "q_learning":
            return QLearningConfig(episodes=120, eval_episodes=5, seed=seed, output_dir=str(output_dir))
        if algorithm == "dqn":
            return DQNConfig(
                episodes=120,
                warmup_steps=500,
                target_update=500,
                eval_episodes=5,
                seed=seed,
                device=device,
                output_dir=str(output_dir),
            )
        if algorithm == "ppo":
            return PPOConfig(
                updates=40,
                rollout_steps=512,
                minibatch_size=128,
                eval_episodes=5,
                seed=seed,
                device=device,
                output_dir=str(output_dir),
            )
        return ActorCriticConfig(episodes=120, eval_episodes=5, seed=seed, device=device, output_dir=str(output_dir))

    if algorithm == "q_learning":
        return QLearningConfig(seed=seed, output_dir=str(output_dir))
    if algorithm == "dqn":
        return DQNConfig(seed=seed, device=device, output_dir=str(output_dir))
    if algorithm == "ppo":
        return PPOConfig(seed=seed, device=device, output_dir=str(output_dir))
    return ActorCriticConfig(seed=seed, device=device, output_dir=str(output_dir))


def run_one(algorithm: str, config) -> dict[str, float]:
    if algorithm == "q_learning":
        return train_q_learning(config)
    if algorithm == "dqn":
        return train_dqn(config)
    if algorithm == "ppo":
        return train_ppo(config)
    return train_actor_critic(config)


def aggregate(results: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for algorithm in ALGORITHMS:
        rows = [row for row in results if row["algorithm"] == algorithm]
        if not rows:
            continue
        means = [float(row["metrics"]["mean_return"]) for row in rows]
        stds = [float(row["metrics"]["std_return"]) for row in rows]
        summary[algorithm] = {
            "runs": len(rows),
            "mean_of_mean_return": mean(means),
            "std_of_mean_return": pstdev(means) if len(means) > 1 else 0.0,
            "mean_eval_std_return": mean(stds),
            "best_mean_return": max(means),
            "worst_mean_return": min(means),
        }
    return summary


def write_markdown_table(summary: dict[str, dict[str, float]], path: Path) -> None:
    lines = [
        "| 方法 | runs | mean eval return | across-seed std | mean episode std | best | worst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm, metrics in summary.items():
        lines.append(
            f"| {algorithm} | {int(metrics['runs'])} | {metrics['mean_of_mean_return']:.2f} | "
            f"{metrics['std_of_mean_return']:.2f} | {metrics['mean_eval_std_return']:.2f} | "
            f"{metrics['best_mean_return']:.2f} | {metrics['worst_mean_return']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible baseline LunarLander-v3 experiments.")
    parser.add_argument("--profile", choices=["smoke", "course", "formal"], default="course")
    parser.add_argument("--algorithms", type=parse_algorithm_list, default=list(ALGORITHMS))
    parser.add_argument("--seeds", type=parse_int_list, default=[42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/baselines_course")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    all_results = []

    for seed in args.seeds:
        for algorithm in args.algorithms:
            run_dir = output_dir / algorithm / f"seed_{seed}"
            config = build_config(algorithm, args.profile, seed, run_dir, args.device)
            print(f"\n=== {algorithm} seed={seed} profile={args.profile} ===")
            metrics = run_one(algorithm, config)
            all_results.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "profile": args.profile,
                    "output_dir": str(run_dir),
                    "config": asdict(config),
                    "metrics": metrics,
                }
            )

    summary = aggregate(all_results)
    save_json({"runs": all_results, "summary": summary}, output_dir / "summary.json")
    write_markdown_table(summary, output_dir / "summary.md")

    print("\nAggregate summary")
    for algorithm, metrics in summary.items():
        print(
            f"{algorithm:14s} runs={int(metrics['runs'])} "
            f"mean={metrics['mean_of_mean_return']:8.1f} "
            f"seed_std={metrics['std_of_mean_return']:7.1f} "
            f"best={metrics['best_mean_return']:8.1f}"
        )


if __name__ == "__main__":
    main()
