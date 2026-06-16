from __future__ import annotations

import argparse
from pathlib import Path

from .actor_critic import ActorCriticConfig, train as train_actor_critic
from .common import ensure_dir, save_json
from .dqn import DQNConfig, train as train_dqn
from .ppo import PPOConfig, train as train_ppo
from .q_learning import QLearningConfig, train as train_q_learning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all LunarLander-v3 RL demos and compare evaluation scores.")
    parser.add_argument("--quick", action="store_true", help="Use short smoke-test settings.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if args.quick:
        configs = {
            "q_learning": QLearningConfig(episodes=5, eval_episodes=2, seed=args.seed, output_dir=str(output_dir / "q_learning")),
            "dqn": DQNConfig(
                episodes=5,
                warmup_steps=64,
                batch_size=32,
                eval_episodes=2,
                seed=args.seed,
                device=args.device,
                output_dir=str(output_dir / "dqn"),
            ),
            "ppo": PPOConfig(
                updates=2,
                rollout_steps=128,
                minibatch_size=64,
                eval_episodes=2,
                seed=args.seed,
                device=args.device,
                output_dir=str(output_dir / "ppo"),
            ),
            "actor_critic": ActorCriticConfig(
                episodes=5,
                eval_episodes=2,
                seed=args.seed,
                device=args.device,
                output_dir=str(output_dir / "actor_critic"),
            ),
        }
    else:
        configs = {
            "q_learning": QLearningConfig(seed=args.seed, output_dir=str(output_dir / "q_learning")),
            "dqn": DQNConfig(seed=args.seed, device=args.device, output_dir=str(output_dir / "dqn")),
            "ppo": PPOConfig(seed=args.seed, device=args.device, output_dir=str(output_dir / "ppo")),
            "actor_critic": ActorCriticConfig(seed=args.seed, device=args.device, output_dir=str(output_dir / "actor_critic")),
        }

    results = {
        "q_learning": train_q_learning(configs["q_learning"]),
        "dqn": train_dqn(configs["dqn"]),
        "ppo": train_ppo(configs["ppo"]),
        "actor_critic": train_actor_critic(configs["actor_critic"]),
    }
    save_json({"results": results}, Path(output_dir) / "summary.json")

    print("\nEvaluation summary")
    for name, metrics in results.items():
        print(
            f"{name:14s} mean={metrics['mean_return']:8.1f} "
            f"std={metrics['std_return']:7.1f} min={metrics['min_return']:8.1f} max={metrics['max_return']:8.1f}"
        )


if __name__ == "__main__":
    main()

