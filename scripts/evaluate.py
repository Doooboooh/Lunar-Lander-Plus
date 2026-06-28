import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunar_lander_rl.experiments import evaluate_model, evaluate_obstacle_model, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["a2c", "dqn", "ppo"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", default="BaseLunarLander-v0")
    parser.add_argument("--vec-normalize", default=None)
    parser.add_argument("--env-kwargs", default=None, help="JSON string of env kwargs, e.g. '{\"random_obstacles\": true}'")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = load_model(args.algorithm, args.model)
    env_kwargs = json.loads(args.env_kwargs) if args.env_kwargs else None

    if args.env.startswith("Obstacle"):
        metrics = evaluate_obstacle_model(
            model=model,
            env_id=args.env,
            episodes=args.episodes,
            seed=args.seed,
            env_kwargs=env_kwargs,
            vec_normalize_path=args.vec_normalize,
        )
        print(
            f"{args.env}: mean_return={metrics['mean_return']:.2f} "
            f"std={metrics['std_return']:.2f} "
            f"collision_rate={metrics['collision_rate']:.2%} "
            f"success_rate={metrics['success_rate']:.2%}"
        )
    else:
        mean_reward, std_reward = evaluate_model(
            model=model,
            env_id=args.env,
            episodes=args.episodes,
            seed=args.seed,
            vec_normalize_path=args.vec_normalize,
            env_kwargs=env_kwargs,
        )
        print(f"{args.env}: mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}")


if __name__ == "__main__":
    main()
