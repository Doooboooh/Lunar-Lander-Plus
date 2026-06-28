import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunar_lander_rl.experiments import evaluate_model_detailed, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["a2c", "dqn", "ppo"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", default="BaseLunarLander-v0")
    parser.add_argument("--vec-normalize", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = load_model(args.algorithm, args.model)
    metrics = evaluate_model_detailed(
        model=model,
        env_id=args.env,
        episodes=args.episodes,
        seed=args.seed,
        vec_normalize_path=args.vec_normalize,
    )
    print(
        f"{args.env}: "
        f"mean_reward={metrics['mean_reward']:.2f} "
        f"std_reward={metrics['std_reward']:.2f} "
        f"mean_episode_length={metrics['mean_episode_length']:.1f} "
        f"mean_completed_waypoints={metrics['mean_completed_waypoints']:.2f} "
        f"waypoint_success_rate={metrics['waypoint_success_rate']:.2f} "
        f"time_limit_truncated_rate={metrics['time_limit_truncated_rate']:.2f}"
    )


if __name__ == "__main__":
    main()
