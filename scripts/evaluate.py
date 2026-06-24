import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunar_lander_rl.experiments import evaluate_model, load_model


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
    mean_reward, std_reward = evaluate_model(
        model=model,
        env_id=args.env,
        episodes=args.episodes,
        seed=args.seed,
        vec_normalize_path=args.vec_normalize,
    )
    print(f"{args.env}: mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}")


if __name__ == "__main__":
    main()
