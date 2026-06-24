import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunar_lander_rl.config import load_config
from lunar_lander_rl.experiments import train_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_ppo.json")
    parser.add_argument("--output-dir", default="outputs/debug")
    args = parser.parse_args()

    model_path = train_from_config(load_config(args.config), args.output_dir)
    print(f"saved model to {model_path}")


if __name__ == "__main__":
    main()
