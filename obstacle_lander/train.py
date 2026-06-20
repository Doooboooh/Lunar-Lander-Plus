"""Train DQN/PPO/Actor-Critic on the obstacle LunarLander env.

Default hyperparameters are tuned for the 14-dim obstacle task — they differ
from the base project's defaults, which were calibrated for plain LunarLander.

Examples:
    python -m obstacle_lander.train --algo dqn --episodes 400
    python -m obstacle_lander.train --algo ppo --random-obstacles
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from lunar_lander_rl.actor_critic import ActorCriticConfig
from lunar_lander_rl.dqn import DQNConfig
from lunar_lander_rl.ppo import PPOConfig

from .env import make_obstacle_env


ALGOS = ("dqn", "ppo", "actor_critic")

# Tuned defaults for the 14-dim obstacle env.
# - PPO base defaults (200 updates, entropy 0.01, hidden 128) collapse on
#   LunarLander; raise updates, kill entropy bonus, widen the net.
# - Vanilla AC base defaults (600 episodes, lr 1e-3) flat-line; give it more
#   episodes and a smaller step size.
PPO_TUNED = dict(updates=500, hidden_dim=256, entropy_coef=0.0)
AC_TUNED = dict(episodes=1500, lr=3e-4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents on obstacle LunarLander.")
    parser.add_argument("--algo", required=True, choices=ALGOS)
    parser.add_argument("--episodes", type=int, default=None, help="DQN / Actor-Critic episodes")
    parser.add_argument("--updates", type=int, default=None, help="PPO updates")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--random-obstacles", action="store_true")
    parser.add_argument("--n-obstacles", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.12)
    return parser.parse_args()


def _make_env_factory(args: argparse.Namespace):
    return make_obstacle_env(
        n_obstacles=args.n_obstacles,
        radius=args.radius,
        random_obstacles=args.random_obstacles,
    )


def run_dqn(args: argparse.Namespace) -> None:
    overrides = {"seed": args.seed, "device": args.device, "output_dir": args.output_dir or "obstacle_lander/outputs/dqn"}
    if args.episodes is not None:
        overrides["episodes"] = args.episodes
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.lr is not None:
        overrides["lr"] = args.lr
    cfg = DQNConfig(**overrides)
    from .training import train_dqn

    train_dqn(cfg, env_factory=_make_env_factory(args))


def run_ppo(args: argparse.Namespace) -> None:
    overrides = {"seed": args.seed, "device": args.device, "output_dir": args.output_dir or "obstacle_lander/outputs/ppo", **PPO_TUNED}
    if args.updates is not None:
        overrides["updates"] = args.updates
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.lr is not None:
        overrides["lr"] = args.lr
    cfg = PPOConfig(**overrides)
    from .training import train_ppo

    train_ppo(cfg, env_factory=_make_env_factory(args))


def run_actor_critic(args: argparse.Namespace) -> None:
    overrides = {"seed": args.seed, "device": args.device, "output_dir": args.output_dir or "obstacle_lander/outputs/actor_critic", **AC_TUNED}
    if args.episodes is not None:
        overrides["episodes"] = args.episodes
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.lr is not None:
        overrides["lr"] = args.lr
    cfg = ActorCriticConfig(**overrides)
    from .training import train_actor_critic

    train_actor_critic(cfg, env_factory=_make_env_factory(args))


def main() -> None:
    args = parse_args()
    if args.algo == "dqn":
        run_dqn(args)
    elif args.algo == "ppo":
        run_ppo(args)
    elif args.algo == "actor_critic":
        run_actor_critic(args)
    else:
        raise ValueError(f"unknown algo: {args.algo}")


if __name__ == "__main__":
    main()
