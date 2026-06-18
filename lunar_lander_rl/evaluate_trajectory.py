from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .actor_critic import ActorCriticNet
from .common import get_device, mlp, save_json
from .ppo import ActorCriticNet as PPONet
from .trajectory_env import ROUTES, load_waypoints, make_waypoint_env, parse_waypoint_text
from .trajectory_eval import evaluate_waypoint_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on waypoint LunarLander metrics.")
    parser.add_argument("algorithm", choices=["dqn", "ppo", "actor_critic"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", default="best_policy.pt")
    parser.add_argument("--task", default="two_waypoint", help=f"Named route, one of {sorted(ROUTES)}, unless custom waypoints are passed.")
    parser.add_argument("--waypoints", help='Custom route as JSON or "x,y;x,y".')
    parser.add_argument("--waypoints-file", help="JSON/text file containing a custom waypoint path.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--radius", type=float, default=0.16)
    parser.add_argument("--progress-reward", type=float, default=8.0)
    parser.add_argument("--waypoint-bonus", type=float, default=45.0)
    parser.add_argument("--route-bonus", type=float, default=80.0)
    parser.add_argument("--early-landing-penalty", type=float, default=160.0)
    parser.add_argument("--landing-after-route-bonus", type=float, default=40.0)
    parser.add_argument("--base-reward-scale-before-route", type=float, default=0.2)
    parser.add_argument("--base-reward-scale-after-route", type=float, default=1.0)
    parser.add_argument("--gif", help="Optional GIF path for the first evaluated episode.")
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--output", help="Optional JSON path for metrics.")
    return parser.parse_args()


def make_reward_kwargs(args: argparse.Namespace) -> dict:
    return {
        "progress_reward": args.progress_reward,
        "waypoint_bonus": args.waypoint_bonus,
        "route_bonus": args.route_bonus,
        "early_landing_penalty": args.early_landing_penalty,
        "landing_after_route_bonus": args.landing_after_route_bonus,
        "base_reward_scale_before_route": args.base_reward_scale_before_route,
        "base_reward_scale_after_route": args.base_reward_scale_after_route,
    }


def load_policy(args: argparse.Namespace, reward_kwargs: dict, custom_waypoints):
    env = make_waypoint_env(route_name=args.task, custom_waypoints=custom_waypoints, radius=args.radius, **reward_kwargs)
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    finally:
        env.close()

    device = get_device(args.device)
    if args.algorithm == "dqn":
        model = mlp(obs_dim, action_dim, args.hidden_dim).to(device)
    elif args.algorithm == "ppo":
        model = PPONet(obs_dim, action_dim, args.hidden_dim).to(device)
    else:
        model = ActorCriticNet(obs_dim, action_dim, args.hidden_dim).to(device)

    checkpoint = Path(args.model_dir) / args.checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if args.algorithm == "dqn":
                logits = model(obs_tensor)
            else:
                logits, _ = model(obs_tensor)
            return int(torch.argmax(logits, dim=1).item())

    return act


def main() -> None:
    args = parse_args()
    if args.waypoints and args.waypoints_file:
        raise SystemExit("Use only one of --waypoints or --waypoints-file.")
    custom_waypoints = None
    if args.waypoints_file:
        custom_waypoints = load_waypoints(args.waypoints_file)
    elif args.waypoints:
        custom_waypoints = parse_waypoint_text(args.waypoints)
    reward_kwargs = make_reward_kwargs(args)
    policy = load_policy(args, reward_kwargs, custom_waypoints)
    metrics = evaluate_waypoint_policy(
        policy,
        route_name=args.task,
        custom_waypoints=custom_waypoints,
        episodes=args.episodes,
        seed=args.seed,
        radius=args.radius,
        gif_path=args.gif,
        gif_fps=args.gif_fps,
        env_kwargs=reward_kwargs,
    )
    payload = {
        "algorithm": args.algorithm,
        "model_dir": args.model_dir,
        "checkpoint": args.checkpoint,
        "task": args.task,
        "custom_waypoints": custom_waypoints,
        "radius": args.radius,
        "reward_config": reward_kwargs,
        "metrics": metrics,
    }
    if args.output:
        save_json(payload, args.output)

    print(
        f"{args.algorithm} {args.checkpoint}: mean={metrics['mean_return']:.1f}, "
        f"wp={metrics['mean_waypoints_completed']:.1f}/{metrics['waypoint_count']:.0f}, "
        f"route={metrics['route_completion_rate']:.2f}, "
        f"land_after_route={metrics['landed_after_route_rate']:.2f}"
    )


if __name__ == "__main__":
    main()
