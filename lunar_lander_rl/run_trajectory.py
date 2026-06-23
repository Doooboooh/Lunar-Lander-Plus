from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .actor_critic import ActorCriticConfig, ActorCriticNet, train as train_actor_critic
from .common import ensure_dir, get_device, mlp, save_json
from .dqn import DQNConfig, train as train_dqn
from .ppo import ActorCriticNet as PPONet
from .ppo import PPOConfig, load_obs_normalizer, train as train_ppo
from .trajectory_env import ROUTES, load_waypoints, make_waypoint_env, parse_waypoint_text, resolve_waypoints
from .trajectory_eval import evaluate_waypoint_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural RL baselines on waypoint LunarLander extensions.")
    parser.add_argument("--algorithm", choices=["dqn", "ppo", "actor_critic", "all"], default="all")
    parser.add_argument("--task", default="two_waypoint", help=f"Named route, one of {sorted(ROUTES)}, unless custom waypoints are passed.")
    parser.add_argument("--waypoints", help='Custom route as JSON or "x,y;x,y", for example "-0.4,1.1;0.4,1.1".')
    parser.add_argument("--waypoints-file", help="JSON/text file containing a custom waypoint path.")
    parser.add_argument("--route-label", help="Output directory label for a custom route.")
    parser.add_argument("--quick", action="store_true", help="Use short smoke-test settings.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--radius", type=float, default=0.16)
    parser.add_argument("--progress-reward", type=float, default=8.0)
    parser.add_argument("--waypoint-bonus", type=float, default=45.0)
    parser.add_argument("--route-bonus", type=float, default=80.0)
    parser.add_argument("--early-landing-penalty", type=float, default=160.0)
    parser.add_argument("--landing-after-route-bonus", type=float, default=40.0)
    parser.add_argument("--base-reward-scale-before-route", type=float, default=0.2)
    parser.add_argument("--base-reward-scale-after-route", type=float, default=1.0)
    parser.add_argument("--trajectory-eval-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override reward-evaluation episodes inside each trainer.")
    parser.add_argument("--dqn-episodes", type=int, default=None)
    parser.add_argument("--ppo-updates", type=int, default=None)
    parser.add_argument("--ppo-rollout-steps", type=int, default=None)
    parser.add_argument("--actor-critic-episodes", type=int, default=None)
    parser.add_argument("--save-gif", action="store_true", help="Save one trajectory evaluation GIF per selected algorithm.")
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/trajectory")
    return parser.parse_args()


def load_policy(algorithm: str, model_dir: Path, env_factory, hidden_dim: int, device_name: str):
    env = env_factory(seed=None)
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    finally:
        env.close()

    device = get_device(device_name)
    if algorithm == "dqn":
        model = mlp(obs_dim, action_dim, hidden_dim).to(device)
    elif algorithm == "ppo":
        model = PPONet(obs_dim, action_dim, hidden_dim).to(device)
    else:
        model = ActorCriticNet(obs_dim, action_dim, hidden_dim).to(device)

    model.load_state_dict(torch.load(model_dir / "best_policy.pt", map_location=device))
    model.eval()
    obs_rms = load_obs_normalizer(model_dir / "obs_norm.npz") if algorithm == "ppo" else None

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            model_obs = obs_rms.normalize(obs) if obs_rms is not None else obs
            obs_tensor = torch.tensor(model_obs, dtype=torch.float32, device=device).unsqueeze(0)
            if algorithm == "dqn":
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
    waypoints = resolve_waypoints(args.task, custom_waypoints)
    route_label = args.route_label or (f"custom_{len(waypoints)}" if custom_waypoints is not None else args.task)
    root = ensure_dir(Path(args.output_dir) / route_label)

    reward_kwargs = {
        "progress_reward": args.progress_reward,
        "waypoint_bonus": args.waypoint_bonus,
        "route_bonus": args.route_bonus,
        "early_landing_penalty": args.early_landing_penalty,
        "landing_after_route_bonus": args.landing_after_route_bonus,
        "base_reward_scale_before_route": args.base_reward_scale_before_route,
        "base_reward_scale_after_route": args.base_reward_scale_after_route,
    }

    def env_factory(render: bool = False, seed: int | None = None):
        return make_waypoint_env(
            render=render,
            seed=seed,
            route_name=args.task,
            custom_waypoints=custom_waypoints,
            radius=args.radius,
            **reward_kwargs,
        )

    if args.quick:
        configs = {
            "dqn": DQNConfig(
                episodes=4,
                warmup_steps=32,
                batch_size=16,
                eval_episodes=1,
                seed=args.seed,
                device=args.device,
                output_dir=str(root / "dqn"),
            ),
            "ppo": PPOConfig(
                updates=1,
                rollout_steps=96,
                minibatch_size=32,
                eval_episodes=1,
                seed=args.seed,
                device=args.device,
                output_dir=str(root / "ppo"),
            ),
            "actor_critic": ActorCriticConfig(
                episodes=4,
                eval_episodes=1,
                seed=args.seed,
                device=args.device,
                output_dir=str(root / "actor_critic"),
            ),
        }
    else:
        configs = {
            "dqn": DQNConfig(seed=args.seed, device=args.device, output_dir=str(root / "dqn")),
            "ppo": PPOConfig(seed=args.seed, device=args.device, output_dir=str(root / "ppo")),
            "actor_critic": ActorCriticConfig(seed=args.seed, device=args.device, output_dir=str(root / "actor_critic")),
        }

    if args.dqn_episodes is not None:
        configs["dqn"].episodes = args.dqn_episodes
    if args.ppo_updates is not None:
        configs["ppo"].updates = args.ppo_updates
    if args.ppo_rollout_steps is not None:
        configs["ppo"].rollout_steps = args.ppo_rollout_steps
    if args.actor_critic_episodes is not None:
        configs["actor_critic"].episodes = args.actor_critic_episodes
    if args.eval_episodes is not None:
        for cfg in configs.values():
            cfg.eval_episodes = args.eval_episodes

    selected = ["dqn", "ppo", "actor_critic"] if args.algorithm == "all" else [args.algorithm]
    reward_results = {}
    trajectory_results = {}
    trajectory_eval_episodes = args.trajectory_eval_episodes if args.trajectory_eval_episodes is not None else (1 if args.quick else 5)

    for name in selected:
        if name == "dqn":
            reward_results[name] = train_dqn(configs[name], env_factory=env_factory)
        elif name == "ppo":
            reward_results[name] = train_ppo(configs[name], env_factory=env_factory)
        else:
            reward_results[name] = train_actor_critic(configs[name], env_factory=env_factory)

        model_dir = Path(configs[name].output_dir)
        policy = load_policy(name, model_dir, env_factory, configs[name].hidden_dim, args.device)
        gif_path = model_dir / "trajectory_eval.gif" if args.save_gif else None
        trajectory_metrics = evaluate_waypoint_policy(
            policy,
            route_name=args.task,
            custom_waypoints=custom_waypoints,
            episodes=trajectory_eval_episodes,
            seed=args.seed + 20000,
            radius=args.radius,
            gif_path=gif_path,
            gif_fps=args.gif_fps,
            env_kwargs=reward_kwargs,
        )
        trajectory_results[name] = trajectory_metrics
        save_json(trajectory_metrics, model_dir / "trajectory_metrics.json")

    save_json(
        {
            "task": args.task,
            "route_label": route_label,
            "waypoints": waypoints,
            "radius": args.radius,
            "reward_config": reward_kwargs,
            "reward_results": reward_results,
            "trajectory_results": trajectory_results,
            "note": "Q-Learning is intentionally omitted here because the augmented continuous observation makes tabular discretization explode.",
        },
        root / "summary.json",
    )

    print("\nTrajectory evaluation summary")
    for name, metrics in trajectory_results.items():
        print(
            f"{name:14s} mean={metrics['mean_return']:8.1f} "
            f"wp={metrics['mean_waypoints_completed']:.1f}/{metrics['waypoint_count']:.0f} "
            f"route={metrics['route_completion_rate']:.2f} land_after_route={metrics['landed_after_route_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
