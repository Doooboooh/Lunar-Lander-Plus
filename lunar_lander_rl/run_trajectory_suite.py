from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .actor_critic import ActorCriticConfig, ActorCriticNet, train as train_actor_critic
from .common import ensure_dir, get_device, mlp, save_json
from .dqn import DQNConfig, train as train_dqn
from .ppo import ActorCriticNet as PPONet
from .ppo import PPOConfig, train as train_ppo
from .trajectory_env import ROUTES, load_waypoints, make_waypoint_env, parse_waypoint_text, resolve_waypoints
from .trajectory_eval import evaluate_waypoint_policy


ALGORITHMS = ("dqn", "ppo", "actor_critic")


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_reward_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "progress_reward": args.progress_reward,
        "waypoint_bonus": args.waypoint_bonus,
        "route_bonus": args.route_bonus,
        "early_landing_penalty": args.early_landing_penalty,
        "landing_after_route_bonus": args.landing_after_route_bonus,
        "base_reward_scale_before_route": args.base_reward_scale_before_route,
        "base_reward_scale_after_route": args.base_reward_scale_after_route,
    }


def build_config(algorithm: str, profile: str, seed: int, output_dir: Path, device: str):
    if profile == "smoke":
        if algorithm == "dqn":
            return DQNConfig(episodes=4, warmup_steps=32, batch_size=16, eval_episodes=1, seed=seed, device=device, output_dir=str(output_dir))
        if algorithm == "ppo":
            return PPOConfig(updates=1, rollout_steps=96, minibatch_size=32, eval_episodes=1, seed=seed, device=device, output_dir=str(output_dir))
        return ActorCriticConfig(episodes=4, eval_episodes=1, seed=seed, device=device, output_dir=str(output_dir))

    if profile == "probe":
        if algorithm == "dqn":
            return DQNConfig(
                episodes=40,
                warmup_steps=128,
                target_update=250,
                batch_size=32,
                eval_episodes=3,
                seed=seed,
                device=device,
                output_dir=str(output_dir),
            )
        if algorithm == "ppo":
            return PPOConfig(
                updates=8,
                rollout_steps=256,
                minibatch_size=64,
                eval_episodes=3,
                seed=seed,
                device=device,
                output_dir=str(output_dir),
            )
        return ActorCriticConfig(episodes=40, eval_episodes=3, seed=seed, device=device, output_dir=str(output_dir))

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


def train_one(algorithm: str, config, env_factory):
    if algorithm == "dqn":
        return train_dqn(config, env_factory=env_factory)
    if algorithm == "ppo":
        return train_ppo(config, env_factory=env_factory)
    return train_actor_critic(config, env_factory=env_factory)


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

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if algorithm == "dqn":
                logits = model(obs_tensor)
            else:
                logits, _ = model(obs_tensor)
            return int(torch.argmax(logits, dim=1).item())

    return act


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "| task | algorithm | profile | reward mean | waypoints | route completion | landed after route | final target dist |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["trajectory_metrics"]
        lines.append(
            f"| {row['task_label']} | {row['algorithm']} | {row['profile']} | "
            f"{metrics['mean_return']:.2f} | "
            f"{metrics['mean_waypoints_completed']:.2f}/{metrics['waypoint_count']:.0f} | "
            f"{metrics['route_completion_rate']:.2f} | "
            f"{metrics['landed_after_route_rate']:.2f} | "
            f"{metrics['mean_final_target_distance']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact suite of waypoint LunarLander extension experiments.")
    parser.add_argument("--profile", choices=["smoke", "probe", "course"], default="probe")
    parser.add_argument("--tasks", default="single_left,near_two_waypoint,two_waypoint", help=f"Comma-separated named tasks from {sorted(ROUTES)}.")
    parser.add_argument("--algorithms", default="dqn,ppo,actor_critic", help="Comma-separated algorithms: dqn, ppo, actor_critic.")
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
    parser.add_argument("--custom-label", help="Optional label for one custom waypoint route.")
    parser.add_argument("--waypoints", help='Optional custom route as JSON or "x,y;x,y".')
    parser.add_argument("--waypoints-file", help="Optional JSON/text file containing one custom route.")
    parser.add_argument("--output-dir", default="outputs/trajectory_suite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.waypoints and args.waypoints_file:
        raise SystemExit("Use only one of --waypoints or --waypoints-file.")

    algorithms = parse_list(args.algorithms)
    unknown_algorithms = sorted(set(algorithms) - set(ALGORITHMS))
    if unknown_algorithms:
        raise SystemExit(f"Unknown algorithm(s): {', '.join(unknown_algorithms)}")

    task_specs: list[tuple[str, str, tuple[tuple[float, float], ...] | None]] = []
    for task in parse_list(args.tasks):
        resolve_waypoints(task)
        task_specs.append((task, task, None))

    if args.waypoints_file:
        custom = load_waypoints(args.waypoints_file)
        task_specs.append((args.custom_label or f"custom_{len(custom)}", "custom", custom))
    elif args.waypoints:
        custom = parse_waypoint_text(args.waypoints)
        task_specs.append((args.custom_label or f"custom_{len(custom)}", "custom", custom))

    output_root = ensure_dir(args.output_dir)
    reward_kwargs = make_reward_kwargs(args)
    eval_episodes = args.trajectory_eval_episodes if args.trajectory_eval_episodes is not None else (1 if args.profile == "smoke" else 3)
    rows: list[dict] = []

    for task_label, route_name, custom_waypoints in task_specs:
        for algorithm in algorithms:
            run_dir = output_root / task_label / algorithm / f"seed_{args.seed}"

            def env_factory(render: bool = False, seed: int | None = None, route_name=route_name, custom_waypoints=custom_waypoints):
                return make_waypoint_env(
                    render=render,
                    seed=seed,
                    route_name=route_name,
                    custom_waypoints=custom_waypoints,
                    radius=args.radius,
                    **reward_kwargs,
                )

            print(f"\n=== task={task_label} algorithm={algorithm} profile={args.profile} ===")
            config = build_config(algorithm, args.profile, args.seed, run_dir, args.device)
            reward_metrics = train_one(algorithm, config, env_factory)
            policy = load_policy(algorithm, run_dir, env_factory, config.hidden_dim, args.device)
            trajectory_metrics = evaluate_waypoint_policy(
                policy,
                route_name=route_name,
                custom_waypoints=custom_waypoints,
                episodes=eval_episodes,
                seed=args.seed + 20000,
                radius=args.radius,
                env_kwargs=reward_kwargs,
            )
            save_json(trajectory_metrics, run_dir / "trajectory_metrics.json")
            rows.append(
                {
                    "task_label": task_label,
                    "route_name": route_name,
                    "custom_waypoints": custom_waypoints,
                    "waypoints": resolve_waypoints(route_name, custom_waypoints),
                    "algorithm": algorithm,
                    "profile": args.profile,
                    "seed": args.seed,
                    "output_dir": str(run_dir),
                    "config": asdict(config),
                    "reward_metrics": reward_metrics,
                    "trajectory_metrics": trajectory_metrics,
                }
            )

    save_json(
        {
            "profile": args.profile,
            "seed": args.seed,
            "radius": args.radius,
            "reward_config": reward_kwargs,
            "runs": rows,
        },
        output_root / "summary.json",
    )
    write_markdown(rows, output_root / "summary.md")

    print("\nTrajectory suite summary")
    for row in rows:
        metrics = row["trajectory_metrics"]
        print(
            f"{row['task_label']:20s} {row['algorithm']:14s} "
            f"mean={metrics['mean_return']:8.1f} "
            f"wp={metrics['mean_waypoints_completed']:.1f}/{metrics['waypoint_count']:.0f} "
            f"route={metrics['route_completion_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
