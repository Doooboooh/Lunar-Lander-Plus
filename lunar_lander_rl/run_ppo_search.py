from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .common import ensure_dir, evaluate_policy, get_device, make_env, save_json
from .ppo import ActorCriticNet, PPOConfig, load_obs_normalizer, train as train_ppo
from .trajectory_env import ROUTES, make_waypoint_env, resolve_waypoints
from .trajectory_eval import evaluate_waypoint_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small PPO hyperparameter and architecture search.")
    parser.add_argument("--mode", choices=["base", "trajectory"], default="base")
    parser.add_argument("--task", default="two_waypoint", help=f"Trajectory task from {sorted(ROUTES)}.")
    parser.add_argument("--profile", choices=["tiny", "probe", "course"], default="probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--names", default=None, help="Comma-separated candidate names to run.")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--selection-eval-episodes", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--trajectory-eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/ppo_search")
    return parser.parse_args()


def base_config_for_profile(profile: str, seed: int, device: str, output_dir: Path) -> PPOConfig:
    if profile == "tiny":
        return PPOConfig(
            updates=4,
            rollout_steps=256,
            minibatch_size=64,
            eval_episodes=2,
            selection_eval_episodes=2,
            eval_interval=2,
            seed=seed,
            device=device,
            output_dir=str(output_dir),
        )
    if profile == "probe":
        return PPOConfig(
            updates=10,
            rollout_steps=512,
            minibatch_size=128,
            eval_episodes=3,
            selection_eval_episodes=2,
            eval_interval=5,
            seed=seed,
            device=device,
            output_dir=str(output_dir),
        )
    return PPOConfig(
        updates=40,
        rollout_steps=512,
        minibatch_size=128,
        eval_episodes=5,
        selection_eval_episodes=3,
        eval_interval=10,
        seed=seed,
        device=device,
        output_dir=str(output_dir),
    )


def candidate_overrides() -> list[dict]:
    return [
        {
            "name": "baseline_tanh_128",
            "hidden_dim": 128,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 3e-4,
            "entropy_coef": 0.01,
            "clip_coef": 0.2,
            "update_epochs": 4,
        },
        {
            "name": "relu_128_low_entropy",
            "hidden_dim": 128,
            "hidden_layers": 2,
            "activation": "relu",
            "lr": 3e-4,
            "entropy_coef": 0.003,
            "clip_coef": 0.2,
            "update_epochs": 4,
        },
        {
            "name": "tanh_256_low_lr",
            "hidden_dim": 256,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 1e-4,
            "entropy_coef": 0.005,
            "clip_coef": 0.2,
            "update_epochs": 4,
        },
        {
            "name": "elu_256",
            "hidden_dim": 256,
            "hidden_layers": 2,
            "activation": "elu",
            "lr": 3e-4,
            "entropy_coef": 0.01,
            "clip_coef": 0.2,
            "update_epochs": 4,
        },
        {
            "name": "deep_tanh_128",
            "hidden_dim": 128,
            "hidden_layers": 3,
            "activation": "tanh",
            "lr": 3e-4,
            "entropy_coef": 0.005,
            "clip_coef": 0.2,
            "update_epochs": 4,
        },
        {
            "name": "conservative_clip",
            "hidden_dim": 128,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 1e-4,
            "entropy_coef": 0.003,
            "clip_coef": 0.1,
            "update_epochs": 6,
        },
        {
            "name": "more_exploration",
            "hidden_dim": 128,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 3e-4,
            "entropy_coef": 0.03,
            "clip_coef": 0.2,
            "update_epochs": 4,
        },
        {
            "name": "longer_rollout",
            "hidden_dim": 128,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 3e-4,
            "entropy_coef": 0.005,
            "clip_coef": 0.2,
            "rollout_steps": 1024,
            "minibatch_size": 256,
            "update_epochs": 4,
        },
        {
            "name": "relu_256_low_lr",
            "hidden_dim": 256,
            "hidden_layers": 2,
            "activation": "relu",
            "lr": 1e-4,
            "entropy_coef": 0.003,
            "clip_coef": 0.2,
            "update_epochs": 6,
        },
        {
            "name": "relu_256_longer_rollout",
            "hidden_dim": 256,
            "hidden_layers": 2,
            "activation": "relu",
            "lr": 1e-4,
            "entropy_coef": 0.003,
            "clip_coef": 0.2,
            "rollout_steps": 1024,
            "minibatch_size": 256,
            "update_epochs": 6,
        },
        {
            "name": "deep_relu_256",
            "hidden_dim": 256,
            "hidden_layers": 3,
            "activation": "relu",
            "lr": 1e-4,
            "entropy_coef": 0.003,
            "clip_coef": 0.15,
            "update_epochs": 6,
        },
        {
            "name": "tanh_256_very_low_entropy",
            "hidden_dim": 256,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 1e-4,
            "entropy_coef": 0.001,
            "clip_coef": 0.2,
            "update_epochs": 6,
        },
        {
            "name": "tanh_256_low_value",
            "hidden_dim": 256,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 1e-4,
            "entropy_coef": 0.003,
            "clip_coef": 0.2,
            "update_epochs": 4,
            "value_coef": 0.25,
        },
        {
            "name": "tanh_256_low_value_fewer_epochs",
            "hidden_dim": 256,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 1e-4,
            "entropy_coef": 0.003,
            "clip_coef": 0.2,
            "update_epochs": 2,
            "value_coef": 0.25,
        },
        {
            "name": "tanh_256_clip01",
            "hidden_dim": 256,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 1e-4,
            "entropy_coef": 0.003,
            "clip_coef": 0.1,
            "update_epochs": 4,
            "value_coef": 0.25,
        },
        {
            "name": "blog_vecnorm_like",
            "hidden_dim": 16,
            "hidden_layers": 3,
            "activation": "tanh",
            "lr": 3e-4,
            "entropy_coef": 0.001,
            "clip_coef": 0.2,
            "update_epochs": 10,
            "rollout_steps": 2048,
            "minibatch_size": 64,
            "normalize_observations": True,
            "normalize_rewards": True,
        },
    ]


def build_policy(model_dir: Path, cfg: PPOConfig, env_factory, device_name: str):
    env = env_factory(seed=None)
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    finally:
        env.close()

    device = get_device(device_name)
    model = ActorCriticNet(obs_dim, action_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.activation).to(device)
    model.load_state_dict(torch.load(model_dir / "best_policy.pt", map_location=device))
    model.eval()
    obs_rms = load_obs_normalizer(model_dir / "obs_norm.npz")

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            model_obs = obs_rms.normalize(obs) if obs_rms is not None else obs
            obs_tensor = torch.tensor(model_obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(obs_tensor)
            return int(torch.argmax(logits, dim=1).item())

    return act


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "| rank | config | eval mean | eval std | route completion | waypoints | final dist | lr | hidden | layers | act | entropy | obs norm | rew norm |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        cfg = row["config"]
        traj = row.get("trajectory_metrics")
        route = "-" if traj is None else f"{traj['route_completion_rate']:.2f}"
        waypoints = "-" if traj is None else f"{traj['mean_waypoints_completed']:.2f}/{traj['waypoint_count']:.0f}"
        final_dist = "-" if traj is None else f"{traj['mean_final_target_distance']:.2f}"
        lines.append(
            f"| {rank} | {row['name']} | {row['metrics']['mean_return']:.2f} | {row['metrics']['std_return']:.2f} | "
            f"{route} | {waypoints} | {final_dist} | {cfg['lr']:.0e} | {cfg['hidden_dim']} | "
            f"{cfg['hidden_layers']} | {cfg['activation']} | {cfg['entropy_coef']:.3f} | "
            f"{'Y' if cfg.get('normalize_observations', False) else 'N'} | "
            f"{'Y' if cfg.get('normalize_rewards', False) else 'N'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_dir) / args.mode / args.profile)
    candidates = candidate_overrides()
    if args.names:
        selected_names = {name.strip() for name in args.names.split(",") if name.strip()}
        candidates = [candidate for candidate in candidates if candidate["name"] in selected_names]
        if not candidates:
            raise SystemExit("No PPO search candidates matched --names.")
    if args.max_configs is not None:
        candidates = candidates[: args.max_configs]

    if args.mode == "trajectory":
        resolve_waypoints(args.task)

        def env_factory(render: bool = False, seed: int | None = None):
            return make_waypoint_env(render=render, seed=seed, route_name=args.task)

    else:

        def env_factory(render: bool = False, seed: int | None = None):
            return make_env(render=render, seed=seed)

    rows: list[dict] = []
    for idx, overrides in enumerate(candidates, start=1):
        name = str(overrides["name"])
        run_dir = output_root / f"{idx:02d}_{name}"
        base_cfg = base_config_for_profile(args.profile, args.seed, args.device, run_dir)
        cfg_kwargs = {k: v for k, v in overrides.items() if k != "name"}
        cfg = replace(base_cfg, **cfg_kwargs)
        if args.updates is not None:
            cfg = replace(cfg, updates=args.updates)
        if args.rollout_steps is not None:
            cfg = replace(cfg, rollout_steps=args.rollout_steps)
        if args.selection_eval_episodes is not None:
            cfg = replace(cfg, selection_eval_episodes=args.selection_eval_episodes)
        if args.eval_interval is not None:
            cfg = replace(cfg, eval_interval=args.eval_interval)
        print(f"\n=== PPO search {idx}/{len(candidates)}: {name} ===")
        metrics = train_ppo(cfg, env_factory=env_factory)
        policy = build_policy(run_dir, cfg, env_factory, args.device)
        metrics = evaluate_policy(policy, args.eval_episodes, seed=args.seed + 50000, env_factory=env_factory)
        trajectory_metrics = None
        if args.mode == "trajectory":
            trajectory_metrics = evaluate_waypoint_policy(
                policy,
                route_name=args.task,
                episodes=args.trajectory_eval_episodes,
                seed=args.seed + 60000,
            )
            save_json(trajectory_metrics, run_dir / "trajectory_metrics.json")

        row = {
            "name": name,
            "rank_score": float(metrics["mean_return"]),
            "config": asdict(cfg),
            "metrics": metrics,
            "trajectory_metrics": trajectory_metrics,
            "output_dir": str(run_dir),
        }
        save_json(row, run_dir / "search_result.json")
        rows.append(row)

    if args.mode == "trajectory":
        rows.sort(
            key=lambda row: (
                row["trajectory_metrics"]["route_completion_rate"],
                row["trajectory_metrics"]["mean_waypoints_completed"],
                row["metrics"]["mean_return"],
            ),
            reverse=True,
        )
    else:
        rows.sort(key=lambda row: row["metrics"]["mean_return"], reverse=True)

    save_json({"mode": args.mode, "profile": args.profile, "task": args.task, "runs": rows}, output_root / "summary.json")
    write_markdown(rows, output_root / "summary.md")

    print("\nPPO search summary")
    for rank, row in enumerate(rows, start=1):
        extra = ""
        if row["trajectory_metrics"] is not None:
            t = row["trajectory_metrics"]
            extra = f" route={t['route_completion_rate']:.2f} wp={t['mean_waypoints_completed']:.2f}/{t['waypoint_count']:.0f}"
        print(f"{rank:02d}. {row['name']:22s} mean={row['metrics']['mean_return']:8.1f}{extra}")


if __name__ == "__main__":
    main()
