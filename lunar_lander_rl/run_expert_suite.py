from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import ensure_dir, evaluate_policy, save_json
from .expert_policy import ExpertConfig, RuleBasedLanderPolicy
from .trajectory_env import ROUTES, load_waypoints, parse_waypoint_text, resolve_waypoints
from .trajectory_eval import evaluate_waypoint_policy


DEFAULT_ROUTE_TASKS = ("two_waypoint", "orbit", "figure_eight")


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_overrides(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"override must look like name=value, got {item!r}")
        name, value = item.split("=", maxsplit=1)
        name = name.strip()
        if not hasattr(ExpertConfig, name):
            raise argparse.ArgumentTypeError(f"unknown ExpertConfig field: {name}")
        overrides[name] = float(value)
    return overrides


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Rule Expert Evaluation",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- seed: `{payload['seed']}`",
        f"- episodes: `{payload['episodes']}`",
        f"- radius: `{payload['radius']}`",
        f"- waypoint hit mode: `{payload['waypoint_hit_mode']}`",
        "",
        "## Base Landing",
        "",
        "| mean return | std | min | max |",
        "|---:|---:|---:|---:|",
    ]
    base = payload["base_landing"]
    lines.append(
        f"| {base['mean_return']:.2f} | {base['std_return']:.2f} | "
        f"{base['min_return']:.2f} | {base['max_return']:.2f} |"
    )
    lines.extend(
        [
            "",
            "## Waypoint Routes",
            "",
            "| task | waypoints | mean return | route completion | both-leg land | touchdown | settled | final target dist | episode length |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["trajectory_runs"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['task_label']} | {metrics['mean_waypoints_completed']:.2f}/{metrics['waypoint_count']:.0f} | "
            f"{metrics['mean_return']:.2f} | {metrics['route_completion_rate']:.2f} | "
            f"{metrics['landed_after_route_rate']:.2f} | {metrics['touchdown_after_route_rate']:.2f} | "
            f"{metrics['settled_after_route_rate']:.2f} | {metrics['mean_final_target_distance']:.3f} | "
            f"{metrics['mean_episode_length']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_iteration_record(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        "timestamp": payload["timestamp"],
        "label": payload["label"],
        "seed": payload["seed"],
        "episodes": payload["episodes"],
        "radius": payload["radius"],
        "waypoint_hit_mode": payload["waypoint_hit_mode"],
        "config": payload["expert_config"],
        "base_landing": payload["base_landing"],
        "trajectory_runs": [
            {
                "task_label": row["task_label"],
                "waypoints": row["waypoints"],
                "metrics": row["metrics"],
            }
            for row in payload["trajectory_runs"]
        ],
        "notes": payload["notes"],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(compact, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the hand-written expert policy on LunarLander route tasks.")
    parser.add_argument("--label", default="manual_rules", help="Run label stored in the structured outputs.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_ROUTE_TASKS), help=f"Comma-separated route names from {sorted(ROUTES)}.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--radius", type=float, default=0.16)
    parser.add_argument("--waypoint-hit-mode", choices=["center", "body"], default="body")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--waypoints", help='Optional custom route as JSON or "x,y;x,y".')
    parser.add_argument("--waypoints-file", help="Optional JSON/text file containing one custom waypoint path.")
    parser.add_argument("--custom-label", help="Label for the optional custom route.")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="Override an ExpertConfig float field.")
    parser.add_argument("--notes", default="", help="Short note describing this iteration.")
    parser.add_argument("--output-dir", default="outputs/expert_rules/latest")
    parser.add_argument("--history", default="outputs/expert_rules/iteration_log.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.waypoints and args.waypoints_file:
        raise SystemExit("Use only one of --waypoints or --waypoints-file.")

    overrides = parse_overrides(args.set)
    config = ExpertConfig(**overrides)
    base_policy = RuleBasedLanderPolicy(config)

    trajectory_specs: list[tuple[str, str, tuple[tuple[float, float], ...] | None]] = []
    for task in parse_list(args.tasks):
        resolve_waypoints(task)
        trajectory_specs.append((task, task, None))

    if args.waypoints_file:
        custom = load_waypoints(args.waypoints_file)
        trajectory_specs.append((args.custom_label or f"custom_{len(custom)}", "custom", custom))
    elif args.waypoints:
        custom = parse_waypoint_text(args.waypoints)
        trajectory_specs.append((args.custom_label or f"custom_{len(custom)}", "custom", custom))

    base_landing = evaluate_policy(base_policy, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
    trajectory_runs = []
    for task_label, route_name, custom_waypoints in trajectory_specs:
        waypoints = resolve_waypoints(route_name, custom_waypoints)
        route_policy = RuleBasedLanderPolicy(config, waypoints=waypoints)
        metrics = evaluate_waypoint_policy(
            route_policy,
            route_name=route_name,
            custom_waypoints=custom_waypoints,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            radius=args.radius,
            waypoint_hit_mode=args.waypoint_hit_mode,
        )
        trajectory_runs.append(
            {
                "task_label": task_label,
                "route_name": route_name,
                "custom_waypoints": custom_waypoints,
                "waypoints": waypoints,
                "metrics": metrics,
            }
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "seed": args.seed,
        "episodes": args.episodes,
        "radius": args.radius,
        "waypoint_hit_mode": args.waypoint_hit_mode,
        "max_steps": args.max_steps,
        "expert_config": config.to_dict(),
        "base_landing": base_landing,
        "trajectory_runs": trajectory_runs,
        "notes": args.notes,
    }

    output_dir = ensure_dir(args.output_dir)
    save_json(payload, output_dir / "summary.json")
    write_markdown(payload, output_dir / "summary.md")
    append_iteration_record(payload, Path(args.history))

    print("Rule expert summary")
    print(
        f"base_landing mean={base_landing['mean_return']:.1f} "
        f"std={base_landing['std_return']:.1f} min={base_landing['min_return']:.1f}"
    )
    for row in trajectory_runs:
        metrics = row["metrics"]
        print(
            f"{row['task_label']:16s} mean={metrics['mean_return']:8.1f} "
            f"wp={metrics['mean_waypoints_completed']:.1f}/{metrics['waypoint_count']:.0f} "
            f"route={metrics['route_completion_rate']:.2f} "
            f"land={metrics['landed_after_route_rate']:.2f} "
            f"touch={metrics['touchdown_after_route_rate']:.2f} "
            f"settled={metrics['settled_after_route_rate']:.2f} "
            f"dist={metrics['mean_final_target_distance']:.3f}"
        )


if __name__ == "__main__":
    main()
