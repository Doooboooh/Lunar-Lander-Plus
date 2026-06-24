from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .common import mlp, register_moving_pad
from .ppo import ActorCriticNet


def load_dqn(path: Path, device: torch.device):
    net = mlp(10, 4, 128).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return int(net(t).argmax(1).item())

    return act


def load_ppo(path: Path, device: torch.device):
    net = ActorCriticNet(10, 4, 128).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(t)
            return int(logits.argmax(1).item())

    return act


def evaluate_policy(name: str, policy_fn, seeds: list[int], env_id: str) -> list[dict]:
    rows = []
    for seed in seeds:
        env = gym.make(env_id)
        obs, _ = env.reset(seed=seed)
        total = 0.0
        terminated = False
        truncated = False
        steps = 0
        for steps in range(1, 1001):
            obs, reward, terminated, truncated, _ = env.step(policy_fn(obs))
            total += float(reward)
            if terminated or truncated:
                break

        u = env.unwrapped
        pos = u.lander.position
        vel = u.lander.linearVelocity
        pad_x, _ = u._pad_center_world()
        speed = math.hypot(vel.x, vel.y)
        both_legs = bool(u.legs[0].ground_contact and u.legs[1].ground_contact)
        on_deck = bool(u._on_deck())
        upright = abs(u.lander.angle) < 0.15

        rows.append(
            {
                "policy": name,
                "seed": seed,
                "return": total,
                "steps": steps,
                "terminated": terminated,
                "truncated": truncated,
                "on_deck": on_deck,
                "both_legs": both_legs,
                "upright": upright,
                "stable_success": bool(on_deck and both_legs and upright and speed < 1.0),
                "rel_x_m": abs(pos.x - pad_x),
                "speed_mps": speed,
                "angle_rad": float(u.lander.angle),
            }
        )
        env.close()
    return rows


def summarize(rows: list[dict]) -> dict:
    returns = np.array([r["return"] for r in rows], dtype=float)
    return {
        "episodes": len(rows),
        "seed_start": int(min(r["seed"] for r in rows)),
        "seed_end": int(max(r["seed"] for r in rows)),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "on_deck_rate": float(np.mean([r["on_deck"] for r in rows])),
        "stable_success_rate": float(np.mean([r["stable_success"] for r in rows])),
        "best_seed": int(rows[int(returns.argmax())]["seed"]),
        "best_return": float(returns.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DQN/PPO policies on MovingPadLunarLander-v0.")
    parser.add_argument("--dqn-model", default="outputs/moving_pad/moving_pad_dqn/best_policy.pt")
    parser.add_argument("--ppo-model", default="outputs/moving_pad/ppo_compare_seed42/best_policy.pt")
    parser.add_argument("--old-ppo-model", default="outputs/moving_pad/moving_pad_ppo/best_policy.pt")
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-json", default="outputs/moving_pad/ppo_comparison_eval.json")
    parser.add_argument("--out-csv", default="outputs/moving_pad/ppo_comparison_eval.csv")
    args = parser.parse_args()

    env_id = register_moving_pad()
    device = torch.device(args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    specs = [
        ("DQN_final", load_dqn(Path(args.dqn_model), device)),
        ("PPO_200u_seed42", load_ppo(Path(args.ppo_model), device)),
    ]
    if Path(args.old_ppo_model).exists():
        specs.append(("PPO_old_400u", load_ppo(Path(args.old_ppo_model), device)))

    all_rows = []
    summary = {}
    for name, policy_fn in specs:
        rows = evaluate_policy(name, policy_fn, seeds, env_id)
        all_rows.extend(rows)
        summary[name] = summarize(rows)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "rows": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
