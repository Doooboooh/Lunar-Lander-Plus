from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


ENV_ID = "LunarLander-v3"


def import_gym() -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit("未找到 gymnasium，请先在 proj 目录运行：pip install -r requirements.txt") from exc
    return gym


def make_env(render: bool = False, seed: int | None = None) -> Any:
    gym = import_gym()
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def evaluate_policy(
    policy_fn: Callable[[np.ndarray], int],
    episodes: int = 5,
    max_steps: int = 1000,
    seed: int = 1234,
    render: bool = False,
) -> dict[str, float]:
    env = make_env(render=render, seed=seed)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            for _ in range(max_steps):
                action = int(policy_fn(obs))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            returns.append(total_reward)
    finally:
        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
    }


def save_history(history: list[dict[str, float]], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    if not history:
        return

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    def convert(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        return value

    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)

