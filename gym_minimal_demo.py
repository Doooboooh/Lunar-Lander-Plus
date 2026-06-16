"""Minimal demos for Pendulum-v1, LunarLander-v3, and Blackjack-v1.

Run examples:
    python gym_minimal_demo.py --env all --episodes 3
    python gym_minimal_demo.py --env LunarLander-v3 --render
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np


ENV_IDS = ("Pendulum-v1", "LunarLander-v3", "Blackjack-v1")


def import_gym() -> Any:
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError as exc:
            raise SystemExit(
                "未找到 gymnasium/gym，请先运行：pip install -r requirements.txt"
            ) from exc
    return gym


def reset_env(env: Any, seed: int | None) -> Any:
    result = env.reset(seed=seed)
    return result[0] if isinstance(result, tuple) else result


def step_env(env: Any, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), info

    obs, reward, done, info = result
    return obs, float(reward), bool(done), info


def choose_action(env_id: str, obs: Any, env: Any) -> Any:
    if env_id == "Blackjack-v1":
        player_sum = obs[0]
        return 0 if player_sum >= 20 else 1  # 0: stick, 1: hit

    if env_id == "Pendulum-v1":
        cos_theta, sin_theta, theta_dot = obs
        theta = math.atan2(sin_theta, cos_theta)
        torque = -2.0 * theta - 0.5 * theta_dot
        return np.array([np.clip(torque, -2.0, 2.0)], dtype=np.float32)

    if env_id == "LunarLander-v3":
        x, y, vx, vy, angle, angular_v, left_leg, right_leg = obs
        target_angle = np.clip(0.5 * x + 1.0 * vx, -0.4, 0.4)
        target_y = 0.55 * abs(x)
        angle_error = (target_angle - angle) * 0.5 - angular_v
        y_error = (target_y - y) * 0.5 - vy * 0.5

        if left_leg or right_leg:
            angle_error = 0.0
            y_error = -vy * 0.5

        if y_error > abs(angle_error) and y_error > 0.05:
            return 2  # main engine
        if angle_error < -0.05:
            return 3  # right engine
        if angle_error > 0.05:
            return 1  # left engine
        return 0

    return env.action_space.sample()


def make_env(gym: Any, env_id: str, render: bool) -> Any:
    render_mode = "human" if render else None
    try:
        return gym.make(env_id, render_mode=render_mode)
    except TypeError:
        return gym.make(env_id)


def run_one_env(env_id: str, episodes: int, max_steps: int, render: bool, seed: int) -> None:
    gym = import_gym()
    env = make_env(gym, env_id, render)
    try:
        for episode in range(1, episodes + 1):
            obs = reset_env(env, seed + episode)
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                action = choose_action(env_id, obs, env)
                obs, reward, done, _ = step_env(env, action)
                total_reward += reward
                if done:
                    break

            print(
                f"{env_id:14s} | episode {episode:02d} | "
                f"steps {step:4d} | reward {total_reward:8.2f}"
            )
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Gym/Gymnasium environment demo.")
    parser.add_argument("--env", choices=(*ENV_IDS, "all"), default="all")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_ids = ENV_IDS if args.env == "all" else (args.env,)

    for env_id in env_ids:
        run_one_env(env_id, args.episodes, args.max_steps, args.render, args.seed)


if __name__ == "__main__":
    main()
