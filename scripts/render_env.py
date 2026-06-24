import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunar_lander_rl.envs import register_custom_lunar_envs


ENV_IDS = [
    "BaseLunarLander-v0",
    "MovingPadLunarLander-v0",
    "ObstacleLunarLander-v0",
    "WaypointLunarLander-v0",
]


def choose_action(env, obs, policy):
    if policy == "random":
        return env.action_space.sample()

    base_env = env.unwrapped.env.unwrapped
    return heuristic(base_env, np.asarray(obs[:8], dtype=np.float32))


def render_episode(env_id, output, steps, seed, policy):
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    frames = []
    total_reward = 0.0
    for _ in range(steps):
        frames.append(env.render())
        action = choose_action(env, obs, policy)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            frames.append(env.render())
            break

    env.close()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=30)
    print(f"{env_id}: episode return {total_reward:.1f}")
    print(f"saved GIF to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="BaseLunarLander-v0",
        choices=[*ENV_IDS, "all"],
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    args = parser.parse_args()

    register_custom_lunar_envs()

    if args.env == "all":
        output_dir = Path(args.output) if args.output else ROOT / "outputs" / "render_all"
        for index, env_id in enumerate(ENV_IDS):
            output = output_dir / f"{env_id}.gif"
            render_episode(env_id, output, args.steps, args.seed + index, args.policy)
        return

    output = args.output or str(ROOT / "outputs" / f"{args.env}.gif")
    render_episode(args.env, output, args.steps, args.seed, args.policy)


if __name__ == "__main__":
    main()
