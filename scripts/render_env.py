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
from lunar_lander_rl.experiments import load_model, make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


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


def render_model_episode(
    env_id,
    output,
    steps,
    seed,
    algorithm,
    model_path,
    vec_normalize_path,
    deterministic,
):
    model = load_model(algorithm, model_path)
    env = make_vec_env(env_id, seed=seed, render_mode="rgb_array")
    if vec_normalize_path is not None:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    obs = env.reset()
    frames = []
    total_reward = 0.0
    try:
        for _ in range(steps):
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            total_reward += float(reward[0])
            if bool(done[0]):
                frames.append(env.render())
                break
    finally:
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
    parser.add_argument("--policy", choices=["heuristic", "random", "model"], default="heuristic")
    parser.add_argument("--algorithm", choices=["a2c", "dqn", "ppo"], default="ppo")
    parser.add_argument("--model", default=None, help="Path to a trained SB3 model zip.")
    parser.add_argument("--vec-normalize", default=None, help="Path to vec_normalize.pkl, if the model was trained with VecNormalize.")
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy instead of using deterministic actions.")
    args = parser.parse_args()

    register_custom_lunar_envs()

    if args.policy == "model":
        if args.env == "all":
            raise SystemExit("--policy model requires one concrete --env, not --env all")
        if args.model is None:
            raise SystemExit("--policy model requires --model")
        output = args.output or str(ROOT / "outputs" / f"{args.env}_model.gif")
        render_model_episode(
            args.env,
            output,
            args.steps,
            args.seed,
            args.algorithm,
            args.model,
            args.vec_normalize,
            deterministic=not args.stochastic,
        )
        return

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
