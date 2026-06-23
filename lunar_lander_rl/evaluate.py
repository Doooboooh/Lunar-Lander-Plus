from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .actor_critic import ActorCriticNet
from .common import evaluate_policy, get_device, import_gym, mlp
from .ppo import ActorCriticNet as PPONet
from .ppo import load_obs_normalizer
from .q_learning import discretize


ENV_ID = "LunarLander-v3"


def env_dims() -> tuple[int, int]:
    gym = import_gym()
    env = gym.make(ENV_ID)
    try:
        return env.observation_space.shape[0], env.action_space.n
    finally:
        env.close()


def load_saved_config(model_dir: Path) -> dict:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("config", {})


def load_policy(args: argparse.Namespace):
    obs_dim, action_dim = env_dims()
    model_dir = Path(args.model_dir)
    saved_config = load_saved_config(model_dir)

    if args.algorithm == "q_learning":
        with (model_dir / "q_table.pkl").open("rb") as f:
            q_table = pickle.load(f)
        return lambda obs: int(np.argmax(q_table[discretize(obs)]))

    device = get_device(args.device)
    if args.algorithm == "dqn":
        model = mlp(obs_dim, action_dim, args.hidden_dim).to(device)
    elif args.algorithm == "ppo":
        hidden_dim = int(saved_config.get("hidden_dim", args.hidden_dim))
        hidden_layers = int(saved_config.get("hidden_layers", 2))
        activation = str(saved_config.get("activation", "tanh"))
        model = PPONet(obs_dim, action_dim, hidden_dim, hidden_layers, activation).to(device)
    else:
        model = ActorCriticNet(obs_dim, action_dim, args.hidden_dim).to(device)

    checkpoint = model_dir / "best_policy.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    obs_rms = load_obs_normalizer(model_dir / "obs_norm.npz") if args.algorithm == "ppo" else None

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            model_obs = obs_rms.normalize(obs) if obs_rms is not None else obs
            obs_tensor = torch.tensor(model_obs, dtype=torch.float32, device=device).unsqueeze(0)
            if args.algorithm == "dqn":
                logits = model(obs_tensor)
            else:
                logits, _ = model(obs_tensor)
            return int(torch.argmax(logits, dim=1).item())

    return act


def save_policy_gif(
    policy,
    gif_path: str | Path,
    seed: int,
    max_steps: int = 1000,
    fps: int = 30,
) -> float:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("未找到 imageio，请先运行：pip install -r requirements.txt") from exc

    gym = import_gym()
    env = gym.make(ENV_ID, render_mode="rgb_array")
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    total_reward = 0.0

    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(max_steps):
            frames.append(env.render())
            action = int(policy(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                frames.append(env.render())
                break
    finally:
        env.close()

    imageio.mimsave(gif_path, frames, fps=fps)
    return total_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or render a trained LunarLander-v3 policy.")
    parser.add_argument("algorithm", choices=["q_learning", "dqn", "ppo", "actor_critic"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gif", help="Save one evaluation episode as a GIF, for example outputs/dqn_lunar_lander.gif")
    parser.add_argument("--gif-fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_policy(args)
    metrics = evaluate_policy(policy, episodes=args.episodes, seed=args.seed, render=args.render)
    print(
        f"{args.algorithm}: mean={metrics['mean_return']:.1f}, "
        f"std={metrics['std_return']:.1f}, min={metrics['min_return']:.1f}, max={metrics['max_return']:.1f}"
    )
    if args.gif:
        gif_return = save_policy_gif(policy, args.gif, seed=args.seed, fps=args.gif_fps)
        print(f"saved GIF to {args.gif} with episode return {gif_return:.1f}")


if __name__ == "__main__":
    main()
