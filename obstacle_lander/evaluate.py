"""Evaluate a trained policy on the obstacle LunarLander env + save a GIF.

Reports return stats and collision_rate. Reuses the saved checkpoint from training.
GIF overlays obstacle circles on each frame (Gymnasium's default renderer
does not draw them) and picks the highest-return episode from a sample so the
demo shows a meaningful trajectory instead of an early crash.

Example:
    python -m obstacle_lander.evaluate --algo dqn \
        --model-dir obstacle_lander/outputs/dqn \
        --episodes 10 --gif obstacle_lander/outputs/dqn_obstacle.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from lunar_lander_rl.actor_critic import ActorCriticNet
from lunar_lander_rl.common import get_device, mlp
from lunar_lander_rl.ppo import ActorCriticNet as PPONet

from .env import N_OBSTACLES_DEFAULT, ObstacleLanderEnv


VIEWPORT_W = 600
VIEWPORT_H = 400
HELIPAD_Y = VIEWPORT_H / 4 / 1  # placeholder, replaced below
# From LunarLander source: H = VIEWPORT_H/SCALE, helipad_y = H/4
# pixel_y(world_y) = VIEWPORT_H - world_y*SCALE
# world_y(obs_y)  = obs_y * (VIEWPORT_H/2) / SCALE + helipad_y  → multiplied by SCALE → obs_y*VIEWPORT_H/2 + helipad_y*SCALE
# helipad_y*SCALE = VIEWPORT_H/4 = 100. So pixel_y(obs_y) = VIEWPORT_H - obs_y*VIEWPORT_H/2 - VIEWPORT_H/4 = 3*VIEWPORT_H/4 - obs_y*VIEWPORT_H/2
PIXEL_X_SCALE = VIEWPORT_W / 2   # obs_x range [-1, 1] → pixel_x range [0, VIEWPORT_W]
PIXEL_X_OFFSET = VIEWPORT_W / 2
PIXEL_Y_SCALE = VIEWPORT_H / 2   # 1 obs_y unit = VIEWPORT_H/2 pixels
PIXEL_Y_OFFSET = 3 * VIEWPORT_H / 4  # obs_y=0 → pixel_y = 3/4 * VIEWPORT_H (helipad line)


def obs_to_pixel(obs_x: float, obs_y: float) -> tuple[float, float]:
    px = obs_x * PIXEL_X_SCALE + PIXEL_X_OFFSET
    py = PIXEL_Y_OFFSET - obs_y * PIXEL_Y_SCALE
    return px, py


def _make_env(args: argparse.Namespace, seed: int, render: bool = False) -> ObstacleLanderEnv:
    return ObstacleLanderEnv(
        n_obstacles=args.n_obstacles,
        radius=args.radius,
        random_obstacles=args.random_obstacles,
        render_mode="rgb_array" if render else None,
        seed=seed,
    )


def _obs_dim(env: ObstacleLanderEnv) -> int:
    return int(env.observation_space.shape[0])


def _action_dim(env: ObstacleLanderEnv) -> int:
    return int(env.action_space.n)


def _resolve_hidden_dim(args: argparse.Namespace) -> int:
    """Auto-detect hidden_dim from metrics.json if user did not pass --hidden-dim."""
    if args.hidden_dim is not None:
        return args.hidden_dim
    metrics_path = Path(args.model_dir) / "metrics.json"
    if metrics_path.exists():
        import json

        with metrics_path.open() as f:
            data = json.load(f)
        cfg = data.get("config", {})
        if "hidden_dim" in cfg:
            return int(cfg["hidden_dim"])
    return 128


def build_policy(args: argparse.Namespace):
    env = _make_env(args, args.seed)
    obs_dim = _obs_dim(env)
    action_dim = _action_dim(env)
    hidden_dim = _resolve_hidden_dim(args)
    env.close()
    device = get_device(args.device)

    if args.algo == "dqn":
        model = mlp(obs_dim, action_dim, hidden_dim).to(device)
        ckpt = Path(args.model_dir) / "best_policy.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        def act(obs: np.ndarray) -> int:
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                return int(model(t).argmax(dim=1).item())

        return act

    if args.algo == "ppo":
        net_cls = PPONet
    else:
        net_cls = ActorCriticNet

    model = net_cls(obs_dim, action_dim, hidden_dim).to(device)
    ckpt = Path(args.model_dir) / "best_policy.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(t)
            return int(torch.argmax(logits, dim=1).item())

    return act


def evaluate(policy, args: argparse.Namespace) -> dict[str, float]:
    returns: list[float] = []
    collisions = 0
    total_episodes = 0

    for ep in range(args.episodes):
        env = _make_env(args, args.seed + ep)
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        collided = False
        try:
            for _ in range(args.max_steps):
                action = int(policy(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if info.get("collision"):
                    collided = True
                if terminated or truncated:
                    break
        finally:
            env.close()
        returns.append(total_reward)
        if collided:
            collisions += 1
        total_episodes += 1

    arr = np.asarray(returns, dtype=np.float64) if returns else np.zeros(0)
    return {
        "episodes": float(total_episodes),
        "mean_return": float(arr.mean()) if arr.size else 0.0,
        "std_return": float(arr.std()) if arr.size else 0.0,
        "min_return": float(arr.min()) if arr.size else 0.0,
        "max_return": float(arr.max()) if arr.size else 0.0,
        "collision_rate": float(collisions / total_episodes) if total_episodes else 0.0,
    }


def _overlay_obstacles(frame: np.ndarray, obstacles, radius: float) -> np.ndarray:
    """Draw each obstacle on a copy of the frame: red core, yellow warning ring."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for cx, cy, r in obstacles:
        r_eff = r if r > 0 else radius
        # obs-space circle is an ellipse in pixel space because x/y have different scales.
        warn_r = r_eff * 2.0
        cx_px_core = cx * PIXEL_X_SCALE + PIXEL_X_OFFSET
        cy_px_core = PIXEL_Y_OFFSET - cy * PIXEL_Y_SCALE
        # warning ring (2x radius)
        x0 = cx_px_core - warn_r * PIXEL_X_SCALE
        x1 = cx_px_core + warn_r * PIXEL_X_SCALE
        y0 = cy_px_core - warn_r * PIXEL_Y_SCALE
        y1 = cy_px_core + warn_r * PIXEL_Y_SCALE
        draw.ellipse((x0, y0, x1, y1), outline=(255, 200, 0), width=2)
        # collision core (1x radius), filled red
        x0c = cx_px_core - r_eff * PIXEL_X_SCALE
        x1c = cx_px_core + r_eff * PIXEL_X_SCALE
        y0c = cy_px_core - r_eff * PIXEL_Y_SCALE
        y1c = cy_px_core + r_eff * PIXEL_Y_SCALE
        draw.ellipse((x0c, y0c, x1c, y1c), fill=(220, 40, 40), outline=(160, 0, 0))
    return np.array(img)


def _rollout_with_render(policy, args: argparse.Namespace, seed: int) -> tuple[list, float, bool]:
    env = _make_env(args, seed, render=True)
    frames: list = []
    total_reward = 0.0
    collided = False
    obstacles = []
    try:
        obs, _ = env.reset(seed=seed)
        obstacles = list(env.obstacles)
        for _ in range(args.max_steps):
            frame = env.render()
            frames.append(_overlay_obstacles(frame, obstacles, args.radius))
            action = int(policy(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if info.get("collision"):
                collided = True
            if terminated or truncated:
                frames.append(_overlay_obstacles(env.render(), obstacles, args.radius))
                break
    finally:
        env.close()
    return frames, total_reward, collided


def save_gif(policy, args: argparse.Namespace) -> float:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("未找到 imageio，请先运行：pip install -r requirements.txt") from exc

    # Run several candidates and keep the best (highest return) so the GIF
    # demonstrates a meaningful trajectory instead of an unlucky early crash.
    candidate_seeds = [args.seed + i for i in range(args.gif_candidates)]
    best_frames: list | None = None
    best_return = float("-inf")
    best_seed = candidate_seeds[0]
    for sd in candidate_seeds:
        frames, total_reward, _collided = _rollout_with_render(policy, args, sd)
        if total_reward > best_return:
            best_return = total_reward
            best_frames = frames
            best_seed = sd

    assert best_frames is not None
    gif_path = Path(args.gif)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, best_frames, fps=args.gif_fps)
    print(f"GIF seed={best_seed} return={best_return:.1f} (picked from {len(candidate_seeds)} candidates)")
    return best_return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on the obstacle env.")
    parser.add_argument("--algo", required=True, choices=["dqn", "ppo", "actor_critic"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Network hidden dim. Auto-detected from metrics.json if omitted.")
    parser.add_argument("--n-obstacles", type=int, default=N_OBSTACLES_DEFAULT)
    parser.add_argument("--radius", type=float, default=0.12)
    parser.add_argument("--random-obstacles", action="store_true")
    parser.add_argument("--gif", help="Save one evaluation episode as a GIF")
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--gif-candidates", type=int, default=10,
                        help="Number of episodes to try; pick the highest-return one for the GIF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = build_policy(args)
    metrics = evaluate(policy, args)
    print(
        f"[{args.algo}] episodes={int(metrics['episodes'])} "
        f"mean={metrics['mean_return']:.1f} std={metrics['std_return']:.1f} "
        f"min={metrics['min_return']:.1f} max={metrics['max_return']:.1f} "
        f"collision_rate={metrics['collision_rate']:.2%}"
    )
    if args.gif:
        gif_return = save_gif(policy, args)
        print(f"saved GIF to {args.gif}")


if __name__ == "__main__":
    main()
