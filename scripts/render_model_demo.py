"""Roll out a trained policy to a GIF, preferring a layout where an obstacle
sits just above the landing pad (the interesting avoidance case).

Unlike ``render_env.py`` (heuristic/random policy), this drives a trained SB3
model. It scans reset seeds for one whose random-obstacle layout has an
obstacle in the column above the pad (``|x| < x_tol`` and ``y < y_max``), then
renders that episode with the policy.
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from lunar_lander_rl.envs import register_custom_lunar_envs  # noqa: E402
from lunar_lander_rl.experiments import ALGORITHMS, make_vec_env  # noqa: E402


def find_above_pad_seed(env_id, env_kwargs, seed_lo, seed_hi, x_tol, y_max):
    """Return the first seed whose layout has an obstacle above the pad."""
    env = gym.make(env_id, **env_kwargs)
    for seed in range(seed_lo, seed_hi):
        env.reset(seed=seed)
        for o in env.unwrapped.obstacles:
            if abs(o.x) < x_tol and o.y < y_max:
                env.close()
                return seed, (float(o.x), float(o.y), float(o.radius))
    env.close()
    return None, None


def underlying_env(venv):
    """Walk the vec-env wrapper chain down to the raw gym env."""
    node = venv
    while not isinstance(node, DummyVecEnv):
        node = node.venv
    return node.envs[0].unwrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=list(ALGORITHMS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--vec-normalize", default=None)
    parser.add_argument("--env", default="ObstacleLunarLander-v0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-kwargs", default='{"random_obstacles": true}')
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seed-lo", type=int, default=0)
    parser.add_argument("--seed-hi", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None,
                        help="pin a specific reset seed; skips the above-pad layout search")
    parser.add_argument("--x-tol", type=float, default=0.3, help="obstacle |x| band to count as 'above pad'")
    parser.add_argument("--y-max", type=float, default=0.6, help="obstacle y ceiling to count as 'above pad'")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-png", default=None, help="optional path to also dump the first frame as PNG")
    args = parser.parse_args()

    env_kwargs = json.loads(args.env_kwargs)
    register_custom_lunar_envs()

    if args.seed is not None:
        seed = args.seed
        print(f"using pinned seed {seed}")
    else:
        seed, ob = find_above_pad_seed(args.env, env_kwargs, args.seed_lo, args.seed_hi, args.x_tol, args.y_max)
        if seed is None:
            print(f"no above-pad layout in [{args.seed_lo},{args.seed_hi}); falling back to seed {args.seed_lo}")
            seed = args.seed_lo
        else:
            print(f"picked seed {seed}: obstacle above pad at x={ob[0]:+.2f} y={ob[1]:+.2f} r={ob[2]:.2f}")

    venv = make_vec_env(args.env, seed=seed, n_envs=1, render_mode="rgb_array", env_kwargs=env_kwargs)
    if args.vec_normalize:
        venv = VecNormalize.load(args.vec_normalize, venv)
        venv.training = False
        venv.norm_reward = False
    model = ALGORITHMS[args.algo].load(args.model)

    # Pin the obstacle RNG to the chosen seed so the rendered layout matches the
    # one we selected (reset otherwise uses the env's advancing persistent RNG).
    raw = underlying_env(venv)
    raw._rng = np.random.default_rng(seed)
    obs = venv.reset()

    frames = [raw.render()]
    total = 0.0
    for _ in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        total += float(rewards[0])
        frames.append(raw.render())
        if dones[0]:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)
    if args.frame_png:
        imageio.imwrite(args.frame_png, frames[0])
    hit = bool(infos[0].get("hit_obstacle"))
    print(f"{args.algo}: seed {seed}, {len(frames)} frames, return {total:+.1f}, hit_obstacle={hit}")
    print(f"saved GIF to {out}")
    venv.close()


if __name__ == "__main__":
    main()
