"""加载 best 模型 + VecNormalize 统计，渲染几个 episode 为 GIF。

注意：训练时 obs 被 VecNormalize 归一化过，推理时必须用相同统计归一化。
"""
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "best" / "best_model.zip"
VN = HERE / "models" / "vecnormalize.pkl"
OUT = HERE / "lunar_lander.gif"

N_EPISODES = 5
FPS = 30
CLIP_OBS = 10.0


def main() -> None:
    vn = VecNormalize.load(str(VN), DummyVecEnv([lambda: gym.make("LunarLander-v3")]))
    mean = vn.obs_rms.mean
    std = np.sqrt(vn.obs_rms.var + 1e-8)

    def normalize(obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - mean) / std, -CLIP_OBS, CLIP_OBS).astype(np.float32)

    model = PPO.load(MODEL, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    frames: list[np.ndarray] = []
    rewards: list[float] = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=1234 + ep)
        frames.append(env.render())
        done = False
        ep_r = 0.0
        while not done:
            action, _ = model.predict(normalize(obs), deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action.item()))
            frames.append(env.render())
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
        print(f"episode {ep + 1}/{N_EPISODES}  reward={ep_r:+.2f}  frames={len(frames)}")

    env.close()
    imageio.mimsave(str(OUT), frames, fps=FPS, subrectangles=True)
    size_kb = OUT.stat().st_size / 1024
    print(f"\nsaved: {OUT}")
    print(f"frames: {len(frames)}  size: {size_kb:.1f} KB  fps: {FPS}")
    print(f"rewards: {[f'{r:+.1f}' for r in rewards]}")
    print(f"mean reward: {np.mean(rewards):+.2f}")


if __name__ == "__main__":
    main()
