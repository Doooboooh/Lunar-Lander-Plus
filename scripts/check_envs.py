import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import sys
from pathlib import Path

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


def main():
    register_custom_lunar_envs()

    for env_id in ENV_IDS:
        env = gym.make(env_id)
        check_env(env.unwrapped, warn=True, skip_render_check=True)

        obs, info = env.reset(seed=42)
        total_reward = 0.0
        for _ in range(32):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            assert env.observation_space.contains(obs), f"{env_id} produced invalid obs"
            if terminated or truncated:
                break

        print(
            f"{env_id}: OK | obs_shape={env.observation_space.shape} "
            f"action_space={env.action_space} rollout_return={total_reward:.2f}"
        )
        env.close()


if __name__ == "__main__":
    main()
