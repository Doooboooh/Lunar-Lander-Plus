from gymnasium.envs.registration import register, registry

from lunar_lander_rl.envs.base import BaseLunarLanderEnv
from lunar_lander_rl.envs.moving_pad import MovingPadLunarLanderEnv
from lunar_lander_rl.envs.obstacle import Obstacle, ObstacleLunarLanderEnv
from lunar_lander_rl.envs.waypoint import WaypointLunarLanderEnv


def register_custom_lunar_envs():
    env_specs = {
        "BaseLunarLander-v0": "lunar_lander_rl.envs.base:BaseLunarLanderEnv",
        "MovingPadLunarLander-v0": "lunar_lander_rl.envs.moving_pad:MovingPadLunarLanderEnv",
        "ObstacleLunarLander-v0": "lunar_lander_rl.envs.obstacle:ObstacleLunarLanderEnv",
        "WaypointLunarLander-v0": "lunar_lander_rl.envs.waypoint:WaypointLunarLanderEnv",
    }
    for env_id, entry_point in env_specs.items():
        if env_id not in registry:
            register(id=env_id, entry_point=entry_point, max_episode_steps=1000)


__all__ = [
    "BaseLunarLanderEnv",
    "MovingPadLunarLanderEnv",
    "Obstacle",
    "ObstacleLunarLanderEnv",
    "WaypointLunarLanderEnv",
    "register_custom_lunar_envs",
]
