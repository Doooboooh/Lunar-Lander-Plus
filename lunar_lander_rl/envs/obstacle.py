from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lunar_lander_rl.envs.common import (
    FPS,
    VIEWPORT_W,
    close_window,
    draw_circle,
    draw_game_over,
    make_base_env,
    present_frame,
    state_to_pixel,
)


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


DEFAULT_OBSTACLES = (
    Obstacle(-0.55, 0.45, 0.11),
    Obstacle(0.20, 0.72, 0.12),
    Obstacle(0.62, 0.34, 0.10),
)


class ObstacleLunarLanderEnv(gym.Env):
    """LunarLander-v3 wrapper that appends obstacle coordinates and penalizes collisions."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        continuous: bool = False,
        obstacles: tuple[Obstacle, ...] = DEFAULT_OBSTACLES,
        obstacle_hit_radius: float = 0.08,
        collision_penalty: float = -100.0,
        **kwargs,
    ):
        self.env = make_base_env(render_mode=render_mode, continuous=continuous, **kwargs)
        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self.obstacles = tuple(obstacles)
        self.obstacle_hit_radius = float(obstacle_hit_radius)
        self.collision_penalty = float(collision_penalty)
        self.last_game_over = False
        self._screen = None
        self._clock = None

        base_low = self.env.observation_space.low.astype(np.float32)
        base_high = self.env.observation_space.high.astype(np.float32)
        obs_low = np.tile(np.array([-3.0, -3.0, 0.0], dtype=np.float32), len(self.obstacles))
        obs_high = np.tile(np.array([3.0, 3.0, 1.0], dtype=np.float32), len(self.obstacles))
        self.observation_space = spaces.Box(
            np.concatenate([base_low, obs_low]),
            np.concatenate([base_high, obs_high]),
            dtype=np.float32,
        )

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_game_over = False
        return self._augment_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        hit_obstacle = self._hit_obstacle(obs)
        if hit_obstacle:
            terminated = True
            reward = self.collision_penalty

        info = {**info, "hit_obstacle": hit_obstacle}
        self.last_game_over = bool(terminated or truncated)
        if self.render_mode == "human":
            self.render()
        return self._augment_observation(obs), float(reward), terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
        frame = np.array(frame, copy=True)
        self._draw_overlays(frame)
        if self.last_game_over:
            draw_game_over(frame)
        if self.render_mode == "rgb_array":
            return frame
        present_frame(self, frame)
        return None

    def close(self):
        self.env.close()
        close_window(self)

    def _augment_observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        obstacle_obs = []
        for obstacle in self.obstacles:
            obstacle_obs.extend([obstacle.x - obs[0], obstacle.y - obs[1], obstacle.radius])
        return np.concatenate([obs, np.array(obstacle_obs, dtype=np.float32)]).astype(np.float32)

    def _hit_obstacle(self, obs):
        pos = np.asarray(obs[:2], dtype=np.float32)
        for obstacle in self.obstacles:
            center = np.array([obstacle.x, obstacle.y], dtype=np.float32)
            if np.linalg.norm(pos - center) <= obstacle.radius + self.obstacle_hit_radius:
                return True
        return False

    def _draw_overlays(self, frame):
        for obstacle in self.obstacles:
            cx, cy = state_to_pixel(self.env, obstacle.x, obstacle.y)
            radius = max(3, int(obstacle.radius * VIEWPORT_W / 2))
            draw_circle(frame, cx, cy, radius, (215, 48, 39))
            draw_circle(frame, cx, cy, max(1, radius - 4), (245, 142, 132))
