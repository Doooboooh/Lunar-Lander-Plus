import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lunar_lander_rl.envs.common import (
    FPS,
    VIEWPORT_W,
    close_window,
    draw_circle,
    draw_circle_outline,
    draw_game_over,
    make_base_env,
    present_frame,
    state_to_pixel,
)


DEFAULT_WAYPOINTS = ((-0.45, 0.85), (0.35, 0.58), (0.05, 0.30))


class WaypointLunarLanderEnv(gym.Env):
    """LunarLander-v3 wrapper that requires visiting ordered waypoints before landing."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        continuous: bool = False,
        waypoints: tuple[tuple[float, float], ...] = DEFAULT_WAYPOINTS,
        waypoint_radius: float = 0.12,
        waypoint_bonus: float = 35.0,
        early_landing_penalty: float = -100.0,
        **kwargs,
    ):
        self.env = make_base_env(render_mode=render_mode, continuous=continuous, **kwargs)
        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self.waypoints = tuple(waypoints)
        self.waypoint_radius = float(waypoint_radius)
        self.waypoint_bonus = float(waypoint_bonus)
        self.early_landing_penalty = float(early_landing_penalty)
        self.active_waypoint = 0
        self.last_game_over = False
        self._screen = None
        self._clock = None

        base_low = self.env.observation_space.low.astype(np.float32)
        base_high = self.env.observation_space.high.astype(np.float32)
        waypoint_low = np.array([-3.0, -3.0, 0.0], dtype=np.float32)
        waypoint_high = np.array([3.0, 3.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            np.concatenate([base_low, waypoint_low]),
            np.concatenate([base_high, waypoint_high]),
            dtype=np.float32,
        )

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.active_waypoint = 0
        self.last_game_over = False
        return self._augment_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        hit_waypoint = False
        if self.active_waypoint < len(self.waypoints):
            target = np.array(self.waypoints[self.active_waypoint], dtype=np.float32)
            if np.linalg.norm(obs[:2] - target) <= self.waypoint_radius:
                self.active_waypoint += 1
                hit_waypoint = True
                reward += self.waypoint_bonus

        if terminated and self.active_waypoint < len(self.waypoints):
            reward = self.early_landing_penalty

        info = {
            **info,
            "target": self._current_target(),
            "active_waypoint": self.active_waypoint,
            "hit_waypoint": hit_waypoint,
        }
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
        target = np.array(self._current_target(), dtype=np.float32)
        progress = self.active_waypoint / max(1, len(self.waypoints))
        waypoint_obs = np.array([target[0] - obs[0], target[1] - obs[1], progress], dtype=np.float32)
        return np.concatenate([obs, waypoint_obs]).astype(np.float32)

    def _current_target(self):
        if self.active_waypoint < len(self.waypoints):
            return self.waypoints[self.active_waypoint]
        return (0.0, 0.0)

    def _draw_overlays(self, frame):
        for idx, waypoint in enumerate(self.waypoints):
            cx, cy = state_to_pixel(self.env, waypoint[0], waypoint[1])
            color = (95, 95, 95)
            if idx == self.active_waypoint:
                color = (31, 143, 230)
            elif idx < self.active_waypoint:
                color = (56, 166, 95)
            draw_circle_outline(frame, cx, cy, int(self.waypoint_radius * VIEWPORT_W / 2), color)
            draw_circle(frame, cx, cy, 4, (255, 255, 255))
