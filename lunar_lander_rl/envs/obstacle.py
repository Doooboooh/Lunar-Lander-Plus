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

# Obstacle placement / observation config.
N_OBSTACLES_DEFAULT = len(DEFAULT_OBSTACLES)
OBSTACLE_RADIUS_DEFAULT = 0.12
# The lander spawns centered at the top (0, ~1.4) and the helipad (the "landing
# zone") sits at (0, 0), spanning obs-x [-0.2, 0.2]. With random_obstacles each
# obstacle is placed inside the landing zone's x-extent, at an altitude above
# the pad, so it blocks the direct descent (the lander must detour around it).
OBSTACLE_LANDING_X_RANGE = (-0.2, 0.2)  # helipad obs-x extent (the landing zone)
OBSTACLE_Y_RANGE = (0.4, 1.2)           # altitude band above the pad surface (y=0)
OBSTACLE_SAMPLE_TRIES = 50
EXT_BOUND = 3.0  # bounds for the relative-coord part of the observation


class ObstacleLunarLanderEnv(gym.Env):
    """LunarLander-v3 wrapper that appends obstacle coordinates and penalizes collisions.

    Observation: base 8-dim + 3-dim per obstacle (relative x, relative y, radius).
    Reward shaping: a smooth penalty inside the warning band
    ``[radius, 2*radius]`` and a large additive penalty + termination on
    collision. Collisions compare against ``obstacle.radius`` only — the lander
    is treated as a point mass.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        continuous: bool = False,
        obstacles: tuple[Obstacle, ...] | None = None,
        n_obstacles: int = N_OBSTACLES_DEFAULT,
        obstacle_radius: float = OBSTACLE_RADIUS_DEFAULT,
        random_obstacles: bool = False,
        collision_penalty: float = -100.0,
        shaping_coef: float = 0.5,
        **kwargs,
    ):
        self.env = make_base_env(render_mode=render_mode, continuous=continuous, **kwargs)
        self.render_mode = render_mode
        self.action_space = self.env.action_space

        self.random_obstacles = bool(random_obstacles)
        self.obstacle_radius = float(obstacle_radius)
        self.collision_penalty = float(collision_penalty)
        self.shaping_coef = float(shaping_coef)

        # The fixed layout is used in non-random mode and as the fallback when
        # random sampling fails. Its length always equals n_obstacles, which
        # keeps the observation space dimension-stable (avoids the lc-branch
        # bug where space and observation could disagree).
        if obstacles is not None:
            self._default_obstacles: tuple[Obstacle, ...] = tuple(obstacles)
        else:
            self._default_obstacles = self._build_fixed_layout(int(n_obstacles))
        self.n_obstacles = len(self._default_obstacles)
        self.obstacles: tuple[Obstacle, ...] = self._default_obstacles
        self._rng = np.random.default_rng()

        self.last_game_over = False
        self._screen = None
        self._clock = None

        base_low = self.env.observation_space.low.astype(np.float32)
        base_high = self.env.observation_space.high.astype(np.float32)
        obs_low = np.tile(np.array([-EXT_BOUND, -EXT_BOUND, 0.0], dtype=np.float32), self.n_obstacles)
        obs_high = np.tile(np.array([EXT_BOUND, EXT_BOUND, 1.0], dtype=np.float32), self.n_obstacles)
        self.observation_space = spaces.Box(
            np.concatenate([base_low, obs_low]),
            np.concatenate([base_high, obs_high]),
            dtype=np.float32,
        )

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.random_obstacles:
            rng = np.random.default_rng(seed) if seed is not None else self._rng
            self.obstacles = self._sample_obstacles(rng)
        else:
            self.obstacles = self._default_obstacles
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_game_over = False
        return self._augment_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus, collided = self._obstacle_bonus(obs)
        reward = float(reward) + bonus
        if collided:
            terminated = True
        info = {**info, "hit_obstacle": bool(collided)}
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
            rel_x = float(np.clip(obstacle.x - obs[0], -EXT_BOUND, EXT_BOUND))
            rel_y = float(np.clip(obstacle.y - obs[1], -EXT_BOUND, EXT_BOUND))
            obstacle_obs.extend([rel_x, rel_y, obstacle.radius])
        return np.concatenate([obs, np.array(obstacle_obs, dtype=np.float32)]).astype(np.float32)

    def _obstacle_bonus(self, obs):
        """Return ``(reward_delta, collided)`` for the current lander position."""
        x, y = float(obs[0]), float(obs[1])
        total = 0.0
        for obstacle in self.obstacles:
            dist = float(np.hypot(x - obstacle.x, y - obstacle.y))
            if dist < obstacle.radius:
                return self.collision_penalty, True
            if dist < 2.0 * obstacle.radius:
                total -= self.shaping_coef * (1.0 - dist / (2.0 * obstacle.radius))
        return total, False

    def _build_fixed_layout(self, n_obstacles: int) -> tuple[Obstacle, ...]:
        """Deterministic layout of length ``n_obstacles`` for non-random mode."""
        if n_obstacles == len(DEFAULT_OBSTACLES):
            return DEFAULT_OBSTACLES
        rng = np.random.default_rng(2024)
        placed: list[Obstacle] = []
        if n_obstacles < len(DEFAULT_OBSTACLES):
            return DEFAULT_OBSTACLES[:n_obstacles]
        while len(placed) < n_obstacles:
            obstacle = self._sample_one(rng, placed)
            if obstacle is None:
                break
            placed.append(obstacle)
        return tuple(placed) if placed else DEFAULT_OBSTACLES

    def _sample_obstacles(self, rng: np.random.Generator) -> tuple[Obstacle, ...]:
        """Reject-sample ``n_obstacles`` obstacles; fall back to the fixed layout
        if any placement fails so the count (and thus obs dim) stays constant."""
        placed: list[Obstacle] = []
        for _ in range(self.n_obstacles):
            obstacle = self._sample_one(rng, placed)
            if obstacle is None:
                return self._default_obstacles
            placed.append(obstacle)
        return tuple(placed)

    def _sample_one(self, rng: np.random.Generator, placed) -> Obstacle | None:
        """Sample one obstacle inside the landing zone, above the pad.

        ``x`` is uniform over the helipad's obs-x extent
        (``OBSTACLE_LANDING_X_RANGE``) and ``y`` is uniform over the altitude
        band above the pad (``OBSTACLE_Y_RANGE``), so each obstacle sits in the
        column directly above the landing zone and blocks the direct descent.
        Samples that overlap an existing obstacle are rejected.

        Returns ``None`` after ``OBSTACLE_SAMPLE_TRIES`` failed attempts.
        """
        x_lo, x_hi = OBSTACLE_LANDING_X_RANGE
        y_lo, y_hi = OBSTACLE_Y_RANGE
        radius = self.obstacle_radius
        for _ in range(OBSTACLE_SAMPLE_TRIES):
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(y_lo, y_hi))
            if any(np.hypot(x - p.x, y - p.y) < 3.0 * radius for p in placed):
                continue
            return Obstacle(x, y, radius)
        return None

    def _draw_overlays(self, frame):
        for obstacle in self.obstacles:
            cx, cy = state_to_pixel(self.env, obstacle.x, obstacle.y)
            radius = max(3, int(obstacle.radius * VIEWPORT_W / 2))
            draw_circle(frame, cx, cy, radius, (215, 48, 39))
            draw_circle(frame, cx, cy, max(1, radius - 4), (245, 142, 132))
