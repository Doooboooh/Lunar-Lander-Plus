from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from lunar_lander_rl.common import import_gym, make_env


DEFAULT_OBSTACLES: tuple[tuple[float, float, float], ...] = (
    (-0.6, 0.7, 0.12),
    (0.4, 0.5, 0.12),
    (0.0, 1.0, 0.12),
)
OBSTACLE_RADIUS_DEFAULT = 0.12
OBSTACLE_Y_RANGE = (0.3, 1.2)
OBSTACLE_SAFE_PAD = 0.05
EXT_BOUND = 3.0
N_OBSTACLES_DEFAULT = 3


class ObstacleLanderEnv:
    """gym.Wrapper around LunarLander-v3 that adds circular obstacles.

    Observation: base 8-dim + 2*n_obstacles relative coords (clip ±3.0).
    Reward shaping: -100 on collision (terminated), smooth penalty in warning band.
    """

    def __init__(
        self,
        obstacles: Iterable[tuple[float, float, float]] | None = None,
        n_obstacles: int = N_OBSTACLES_DEFAULT,
        radius: float = OBSTACLE_RADIUS_DEFAULT,
        random_obstacles: bool = False,
        render: bool = False,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        gym = import_gym()
        self._gym = gym
        if render_mode is not None:
            from lunar_lander_rl.common import ENV_ID

            self.env = gym.make(ENV_ID, render_mode=render_mode)
            if seed is not None:
                self.env.reset(seed=seed)
                self.env.action_space.seed(seed)
        else:
            self.env = make_env(render=render, seed=seed)

        self.random_obstacles = random_obstacles
        self.n_obstacles = n_obstacles
        self.radius = float(radius)
        self._default_obstacles = (
            list(obstacles) if obstacles is not None else list(DEFAULT_OBSTACLES)
        )
        self.obstacles: list[tuple[float, float, float]] = list(self._default_obstacles)

        base_space = self.env.observation_space
        base_low = np.asarray(base_space.low, dtype=np.float32)
        base_high = np.asarray(base_space.high, dtype=np.float32)
        ext_dim = 2 * self.n_obstacles
        ext_low = np.full(ext_dim, -EXT_BOUND, dtype=np.float32)
        ext_high = np.full(ext_dim, EXT_BOUND, dtype=np.float32)
        low = np.concatenate([base_low, ext_low])
        high = np.concatenate([base_high, ext_high])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = self.env.action_space

    def _extend_obs(self, obs: np.ndarray) -> np.ndarray:
        x, y = float(obs[0]), float(obs[1])
        rels: list[float] = []
        for cx, cy, _r in self.obstacles:
            rels.append(float(np.clip(x - cx, -EXT_BOUND, EXT_BOUND)))
            rels.append(float(np.clip(y - cy, -EXT_BOUND, EXT_BOUND)))
        return np.concatenate([np.asarray(obs, dtype=np.float32), np.asarray(rels, dtype=np.float32)])

    def _obstacle_bonus(self, x: float, y: float) -> tuple[float, bool]:
        total = 0.0
        for cx, cy, r in self.obstacles:
            dist = float(np.hypot(x - cx, y - cy))
            if dist < r:
                return -100.0, True
            if dist < 2.0 * r:
                total -= 0.5 * (1.0 - dist / (2.0 * r))
        return total, False

    def _sample_obstacles(self, rng: np.random.Generator) -> list[tuple[float, float, float]]:
        y_lo, y_hi = OBSTACLE_Y_RANGE
        placed: list[tuple[float, float, float]] = []
        for _ in range(self.n_obstacles):
            for _attempt in range(50):
                x = float(rng.uniform(-1.5, 1.5))
                y = float(rng.uniform(y_lo, y_hi))
                if abs(x) < 0.3 + OBSTACLE_SAFE_PAD and y < 0.4:
                    continue
                too_close = False
                for px, py, _pr in placed:
                    if np.hypot(x - px, y - py) < 3.0 * self.radius:
                        too_close = True
                        break
                if too_close:
                    continue
                placed.append((x, y, self.radius))
                break
            else:
                return list(self._default_obstacles)
        return placed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.random_obstacles:
            rng = np.random.default_rng(seed if seed is not None else 0)
            self.obstacles = self._sample_obstacles(rng)
        else:
            self.obstacles = list(self._default_obstacles)
        obs, info = self.env.reset(seed=seed, options=options)
        return self._extend_obs(obs), info

    def step(self, action: Any):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        bonus, collided = self._obstacle_bonus(float(next_obs[0]), float(next_obs[1]))
        reward = float(reward) + bonus
        if collided:
            terminated = True
            info["collision"] = True
        return self._extend_obs(next_obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def spec(self):
        return self.env.spec

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def make_obstacle_env(
    obstacles: Iterable[tuple[float, float, float]] | None = None,
    n_obstacles: int = N_OBSTACLES_DEFAULT,
    radius: float = OBSTACLE_RADIUS_DEFAULT,
    random_obstacles: bool = False,
    render: bool = False,
) -> Callable[[int | None], ObstacleLanderEnv]:
    def factory(seed: int | None = None) -> ObstacleLanderEnv:
        return ObstacleLanderEnv(
            obstacles=obstacles,
            n_obstacles=n_obstacles,
            radius=radius,
            random_obstacles=random_obstacles,
            render=render,
            seed=seed,
        )

    return factory
