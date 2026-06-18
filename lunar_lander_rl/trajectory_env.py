from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .common import import_gym


WAYPOINT_HIT_MODES = ("center", "body")


ROUTES: dict[str, tuple[tuple[float, float], ...]] = {
    "single_left": ((-0.55, 1.05),),
    "near_two_waypoint": ((-0.30, 1.15), (0.30, 1.15)),
    "two_waypoint": ((-0.55, 1.05), (0.55, 1.05)),
    "orbit": ((-0.55, 0.95), (0.0, 1.35), (0.55, 0.95), (0.0, 0.65)),
    "figure_eight": (
        (-0.55, 1.05),
        (0.0, 1.30),
        (0.55, 1.05),
        (0.0, 0.80),
        (-0.55, 1.05),
        (0.0, 0.80),
        (0.55, 1.05),
        (0.0, 1.30),
    ),
}


def parse_waypoint_text(text: str) -> tuple[tuple[float, float], ...]:
    """Parse custom waypoints from JSON or a compact "x,y;x,y" string."""

    text = text.strip()
    if not text:
        raise ValueError("custom waypoint text is empty")

    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("waypoints")
        if data is None:
            raise ValueError("JSON waypoint data must be a list or contain a 'waypoints' field")
        points = data
    else:
        points = []
        for pair in text.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            x_text, y_text = pair.split(",", maxsplit=1)
            points.append((float(x_text), float(y_text)))

    parsed = tuple((float(point[0]), float(point[1])) for point in points)
    if not parsed:
        raise ValueError("at least one waypoint is required")
    return parsed


def load_waypoints(path: str | Path) -> tuple[tuple[float, float], ...]:
    return parse_waypoint_text(Path(path).read_text(encoding="utf-8"))


def resolve_waypoints(
    route_name: str = "two_waypoint",
    custom_waypoints: Sequence[tuple[float, float]] | None = None,
) -> tuple[tuple[float, float], ...]:
    if custom_waypoints is not None:
        return tuple((float(x), float(y)) for x, y in custom_waypoints)
    if route_name not in ROUTES:
        raise ValueError(f"unknown route_name {route_name!r}; choose one of {sorted(ROUTES)} or pass custom_waypoints")
    return ROUTES[route_name]


class WaypointLunarLander:
    """Gymnasium wrapper that turns LunarLander into a waypoint-then-land task.

    The base observation is extended with five values: target dx, target dy,
    target distance, route progress, and a binary landing-phase flag.
    """

    def __init__(
        self,
        env,
        waypoints: Sequence[tuple[float, float]],
        radius: float = 0.16,
        progress_reward: float = 8.0,
        waypoint_bonus: float = 45.0,
        route_bonus: float = 80.0,
        early_landing_penalty: float = 160.0,
        landing_after_route_bonus: float = 40.0,
        base_reward_scale_before_route: float = 0.2,
        base_reward_scale_after_route: float = 1.0,
        waypoint_hit_mode: str = "center",
    ) -> None:
        gym = import_gym()
        self.env = env
        self.waypoints = np.asarray(waypoints, dtype=np.float32)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2 or len(self.waypoints) == 0:
            raise ValueError("waypoints must be a non-empty sequence of (x, y) pairs")
        self.radius = float(radius)
        self.progress_reward = float(progress_reward)
        self.waypoint_bonus = float(waypoint_bonus)
        self.route_bonus = float(route_bonus)
        self.early_landing_penalty = float(early_landing_penalty)
        self.landing_after_route_bonus = float(landing_after_route_bonus)
        self.base_reward_scale_before_route = float(base_reward_scale_before_route)
        self.base_reward_scale_after_route = float(base_reward_scale_after_route)
        if waypoint_hit_mode not in WAYPOINT_HIT_MODES:
            raise ValueError(f"waypoint_hit_mode must be one of {WAYPOINT_HIT_MODES}, got {waypoint_hit_mode!r}")
        self.waypoint_hit_mode = waypoint_hit_mode
        self.next_waypoint_index = 0
        self.last_distance = 0.0

        low = np.concatenate([env.observation_space.low, np.array([-3.0, -3.0, 0.0, 0.0, 0.0], dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.array([3.0, 3.0, 5.0, 1.0, 1.0], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.next_waypoint_index = 0
        self.last_distance = self._distance_to_current_target(obs)
        return self._augment(obs), self._info(info, obs)

    def step(self, action):
        previous_distance = self.last_distance
        obs, reward, terminated, truncated, info = self.env.step(action)
        completed_before = self.route_complete
        reward_scale = self.base_reward_scale_after_route if completed_before else self.base_reward_scale_before_route
        reward = float(reward) * reward_scale

        if not self.route_complete:
            distance = self._distance_to_current_target(obs)
            reward += self.progress_reward * (previous_distance - distance)
            self.last_distance = distance

            if self._current_target_reached(obs, distance):
                reward += self.waypoint_bonus
                self.next_waypoint_index += 1
                if self.route_complete:
                    reward += self.route_bonus
                    self.last_distance = 0.0
                else:
                    self.last_distance = self._distance_to_current_target(obs)

        done = bool(terminated or truncated)
        if done and not self.route_complete:
            reward -= self.early_landing_penalty
        elif done and completed_before and reward > 0.0:
            reward += self.landing_after_route_bonus

        return self._augment(obs), reward, terminated, truncated, self._info(info, obs)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def route_complete(self) -> bool:
        return self.next_waypoint_index >= len(self.waypoints)

    def _current_target(self) -> np.ndarray:
        if self.route_complete:
            return np.array([0.0, 0.0], dtype=np.float32)
        return self.waypoints[self.next_waypoint_index]

    def _distance_to_current_target(self, obs) -> float:
        pos = np.asarray(obs[:2], dtype=np.float32)
        return float(np.linalg.norm(self._current_target() - pos))

    def _current_target_reached(self, obs, distance: float) -> bool:
        if self.waypoint_hit_mode == "center":
            return distance <= self.radius
        return self._target_touches_lander_body()

    def _target_touches_lander_body(self) -> bool:
        """Return True when the current target point is inside the lander body polygon."""

        try:
            from Box2D import b2Vec2
            from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W
        except ImportError:
            return False

        base_env = getattr(self.env, "unwrapped", self.env)
        lander = getattr(base_env, "lander", None)
        if lander is None or not getattr(lander, "fixtures", None):
            return False

        target = self._current_target()
        helipad_y = float(getattr(base_env, "helipad_y", 0.0))
        world_x = float(target[0]) * (VIEWPORT_W / SCALE / 2.0) + (VIEWPORT_W / SCALE / 2.0)
        world_y = float(target[1]) * (VIEWPORT_H / SCALE / 2.0) + helipad_y + LEG_DOWN / SCALE
        point = b2Vec2(world_x, world_y)

        vertices = [lander.transform * vertex for vertex in lander.fixtures[0].shape.vertices]
        return _point_in_polygon(point, vertices)

    def _augment(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        target_delta = self._current_target() - obs[:2]
        distance = float(np.linalg.norm(target_delta))
        progress = self.next_waypoint_index / float(len(self.waypoints))
        landing_phase = 1.0 if self.route_complete else 0.0
        extra = np.array([target_delta[0], target_delta[1], distance, progress, landing_phase], dtype=np.float32)
        return np.concatenate([obs, extra]).astype(np.float32)

    def _info(self, info, obs) -> dict:
        data = dict(info)
        data.update(
            {
                "waypoints_completed": int(min(self.next_waypoint_index, len(self.waypoints))),
                "waypoint_count": int(len(self.waypoints)),
                "route_complete": bool(self.route_complete),
                "target_distance": float(self._distance_to_current_target(obs)),
                "waypoint_hit_mode": self.waypoint_hit_mode,
            }
        )
        return data


def make_waypoint_env(
    render: bool = False,
    seed: int | None = None,
    route_name: str = "two_waypoint",
    custom_waypoints: Sequence[tuple[float, float]] | None = None,
    radius: float = 0.16,
    render_mode: str | None = None,
    progress_reward: float = 8.0,
    waypoint_bonus: float = 45.0,
    route_bonus: float = 80.0,
    early_landing_penalty: float = 160.0,
    landing_after_route_bonus: float = 40.0,
    base_reward_scale_before_route: float = 0.2,
    base_reward_scale_after_route: float = 1.0,
    waypoint_hit_mode: str = "center",
):
    gym = import_gym()
    waypoints = resolve_waypoints(route_name, custom_waypoints)
    mode = render_mode if render_mode is not None else ("human" if render else None)
    env = gym.make("LunarLander-v3", render_mode=mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return WaypointLunarLander(
        env,
        waypoints,
        radius=radius,
        progress_reward=progress_reward,
        waypoint_bonus=waypoint_bonus,
        route_bonus=route_bonus,
        early_landing_penalty=early_landing_penalty,
        landing_after_route_bonus=landing_after_route_bonus,
        base_reward_scale_before_route=base_reward_scale_before_route,
        base_reward_scale_after_route=base_reward_scale_after_route,
        waypoint_hit_mode=waypoint_hit_mode,
    )


def _point_in_polygon(point, vertices) -> bool:
    inside = False
    j = len(vertices) - 1
    for i, vertex in enumerate(vertices):
        other = vertices[j]
        yi = float(vertex.y)
        yj = float(other.y)
        if (yi > point.y) != (yj > point.y):
            xi = float(vertex.x)
            xj = float(other.x)
            x_intersect = (xj - xi) * (point.y - yi) / (yj - yi) + xi
            if point.x < x_intersect:
                inside = not inside
        j = i
    return inside
