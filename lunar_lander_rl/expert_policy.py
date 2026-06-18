from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class ExpertConfig:
    """Rule parameters for the hand-written LunarLander controller."""

    waypoint_angle_limit: float = 0.58
    waypoint_max_vx: float = 1.25
    waypoint_max_vy_up: float = 0.70
    waypoint_max_vy_down: float = 0.62
    waypoint_kx: float = 1.45
    waypoint_kvx: float = 1.15
    waypoint_ky: float = 0.85
    waypoint_kvy: float = 1.20
    waypoint_angle_gain: float = 0.55
    waypoint_angvel_gain: float = 1.00
    waypoint_main_threshold: float = 0.08
    waypoint_side_threshold: float = 0.045
    waypoint_slow_radius: float = 0.28
    waypoint_close_vx: float = 0.30
    waypoint_close_vy: float = 0.22
    waypoint_arrival_radius: float = 0.025
    waypoint_lookahead_radius: float = 0.075
    waypoint_lookahead_speed: float = 0.06
    final_waypoint_slow_radius: float = 0.28
    final_waypoint_close_vx: float = 0.30
    final_waypoint_close_vy: float = 0.22
    landing_angle_limit: float = 0.40
    landing_x_gain: float = 0.50
    landing_vx_gain: float = 1.00
    landing_y_x_gain: float = 0.55
    landing_y_gain: float = 0.50
    landing_vy_gain: float = 0.50
    landing_side_threshold: float = 0.05
    landing_main_threshold: float = 0.05
    route_landing_altitude: float = 0.0
    route_landing_x_altitude_gain: float = 0.0
    route_landing_vx_altitude_gain: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuleBasedLanderPolicy:
    """Hand-written expert policy for base and waypoint LunarLander tasks.

    The policy is intentionally stateless: the waypoint wrapper already exposes
    the current target delta and whether the route is complete.
    """

    def __init__(self, config: ExpertConfig | None = None, waypoints: Any | None = None) -> None:
        self.config = config or ExpertConfig()
        self.waypoints = None if waypoints is None else np.asarray(waypoints, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if len(obs) < 13:
            return self._landing_action(obs[:8], cautious_return=False)
        if obs[12] >= 0.5:
            return self._landing_action(obs[:8], cautious_return=True)
        return self._waypoint_action(obs)

    def _landing_action(self, obs: np.ndarray, cautious_return: bool) -> int:
        cfg = self.config
        x, y, vx, vy, angle, angular_v, left_leg, right_leg = [float(v) for v in obs[:8]]

        target_angle = np.clip(
            cfg.landing_x_gain * x + cfg.landing_vx_gain * vx,
            -cfg.landing_angle_limit,
            cfg.landing_angle_limit,
        )
        target_y = cfg.landing_y_x_gain * abs(x)
        if cautious_return and y > 0.20:
            target_y = max(
                target_y,
                cfg.route_landing_altitude
                + cfg.route_landing_x_altitude_gain * abs(x)
                + cfg.route_landing_vx_altitude_gain * abs(vx),
            )
        angle_error = (target_angle - angle) * cfg.landing_y_gain - angular_v
        y_error = (target_y - y) * cfg.landing_y_gain - vy * cfg.landing_vy_gain

        if left_leg > 0.5 and right_leg > 0.5:
            angle_error = 0.0
            y_error = -vy * cfg.landing_vy_gain
        elif left_leg > 0.5 or right_leg > 0.5:
            angle_error *= 0.45
            y_error = -vy * cfg.landing_vy_gain

        if y_error > abs(angle_error) and y_error > cfg.landing_main_threshold:
            return 2
        if angle_error < -cfg.landing_side_threshold:
            return 3
        if angle_error > cfg.landing_side_threshold:
            return 1
        return 0

    def _waypoint_action(self, obs: np.ndarray) -> int:
        cfg = self.config
        x, y, vx, vy, angle, angular_v = [float(v) for v in obs[:6]]
        target_dx, target_dy, target_dist = [float(v) for v in obs[8:11]]
        next_direction = self._next_segment_direction(obs)

        slow_radius = cfg.waypoint_slow_radius
        close_vx = cfg.waypoint_close_vx
        close_vy = cfg.waypoint_close_vy
        if next_direction is None:
            slow_radius = max(slow_radius, cfg.final_waypoint_slow_radius)
            close_vx = min(close_vx, cfg.final_waypoint_close_vx)
            close_vy = min(close_vy, cfg.final_waypoint_close_vy)

        speed_scale = min(1.0, max(0.25, target_dist / slow_radius))
        max_vx = cfg.waypoint_max_vx * speed_scale
        max_vy_up = cfg.waypoint_max_vy_up * speed_scale
        max_vy_down = cfg.waypoint_max_vy_down * speed_scale
        if target_dist < slow_radius:
            max_vx = min(max_vx, close_vx)
            max_vy_up = min(max_vy_up, close_vy)
            max_vy_down = min(max_vy_down, close_vy)

        desired_vx = float(np.clip(cfg.waypoint_kx * target_dx, -max_vx, max_vx))
        desired_vy = float(np.clip(cfg.waypoint_ky * target_dy, -max_vy_down, max_vy_up))

        if next_direction is not None and target_dist < cfg.waypoint_lookahead_radius:
            span = max(cfg.waypoint_lookahead_radius - cfg.waypoint_arrival_radius, 1e-6)
            lookahead_scale = 1.0 - np.clip((target_dist - cfg.waypoint_arrival_radius) / span, 0.0, 1.0)
            desired_vx += float(next_direction[0]) * cfg.waypoint_lookahead_speed * lookahead_scale
            desired_vy += float(next_direction[1]) * cfg.waypoint_lookahead_speed * lookahead_scale
            desired_vx = float(np.clip(desired_vx, -max_vx, max_vx))
            desired_vy = float(np.clip(desired_vy, -max_vy_down, max_vy_up))

        target_angle = np.clip(
            cfg.waypoint_kvx * (vx - desired_vx),
            -cfg.waypoint_angle_limit,
            cfg.waypoint_angle_limit,
        )
        angle_error = (target_angle - angle) * cfg.waypoint_angle_gain - angular_v * cfg.waypoint_angvel_gain
        vertical_error = cfg.waypoint_kvy * (desired_vy - vy)

        if y < 0.45 and vy < 0.05:
            vertical_error += 0.35
        if target_dy > 0.10 and vy < desired_vy:
            vertical_error += 0.20 * target_dy
        if target_dy < -0.20 and vy > desired_vy:
            vertical_error -= 0.08

        if vertical_error > max(cfg.waypoint_main_threshold, abs(angle_error) * 0.65):
            return 2
        if angle_error < -cfg.waypoint_side_threshold:
            return 3
        if angle_error > cfg.waypoint_side_threshold:
            return 1
        if vertical_error > cfg.waypoint_main_threshold:
            return 2
        return 0

    def _next_segment_direction(self, obs: np.ndarray) -> np.ndarray | None:
        if self.waypoints is None or len(self.waypoints) < 2:
            return None
        progress = float(obs[11])
        current_index = int(round(progress * len(self.waypoints)))
        if current_index < 0 or current_index >= len(self.waypoints) - 1:
            return None
        segment = self.waypoints[current_index + 1] - self.waypoints[current_index]
        norm = float(np.linalg.norm(segment))
        if norm <= 1e-6:
            return None
        return segment / norm
