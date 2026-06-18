from __future__ import annotations

from pathlib import Path
from typing import Callable
from collections.abc import Sequence

import numpy as np

from .trajectory_env import make_waypoint_env


def observation_point_to_pixel(env, point: Sequence[float], frame_shape: tuple[int, ...]) -> tuple[int, int]:
    """Map LunarLander observation-space x/y to rgb_array frame pixels."""

    try:
        from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W
    except ImportError:
        LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W = 18, 30.0, 400, 600

    base_env = getattr(env, "env", env)
    helipad_y = float(getattr(base_env, "helipad_y", 0.0))
    world_x = float(point[0]) * (VIEWPORT_W / SCALE / 2.0) + (VIEWPORT_W / SCALE / 2.0)
    world_y = float(point[1]) * (VIEWPORT_H / SCALE / 2.0) + helipad_y + LEG_DOWN / SCALE
    x = int(round(world_x * SCALE))
    y = int(round(VIEWPORT_H - world_y * SCALE))

    height, width = frame_shape[:2]
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))


def observation_radius_to_pixel_radii(radius: float) -> tuple[int, int]:
    """Map an observation-space waypoint radius to x/y pixel radii."""

    try:
        from gymnasium.envs.box2d.lunar_lander import VIEWPORT_H, VIEWPORT_W
    except ImportError:
        VIEWPORT_H, VIEWPORT_W = 400, 600

    return int(round(radius * VIEWPORT_W / 2.0)), int(round(radius * VIEWPORT_H / 2.0))


def blend_circle(frame: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int], alpha: float) -> None:
    x0, y0 = center
    height, width = frame.shape[:2]
    x_min = max(0, x0 - radius)
    x_max = min(width - 1, x0 + radius)
    y_min = max(0, y0 - radius)
    y_max = min(height - 1, y0 + radius)
    if x_min > x_max or y_min > y_max:
        return

    ys, xs = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
    mask = (xs - x0) ** 2 + (ys - y0) ** 2 <= radius**2
    patch = frame[y_min : y_max + 1, x_min : x_max + 1].astype(np.float32)
    patch[mask] = patch[mask] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    frame[y_min : y_max + 1, x_min : x_max + 1] = patch.astype(np.uint8)


def blend_ellipse(
    frame: np.ndarray,
    center: tuple[int, int],
    radius_x: int,
    radius_y: int,
    color: tuple[int, int, int],
    alpha: float,
    outline_width: int = 0,
) -> None:
    x0, y0 = center
    height, width = frame.shape[:2]
    x_min = max(0, x0 - radius_x)
    x_max = min(width - 1, x0 + radius_x)
    y_min = max(0, y0 - radius_y)
    y_max = min(height - 1, y0 + radius_y)
    if x_min > x_max or y_min > y_max or radius_x <= 0 or radius_y <= 0:
        return

    ys, xs = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
    distance = ((xs - x0) / radius_x) ** 2 + ((ys - y0) / radius_y) ** 2
    if outline_width > 0:
        inner_x = max(radius_x - outline_width, 1)
        inner_y = max(radius_y - outline_width, 1)
        inner_distance = ((xs - x0) / inner_x) ** 2 + ((ys - y0) / inner_y) ** 2
        mask = (distance <= 1.0) & (inner_distance >= 1.0)
    else:
        mask = distance <= 1.0

    patch = frame[y_min : y_max + 1, x_min : x_max + 1].astype(np.float32)
    patch[mask] = patch[mask] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    frame[y_min : y_max + 1, x_min : x_max + 1] = patch.astype(np.uint8)


def blend_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
    width: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    distance = max(abs(x1 - x0), abs(y1 - y0), 1)
    for t in np.linspace(0.0, 1.0, distance + 1):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        blend_circle(frame, (x, y), width, color, alpha)


def overlay_waypoint_route(frame: np.ndarray, env, completed: int = 0) -> np.ndarray:
    """Overlay the target waypoint path as a translucent background guide."""

    waypoints = getattr(env, "waypoints", None)
    if waypoints is None or len(waypoints) == 0:
        return frame

    annotated = np.array(frame, copy=True)
    pixels = [observation_point_to_pixel(env, point, annotated.shape) for point in waypoints]
    hit_radius = float(getattr(env, "radius", 0.0))
    hit_mode = getattr(env, "waypoint_hit_mode", "center")
    hit_radius_x, hit_radius_y = observation_radius_to_pixel_radii(hit_radius)

    route_color = (64, 128, 255)
    completed_color = (52, 168, 83)
    target_color = (255, 193, 7)
    point_color = (32, 96, 220)

    for idx in range(len(pixels) - 1):
        color = completed_color if idx < max(0, completed - 1) else route_color
        alpha = 0.26 if idx < max(0, completed - 1) else 0.18
        blend_line(annotated, pixels[idx], pixels[idx + 1], color, alpha=alpha, width=3)

    for idx, pixel in enumerate(pixels):
        if hit_mode == "center" and hit_radius > 0.0:
            blend_ellipse(annotated, pixel, hit_radius_x, hit_radius_y, color=(255, 255, 255), alpha=0.08)
            blend_ellipse(annotated, pixel, hit_radius_x, hit_radius_y, color=(255, 255, 255), alpha=0.20, outline_width=2)
        if idx < completed:
            blend_circle(annotated, pixel, radius=13, color=completed_color, alpha=0.28)
            blend_circle(annotated, pixel, radius=6, color=completed_color, alpha=0.45)
        elif idx == completed:
            blend_circle(annotated, pixel, radius=15, color=target_color, alpha=0.24)
            blend_circle(annotated, pixel, radius=7, color=target_color, alpha=0.45)
        else:
            blend_circle(annotated, pixel, radius=11, color=point_color, alpha=0.22)
            blend_circle(annotated, pixel, radius=5, color=point_color, alpha=0.35)

    return annotated


def evaluate_waypoint_policy(
    policy_fn: Callable[[np.ndarray], int],
    route_name: str,
    custom_waypoints: Sequence[tuple[float, float]] | None = None,
    episodes: int = 5,
    max_steps: int = 1000,
    seed: int = 1234,
    radius: float = 0.16,
    waypoint_hit_mode: str = "center",
    gif_path: str | Path | None = None,
    gif_fps: int = 30,
    env_kwargs: dict | None = None,
) -> dict[str, float]:
    """Evaluate a policy with waypoint-specific success metrics."""

    returns: list[float] = []
    completed_counts: list[int] = []
    route_complete_flags: list[float] = []
    landed_after_route_flags: list[float] = []
    touchdown_after_route_flags: list[float] = []
    settled_after_route_flags: list[float] = []
    final_distances: list[float] = []
    episode_lengths: list[int] = []
    gif_frames = []
    env_kwargs = env_kwargs or {}

    for episode in range(episodes):
        capture_gif = gif_path is not None and episode == 0
        env = make_waypoint_env(
            seed=seed + episode,
            route_name=route_name,
            custom_waypoints=custom_waypoints,
            radius=radius,
            waypoint_hit_mode=waypoint_hit_mode,
            render_mode="rgb_array" if capture_gif else None,
            **env_kwargs,
        )
        total_reward = 0.0
        final_info = {}
        final_obs = None
        steps = 0
        try:
            obs, info = env.reset(seed=seed + episode)
            final_info = info
            final_obs = obs
            for steps in range(1, max_steps + 1):
                if capture_gif:
                    gif_frames.append(overlay_waypoint_route(env.render(), env, int(info.get("waypoints_completed", 0))))
                action = int(policy_fn(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                final_info = info
                final_obs = obs
                if terminated or truncated:
                    if capture_gif:
                        gif_frames.append(overlay_waypoint_route(env.render(), env, int(info.get("waypoints_completed", 0))))
                    break
        finally:
            env.close()

        waypoint_count = int(final_info.get("waypoint_count", 0))
        completed = int(final_info.get("waypoints_completed", 0))
        route_complete = bool(final_info.get("route_complete", False))
        leg_contact = bool(final_obs is not None and len(final_obs) >= 8 and final_obs[6] > 0.5 and final_obs[7] > 0.5)
        any_leg_contact = bool(final_obs is not None and len(final_obs) >= 8 and (final_obs[6] > 0.5 or final_obs[7] > 0.5))
        settled = False
        if final_obs is not None and len(final_obs) >= 8:
            x, y, vx, vy, angle = [float(v) for v in final_obs[:5]]
            settled = bool(
                y <= 0.12
                and abs(x) <= 0.35
                and abs(vx) <= 0.40
                and abs(vy) <= 0.30
                and abs(angle) <= 0.35
                and any_leg_contact
            )

        returns.append(total_reward)
        completed_counts.append(completed)
        route_complete_flags.append(float(route_complete))
        landed_after_route_flags.append(float(route_complete and leg_contact))
        touchdown_after_route_flags.append(float(route_complete and any_leg_contact))
        settled_after_route_flags.append(float(route_complete and settled))
        final_distances.append(float(final_info.get("target_distance", 0.0)))
        episode_lengths.append(steps)

    if gif_path is not None and gif_frames:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise SystemExit("未找到 imageio，请先运行：pip install -r requirements.txt") from exc
        path = Path(gif_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, gif_frames, fps=gif_fps)

    return {
        "episodes": float(episodes),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_waypoints_completed": float(np.mean(completed_counts)),
        "max_waypoints_completed": float(np.max(completed_counts)),
        "route_completion_rate": float(np.mean(route_complete_flags)),
        "landed_after_route_rate": float(np.mean(landed_after_route_flags)),
        "touchdown_after_route_rate": float(np.mean(touchdown_after_route_flags)),
        "settled_after_route_rate": float(np.mean(settled_after_route_flags)),
        "mean_final_target_distance": float(np.mean(final_distances)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "waypoint_count": float(waypoint_count),
    }
