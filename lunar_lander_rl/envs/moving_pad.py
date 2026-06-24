import math

import gymnasium as gym
import numpy as np
from Box2D.b2 import fixtureDef, polygonShape
from gymnasium import spaces

from lunar_lander_rl.envs.common import (
    FPS,
    VIEWPORT_W,
    close_window,
    draw_game_over,
    draw_rect,
    make_base_env,
    present_frame,
    state_to_pixel,
    state_to_world,
)


class MovingPadLunarLanderEnv(gym.Env):
    """LunarLander-v3 wrapper with a horizontal moving landing pad."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        continuous: bool = False,
        pad_amplitude: float = 0.45,
        pad_period: int = 240,
        pad_width: float = 3.0,
        pad_height: float = 0.16,
        collision_penalty: float = -100.0,
        **kwargs,
    ):
        self.env = make_base_env(render_mode=render_mode, continuous=continuous, **kwargs)
        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self.pad_amplitude = float(pad_amplitude)
        self.pad_period = int(pad_period)
        self.pad_width = float(pad_width)
        self.pad_height = float(pad_height)
        self.collision_penalty = float(collision_penalty)
        self.episode_step = 0
        self.pad_phase = 0.0
        self.platform_body = None
        self.platform_top_contact = False
        self.platform_bad_contact = False
        self.last_game_over = False
        self._screen = None
        self._clock = None

        base_low = self.env.observation_space.low.astype(np.float32)
        base_high = self.env.observation_space.high.astype(np.float32)
        pad_low = np.array([-1.0, -1.0, -3.0], dtype=np.float32)
        pad_high = np.array([1.0, 1.0, 3.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            np.concatenate([base_low, pad_low]),
            np.concatenate([base_high, pad_high]),
            dtype=np.float32,
        )

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.episode_step = 0
        self.pad_phase = float(self.env.unwrapped.np_random.uniform(0.0, 2.0 * math.pi))
        self._create_platform_body()
        self.platform_top_contact = False
        self.platform_bad_contact = False
        self.last_game_over = False
        return self._augment_observation(obs), info

    def step(self, action):
        self._sync_platform_body()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        self._sync_platform_body()
        self._update_platform_contacts()

        hit_platform = self.platform_bad_contact
        if hit_platform:
            terminated = True
            reward = self.collision_penalty

        info = {
            **info,
            "target": self._landing_target(),
            "hit_platform": hit_platform,
            "platform_top_contact": self.platform_top_contact,
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
        if self.platform_body is None:
            self._draw_landing_pad(frame)
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
        target_x, target_vx = self._pad_state()
        pad_obs = np.array([target_x, target_vx, obs[0] - target_x], dtype=np.float32)
        return np.concatenate([obs, pad_obs]).astype(np.float32)

    def _landing_target(self):
        target_x, _ = self._pad_state()
        return (target_x, 0.0)

    def _pad_state(self):
        omega = 2.0 * math.pi / self.pad_period
        phase = omega * self.episode_step + self.pad_phase
        x = self.pad_amplitude * math.sin(phase)
        vx = self.pad_amplitude * omega * math.cos(phase)
        return float(x), float(vx)

    def _create_platform_body(self):
        self.platform_body = None
        world_x, world_y = state_to_world(self.env, *self._landing_target())
        self.platform_body = self.env.unwrapped.world.CreateKinematicBody(
            position=(world_x, world_y + self.pad_height / 2.0),
            fixtures=fixtureDef(
                shape=polygonShape(box=(self.pad_width / 2.0, self.pad_height / 2.0)),
                density=0.0,
                friction=0.9,
                categoryBits=0x001,
                maskBits=0x0030,
                restitution=0.0,
            ),
        )
        self.platform_body.color1 = (32, 92, 172)
        self.platform_body.color2 = (236, 244, 255)
        self.env.unwrapped.drawlist.append(self.platform_body)
        self._sync_platform_body()

    def _sync_platform_body(self):
        if self.platform_body is None:
            return
        target_x, target_vx = self._pad_state()
        world_x, world_y = state_to_world(self.env, target_x, 0.0)
        self.platform_body.position = (world_x, world_y + self.pad_height / 2.0)
        self.platform_body.linearVelocity = (target_vx * (VIEWPORT_W / 30.0 / 2.0), 0.0)

    def _update_platform_contacts(self):
        self.platform_top_contact = False
        self.platform_bad_contact = False
        if self.platform_body is None:
            return

        lander = self.env.unwrapped.lander
        legs = tuple(getattr(self.env.unwrapped, "legs", ()))
        platform_top = self.platform_body.position.y + self.pad_height / 2.0
        platform_left = self.platform_body.position.x - self.pad_width / 2.0
        platform_right = self.platform_body.position.x + self.pad_width / 2.0
        top_slop = 0.12
        side_slop = 0.08

        for contact in self.env.unwrapped.world.contacts:
            if not contact.touching:
                continue
            bodies = (contact.fixtureA.body, contact.fixtureB.body)
            if self.platform_body not in bodies:
                continue

            other = bodies[1] if bodies[0] is self.platform_body else bodies[0]
            if other is lander:
                self.platform_bad_contact = True
                continue
            if other not in legs:
                continue

            legal_top_touch = False
            for point in contact.worldManifold.points:
                if (
                    platform_top - top_slop <= point[1] <= platform_top + top_slop
                    and platform_left + side_slop <= point[0] <= platform_right - side_slop
                ):
                    legal_top_touch = True

            if legal_top_touch:
                self.platform_top_contact = True
            else:
                self.platform_bad_contact = True

    def _draw_landing_pad(self, frame):
        target_x, _ = self._pad_state()
        cx, cy = state_to_pixel(self.env, target_x, 0.0)
        half_width = 58
        draw_rect(frame, cx - half_width, cy - 6, cx + half_width, cy + 4, (32, 92, 172))
        draw_rect(frame, cx - half_width, cy - 12, cx - half_width + 4, cy + 28, (255, 255, 255))
        draw_rect(frame, cx + half_width - 4, cy - 12, cx + half_width, cy + 28, (255, 255, 255))
