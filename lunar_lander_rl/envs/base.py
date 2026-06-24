import gymnasium as gym

from lunar_lander_rl.envs.common import FPS, close_window, make_base_env


class BaseLunarLanderEnv(gym.Env):
    """Minimal project wrapper around Gymnasium LunarLander-v3."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, *, render_mode: str | None = None, continuous: bool = False, **kwargs):
        self.env = make_base_env(render_mode=render_mode, continuous=continuous, **kwargs)
        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._screen = None
        self._clock = None

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
        close_window(self)
