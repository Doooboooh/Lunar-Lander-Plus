"""目标条件的 LunarLander 环境（用于拓展3 重训策略）。

为什么需要它
------------
原版 LunarLander 的 shaping 用"相对平台中心的 x"做距离惩罚，所以预训练策略是
"回中器"——无论从哪起飞都落回中心，无法让多架飞船错开落点。

本环境做两件事让策略"能落到指定目标点"：
  1. 观测增加一维 goal_x（相对平台中心的目标落点 x），共 9 维；策略据此条件化。
  2. shaping 里把"相对中心的 x"替换为"相对目标 goal_x 的 x"，引导策略朝目标落；
     并在最终着陆奖励里要求落点接近 goal_x。

同时支持"已停泊飞船障碍"：reset 时可传入 landed_obstacles（已降落飞船的相对 x 列表），
本环境的飞船与它们发生 Box2D 碰撞会触发 game_over（视为撞机），从而让后一架真避让。
（障碍物的物理创建见 multi_lander.env 的停泊飞船管理；本类只负责目标条件化奖励。）
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.box2d.lunar_lander import (
    VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN, INITIAL_RANDOM, FPS,
)
from gymnasium.spaces import Box

_CENTER_X = VIEWPORT_W / SCALE / 2.0
GOAL_LIMIT = 0.9  # 目标 x 的采样范围（相对平台中心，米）


class GoalConditionedLunarLander(LunarLander):
    """9 维观测（末位为 goal_x）、目标条件 shaping 的 LunarLander。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, goal_x: float = 0.0,
                 landed_obstacles: Sequence[float] = (),
                 render_mode: Optional[str] = None, **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self.goal_x = float(np.clip(goal_x, -GOAL_LIMIT, GOAL_LIMIT))
        self.target_goal_x = self.goal_x  # reset 时不被覆盖的"本局目标"
        # 扩展观测空间：原 8 维 + goal_x
        low = np.append(self.observation_space.low, -GOAL_LIMIT).astype(np.float32)
        high = np.append(self.observation_space.high, GOAL_LIMIT).astype(np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------
    def _build_state(self, state8: np.ndarray) -> np.ndarray:
        """在原 8 维观测后追加 goal_x。"""
        return np.append(state8, self.target_goal_x).astype(np.float32)

    def _read_state8(self) -> np.ndarray:
        """读取当前 lander 的标准 8 维状态（与父类 LunarLander.step 归一化一致）。"""
        from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, VIEWPORT_H, LEG_DOWN, FPS
        p, v = self.lander.position, self.lander.linearVelocity
        cx = VIEWPORT_W / SCALE / 2.0
        w2, h2 = VIEWPORT_W / SCALE / 2.0, VIEWPORT_H / SCALE / 2.0
        return np.array([
            (p.x - cx) / w2,
            (p.y - (self.helipad_y + LEG_DOWN / SCALE)) / h2,
            v.x * w2 / FPS, v.y * h2 / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ], dtype=np.float32)

    def step(self, action):
        # 复用父类的物理推进与 8 维 state 组装，但重写奖励中的 shaping/着陆判定。
        # 为避免大段复制父类代码，采用：先记录 lander 状态 -> 调父类 step ->
        # 拿到 8 维 state 与父类算的 reward -> 用"目标条件 shaping"覆盖 reward。
        obs8_full = super().step(action)  # 返回 (state8, reward, term, trunc, info)
        state8, reward_parent, terminated, truncated, info = obs8_full
        state8 = np.asarray(state8, dtype=np.float32)

        # ---- 目标条件 shaping（替换原版回中 shaping）----
        # state[0] 是相对平台中心的 x；相对目标的 x = state[0] - goal_x
        rel_x_to_goal = state8[0] - self.target_goal_x
        shaping = (
            -100 * np.sqrt(rel_x_to_goal * rel_x_to_goal + state8[1] * state8[1])
            - 100 * np.sqrt(state8[2] * state8[2] + state8[3] * state8[3])
            - 100 * abs(state8[4])
            + 10 * state8[6]
            + 10 * state8[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        else:
            reward = 0.0
        self.prev_shaping = shaping

        # 父类 step 里已经扣了燃料（m_power*0.3 等），这里补不回来；为干净起见，
        # 我们直接用 shaping 差分作为主奖励，并显式扣燃料：从 info 里取不到，
        # 所以采用近似的"主发动机会被父类计入 terminated 判断"，这里不重复扣燃料，
        # 改为对落点精度给奖惩（见下）。

        # ---- 着陆判定（覆盖父类的 +100/-100）----
        if self.game_over or abs(state8[0]) >= 1.0:
            terminated = True
            reward = -100.0
        elif not self.lander.awake:
            terminated = True
            # 平稳落地：奖励 = 100 - 落点离目标的距离惩罚
            reward = 100.0 - 50.0 * abs(rel_x_to_goal)
            # 双足都接地且速度小 -> 额外稳定奖励
            both = self.legs[0].ground_contact and self.legs[1].ground_contact
            speed = math.hypot(state8[2], state8[3])
            if both and speed < 1.0:
                reward += 30.0

        return self._build_state(state8), float(reward), terminated, truncated, info
