"""三飞船有序降落环境（拓展3 回归版）。

设计：
  - 同一 Box2D world 里 3 艘 lander（继承 MultiAgentLunarLander 的多体物理）；
  - reset 时三艘都重定位到高空悬停点（水平错开），用 HoverPDController 主动悬停；
  - 依次降落：当前 phase 的飞船用外部传入的 DQN 动作，其余两艘用 PD 悬停；
  - 当前飞船平稳落到自己 goal_x → 停泊（parked），phase 推进；后降飞船靠 goal_x
    错开 + 垂直路径自然避开已停泊飞船；撞击任意他船 = 失败。

任意时刻只有一艘在降落，其余悬停 → 单智能体控制 + 静态障碍，收敛可靠。
observation_space = Box(9,)（当前降落船 8 维 + 自己的 goal_x），action_space = Discrete(4)。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN, FPS

from .multi_agent_env import MultiAgentLunarLander
from .hover_controller import HoverForceController

_CENTER_X = VIEWPORT_W / SCALE / 2.0


class SequentialLandingEnv(MultiAgentLunarLander):
    """三飞船有序降落：悬停 → 依次降落到错开落点。"""

    # 简化方案：三艘都落中心附近（基础 DQN 能稳定做到）。为避免已停泊飞船挡道，
    # 每艘落定后传送到画面顶部"停泊区"继续悬停，腾出落点给下一艘。
    GOAL_X = [0.0, 0.0, 0.0]
    OBS_BIAS = [0.0, 0.0, 0.0]         # 不偏置（基础 DQN 直接落中心）
    START_X_HOVER = [-0.9, 0.0, +0.9]  # 悬停时水平错开（避免初始碰撞）
    HOVER_Y = [2.0, 2.6, 2.0]          # 悬停高度错开
    # 已停泊飞船传送到顶部停泊区的位置（水平错开，避免叠在一起）
    PARK_X = [-1.2, 0.0, +1.2]
    PARK_Y = 3.0
    MAX_STEPS_PER_SHIP = 700
    WARMUP_STEPS = 40

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__(render_mode=render_mode, demo_mode=True)
        # 本环境用基础 DQN（8 维观测），通过观测偏置让三艘落不同位置
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = Discrete(4)
        self.hovers: list[HoverForceController] = []
        self.phase = 0                       # 当前第几艘在降落
        self.parked = [False, False, False]
        self.landed_x = [None, None, None]
        self.ship_step = [0, 0, 0]
        self.success = None

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 父类 reset 建 world + 3 lander（用我们的落点作 goal）
        super().reset(seed=seed, options={"goals": list(self.GOAL_X)})
        # 把三艘重定位到高空悬停点，清零速度/姿态
        # 用 SetTransform 直接设定位姿（直接赋 .position 在某些 Box2D 版本不可靠）
        for i, ld in enumerate(self.landers):
            tx = _CENTER_X + self.START_X_HOVER[i]
            ty = self.helipad_y + LEG_DOWN / SCALE + self.HOVER_Y[i]
            ld["lander"].transform = ((tx, ty), 0.0)
            ld["lander"].linearVelocity = (0.0, 0.0)
            ld["lander"].angularVelocity = 0.0
            for leg in ld["legs"]:
                leg.transform = ((tx, ty), 0.0)
                leg.linearVelocity = (0.0, 0.0)
            ld["alive"] = True
        # 每艘一个力施加悬停控制器（目标=各自的悬停点）
        self.hovers = [
            HoverForceController(self.START_X_HOVER[i]) for i in range(3)
        ]
        self.phase = 0
        self.parked = [False, False, False]
        self.landed_x = [None, None, None]
        self.ship_step = [0, 0, 0]
        self.success = None
        self.game_over = False
        # warmup：三艘只施加悬停力空跑，进入稳定悬停（渲染时这几帧=喷火悬停）
        for _ in range(self.WARMUP_STEPS):
            self._physics_step(dqn_idx=None, dqn_action=0)
        # 给第一艘降落船一个向下初速（解除悬停、进入下坠），匹配 goal_dqn 训练分布
        # （goal_dqn 训练起点是高空自由落体；静止悬停对它是 OOD，会失控侧推）
        self._drop_kick(0)
        return self._current_obs(), {"phase": self.phase}

    def _drop_kick(self, idx: int) -> None:
        """让第 idx 艘飞船解除悬停、传送到高空标准起点（goal_x 正上方 y≈13），
        从静止开始自由落体，匹配基础 DQN 训练分布。

        注意：原版 INITIAL_RANDOM=1000 是 ApplyForceToCenter 的"力"不是速度，
        不能当速度用（会给飞船 ~150 m/s 初速直接飞出界）。这里给很小的随机初速。
        """
        ld = self.landers[idx]
        gx = ld["goal_x"]
        tx = _CENTER_X + gx
        ty = VIEWPORT_H / SCALE          # 高空标准起点
        lander = ld["lander"]
        lander.position = (tx, ty)
        lander.linearVelocity = (self.np_random.uniform(-0.5, 0.5),
                                 self.np_random.uniform(-0.5, 0.5))
        lander.angle = 0.0
        lander.angularVelocity = 0.0
        for leg in ld["legs"]:
            leg.position = (tx, ty)
            leg.linearVelocity = (0.0, 0.0)
        ld["alive"] = True

    def _park_at_top(self, idx: int) -> None:
        """把已降落飞船传送到顶部停泊区，切换其悬停控制器目标到停泊位置。

        落定后移走，腾出落点给下一艘；停泊区飞船继续悬停（喷火）展示"已完成"。
        """
        ld = self.landers[idx]
        tx = _CENTER_X + self.PARK_X[idx]
        ty = self.helipad_y + LEG_DOWN / SCALE + self.PARK_Y
        ld["lander"].position = (tx, ty)
        ld["lander"].linearVelocity = (0.0, 0.0)
        ld["lander"].angle = 0.0
        ld["lander"].angularVelocity = 0.0
        for leg in ld["legs"]:
            leg.position = (tx, ty)
            leg.linearVelocity = (0.0, 0.0)
            leg.ground_contact = False
        # 该艘的悬停控制器目标改为停泊位置
        self.hovers[idx] = HoverForceController(self.PARK_X[idx])

    # ------------------------------------------------------------------
    def _physics_step(self, dqn_idx: Optional[int], dqn_action: int) -> None:
        """推进一步物理：
        - 编号 dqn_idx 的飞船（当前降落船）用 DQN 离散动作 _apply_engine；
        - 其余未停泊的飞船施加悬停力（HoverForceController）；
        - 已停泊的飞船在顶部停泊区继续悬停（施加悬停力，不挡落点）。
        """
        for i, ld in enumerate(self.landers):
            if not ld["alive"]:
                continue
            if i == dqn_idx:
                self._apply_engine(i, int(dqn_action))   # 降落船：DQN 控制
            else:
                self.hovers[i].apply(ld["lander"], ld["legs"])  # 悬停/停泊船：施力
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

    def _state8(self, idx: int) -> np.ndarray:
        """第 idx 艘的标准 8 维状态（与 gymnasium LunarLander.step 归一化完全一致）。

        必须和 goal_dqn 训练时（goal_env，其 state 来自父类 LunarLander）一致，否则
        策略失效。goal_x 不混入 state[0]，而是单独作为第 9 维（见 _current_obs）。
        """
        ld = self.landers[idx]
        p, v = ld["lander"].position, ld["lander"].linearVelocity
        W2 = VIEWPORT_W / SCALE / 2.0
        H2 = VIEWPORT_H / SCALE / 2.0
        return np.array([
            (p.x - _CENTER_X) / W2,
            (p.y - (self.helipad_y + LEG_DOWN / SCALE)) / H2,
            v.x * W2 / FPS,
            v.y * H2 / FPS,
            ld["lander"].angle,
            20.0 * ld["lander"].angularVelocity / FPS,
            1.0 if ld["legs"][0].ground_contact else 0.0,
            1.0 if ld["legs"][1].ground_contact else 0.0,
        ], dtype=np.float32)

    def _current_obs(self) -> np.ndarray:
        """返回当前降落船的 8 维观测，state[0] 加偏置让基础 DQN 落到目标 x。

        DQN 是回中器（想让 state[0]→0）。给它看的 state[0] = 真实 + bias，
        它会朝使"看到 state[0]=0"的方向飞，即真实位置落到 -bias = target_x。
        """
        if self.phase >= 3:
            return np.zeros(8, dtype=np.float32)
        s8 = self._state8(self.phase).copy()
        s8[0] += self.OBS_BIAS[self.phase]
        return s8.astype(np.float32)

    def _landed_cleanly(self, idx: int) -> bool:
        ld = self.landers[idx]
        if not ld["alive"]:
            return False
        p, v = ld["lander"].position, ld["lander"].linearVelocity
        rel_x = (p.x - _CENTER_X)
        speed = math.hypot(v.x, v.y)          # m/s
        both = ld["legs"][0].ground_contact and ld["legs"][1].ground_contact
        in_plat = -1.0 <= rel_x <= 1.0        # 落在平台范围内即可（基础 DQN 落点不精确）
        return bool(both and speed < 2.5 and in_plat)

    # ------------------------------------------------------------------
    def step(self, action):
        cur = self.phase
        if cur >= 3:
            return self._current_obs(), 0.0, True, False, {"success": self.success}

        # 当前降落船(cur)用 DQN 动作，其余悬停船施加悬停力，已停泊不动
        self._physics_step(dqn_idx=cur, dqn_action=int(action))
        self.ship_step[cur] += 1

        # 清空父类的碰撞事件缓冲（_physics_step 里 ContactDetector 会写入）
        self.collision_events = getattr(self, "collision_events", [])
        had_impact_cur = any(
            ev.get("impact") and cur in ev.get("pair", ()) for ev in self.collision_events
        )

        terminated = False
        info = {"phase": cur, "phase_advanced": False, "success": None,
                "collision": had_impact_cur}

        # 当前船平稳落地 → 停泊：传送到顶部停泊区悬停，腾出落点给下一艘
        if self._landed_cleanly(cur):
            p = self.landers[cur]["lander"].position
            self.landed_x[cur] = (p.x - _CENTER_X)
            self.parked[cur] = True
            self._park_at_top(cur)            # 移到顶部停泊区
            self.phase += 1
            info["phase_advanced"] = True
            if self.phase < 3:
                self._drop_kick(self.phase)   # 下一艘解除悬停开始降落
        # 撞击 → 失败
        if had_impact_cur:
            terminated = True
            self.success = False
        # 全部停泊 → 成功
        if all(self.parked):
            terminated = True
            self.success = True
        # 当前船超时/出界 → 失败
        rel_x_cur = (self.landers[cur]["lander"].position.x - _CENTER_X)
        if (not self.parked[cur]) and (self.ship_step[cur] > self.MAX_STEPS_PER_SHIP
                                       or abs(rel_x_cur) >= 2.0):
            terminated = True
            self.success = False

        if self.render_mode == "human":
            self.render()
        self.collision_events = []
        return self._current_obs(), 0.0, terminated, False, info

    # ------------------------------------------------------------------
    def render(self):
        """父类渲染后，给悬停中的飞船（未停泊且非当前降落船）画喷火特效。

        悬停飞船靠施加力抵消重力（非离散主推动作），所以没有原生喷火粒子。
        这里在它的主发动机位置画一段亮黄/橙喷火，让"悬停喷火"视觉可见。
        """
        import pygame
        frame = super().render()
        if frame is None:
            return None
        if not getattr(self, "landers", None):
            return frame
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        rng = getattr(self, "_flame_rng", None)
        if rng is None:
            rng = np.random.default_rng(0)
            self._flame_rng = rng
        for i, ld in enumerate(self.landers):
            # 悬停中：未停泊 且 不是当前降落船
            if self.parked[i] or i == self.phase:
                continue
            lander = ld["lander"]
            p = lander.position
            # 主发动机喷口在机身下方，沿机身朝向
            import math as _m
            angle = lander.angle
            ox = _m.sin(angle) * 14 / SCALE      # 沿机身向"下"的世界偏移
            oy = -_m.cos(angle) * 14 / SCALE
            wx, wy = p.x + ox, p.y + oy
            sx = int(wx * SCALE)
            sy = int(VIEWPORT_H - wy * SCALE)
            # 喷火：随机长度的橙黄渐变三角形
            length = 10 + int(rng.uniform(0, 8))
            end_x = sx + int(_m.sin(angle) * length)
            end_y = sy - int(-_m.cos(angle) * length)
            pygame.draw.polygon(surf, (255, 180, 40),
                                [(sx - 4, sy), (sx + 4, sy), (end_x, end_y)])
            pygame.draw.polygon(surf, (255, 240, 200),
                                [(sx - 2, sy), (sx + 2, sy),
                                 (sx + int(_m.sin(angle) * length * 0.5),
                                  sy - int(-_m.cos(angle) * length * 0.5))])
        return pygame.surfarray.array3d(surf).swapaxes(0, 1)
