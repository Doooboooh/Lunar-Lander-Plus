"""拓展变体：移动着陆平台（Moving Pad）—— 真实体、悬空、可降落的 LunarLander。

现实意义：对应海上回收船 / 航母甲板等"移动平台降落"（如 SpaceX 无人船）。

核心设计（这次是"真实体"）
--------------------------
平台是一个 **KinematicBody 矩形，悬在地表之上**（不嵌在地下），飞船真的会落到它的
甲板顶面上（不会穿透、不会落到地面）。为做到这一点：
  - 平台 fixture：categoryBits=0x0004，maskBits=0x0010|0x0020（被 lander/leg 碰）；
  - 关键：reset 创建 lander 后，把 lander 和 legs 的 maskBits 改成 0x001 | 0x0004，
    让飞船也碰平台（原版 maskBits=0x001 只碰地面，会穿透平台）。
  - 已验证：改 maskBits 后飞船稳稳停在平台甲板上。

平台沿水平方向左右移动（正弦 + 随机相位），飞船要实时跟踪落在甲板上。
- 观测：原 8 维 + 平台相对飞船的位置 2 维（10 维）。
- reward：shaping 用"飞船相对平台顶面中心"的距离；着陆判据 = 站在平台甲板上。
- 渲染：画一个宽大、悬空的实体甲板（带支柱），取代原版地面旗帜。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import Box2D
from Box2D.b2 import fixtureDef, polygonShape
from gymnasium.envs.box2d.lunar_lander import LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, FPS, LEG_DOWN
from gymnasium.spaces import Box

_CENTER_X = VIEWPORT_W / SCALE / 2.0
_HELIPAD_Y = VIEWPORT_H / SCALE / 4.0
PAD_HALF_W = 1.3        # 平台半宽（米）—— 适当加宽，飞船落在中央不撞边弹飞
PAD_THICK = 0.35        # 平台甲板厚度（米）
PAD_LIFT = 0.9          # 平台顶面高于地表 helipad_y 的距离（米）→ 悬空
PAD_X_AMP = 2.2         # 平台水平往返幅度（米，相对中心）—— 覆盖画面最左到最右的大部分宽度
PAD_CAT = 0x0004        # 平台的碰撞 category
LANDER_CAT = 0x0010
LEG_CAT = 0x0020


class MovingPadLunarLander(LunarLander):
    """移动着陆平台：悬空实体甲板，飞船落到甲板顶面上。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode: Optional[str] = None,
                 pad_x_amp: float = PAD_X_AMP, **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self.pad_x_amp = float(pad_x_amp)
        # 扩展观测：原 8 维 + 平台相对位置 2 维
        low = np.append(self.observation_space.low, [-np.inf, -np.inf]).astype(np.float32)
        high = np.append(self.observation_space.high, [np.inf, np.inf]).astype(np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)
        self.ship_omega = 0.5
        self.ship_phase = 0.0
        self.ship = None
        self.ship_x = 0.0          # 平台中心 x（相对场景中心，米）
        self.deck_top_y = _HELIPAD_Y + PAD_LIFT    # 甲板顶面世界 y（固定高度，悬空）
        self.t = 0
        self._prev_goal_shaping = None

    # ------------------------------------------------------------------
    def _pad_center_world(self) -> tuple[float, float]:
        return (_CENTER_X + self.ship_x, self.deck_top_y)

    def _create_ship(self) -> None:
        """创建悬空实体平台（KinematicBody）。"""
        cx = _CENTER_X
        self.ship = self.world.CreateKinematicBody(
            position=(cx, self.deck_top_y),
            fixtures=fixtureDef(
                shape=polygonShape(box=(PAD_HALF_W, PAD_THICK / 2.0)),
                density=0.0, friction=2.0,
                categoryBits=PAD_CAT,
                maskBits=LANDER_CAT | LEG_CAT,   # 被 lander/leg 碰
                restitution=0.0,
            ),
        )
        self.ship.color1 = (70, 80, 95)
        self.ship.color2 = (45, 52, 62)

    def _patch_lander_mask(self) -> None:
        """让飞船(lander+legs)也碰平台。原版 maskBits=0x001 只碰地面。"""
        for f in self.lander.fixtures:
            f.filterData.maskBits = 0x001 | PAD_CAT
        for leg in self.legs:
            for f in leg.fixtures:
                f.filterData.maskBits = 0x001 | PAD_CAT

    def _move_ship(self) -> None:
        self.ship_x = self.pad_x_amp * math.sin(self.ship_omega * self.t / FPS + self.ship_phase)
        if self.ship is not None:
            self.ship.position = (_CENTER_X + self.ship_x, self.deck_top_y)
            # 速度（供接触响应；位置驱动）
            vx = self.pad_x_amp * self.ship_omega * math.cos(self.ship_omega * self.t / FPS + self.ship_phase)
            self.ship.linearVelocity = (vx, 0.0)

    def _stick_to_deck(self) -> None:
        """若飞船落在平台甲板上（双足接地 + 在平台宽度内 + 接近甲板顶），
        把它的速度强制设为平台速度（跟随平台移动），杜绝高速撞击弹飞。

        相当于"着陆锁定"：一旦落上甲板，飞船就粘住平台一起运动，不会被弹开。
        """
        if self.ship is None:
            return
        pos = self.lander.position
        sx, sy = self._pad_center_world()
        deck_top = sy + PAD_THICK / 2.0
        rel_x = pos.x - sx
        rel_y = pos.y - deck_top
        both = self.legs[0].ground_contact and self.legs[1].ground_contact
        on_deck = both and abs(rel_x) < PAD_HALF_W - 0.05 and abs(rel_y) < 0.5
        if on_deck:
            # 强制飞船跟随平台速度（平台水平速度），垂直清零
            pad_vx = self.pad_x_amp * self.ship_omega * math.cos(
                self.ship_omega * self.t / FPS + self.ship_phase)
            self.lander.linearVelocity = (pad_vx, 0.0)
            # 腿也跟随，避免关节把船拽偏
            for leg in self.legs:
                leg.linearVelocity = (pad_vx, 0.0)

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 关键：在调 super().reset() 之前先把 ship 引用置空。super().reset() 内部会调
        # 一次 self.step(0)，此时若 self.ship 还指向旧 world 的 body，step 里访问它就
        # 会段错误（C 层面，try 接不住）。置空后 step 走"ship is None"分支只跑父类物理。
        self.ship = None
        # options 可传 gravity 覆盖默认重力（值越小下落越慢，用于"更快平台"演示）
        opts = options or {}
        if "gravity" in opts:
            self.gravity = float(opts["gravity"])
        super().reset(seed=seed)   # 重建 world + lander（内部调的 step(0) 此时安全）
        rng = np.random.default_rng(seed if seed is not None else 0)
        # options 可传 omega 覆盖默认随机速度（用于单独生成"更快版"演示而不改全局默认）
        opts = options or {}
        if "omega" in opts:
            self.ship_omega = float(opts["omega"])
        else:
            self.ship_omega = float(rng.uniform(1.2, 1.6))   # 1.3倍速，飞船能学会匹配速度平稳落
        self.ship_phase = float(rng.uniform(0, 2 * math.pi))
        self.t = 0
        self._prev_goal_shaping = None
        # 在新 world 里建平台 + 让飞船能碰它
        self._create_ship()
        self._patch_lander_mask()
        self._move_ship()
        return self._build_obs(self._current_state8()), {}

    def _current_state8(self) -> np.ndarray:
        pos = self.lander.position
        vel = self.lander.linearVelocity
        return np.array([
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ], dtype=np.float32)

    def _build_obs(self, obs8) -> np.ndarray:
        pos = self.lander.position
        sx, sy = self._pad_center_world()
        dx = (sx - pos.x) / SCALE
        dy = (sy - pos.y) / SCALE
        dx_norm = dx / (VIEWPORT_W / SCALE / 2.0)
        dy_norm = dy / (VIEWPORT_H / SCALE / 2.0)
        return np.append(np.asarray(obs8, dtype=np.float32), [dx_norm, dy_norm]).astype(np.float32)

    # ------------------------------------------------------------------
    def _on_deck(self) -> bool:
        """飞船是否站在平台甲板上（双足接触 + 落点在平台宽度内 + 高度接近甲板顶）。"""
        if self.ship is None:
            return False
        pos = self.lander.position
        sx, sy = self._pad_center_world()
        rel_x = pos.x - sx
        rel_y = pos.y - (sy + PAD_THICK / 2.0)   # 相对甲板顶面
        both = self.legs[0].ground_contact and self.legs[1].ground_contact
        on_x = abs(rel_x) < PAD_HALF_W - 0.1
        near_top = abs(rel_y) < 0.6
        return bool(both and on_x and near_top)

    def step(self, action):
        # super().reset() 内部会调一次 step(0)，此时 self.ship 可能指向已销毁旧 world
        # 的 body（访问 position 即崩）。用 try 判断 ship 是否仍可用。
        ship_ok = self.ship is not None
        if ship_ok:
            try:
                _ = self.ship.position
            except Exception:
                ship_ok = False
        if not ship_ok:
            obs8, r, term, trunc, info = super().step(action)
            return self._build_obs(obs8), float(r), term, trunc, info

        self.t += 1
        obs8, reward_raw, terminated, truncated, info = super().step(action)
        self._move_ship()

        # ---- 用"飞船相对平台甲板中心"的 shaping 修正 reward ----
        pos = self.lander.position
        vel = self.lander.linearVelocity
        sx, sy = self._pad_center_world()
        deck_top = sy + PAD_THICK / 2.0
        rel_x = pos.x - sx
        rel_y = pos.y - deck_top
        legs = self.legs
        # 平台当前水平速度（飞船要匹配它，否则落甲板会被弹飞）
        pad_vx = self.pad_x_amp * self.ship_omega * math.cos(
            self.ship_omega * self.t / FPS + self.ship_phase)
        rel_vx = vel.x - pad_vx     # 飞船相对平台的水平速度
        shaping = (
            -100 * np.sqrt(rel_x * rel_x + rel_y * rel_y)
            - 100 * math.hypot(vel.x, vel.y)
            - 300 * abs(self.lander.angle)          # 加大姿态惩罚，促使竖直着陆
            - 150 * abs(rel_vx)                      # 关键：惩罚水平相对速度，迫使飞船匹配平台速度后落地
            + 10 * (1.0 if legs[0].ground_contact else 0.0)
            + 10 * (1.0 if legs[1].ground_contact else 0.0)
        )
        if self._prev_goal_shaping is not None:
            reward = shaping - self._prev_goal_shaping
        else:
            reward = 0.0
        self._prev_goal_shaping = shaping

        # ---- 着陆判定 ----
        # 出界判定放宽：平台移动范围加大后（amp≈2.2），飞船追平台可能到 ±3，不能误判出界
        if self.game_over or abs((pos.x - _CENTER_X)) >= 4.0:
            terminated = True
            reward = -100.0 if not self._on_deck() else +120.0
        elif not self.lander.awake:
            terminated = True
            speed = math.hypot(vel.x, vel.y)
            upright = abs(self.lander.angle) < 0.15   # 接近竖直才算"平稳"
            if self._on_deck():
                reward = +120.0 if (speed < 1.0 and upright) else (+40.0 if upright else -10.0)
            else:
                reward = -30.0    # 落到地面（没上平台）

        return self._build_obs(obs8), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        """渲染：调父类画飞船/地形/粒子，再画实体悬空甲板（盖住原版地面旗帜）。"""
        frame = super().render()
        if frame is None:
            return None
        import pygame
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        sx, sy = self._pad_center_world()
        deck_top = sy + PAD_THICK / 2.0
        cx = int(sx * SCALE)
        top_y = int(VIEWPORT_H - deck_top * SCALE)          # 甲板顶面屏幕 y
        bot_y = int(VIEWPORT_H - (sy - PAD_THICK / 2.0) * SCALE)  # 甲板底面屏幕 y
        half_w_px = int(PAD_HALF_W * SCALE)
        thick_px = max(8, bot_y - top_y)

        # 用"天空蓝"矩形擦掉甲板下方的地面旗帜（盖住原版旗帜区域）
        ground_y = int(VIEWPORT_H - self.helipad_y * SCALE)
        # 甲板底面到地面之间画支柱 + 擦旗帜
        # 支柱（左右两根，连接甲板底面到地面）
        pillar = (50, 58, 68)
        pw = 8
        pygame.draw.rect(surf, pillar,
                         pygame.Rect(cx - half_w_px + 10, bot_y, pw, ground_y - bot_y))
        pygame.draw.rect(surf, pillar,
                         pygame.Rect(cx + half_w_px - 10 - pw, bot_y, pw, ground_y - bot_y))
        # 用天空色覆盖甲板下方区域（擦掉地面伸上来的旗帜）
        SKY = (120, 160, 200)
        pygame.draw.rect(surf, SKY,
                         pygame.Rect(cx - half_w_px, bot_y, 2 * half_w_px, ground_y - bot_y))

        # 甲板本体（深色实体 + 亮黄边线）
        deck = (75, 85, 100)
        edge = (255, 215, 0)
        pygame.draw.rect(surf, deck,
                         pygame.Rect(cx - half_w_px, top_y, 2 * half_w_px, thick_px))
        pygame.draw.rect(surf, edge,
                         pygame.Rect(cx - half_w_px, top_y, 2 * half_w_px, thick_px), 3)
        # 甲板跑道中线（白色虚线）
        for i in range(-half_w_px + 14, half_w_px, 22):
            pygame.draw.line(surf, (240, 240, 240),
                             (cx + i, top_y + thick_px // 2),
                             (cx + i + 11, top_y + thick_px // 2), 3)
        return pygame.surfarray.array3d(surf).swapaxes(0, 1)
