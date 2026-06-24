"""拓展3：多个飞船顺序降落环境。

核心思想
--------
不重新训练飞船控制。把一个完整的多飞船任务看成一个 episode：
  for k in range(num_landers):
      1. 调度器为第 k 架飞船规划目标落点 + 起飞点（避开已降落飞船）
      2. 用复用的单飞船策略（默认 DQN）控制这架飞船完整降落
      3. 记录它的最终位置（落点），作为后续飞船的"占用区"约束
      4. 计算这架的奖励：原生 reward + 顺序/避碰奖惩
全部飞船都平稳落地 -> 全局 bonus。

飞船间约束通过两处体现：
  (a) 调度层：用 LanderScheduler 给后续飞船推远目标落点（见 scheduler.py）
  (b) 奖励层：落点离已降落飞船 < safe_dist -> collision_penalty
        过程中飞船与已降落飞船碰撞 -> 也算 collision_penalty

环境对外提供 run() 一次性跑完整个多飞船 episode（顺序决策，无需交互式 step），
以及 step()-style 接口（把"当前飞船的一步"作为一个 step），方便后续做
端到端强化学习扩展。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..common import import_gym
from .config import MultiLanderConfig
from .policy import SingleLanderPolicy
from .scheduler import LanderScheduler, ShipPlan

ENV_ID = "LunarLander-v3"

# LunarLander 的世界坐标常量（平台中心的世界 x = VIEWPORT_W/SCALE/2 = 10.0）
from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN
_CENTER_X = VIEWPORT_W / SCALE / 2.0  # 平台中心的世界 x 坐标


class _RelocatableLunarLander:
    """对 LunarLander 的薄包装：reset 后把飞船重定位到指定 (x, y)。

    LunarLander-v3 的初始位置在 reset() 内部硬编码（屏幕中心上方），无法通过
    参数传入。我们在 reset 之后用 Box2D 接口把 lander 与两条腿平移到目标起飞点，
    并清零速度，从而实现"调度器指定起飞点"。
    """

    def __init__(self, render: bool = False) -> None:
        gym = import_gym()
        self.env = gym.make(ENV_ID, render_mode="human" if render else None)
        self._unwrapped = self.env.unwrapped

    def reset_and_relocate(self, seed: int, init_x: float, init_y: float = 0.0) -> np.ndarray:
        """reset 后把飞船平移到相对平台中心 init_x 处。

        init_x : 相对平台中心的横向偏移（米，观测系），调度器据此错开各飞船起飞点。
        init_y : 对原生起飞高度的额外偏移（世界坐标，米）；默认 0 表示沿用原生高度。
                 我们不强行改高度——单飞船策略是在原生动力学下训练的，保持高度才能
                 让它正常工作，调度只干预横向位置。
        """
        obs, _ = self.env.reset(seed=seed)
        lander = self._unwrapped.lander
        cx = lander.position.x          # 原生世界 x（≈ 平台中心）
        native_y = lander.position.y    # 原生起飞高度，保留
        new_pos = (cx + init_x, native_y + init_y)
        lander.position = new_pos
        lander.linearVelocity = (0.0, 0.0)
        lander.angle = 0.0
        lander.angularVelocity = 0.0
        # 两条腿跟着平移到新位置（保持相对 lander 的关节关系）
        for leg in self._unwrapped.legs:
            leg.position = new_pos
            leg.linearVelocity = (0.0, 0.0)
        obs = self._read_obs()
        return obs

    def _read_obs(self) -> np.ndarray:
        u = self._unwrapped
        pos = u.lander.position
        vel = u.lander.linearVelocity
        state = [
            (pos.x - _CENTER_X) / SCALE,   # 相对平台中心的 x（米）
            (pos.y - (u.helipad_y + LEG_DOWN / SCALE)) / SCALE,
            vel.x / SCALE,
            vel.y / SCALE,
            u.lander.angle,
            u.lander.angularVelocity,
            1.0 if u.legs[0].ground_contact else 0.0,
            1.0 if u.legs[1].ground_contact else 0.0,
        ]
        assert np.all(np.isfinite(state))
        return np.array(state, dtype=np.float32)

    def step(self, action: int):
        return self.env.step(action)

    @property
    def unwrapped(self):
        return self._unwrapped

    def close(self):
        self.env.close()


# ----------------------------------------------------------------------------
# 单架飞船的结果记录
# ----------------------------------------------------------------------------
@dataclass
class LanderResult:
    index: int
    plan: ShipPlan
    landed: bool                 # 是否平稳落地（双足接地 + 速度小 + 在平台内）
    in_platform: bool            # 落点是否在平台 x 范围内
    collided: bool               # 是否与已降落飞船过近/碰撞
    final_x: float               # 最终落点 x（相对平台中心，米）
    final_y: float
    raw_return: float            # LunarLander 原生累计 reward
    bonus: float                 # 任务级额外奖惩（顺序/避碰/全局）
    steps: int


@dataclass
class EpisodeReport:
    num_landers: int
    per_lander: list[LanderResult] = field(default_factory=list)
    total_return: float = 0.0          # 所有飞船 reward 之和
    n_landed: int = 0                  # 平稳落地数量
    n_collided: int = 0
    all_success: bool = False          # 是否全部平稳且无碰撞


class SequentialMultiLanderEnv:
    """多个飞船顺序降落环境。"""

    def __init__(self, cfg: Optional[MultiLanderConfig] = None) -> None:
        self.cfg = cfg or MultiLanderConfig()
        self.scheduler = LanderScheduler(
            spacing=self.cfg.target_spacing,
            safe_dist=self.cfg.safe_dist,
            init_x_jitter=self.cfg.init_x_jitter,
            init_y=self.cfg.init_y,
            seed=self.cfg.seed,
        )
        self._policy: Optional[SingleLanderPolicy] = None
        self._env: Optional[_RelocatableLunarLander] = None

    # ------------------------------------------------------------------
    def _ensure_policy(self) -> SingleLanderPolicy:
        if self._policy is None:
            self._policy = SingleLanderPolicy(
                algorithm=self.cfg.algorithm,
                model_dir=self.cfg.model_dir,
                hidden_dim=self.cfg.hidden_dim,
                device=self.cfg.device,
            )
        return self._policy

    def _ensure_env(self) -> _RelocatableLunarLander:
        if self._env is None:
            self._env = _RelocatableLunarLander(render=self.cfg.render)
        return self._env

    # ------------------------------------------------------------------
    def run_one_lander(self, index: int, num: int,
                       landed_xs: list[float], seed: int) -> LanderResult:
        """控制单架飞船从起飞到结束，返回结果。"""
        cfg = self.cfg
        policy = self._ensure_policy()
        env = self._ensure_env()

        plan = self.scheduler.plan_next(index, num, landed_xs, mode=self.cfg.mode)
        obs = env.reset_and_relocate(seed=seed, init_x=plan.init_x, init_y=plan.init_y)

        raw_return = 0.0
        steps = 0
        terminated = truncated = False
        for steps in range(1, cfg.max_steps_per_lander + 1):
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            raw_return += float(reward)
            if terminated or truncated:
                break

        # 判定落地状态
        u = env.unwrapped
        rel_x = (u.lander.position.x - _CENTER_X) / SCALE
        speed = float(np.hypot(u.lander.linearVelocity.x, u.lander.linearVelocity.y) / SCALE)
        both_legs = u.legs[0].ground_contact and u.legs[1].ground_contact
        from .scheduler import PLATFORM_X_MIN, PLATFORM_X_MAX
        in_platform = PLATFORM_X_MIN <= rel_x <= PLATFORM_X_MAX
        # 平稳落地：双足接地 + 速度小 + 在平台内 + 非 game_over
        landed = bool(both_legs and speed < cfg.flat_vel_threshold and in_platform
                      and not getattr(u, "game_over", False))

        # 飞船间避碰：最终落点离最近已降落飞船 < safe_dist
        collided = False
        if landed_xs:
            min_dist = min(abs(rel_x - lx) for lx in landed_xs)
            collided = min_dist < cfg.safe_dist

        # 任务级奖惩
        bonus = 0.0
        if collided:
            bonus -= cfg.collision_penalty
        if landed and not collided:
            bonus += cfg.order_bonus          # 按顺序平稳降落
        # 若没平稳落地（坠毁/翻倒），原生 reward 已经很负，这里不再额外加

        final_x = rel_x
        final_y = (u.lander.position.y - (u.helipad_y + LEG_DOWN / SCALE)) / SCALE
        return LanderResult(
            index=index, plan=plan, landed=landed, in_platform=in_platform,
            collided=collided, final_x=final_x, final_y=final_y,
            raw_return=raw_return, bonus=bonus, steps=steps,
        )

    # ------------------------------------------------------------------
    def run_episode(self, seed: Optional[int] = None) -> EpisodeReport:
        """跑完整的多飞船顺序降落 episode。"""
        cfg = self.cfg
        ep_seed = seed if seed is not None else cfg.seed
        report = EpisodeReport(num_landers=cfg.num_landers)

        landed_xs: list[float] = []
        try:
            for k in range(cfg.num_landers):
                # 时序模式：前一架完成后物理清除，下一架进入时不带历史占用；
                # 空间模式：累积已降落落点，作为后续避障约束。
                active_landed = [] if cfg.mode == "sequential" else landed_xs
                res = self.run_one_lander(k, cfg.num_landers, active_landed, seed=ep_seed + k)
                report.per_lander.append(res)
                report.total_return += res.raw_return + res.bonus
                if res.landed and not res.collided:
                    report.n_landed += 1
                    landed_xs.append(res.final_x)   # 记录每架落点（供报告/展示）
                if res.collided:
                    report.n_collided += 1
            report.all_success = (report.n_landed == cfg.num_landers and report.n_collided == 0)
            if report.all_success:
                report.total_return += cfg.all_success_bonus
        finally:
            if self._env is not None:
                self._env.close()
                self._env = None
        return report

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
