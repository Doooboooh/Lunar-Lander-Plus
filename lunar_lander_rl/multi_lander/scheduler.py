"""调度层：为每架飞船规划"目标落点"和"初始位置"，避开已降落飞船。

设计思路
--------
LunarLander-v3 的飞船从屏幕上方（x≈0, y≈1.4）自由下落，最终落在 x∈[-1,1]、
y≈0 的平台上。多架飞船顺序降落时，后一架不能落在前一架的位置上，否则就碰撞。

这里用一个简单而显式的调度规则（不学习，作为"调度策略"的基线）：
  1. 给每架飞船分配一个"目标落点 x"：围绕 0 对称排开，间距 target_spacing。
     例如 num=3 -> 目标 x = [-0.55, 0, +0.55]。
  2. 调度器在安排下一架时，再根据"已降落飞船"的落点做一次最近邻避让：
     如果目标 x 离任何已降落飞船 < safe_dist，就把它沿 x 推远，直到满足约束
     或到达平台边界（超出边界则记为"无安全落点"，留给奖励函数惩罚）。
  3. 初始 x 给一点随机扰动，模拟"调度 + 控制"的联合难度。

这个调度器是可替换的：报告里可以换成"扫描线""贪心最大间距"等更复杂策略做对比。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# LunarLander 平台大致范围（米）。超出则视为飞出平台。
PLATFORM_X_MIN, PLATFORM_X_MAX = -1.0, 1.0


@dataclass
class ShipPlan:
    """单架飞船的调度计划。"""
    index: int                 # 顺序序号（0-based）
    target_x: float            # 期望落点的 x
    init_x: float              # 初始 x（起飞点）
    init_y: float              # 初始 y（起飞高度）
    note: str = ""             # 调度备注（如"被避让推远""超出平台"）


class LanderScheduler:
    """顺序降落调度器：依次为每架飞船生成起飞点与目标落点。"""

    def __init__(self, spacing: float = 0.55, safe_dist: float = 0.45,
                 init_x_jitter: float = 0.6, init_y: float = 1.4,
                 seed: int = 0) -> None:
        self.spacing = spacing
        self.safe_dist = safe_dist
        self.init_x_jitter = init_x_jitter
        self.init_y = init_y
        self._rng = np.random.default_rng(seed)

    def _base_targets(self, num: int) -> list[float]:
        """围绕 0 对称排开的目标 x 列表。"""
        offsets = []
        # 0, +1, -1, +2, -2 ... 对称交错，保证中间飞船最先落
        for i in range(num):
            sign = 1 if i % 2 == 1 else -1
            mag = (i + 1) // 2
            offsets.append(sign * mag * self.spacing)
        return offsets

    def plan_next(self, index: int, num: int,
                  landed_xs: Sequence[float], mode: str = "sequential") -> ShipPlan:
        """为第 index 架飞船（共 num 架）生成起飞点与（记录用）目标落点。

        mode:
          - "sequential"（默认，时序错开）：起飞 x 给随机扰动，落点交给策略自然产生。
            因预训练策略回中，多架靠"依次进入/离开"时序错开，不依赖空间分离。
          - "spatial"（空间错开，需重训的目标条件策略）：分配错开的目标 x 并据此选起飞点。
        """
        note = ""
        if mode == "spatial":
            base = self._base_targets(num)
            target_x = base[index] if index < len(base) else 0.0
            # 避让已降落飞船
            if landed_xs:
                arr = np.asarray(landed_xs, dtype=float)
                for _ in range(20):
                    nearest = arr[np.argmin(np.abs(arr - target_x))]
                    if abs(target_x - nearest) < self.safe_dist:
                        push = self.safe_dist - abs(target_x - nearest) + 0.05
                        target_x += push if target_x >= nearest else -push
                        note = "避让已降落飞船"
                    else:
                        break
            if target_x > PLATFORM_X_MAX or target_x < PLATFORM_X_MIN:
                target_x = float(np.clip(target_x, PLATFORM_X_MIN, PLATFORM_X_MAX))
                note = "目标点超出平台，已夹紧"
            init_x = float(np.clip(
                target_x + self._rng.uniform(-self.init_x_jitter, self.init_x_jitter),
                PLATFORM_X_MIN - 0.5, PLATFORM_X_MAX + 0.5,
            ))
        else:
            # 时序错开：起飞点小幅随机偏移，目标记录为中心（策略会回中）
            target_x = 0.0
            init_x = float(self._rng.uniform(-self.init_x_jitter, self.init_x_jitter))
            note = "时序错开：随机入射，策略自然回中"

        return ShipPlan(index=index, target_x=target_x,
                        init_x=init_x, init_y=self.init_y, note=note)
