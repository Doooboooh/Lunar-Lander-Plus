"""悬停控制器（力施加版，确定性，不学习）。

之前用离散动作 bang-bang 控主/侧推，但离散动作很难精确抵消重力（主推冲量与重力
接近，时序一错位飞船就大幅下沉或飞走）。实测最可靠的悬停方式：直接给悬停飞船的
每个 body（lander + 2 legs）施加等于其重力的向上力（= 主动喷火抵消重力的物理等价），
再加一个水平速度阻尼力防止漂移。

这样悬停飞船物理上稳定悬停，视觉上由环境的渲染层给它画喷火特效（见
sequential_landing_env.render 的喷火绘制），既满足"画面看到喷火"，又保证稳定。

自检：单船悬停 300 步，rel_y/rel_x 漂移在 ±0.4m 内。
"""
from __future__ import annotations

from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, SCALE, LEG_DOWN

_CENTER_X = VIEWPORT_W / SCALE / 2.0
GRAVITY = 10.0
HORIZ_DAMP = 5.0   # 水平阻尼系数（越大回正越快）


class HoverForceController:
    """对一艘悬停飞船的 lander+legs 施加反重力力 + 水平阻尼，使其悬停在 hover_x。"""

    def __init__(self, hover_x: float, gravity: float = GRAVITY, damp: float = HORIZ_DAMP):
        self.hover_x = float(hover_x)
        self.gravity = float(gravity)
        self.damp = float(damp)

    def apply(self, lander_body, legs, world_center_x: float = _CENTER_X) -> None:
        """在 world.Step 之前调用：给 lander 与每条腿施加向上补偿力 + 水平阻尼。"""
        # 反重力：每个 body 施加 m*g 向上
        lander_body.ApplyForceToCenter((0.0, lander_body.mass * self.gravity), True)
        for leg in legs:
            leg.ApplyForceToCenter((0.0, leg.mass * self.gravity), True)
        # 水平：拉回 hover_x 的弹簧力 + 速度阻尼
        rel_x = lander_body.position.x - world_center_x - self.hover_x
        total_mass = lander_body.mass + sum(l.mass for l in legs)
        # 弹簧（拉回目标 x）+ 阻尼（抑制速度）
        fx = -self.damp * lander_body.linearVelocity.x * total_mass * 0.2 - 6.0 * rel_x * total_mass
        lander_body.ApplyForceToCenter((fx, 0.0), True)


# ----------------------------------------------------------------------
def _self_check():
    """单船悬停自检：标准 LunarLander-v3，定位到高空，用 HoverForceController 悬停 300 步。"""
    import numpy as np
    import gymnasium as gym

    env = gym.make("LunarLander-v3")
    env.reset(seed=0)
    u = env.unwrapped
    hover_x, hover_y = 0.0, 1.5
    u.lander.position = (_CENTER_X + hover_x, u.helipad_y + LEG_DOWN / SCALE + hover_y)
    u.lander.linearVelocity = (0.0, 0.0)

    ctrl = HoverForceController(hover_x)
    rel_xs, rel_ys = [], []
    for _ in range(300):
        ctrl.apply(u.lander, u.legs)
        env.step(0)
        p = u.lander.position
        rel_xs.append((p.x - _CENTER_X) - hover_x)
        rel_ys.append((p.y - (u.helipad_y + LEG_DOWN / SCALE)) - hover_y)
    env.close()

    rel_xs, rel_ys = np.array(rel_xs), np.array(rel_ys)
    print(f"[hover self-check] 300 步力施加悬停")
    print(f"  rel_x: mean={rel_xs.mean():+.3f} std={rel_xs.std():.3f} "
          f"range=[{rel_xs.min():+.3f},{rel_xs.max():+.3f}]")
    print(f"  rel_y: mean={rel_ys.mean():+.3f} std={rel_ys.std():.3f} "
          f"range=[{rel_ys.min():+.3f},{rel_ys.max():+.3f}]")
    ok = (abs(rel_xs).max() < 0.4 and abs(rel_ys).max() < 0.5)
    print(f"  => 判据(|rel_x|<0.4, |rel_y|<0.5): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _self_check()
