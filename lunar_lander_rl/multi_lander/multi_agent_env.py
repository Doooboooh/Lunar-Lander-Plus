"""拓展3 真多体环境：三艘飞船同时在同一 Box2D 世界里飞行 + 真实碰撞。

与之前的"轮流降落 + 贴标记"不同，这里是真正的多智能体物理：
  - 同一个 Box2D world 里创建 3 个 lander body + 各 2 条腿；
  - 每步用参数共享策略给每艘各选一个动作，同时施加推力；
  - 飞船之间通过修改 maskBits 实现真实碰撞；
  - MultiContactDetector 检测"不同飞船间接触"并判定撞击/接地。

继承 gymnasium LunarLander 以复用其 render()（遍历 self.drawlist 画所有 body）、
粒子系统、close()。重写 reset/step/_destroy。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import Box2D
from Box2D.b2 import (
    edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef,
    contactListener,
)
from gymnasium.envs.box2d.lunar_lander import (
    LunarLander, ContactDetector,
    VIEWPORT_W, VIEWPORT_H, SCALE, FPS,
    INITIAL_RANDOM, LANDER_POLY, LEG_W, LEG_H, LEG_AWAY, LEG_DOWN,
    LEG_SPRING_TORQUE, MAIN_ENGINE_POWER, MAIN_ENGINE_Y_LOCATION,
    SIDE_ENGINE_POWER, SIDE_ENGINE_AWAY, SIDE_ENGINE_HEIGHT,
)
from gymnasium.spaces import Box, Discrete

_CENTER_X = VIEWPORT_W / SCALE / 2.0
GOAL_LIMIT = 0.9
OBS_DIM = 23  # 8(自身) + 1(goal) + 7(船j) + 7(船k)

# 三艘飞船的颜色（便于同框区分）
SHIP_COLORS = [
    ((230, 102, 102), (128, 77, 77)),   # 红
    ((102, 200, 102), (77, 128, 77)),   # 绿
    ((102, 128, 230), (77, 77, 128)),   # 蓝
]


class MultiContactDetector(contactListener):
    """多体碰撞检测：识别哪艘的腿接地、哪些不同飞船间发生接触。"""

    def __init__(self, env: "MultiAgentLunarLander") -> None:
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact) -> None:
        a, b = contact.fixtureA.body, contact.fixtureB.body
        env = self.env
        # 1) 任何腿接触任意物体 -> 该腿 ground_contact=True
        for ld in env.landers:
            for leg in ld["legs"]:
                if leg is a or leg is b:
                    leg.ground_contact = True
        # 2) 不同飞船之间接触 -> 记录碰撞事件
        idA = env._body_owner(a)
        idB = env._body_owner(b)
        if idA is not None and idB is not None and idA != idB:
            a_is_leg = a in (env.landers[idA]["legs"][0], env.landers[idA]["legs"][1])
            b_is_leg = b in (env.landers[idB]["legs"][0], env.landers[idB]["legs"][1])
            impact = not (a_is_leg and b_is_leg)   # 任一是机身即算撞击
            env.collision_events.append({"pair": (idA, idB), "impact": impact})
            if impact:
                env.game_over = True

    def EndContact(self, contact) -> None:
        a, b = contact.fixtureA.body, contact.fixtureB.body
        for ld in self.env.landers:
            for leg in ld["legs"]:
                if leg is a or leg is b:
                    leg.ground_contact = False


class MultiAgentLunarLander(LunarLander):
    """三飞船同时同框 + 真实碰撞的多智能体 LunarLander。"""

    NUM_LANDERS = 3
    # 起飞位置错开（相对平台中心，米）+ 高度偏移
    START_X = [-0.9, 0.0, 0.9]
    START_Y_OFF = [0.4, 0.0, 0.4]

    def __init__(self, render_mode: Optional[str] = None, demo_mode: bool = False) -> None:
        # continuous=False（离散4动作），不开风
        super().__init__(render_mode=render_mode, continuous=False,
                         enable_wind=False)
        self.demo_mode = demo_mode
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space_single = Discrete(4)
        self.landers: list[dict] = []
        self.collision_events: list[dict] = []
        self.done_flags: list[bool] = []
        self.game_over = False
        # 占位 lander/legs 给父类（避免父类某些代码访问时报错）
        self.lander = None
        self.legs = []
        self.step_count = 0

    # ------------------------------------------------------------------
    def _destroy(self) -> None:
        """重写：销毁所有 lander/leg（父类只清单 lander）。"""
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        for ld in getattr(self, "landers", []):
            try:
                self.world.DestroyBody(ld["lander"])
            except Exception:
                pass
            for leg in ld["legs"]:
                try:
                    self.world.DestroyBody(leg)
                except Exception:
                    pass
        self.landers = []

    # ------------------------------------------------------------------
    def _build_terrain(self) -> None:
        """建地形（从父类 reset 抽出）。"""
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        for k in range(-2, 3):
            height[CHUNKS // 2 + k] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i] + height[i + 1]) for i in range(CHUNKS)]
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

    def _create_one_lander(self, idx: int, rel_x: float, rel_y_off: float, goal_x: float) -> dict:
        """创建一艘 lander + 2 条腿。位置错开，maskBits 让飞船间可碰。"""
        initial_y = VIEWPORT_H / SCALE + rel_y_off
        initial_x = _CENTER_X + rel_x
        c1, c2 = SHIP_COLORS[idx % len(SHIP_COLORS)]

        lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y), angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0, friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001 | 0x0010 | 0x0020,   # 地面 + 任意 lander + 任意 leg
                restitution=0.0,
            ),
        )
        lander.color1 = c1
        lander.color2 = c2
        lander.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
             self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)),
            True,
        )

        legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0, restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001 | 0x0010 | 0x0020,
                ),
            )
            leg.ground_contact = False
            leg.color1 = c1
            leg.color2 = c2
            rjd = revoluteJointDef(
                bodyA=lander, bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True, enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE, motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            legs.append(leg)

        return {"lander": lander, "legs": legs, "goal_x": float(goal_x),
                "prev_shaping": None, "alive": True, "index": idx}

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 跳过 LunarLander.reset（它末尾会调 self.step(0)，与多体 step 不兼容），
        # 直接用 gym.Env.reset 仅完成种子设置。
        from gymnasium import Env
        Env.reset(self, seed=seed)
        self._destroy()
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = MultiContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.collision_events = []
        self.done_flags = [False] * self.NUM_LANDERS
        self.step_count = 0

        self._build_terrain()
        goals = (options or {}).get("goals", [-0.7, 0.0, 0.7])
        self.landers = [
            self._create_one_lander(i, self.START_X[i], self.START_Y_OFF[i], goals[i])
            for i in range(self.NUM_LANDERS)
        ]
        # drawlist 含所有 lander + leg，供父类 render 同框绘制
        self.drawlist = []
        for ld in self.landers:
            self.drawlist += [ld["lander"]] + ld["legs"]

        if self.render_mode == "human":
            self.render()
        return [self._build_obs(i) for i in range(self.NUM_LANDERS)], {}

    # ------------------------------------------------------------------
    def _body_owner(self, body) -> Optional[int]:
        for i, ld in enumerate(self.landers):
            if body is ld["lander"] or body in ld["legs"]:
                return i
        return None

    def _apply_engine(self, idx: int, action: int) -> None:
        """对第 idx 艘施加引擎推力（搬父类 step 引擎逻辑）。"""
        ld = self.landers[idx]
        lander = ld["lander"]
        tip = (math.sin(lander.angle), math.cos(lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, 1.0) / SCALE for _ in range(2)]

        # 主发动机（action==2）
        if action == 2:
            m_power = 1.0
            ox = (tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                  + side[0] * dispersion[1])
            oy = (-tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                  - side[1] * dispersion[1])
            impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power,
                                      oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
            lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power,
                                       -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)

        # 侧发动机（action==1 左，action==3 右）
        if action in (1, 3):
            direction = action - 2
            s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (lander.position[0] + ox - tip[0] * 17 / SCALE,
                           lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power,
                                      oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)
            lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power,
                                       -oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)

    # ------------------------------------------------------------------
    def _state8(self, idx: int) -> np.ndarray:
        """第 idx 艘的标准 8 维状态（x 相对各自 goal）。"""
        ld = self.landers[idx]
        p = ld["lander"].position
        v = ld["lander"].linearVelocity
        s = [
            (p.x - _CENTER_X) / SCALE - ld["goal_x"],   # 相对自己目标
            (p.y - (self.helipad_y + LEG_DOWN / SCALE)) / SCALE,
            v.x / SCALE, v.y / SCALE,
            ld["lander"].angle, ld["lander"].angularVelocity,
            1.0 if ld["legs"][0].ground_contact else 0.0,
            1.0 if ld["legs"][1].ground_contact else 0.0,
        ]
        return np.array(s, dtype=np.float32)

    def _build_obs(self, idx: int) -> np.ndarray:
        """23 维观测：自身8 + goal1 + 另两艘各7。另两艘按水平距离排序保证置换不变。"""
        ld = self.landers[idx]
        me = ld["lander"].position
        me_v = ld["lander"].linearVelocity
        state8 = self._state8(idx)

        # 收集其他船的相对信息
        others = []
        for j, od in enumerate(self.landers):
            if j == idx:
                continue
            op = od["lander"].position
            ov = od["lander"].linearVelocity
            dx = (op.x - me.x) / SCALE
            dy = (op.y - me.y) / SCALE
            dvx = (ov.x - me_v.x) / SCALE
            dvy = (ov.y - me_v.y) / SCALE
            dangle = od["lander"].angle - ld["lander"].angle
            alive = 1.0 if od["alive"] else 0.0
            others.append((abs(dx), [dx, dy, dvx, dvy, dangle, alive, 0.0]))
        others.sort(key=lambda t: t[0])   # 按水平距离排序
        # 补足到 2 个槽位（NUM_LANDERS=3 时正好 2 个）
        while len(others) < 2:
            others.append((0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        obs = np.concatenate([
            state8,
            np.array([ld["goal_x"]], dtype=np.float32),
            np.array(others[0][1], dtype=np.float32),
            np.array(others[1][1], dtype=np.float32),
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    def _compute_reward(self, idx: int, action: int) -> tuple[float, float, float]:
        """返回 (reward, m_power, s_power)，用于扣燃料。"""
        ld = self.landers[idx]
        if not ld["alive"]:
            return 0.0, 0.0, 0.0
        state8 = self._state8(idx)
        shaping = (
            -100 * np.sqrt(state8[0] ** 2 + state8[1] ** 2)
            - 100 * np.sqrt(state8[2] ** 2 + state8[3] ** 2)
            - 100 * abs(state8[4])
            + 10 * state8[6] + 10 * state8[7]
        )
        if ld["prev_shaping"] is not None:
            reward = shaping - ld["prev_shaping"]
        else:
            reward = 0.0
        ld["prev_shaping"] = shaping

        # 燃料
        m_power = 1.0 if action == 2 else 0.0
        s_power = 1.0 if action in (1, 3) else 0.0
        reward -= m_power * 0.30 + s_power * 0.03

        # 与他船过近的连续惩罚（引导避让）
        me = ld["lander"].position
        safe = 0.9
        for j, od in enumerate(self.landers):
            if j == idx:
                continue
            op = od["lander"].position
            d = math.hypot(op.x - me.x, op.y - me.y) / SCALE
            if d < safe:
                reward -= (safe - d) / safe * 1.0
        return float(reward), m_power, s_power

    def _landed_cleanly(self, idx: int) -> bool:
        ld = self.landers[idx]
        if not ld["alive"]:
            return False
        p = ld["lander"].position
        v = ld["lander"].linearVelocity
        rel_x = (p.x - _CENTER_X) / SCALE
        speed = math.hypot(v.x, v.y) / SCALE
        both = ld["legs"][0].ground_contact and ld["legs"][1].ground_contact
        in_plat = -1.0 <= rel_x <= 1.0
        near_goal = abs(rel_x - ld["goal_x"]) < 0.5
        return bool(both and speed < 2.0 and in_plat and near_goal)

    # ------------------------------------------------------------------
    def step(self, actions: list[int]):
        self.step_count += 1
        # 施加推力（仅对 alive 的船）
        for i, a in enumerate(actions):
            if self.landers[i]["alive"]:
                self._apply_engine(i, int(a))
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # 计算每艘 reward
        rewards = [0.0] * self.NUM_LANDERS
        for i in range(self.NUM_LANDERS):
            rewards[i], _, _ = self._compute_reward(i, int(actions[i]))

        # 处理碰撞事件：撞击涉及的船判坠毁
        impacted_ships = set()
        for ev in self.collision_events:
            if ev["impact"]:
                impacted_ships.update(ev["pair"])
        for i in impacted_ships:
            if self.landers[i]["alive"]:
                rewards[i] = -100.0
                self.landers[i]["alive"] = False

        # 平稳落地
        terminated = False
        for i in range(self.NUM_LANDERS):
            if self._landed_cleanly(i) and self.landers[i]["alive"]:
                self.landers[i]["alive"] = False
                self.done_flags[i] = True
                rewards[i] += 250.0          # 提高落地奖励，强激励降落（而非悬停）

        # 出界 / 坠落到非平台 -> 失败
        for i in range(self.NUM_LANDERS):
            if self.landers[i]["alive"]:
                p = self.landers[i]["lander"].position
                rel_x = (p.x - _CENTER_X) / SCALE
                if abs(rel_x) >= 1.5:
                    self.landers[i]["alive"] = False
                    rewards[i] -= 50.0

        # 全部平稳 = 全局成功
        all_success = all(self.done_flags) and not impacted_ships
        if all_success:
            for i in range(self.NUM_LANDERS):
                rewards[i] += 50.0
            terminated = True

        # 是否还有活的船
        any_alive = any(ld["alive"] for ld in self.landers)
        truncated = self.step_count >= 1000
        if self.game_over:
            terminated = True
        if not any_alive:
            terminated = True

        if self.demo_mode:
            terminated = False   # 演示模式：即使撞了也继续到所有船落地

        if self.render_mode == "human":
            self.render()

        obs = [self._build_obs(i) for i in range(self.NUM_LANDERS)]
        info = {"collision_events": self.collision_events, "all_success": all_success,
                "done_flags": list(self.done_flags)}
        self.collision_events = []   # 每步清空事件列表（已传入 info）
        return obs, rewards, terminated, truncated, info
