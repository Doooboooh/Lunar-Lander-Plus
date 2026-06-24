"""最小可行验证：三飞船同时同框 + 真实碰撞的物理与渲染。

不训练。故意让三船乱飞（随机动作 + 有时全开主发动机），最大化碰撞概率。
验收标准：
  1. GIF 同一帧能同时看到三艘飞船（红/绿/蓝）。
  2. 至少出现一次 collision_events（不同飞船间接触被检测到）。
  3. 物理不崩（观测全 finite）。
  4. 渲染不报错。

用法：
    python -m lunar_lander_rl.multi_lander.collision_smoke_test
"""
from __future__ import annotations

import numpy as np

from .multi_agent_env import MultiAgentLunarLander


def main() -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("需要 imageio：pip install imageio") from exc

    env = MultiAgentLunarLander(render_mode="rgb_array", demo_mode=True)
    obs, _ = env.reset(seed=0, options={"goals": [-0.6, 0.0, 0.6]})
    print(f"[smoke] 初始 obs 数量={len(obs)}, 每艘维度={obs[0].shape}")

    frames = []
    collision_count = 0
    finite_ok = True
    rng = np.random.default_rng(1)

    for t in range(250):
        # 随机动作，但偏向主发动机/侧推以制造碰撞
        actions = []
        for _ in range(env.NUM_LANDERS):
            if rng.random() < 0.5:
                actions.append(int(rng.choice([1, 2, 3])))   # 乱飞
            else:
                actions.append(2)   # 主发动机
        obs, rewards, term, trunc, info = env.step(actions)
        frame = env.render()
        frames.append(frame)
        if info["collision_events"]:
            collision_count += len(info["collision_events"])
            print(f"  t={t:3d} 碰撞! events={info['collision_events']} rewards={[round(r,1) for r in rewards]}")
        # 检查 finite
        for o in obs:
            if not np.all(np.isfinite(o)):
                finite_ok = False
                print(f"  t={t} 出现非 finite 观测!")
                break
        if not any(ld["alive"] for ld in env.landers):
            print(f"  t={t} 所有船已结束（alive 全 False）")
            # 多渲染几帧看落定状态后退出
            for _ in range(10):
                frames.append(env.render())
            break

    env.close()
    imageio.mimsave("outputs/multi_collision_smoke.gif", frames, fps=30)
    print(f"\n[smoke] 完成。帧数={len(frames)} 碰撞事件总数={collision_count} finite_ok={finite_ok}")
    print(f"[smoke] GIF -> outputs/multi_collision_smoke.gif")
    if collision_count == 0:
        print("[smoke] ⚠️ 未检测到任何碰撞，需检查 maskBits/ContactDetector")
    if not finite_ok:
        print("[smoke] ⚠️ 物理出现 NaN，需检查")


if __name__ == "__main__":
    main()
