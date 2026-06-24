"""拓展3 舰队演示：三艘飞船在同一画面里依次降落、并排停泊。

视觉效果
--------
解决"画面里只有一艘"的问题：每架飞船独立跑完整降落过程并渲染；落地后把它的落点
记录下来，在后续每一帧上用 pygame 画出紫色飞船标记，表示"这架已停泊在此"。
于是观众看到：
  - 第一架从上方飞下、落地，停在画面里；
  - 第二架飞下，第一架还在原位，第二架落在旁边；
  - 第三架飞下，前两架都在，第三架落定；
  - 最终画面：三艘飞船并排停泊在平台上。

控制策略
--------
优先用目标条件策略（outputs/goal_dqn/best_policy.pt，9 维观测），让每架落到不同的
指定目标点，实现真正的"并排"。若该策略不存在，回退基础 DQN（8 维，回中）。

用法（项目根目录，venv 已激活）：
    python -m lunar_lander_rl.multi_lander.fleet_demo --num-landers 3 \
        --gif outputs/fleet_landing.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN
from gymnasium.envs.registration import register

_CENTER_X = VIEWPORT_W / SCALE / 2.0

try:
    register(id="GoalLunarLander-v0",
             entry_point="lunar_lander_rl.multi_lander.goal_env:GoalConditionedLunarLander")
except Exception:
    pass


def _load_goal_policy(model_path: str, device: str = "cpu"):
    from lunar_lander_rl.multi_lander.goal_conditioned_dqn import mlp
    p = Path(model_path)
    if not p.exists():
        return None
    dev = torch.device(device)
    net = mlp(9, 4, 128).to(dev)
    net.load_state_dict(torch.load(p, map_location=dev))
    net.eval()
    def act(obs9):
        with torch.no_grad():
            t = torch.tensor(obs9, dtype=torch.float32, device=dev).unsqueeze(0)
            return int(net(t).argmax(1).item())
    return act


def _load_base_policy(model_dir: str, device: str = "cpu"):
    from lunar_lander_rl.common import mlp
    dev = torch.device(device)
    net = mlp(8, 4, 128).to(dev)
    net.load_state_dict(torch.load(Path(model_dir) / "best_policy.pt", map_location=dev))
    net.eval()
    def act(obs8):
        with torch.no_grad():
            t = torch.tensor(obs8, dtype=torch.float32, device=dev).unsqueeze(0)
            return int(net(t).argmax(1).item())
    return act


def _read_obs8(u) -> np.ndarray:
    p = u.lander.position
    v = u.lander.linearVelocity
    return np.array([
        (p.x - _CENTER_X) / SCALE,
        (p.y - (u.helipad_y + LEG_DOWN / SCALE)) / SCALE,
        v.x / SCALE, v.y / SCALE,
        u.lander.angle, u.lander.angularVelocity,
        1.0 if u.legs[0].ground_contact else 0.0,
        1.0 if u.legs[1].ground_contact else 0.0,
    ], dtype=np.float32)


def _stamp_parked(frame: np.ndarray, parked: list[tuple[float, float]]) -> np.ndarray:
    """在帧上画出所有已停泊飞船的紫色标记，保证它们始终可见、位置精确。"""
    import pygame
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    BODY = (128, 102, 230)
    for (rx, ry) in parked:
        # rel_x 是相对平台中心的米数；平台中心像素 = VIEWPORT_W/2
        px = int(VIEWPORT_W / 2 + rx * SCALE)
        # 平台地面像素 ≈ VIEWPORT_H*3/4（LunarLander 的 helipad_y 对应这里）
        py = int(VIEWPORT_H * 3 / 4 - ry * SCALE)
        pygame.draw.polygon(surf, BODY, [(px, py - 14), (px - 10, py + 6), (px + 10, py + 6)])
        pygame.draw.line(surf, BODY, (px - 10, py + 6), (px - 14, py + 14), 2)
        pygame.draw.line(surf, BODY, (px + 10, py + 6), (px + 14, py + 14), 2)
        pygame.draw.line(surf, BODY, (px - 14, py + 14), (px - 14, py + 18), 2)
        pygame.draw.line(surf, BODY, (px + 14, py + 14), (px + 14, py + 18), 2)
    return pygame.surfarray.array3d(surf).swapaxes(0, 1)


def run_fleet(num_landers: int = 3, gif_path: Optional[str] = None,
              seed: int = 0, fps: int = 30,
              goal_model: str = "outputs/goal_dqn/best_policy.pt",
              base_model_dir: str = "outputs/dqn",
              max_steps: int = 1000) -> dict:
    goal_policy = _load_goal_policy(goal_model)
    use_goal = goal_policy is not None
    if use_goal:
        print(f"[fleet] 使用目标条件策略 {goal_model}（每架落到指定点，真并排）")
        base_policy = None
    else:
        print(f"[fleet] 未找到目标条件策略，回退基础 DQN {base_model_dir}（回中现象）")
        base_policy = _load_base_policy(base_model_dir)

    # 目标 x：围绕 0 对称排开
    spacing = 0.55
    targets = []
    for i in range(num_landers):
        sign = 1 if i % 2 == 1 else -1
        mag = (i + 1) // 2
        targets.append(sign * mag * spacing)

    env = gym.make("GoalLunarLander-v0", render_mode="rgb_array")
    u = env.unwrapped
    frames: list[np.ndarray] = []
    parked: list[tuple[float, float]] = []
    results = []

    try:
        for k in range(num_landers):
            target = targets[k]
            u.target_goal_x = target
            env.reset(seed=seed + k)

            obs8 = _read_obs8(u)
            obs_goal = np.append(obs8, target).astype(np.float32) if use_goal else None

            for step in range(max_steps):
                if use_goal:
                    action = goal_policy(obs_goal)
                    obs_goal, _, term, trunc, _ = env.step(action)
                else:
                    action = base_policy(obs8)
                    obs_ret, _, term, trunc, _ = env.step(action)
                    obs8 = obs_ret[:8].astype(np.float32)
                frames.append(_stamp_parked(env.render(), parked))
                if term or trunc:
                    break

            rel_x = (u.lander.position.x - _CENTER_X) / SCALE
            rel_y = (u.lander.position.y - (u.helipad_y + LEG_DOWN / SCALE)) / SCALE
            speed = float(np.hypot(u.lander.linearVelocity.x, u.lander.linearVelocity.y) / SCALE)
            both = u.legs[0].ground_contact and u.legs[1].ground_contact
            in_plat = -1.0 <= rel_x <= 1.0
            landed = bool(both and speed < 3.0 and in_plat)
            err = abs(rel_x - target)
            results.append({"index": k, "target": target, "landed_x": rel_x,
                            "error": err, "landed": landed})
            print(f"  飞船{k}: 目标x={target:+.2f} 落点x={rel_x:+.2f} "
                  f"误差={err:.2f} {'平稳落地' if landed else '未平稳'}")
            # 停泊展示位置用 target（让三艘并排错开显示），y 用真实落点高度（接近地面）
            parked.append((target, rel_y))
            # 停几帧便于观察这架到位（此时这架自己也被画成标记了）
            last = _stamp_parked(env.render(), parked)
            for _ in range(15):
                frames.append(last)
    finally:
        env.close()

    n_landed = sum(r["landed"] for r in results)
    print(f"\n[fleet] 平稳落地 {n_landed}/{num_landers}")

    if gif_path:
        import imageio.v2 as imageio
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"[fleet] 已保存舰队降落 GIF -> {gif_path}（{len(frames)} 帧）")

    return {"results": results, "n_landed": n_landed, "num_landers": num_landers,
            "policy": "goal_conditioned" if use_goal else "base_dqn"}


def main():
    p = argparse.ArgumentParser(description="拓展3 舰队降落演示（三舰同框并排停泊）")
    p.add_argument("--num-landers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--goal-model", default="outputs/goal_dqn/best_policy.pt")
    p.add_argument("--base-model-dir", default="outputs/dqn")
    args = p.parse_args()
    run_fleet(num_landers=args.num_landers, gif_path=args.gif, seed=args.seed,
              fps=args.fps, goal_model=args.goal_model,
              base_model_dir=args.base_model_dir)


if __name__ == "__main__":
    main()
