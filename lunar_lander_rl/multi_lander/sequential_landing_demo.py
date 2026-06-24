"""拓展3（回归版）三飞船有序降落演示。

三艘飞船一开始都在悬停（力施加抵消重力，渲染时画喷火），然后依次降落到各自
固定落点（左/中/右）。降落用 goal_dqn（含悬停起点分布重训），悬停用 HoverForceController。
后降飞船靠落点错开 + 垂直路径避开已停泊飞船。

用法：
    python -m lunar_lander_rl.multi_lander.sequential_landing_demo \
        --gif outputs/multi_lander/sequential_landing.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .sequential_landing_env import SequentialLandingEnv


def _load_base_policy(model_path: str, device: str = "cpu"):
    """加载基础 DQN（8 维观测，回中器）。环境通过观测偏置让它落到不同 x。"""
    from lunar_lander_rl.common import mlp
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"基础 DQN 不存在：{model_path}")
    dev = torch.device(device)
    net = mlp(8, 4, 128).to(dev)
    net.load_state_dict(torch.load(p, map_location=dev))
    net.eval()
    def act(obs8):
        with torch.no_grad():
            t = torch.tensor(obs8, dtype=torch.float32, device=dev).unsqueeze(0)
            return int(net(t).argmax(1).item())
    return act


def run(gif_path: str = "outputs/multi_lander/sequential_landing.gif",
        model_path: str = "outputs/baseline/dqn/best_policy.pt",
        seed: int = 0, fps: int = 30, max_steps: int = 3000,
        device: str = "cpu") -> dict:
    import imageio.v2 as imageio

    env = SequentialLandingEnv(render_mode="rgb_array")
    policy = _load_base_policy(model_path, device)
    obs, _ = env.reset(seed=seed)
    frames = []
    results = []
    total_steps = 0

    while total_steps < max_steps:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        frames.append(env.render())
        if info.get("phase_advanced"):
            # 某艘刚停泊，记录并插几帧停顿
            idx = info["phase"]   # 刚停泊的是 info["phase"]（已推进前的）
            results.append({"index": info["phase"], "landed_x": env.landed_x[info["phase"]]})
            for _ in range(15):
                frames.append(env.render())
        total_steps += 1
        if term:
            break

    env.close()
    # 汇总每艘落点
    print(f"[sequential] 总步数={total_steps} 总帧数={len(frames)} "
          f"成功={env.success} phase={env.phase}/3")
    for i in range(3):
        lx = env.landed_x[i]
        if lx is not None:
            print(f"  飞船{i}: goal_x={env.GOAL_X[i]:+.2f} 落点={lx:+.2f} "
                  f"误差={abs(lx-env.GOAL_X[i]):.2f}")
        else:
            print(f"  飞船{i}: 未停泊 (goal_x={env.GOAL_X[i]:+.2f})")

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"[sequential] GIF -> {gif_path}")
    return {"success": env.success, "frames": len(frames),
            "landed": [x is not None for x in env.landed_x]}


def main():
    p = argparse.ArgumentParser(description="三飞船有序降落演示")
    p.add_argument("--gif", default="outputs/multi_lander/sequential_landing.gif")
    p.add_argument("--model", default="outputs/baseline/dqn/best_policy.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run(gif_path=args.gif, model_path=args.model, seed=args.seed,
        fps=args.fps, device=args.device)


if __name__ == "__main__":
    main()
