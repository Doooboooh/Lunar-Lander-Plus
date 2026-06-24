"""MAPPO 三飞船演示：加载训练好的分散 actor，三船同时同框飞行 + 真实碰撞，生成 GIF。

每艘飞船用同一个 actor（参数共享）+ 自己的局部观测选动作。三船在同一 Box2D 世界
里同时飞，互碰=坠毁。生成三船同框降落或碰撞的 GIF。

用法：
    python -m lunar_lander_rl.multi_lander.mappo_demo --gif outputs/multi_lander/mappo_landing.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .multi_agent_env import MultiAgentLunarLander, OBS_DIM, N_AGENTS, ACTION_DIM
from .mappo import ActorNet


def _load_actor(model_path: str, device: str = "cpu"):
    dev = torch.device(device)
    net = ActorNet(OBS_DIM, ACTION_DIM, 128).to(dev)
    net.load_state_dict(torch.load(model_path, map_location=dev))
    net.eval()
    def act_all(obs_list):
        with torch.no_grad():
            t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=dev)
            return net(t).argmax(dim=1).cpu().numpy().tolist()
    return act_all


def run(gif_path: str = "outputs/multi_lander/mappo_landing.gif",
        model_path: str = "outputs/multi_lander/mappo/best_actor.pt",
        seed: int = 0, fps: int = 30, max_steps: int = 1000,
        device: str = "cpu") -> dict:
    import imageio.v2 as imageio
    env = MultiAgentLunarLander(render_mode="rgb_array", demo_mode=True)
    policy = _load_actor(model_path, device)
    obs_list, _ = env.reset(seed=seed)
    frames = []
    impact_count = 0
    for t in range(max_steps):
        actions = policy(obs_list)
        obs_list, rewards, term, trunc, info = env.step(actions)
        frames.append(env.render())
        for ev in info.get("collision_events", []):
            if ev.get("impact"):
                impact_count += 1
        if not any(ld["alive"] for ld in env.landers):
            for _ in range(20):
                frames.append(env.render())
            break
    env.close()
    landed = sum(info.get("done_flags", []))
    print(f"[MAPPO-demo] {len(frames)}帧 撞击={impact_count} 平稳落地={landed}/3 -> {gif_path}")
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)
    return {"frames": len(frames), "impacts": impact_count, "landed": landed}


def main():
    p = argparse.ArgumentParser(description="MAPPO 三飞船演示")
    p.add_argument("--gif", default="outputs/multi_lander/mappo_landing.gif")
    p.add_argument("--model", default="outputs/multi_lander/mappo/best_actor.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run(gif_path=args.gif, model_path=args.model, seed=args.seed, fps=args.fps, device=args.device)


if __name__ == "__main__":
    main()
