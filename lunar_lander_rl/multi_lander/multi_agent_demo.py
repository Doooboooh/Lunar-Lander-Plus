"""拓展3 多智能体演示：三舰同时飞 + 真实碰撞，生成 GIF。

两种模式：
  --mode success：加载训练好的共享策略，三舰主动避让、各自落到不同目标点。
  --mode crash  ：弱策略 + 重叠目标，触发真实 Box2D 碰撞，demo_mode 让物理继续，
                  展示三船撞在一起缠坠的场面。

用法：
    python -m lunar_lander_rl.multi_lander.multi_agent_demo --mode success --gif outputs/multi_agent_success.gif
    python -m lunar_lander_rl.multi_lander.multi_agent_demo --mode crash --gif outputs/multi_agent_crash.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .multi_agent_env import MultiAgentLunarLander, OBS_DIM
from .multi_agent_dqn import mlp


def _load_policy(model_path: str, device: str = "cpu"):
    dev = torch.device(device)
    net = mlp(OBS_DIM, 4, 128).to(dev)
    net.load_state_dict(torch.load(model_path, map_location=dev))
    net.eval()
    def act(obs):
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            return int(net(t).argmax(1).item())
    return act


def run_demo(mode: str = "success", gif_path: str = "outputs/multi_agent_success.gif",
             model_path: str = "outputs/multi_agent_dqn/best_policy.pt",
             seed: int = 0, fps: int = 30, max_steps: int = 1200,
             device: str = "cpu") -> dict:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("需要 imageio：pip install imageio") from exc

    env = MultiAgentLunarLander(render_mode="rgb_array", demo_mode=True)

    if mode == "success":
        # 训练好的策略，三舰分散目标，主动避让
        policy = _load_policy(model_path, device)
        goals = [-0.7, 0.0, 0.7]
        print(f"[demo/success] 加载策略 {model_path}，目标={goals}")
    else:
        # crash 模式：用贪婪策略但故意给重叠目标，逼出碰撞
        policy = _load_policy(model_path, device) if Path(model_path).exists() else None
        # 重叠目标：三舰都想落中心，必然挤碰
        goals = [0.0, 0.1, -0.1]
        print(f"[demo/crash] 重叠目标={goals}，制造碰撞")

    obs_list, _ = env.reset(seed=seed, options={"goals": goals})
    frames = []
    impact_count = 0
    rng = np.random.default_rng(seed + 99)

    for t in range(max_steps):
        if policy is not None:
            actions = [policy(o) for o in obs_list]
        else:
            # 无策略：乱飞制造碰撞
            actions = [int(rng.choice([1, 2, 3])) for _ in range(3)]
        # crash 模式下偶尔扰动增加碰撞概率
        if mode == "crash" and rng.random() < 0.3:
            actions = [int(rng.choice([1, 2, 3])) for _ in actions]

        obs_list, rewards, term, trunc, info = env.step(actions)
        frames.append(env.render())
        for ev in info["collision_events"]:
            if ev["impact"]:
                impact_count += 1
        # 全部船结束就停
        if not any(ld["alive"] for ld in env.landers):
            for _ in range(20):
                frames.append(env.render())
            break

    env.close()
    landed = sum(info["done_flags"])
    print(f"[demo/{mode}] 帧数={len(frames)} 撞击次数={impact_count} 平稳落地={landed}/3")

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"[demo/{mode}] 已保存 -> {gif_path}")
    return {"frames": len(frames), "impacts": impact_count, "landed": landed, "mode": mode}


def main():
    p = argparse.ArgumentParser(description="拓展3 多智能体演示（三舰同框+碰撞）")
    p.add_argument("--mode", choices=["success", "crash"], default="success")
    p.add_argument("--gif", default="outputs/multi_agent_success.gif")
    p.add_argument("--model", default="outputs/multi_agent_dqn/best_policy.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run_demo(mode=args.mode, gif_path=args.gif, model_path=args.model,
             seed=args.seed, fps=args.fps, device=args.device)


if __name__ == "__main__":
    main()
