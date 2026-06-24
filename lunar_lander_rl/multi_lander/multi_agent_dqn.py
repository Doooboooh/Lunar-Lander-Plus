"""拓展3 多智能体 DQN 训练：参数共享策略同时控制三艘飞船。

每步对每艘飞船各前向一次（同一网络，参数共享），收集 3 条 transition 入同一个
ReplayBuffer。观测 23 维（自身8+goal1+他船14）。动作每艘独立 4 离散。

用法：
    python -m lunar_lander_rl.multi_lander.multi_agent_dqn --episodes 600
"""
from __future__ import annotations

import argparse
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import ensure_dir, get_device, save_history, save_json, set_seed
from .multi_agent_env import MultiAgentLunarLander, OBS_DIM

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.array([x.state for x in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array([x.next_state for x in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([x.done for x in batch], dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class MA_DQNConfig:
    episodes: int = 600
    max_steps: int = 1000
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    replay_size: int = 200_000
    warmup_steps: int = 2000
    target_update: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000
    seed: int = 42
    eval_episodes: int = 5
    device: str = "auto"
    output_dir: str = "outputs/multi_agent_dqn"


def epsilon_by_step(step: int, cfg: MA_DQNConfig) -> float:
    return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-step / cfg.epsilon_decay_steps)


def evaluate(policy_net, device, episodes: int, seed: int) -> dict:
    """评估：随机目标下三船的平均 reward、平稳落地数、撞击数。"""
    env = MultiAgentLunarLander(render_mode=None)
    rng = np.random.default_rng(seed)
    tot_returns, n_landed, n_impacts = [], [], []
    for ep in range(episodes):
        goals = [float(rng.uniform(-0.7, 0.7)) for _ in range(3)]
        obs_list, _ = env.reset(seed=seed + ep, options={"goals": goals})
        ep_rewards = [0.0, 0.0, 0.0]
        ep_impact = 0
        for _ in range(1000):
            actions = []
            with torch.no_grad():
                for o in obs_list:
                    t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    actions.append(int(policy_net(t).argmax(1).item()))
            obs_list, rewards, term, trunc, info = env.step(actions)
            for i in range(3):
                ep_rewards[i] += rewards[i]
            for ev in info["collision_events"]:
                if ev["impact"]:
                    ep_impact += 1
            if term or trunc:
                break
        tot_returns.append(float(np.mean(ep_rewards)))
        n_landed.append(sum(info["done_flags"]))
        n_impacts.append(ep_impact)
    env.close()
    return {"mean_return": float(np.mean(tot_returns)),
            "mean_landed": float(np.mean(n_landed)),
            "mean_impacts": float(np.mean(n_impacts))}


def train(cfg: MA_DQNConfig) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[MA-DQN] device={device}, obs_dim={OBS_DIM}")
    output_dir = ensure_dir(cfg.output_dir)

    policy = mlp(OBS_DIM, 4, cfg.hidden_dim).to(device)
    target = mlp(OBS_DIM, 4, cfg.hidden_dim).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)
    env = MultiAgentLunarLander(render_mode=None)

    global_step = 0
    best_score = float("-inf")
    rng = np.random.default_rng(cfg.seed)
    history: list[dict] = []

    try:
        for episode in range(1, cfg.episodes + 1):
            goals = [float(rng.uniform(-0.7, 0.7)) for _ in range(3)]
            obs_list, _ = env.reset(seed=cfg.seed + episode, options={"goals": goals})
            ep_rewards = [0.0, 0.0, 0.0]

            for step in range(cfg.max_steps):
                epsilon = epsilon_by_step(global_step, cfg)
                actions = []
                for o in obs_list:
                    if random.random() < epsilon:
                        actions.append(env.action_space_single.sample())
                    else:
                        with torch.no_grad():
                            t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                            actions.append(int(policy(t).argmax(1).item()))
                next_obs_list, rewards, term, trunc, info = env.step(actions)
                done = term or trunc
                # 每艘各 push 一条 transition（参数共享）
                for i in range(3):
                    replay.push(obs_list[i], actions[i], rewards[i], next_obs_list[i], done)
                obs_list = next_obs_list
                for i in range(3):
                    ep_rewards[i] += rewards[i]
                global_step += 1

                if len(replay) >= cfg.warmup_steps:
                    s_b, a_b, r_b, ns_b, d_b = replay.sample(cfg.batch_size, device)
                    q = policy(s_b).gather(1, a_b)
                    with torch.no_grad():
                        nq = target(ns_b).max(1, keepdim=True).values
                        tgt = r_b + cfg.gamma * (1 - d_b) * nq
                    loss = F.smooth_l1_loss(q, tgt)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()
                if global_step % cfg.target_update == 0:
                    target.load_state_dict(policy.state_dict())
                if term or trunc:
                    break

            history.append({"episode": episode, "mean_return": float(np.mean(ep_rewards)),
                            "epsilon": epsilon})
            if episode == 1 or episode % 20 == 0:
                print(f"[MA-DQN] ep={episode:04d} mean_return={np.mean(ep_rewards):7.1f} eps={epsilon:.3f}")

            if episode % 40 == 0:
                m = evaluate(policy, device, cfg.eval_episodes, cfg.seed + 10000)
                score = m["mean_return"] + 20 * m["mean_landed"] - 30 * m["mean_impacts"]
                print(f"   eval: return={m['mean_return']:.1f} landed={m['mean_landed']:.1f}/3 "
                      f"impacts={m['mean_impacts']:.1f} score={score:.1f}")
                if score > best_score:
                    best_score = score
                    torch.save(policy.state_dict(), output_dir / "best_policy.pt")
    finally:
        env.close()

    save_history(history, output_dir)
    save_json({"algorithm": "MultiAgent-shared-DQN", "config": cfg,
               "best_score": best_score}, output_dir / "metrics.json")
    print(f"[MA-DQN] done. best_score={best_score:.1f} -> {output_dir}/best_policy.pt")
    return {"best_score": best_score}


def parse_args() -> MA_DQNConfig:
    p = argparse.ArgumentParser(description="Multi-agent shared DQN (拓展3).")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="outputs/multi_agent_dqn")
    a = p.parse_args()
    return MA_DQNConfig(episodes=a.episodes, seed=a.seed, device=a.device, output_dir=a.output_dir)


if __name__ == "__main__":
    train(parse_args())
