"""目标条件 DQN 训练脚本（拓展3）。

在 GoalConditionedLunarLander 上训练 DQN，让策略学会"落到指定目标 x"。
训练时每局随机采样一个 goal_x∈[-GOAL_LIMIT, GOAL_LIMIT]，飞船必须落到该点附近。
训练得到的策略供 multi_lander 的 spatial 模式使用，让多架飞船真正错开落点。

用法（项目根目录，venv 已激活）：
    python -m lunar_lander_rl.multi_lander.goal_conditioned_dqn --episodes 600
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
import torch.nn.functional as F

from ..common import ensure_dir, get_device, save_history, save_json, set_seed
from .goal_env import GoalConditionedLunarLander, GOAL_LIMIT

import gymnasium as gym
from gymnasium.envs.registration import register

# 注册一个可供 gym.make 使用的目标条件环境 id
ENV_ID = "GoalLunarLander-v0"
try:
    register(id=ENV_ID, entry_point="lunar_lander_rl.multi_lander.goal_env:GoalConditionedLunarLander")
except Exception:
    pass  # 已注册则跳过

# 训练时混入"悬停静止起点"的概率。实测 >0 会让整体策略变差，故默认 0（仅标准起点）。
HOVER_START_PROB = 0.0

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


@dataclass
class GC_DQNConfig:
    episodes: int = 600
    max_steps: int = 1000
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_size: int = 100_000
    warmup_steps: int = 1000
    target_update: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000
    seed: int = 42
    eval_episodes: int = 10
    device: str = "auto"
    output_dir: str = "outputs/goal_dqn"


def epsilon_by_step(step: int, cfg: GC_DQNConfig) -> float:
    return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-step / cfg.epsilon_decay_steps)


def mlp(input_dim: int, output_dim: int, hidden_dim: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def make_env(seed: int) -> gym.Env:
    # seed 在 reset 时传（LunarLander.__init__ 不接受 seed 参数）
    return gym.make(ENV_ID)


def _relocate_to_hover(env_unwrapped, goal_x: float, hover_y: float) -> None:
    """把飞船重定位到 goal_x 正上方 hover_y 高度的静止悬停起点。

    用于训练时混入"悬停静止"起点分布，让 goal_dqn 学会从悬停接手降落。
    """
    from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, SCALE, LEG_DOWN
    cx = VIEWPORT_W / SCALE / 2.0
    tx = cx + goal_x
    ty = env_unwrapped.helipad_y + LEG_DOWN / SCALE + hover_y
    env_unwrapped.lander.position = (tx, ty)
    env_unwrapped.lander.linearVelocity = (0.0, 0.0)
    env_unwrapped.lander.angle = 0.0
    env_unwrapped.lander.angularVelocity = 0.0
    # 重置 shaping 基准，避免 reset 后第一步 shaping 差分巨大
    env_unwrapped.prev_shaping = None


def evaluate(policy_net, device, episodes: int, seed: int) -> dict:
    """评估：随机目标下，落点离目标的平均误差 + 平均 reward。"""
    env = make_env(seed)
    returns, errors = [], []
    for ep in range(episodes):
        env.unwrapped.target_goal_x = float(np.random.default_rng(seed + ep).uniform(-GOAL_LIMIT, GOAL_LIMIT))
        s, _ = env.reset(seed=seed + ep)
        # reset 后 goal 已由 env 在 __init__ 固定，这里强制更新目标维度
        s[8] = env.unwrapped.target_goal_x
        tot = 0.0
        for _ in range(1000):
            with torch.no_grad():
                a = int(policy_net(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item())
            s, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc:
                break
        # 落点误差
        u = env.unwrapped
        rel_x = (u.lander.position.x - (VIEWPORT_W / SCALE / 2.0)) / SCALE
        errors.append(abs(rel_x - env.unwrapped.target_goal_x))
        returns.append(tot)
    env.close()
    return {"mean_return": float(np.mean(returns)),
            "mean_abs_landing_error": float(np.mean(errors))}


# 需要导入 VIEWPORT_W/SCALE 用于评估落点
from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, SCALE


def train(cfg: GC_DQNConfig) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[GC-DQN] device = {device}")
    output_dir = ensure_dir(cfg.output_dir)

    env = make_env(cfg.seed)
    obs_dim = env.observation_space.shape[0]   # 9
    action_dim = env.action_space.n            # 4

    policy = mlp(obs_dim, action_dim, cfg.hidden_dim).to(device)
    target = mlp(obs_dim, action_dim, cfg.hidden_dim).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)
    history: list[dict] = []

    global_step = 0
    best_score = float("-inf")
    rng = np.random.default_rng(cfg.seed)
    try:
        for episode in range(1, cfg.episodes + 1):
            # 每局随机一个目标
            goal_x = float(rng.uniform(-GOAL_LIMIT, GOAL_LIMIT))
            env.unwrapped.target_goal_x = goal_x
            state, _ = env.reset(seed=cfg.seed + episode)
            state[8] = goal_x  # 确保观测里的目标维度正确
            # 一定概率把飞船重定位到"悬停静止起点"（goal_x 正上方某高度，速度0）。
            # 实测加入该分布会让整体策略变差（0/9 失败），故默认 HOVER_START_PROB=0，
            # 只用标准高空自由落体起点训练（稳定，落点误差 ~0.18）。
            if rng.random() < HOVER_START_PROB:
                _relocate_to_hover(env.unwrapped, goal_x, float(rng.uniform(1.0, 2.0)))
                state = env.unwrapped._build_state(env.unwrapped._read_state8())
            episode_return = 0.0

            for step in range(1, cfg.max_steps + 1):
                epsilon = epsilon_by_step(global_step, cfg)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = int(policy(st).argmax(1).item())
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay.push(state, action, reward, next_state, done)
                state = next_state
                episode_return += float(reward)
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
                if done:
                    break

            history.append({"episode": episode, "return": episode_return,
                            "goal_x": goal_x, "epsilon": epsilon})
            if episode == 1 or episode % 20 == 0:
                print(f"[GC-DQN] ep={episode:04d} return={episode_return:7.1f} "
                      f"goal_x={goal_x:+.2f} eps={epsilon:.3f}")

            # 定期评估并保存最优（按"落点误差小 + reward 高"综合）
            if episode % 40 == 0:
                m = evaluate(policy, device, cfg.eval_episodes, cfg.seed + 10000)
                score = m["mean_return"] - 30 * m["mean_abs_landing_error"]
                print(f"   eval: return={m['mean_return']:.1f} "
                      f"abs_landing_err={m['mean_abs_landing_error']:.3f} score={score:.1f}")
                if score > best_score:
                    best_score = score
                    torch.save(policy.state_dict(), output_dir / "best_policy.pt")
    finally:
        env.close()

    save_history(history, output_dir)
    save_json({"algorithm": "GoalConditioned-DQN", "config": cfg,
               "best_score": best_score}, output_dir / "metrics.json")
    print(f"[GC-DQN] done. best_score={best_score:.1f}, saved -> {output_dir}/best_policy.pt")
    return {"best_score": best_score}


def parse_args() -> GC_DQNConfig:
    p = argparse.ArgumentParser(description="Goal-conditioned DQN for multi-lander (拓展3).")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="outputs/goal_dqn")
    a = p.parse_args()
    return GC_DQNConfig(episodes=a.episodes, seed=a.seed, device=a.device, output_dir=a.output_dir)


if __name__ == "__main__":
    train(parse_args())
