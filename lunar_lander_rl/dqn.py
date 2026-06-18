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

from .common import EnvFactory, ensure_dir, evaluate_policy, get_device, make_env, mlp, save_history, save_json, set_seed


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
class DQNConfig:
    episodes: int = 400
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
    epsilon_decay_steps: int = 30_000
    seed: int = 42
    eval_episodes: int = 5
    device: str = "auto"
    output_dir: str = "outputs/dqn"


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-step / cfg.epsilon_decay_steps)


def train(
    cfg: DQNConfig,
    env_factory: EnvFactory = make_env,
    eval_env_factory: EnvFactory | None = None,
) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    env = env_factory(seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = mlp(obs_dim, action_dim, cfg.hidden_dim).to(device)
    target = mlp(obs_dim, action_dim, cfg.hidden_dim).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)
    history: list[dict[str, float]] = []

    global_step = 0
    best_return = float("-inf")
    try:
        for episode in range(1, cfg.episodes + 1):
            state, _ = env.reset(seed=cfg.seed + episode)
            episode_return = 0.0

            for step in range(1, cfg.max_steps + 1):
                epsilon = epsilon_by_step(global_step, cfg)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = int(policy(state_tensor).argmax(dim=1).item())

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay.push(state, action, reward, next_state, done)
                state = next_state
                episode_return += float(reward)
                global_step += 1

                if len(replay) >= cfg.warmup_steps:
                    states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size, device)
                    q_values = policy(states).gather(1, actions)
                    with torch.no_grad():
                        next_q = target(next_states).max(dim=1, keepdim=True).values
                        targets = rewards + cfg.gamma * (1.0 - dones) * next_q
                    loss = F.smooth_l1_loss(q_values, targets)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()

                if global_step % cfg.target_update == 0:
                    target.load_state_dict(policy.state_dict())

                if done:
                    break

            if episode_return > best_return:
                best_return = episode_return
                torch.save(policy.state_dict(), output_dir / "best_policy.pt")

            history.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "best_return": best_return,
                    "epsilon": epsilon_by_step(global_step, cfg),
                    "steps": step,
                }
            )
            if episode == 1 or episode % 20 == 0:
                print(
                    f"[DQN] episode={episode:04d} return={episode_return:8.1f} "
                    f"best={best_return:8.1f} epsilon={epsilon_by_step(global_step, cfg):.3f}"
                )
    finally:
        env.close()

    policy.load_state_dict(torch.load(output_dir / "best_policy.pt", map_location=device))
    policy.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return int(policy(obs_tensor).argmax(dim=1).item())

    metrics = evaluate_policy(act, cfg.eval_episodes, seed=cfg.seed + 10000, env_factory=eval_env_factory or env_factory)
    save_history(history, output_dir)
    save_json({"algorithm": "DQN", "config": cfg, "metrics": metrics}, output_dir / "metrics.json")
    print(f"[DQN] eval mean return: {metrics['mean_return']:.1f}")
    return metrics


def parse_args() -> DQNConfig:
    parser = argparse.ArgumentParser(description="Deep Q-Network demo for LunarLander-v3.")
    parser.add_argument("--episodes", type=int, default=DQNConfig.episodes)
    parser.add_argument("--eval-episodes", type=int, default=DQNConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=DQNConfig.seed)
    parser.add_argument("--device", default=DQNConfig.device)
    parser.add_argument("--output-dir", default=DQNConfig.output_dir)
    args = parser.parse_args()
    return DQNConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    train(parse_args())

