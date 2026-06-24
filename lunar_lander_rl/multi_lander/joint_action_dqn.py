"""Centralized DQN for the three-lander task.

This is the most direct "do not change the DQN core, only enlarge input and
output" baseline:
  - input: concat(obs_0, obs_1, obs_2), 3 * 23 = 69 dimensions
  - output: one Q value for each joint action, 4 ** 3 = 64 actions

Each joint action index is decoded into (action_0, action_1, action_2), then all
three landers step in the same Box2D world. The transition reward is the team
mean reward, so the learned policy optimizes a shared team objective.

Usage:
    python -m lunar_lander_rl.multi_lander.joint_action_dqn --episodes 600
"""
from __future__ import annotations

import argparse
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..common import ensure_dir, get_device, save_history, save_json, set_seed
from .multi_agent_env import MultiAgentLunarLander, OBS_DIM
from .multi_agent_dqn import mlp

NUM_LANDERS = MultiAgentLunarLander.NUM_LANDERS
ACTION_DIM_SINGLE = 4
JOINT_OBS_DIM = OBS_DIM * NUM_LANDERS
JOINT_ACTION_DIM = ACTION_DIM_SINGLE ** NUM_LANDERS

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
class JointDQNConfig:
    episodes: int = 600
    max_steps: int = 1000
    hidden_dim: int = 256
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    replay_size: int = 200_000
    warmup_steps: int = 2000
    target_update: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    seed: int = 42
    eval_episodes: int = 5
    eval_every: int = 40
    device: str = "auto"
    output_dir: str = "outputs/multi_lander/joint_action_dqn"


def flatten_obs(obs_list: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(obs_list).astype(np.float32)


def decode_joint_action(index: int) -> list[int]:
    actions = []
    for _ in range(NUM_LANDERS):
        actions.append(index % ACTION_DIM_SINGLE)
        index //= ACTION_DIM_SINGLE
    return actions


def encode_joint_action(actions: list[int]) -> int:
    index = 0
    factor = 1
    for action in actions:
        index += int(action) * factor
        factor *= ACTION_DIM_SINGLE
    return index


def epsilon_by_step(step: int, cfg: JointDQNConfig) -> float:
    return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
        -step / cfg.epsilon_decay_steps
    )


def evaluate(policy_net, device: torch.device, episodes: int, seed: int, max_steps: int) -> dict:
    env = MultiAgentLunarLander(render_mode=None)
    rng = np.random.default_rng(seed)
    returns, landed, impacts, success = [], [], [], []
    try:
        for ep in range(episodes):
            goals = [float(rng.uniform(-0.7, 0.7)) for _ in range(NUM_LANDERS)]
            obs_list, _ = env.reset(seed=seed + ep, options={"goals": goals})
            team_return = 0.0
            impact_count = 0
            info = {"done_flags": [False] * NUM_LANDERS, "all_success": False}

            for _ in range(max_steps):
                state = flatten_obs(obs_list)
                with torch.no_grad():
                    tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_index = int(policy_net(tensor).argmax(1).item())
                obs_list, rewards, terminated, truncated, info = env.step(
                    decode_joint_action(action_index)
                )
                team_return += float(np.mean(rewards))
                impact_count += sum(1 for ev in info["collision_events"] if ev["impact"])
                if terminated or truncated:
                    break

            returns.append(team_return)
            landed.append(sum(info["done_flags"]))
            impacts.append(impact_count)
            success.append(1.0 if info.get("all_success") else 0.0)
    finally:
        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "mean_landed": float(np.mean(landed)),
        "mean_impacts": float(np.mean(impacts)),
        "all_success_rate": float(np.mean(success)),
    }


def train(cfg: JointDQNConfig) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    print(
        f"[Joint-DQN] device={device}, obs_dim={JOINT_OBS_DIM}, "
        f"joint_actions={JOINT_ACTION_DIM}"
    )

    policy = mlp(JOINT_OBS_DIM, JOINT_ACTION_DIM, cfg.hidden_dim).to(device)
    target = mlp(JOINT_OBS_DIM, JOINT_ACTION_DIM, cfg.hidden_dim).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)
    env = MultiAgentLunarLander(render_mode=None)
    rng = np.random.default_rng(cfg.seed)

    global_step = 0
    best_score = float("-inf")
    history: list[dict] = []
    last_eval: dict | None = None

    try:
        for episode in range(1, cfg.episodes + 1):
            goals = [float(rng.uniform(-0.7, 0.7)) for _ in range(NUM_LANDERS)]
            obs_list, _ = env.reset(seed=cfg.seed + episode, options={"goals": goals})
            state = flatten_obs(obs_list)
            team_return = 0.0
            impact_count = 0
            info = {"done_flags": [False] * NUM_LANDERS, "all_success": False}

            for step in range(1, cfg.max_steps + 1):
                epsilon = epsilon_by_step(global_step, cfg)
                if random.random() < epsilon:
                    action_index = random.randrange(JOINT_ACTION_DIM)
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(
                            state, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        action_index = int(policy(state_tensor).argmax(1).item())

                actions = decode_joint_action(action_index)
                next_obs_list, rewards, terminated, truncated, info = env.step(actions)
                next_state = flatten_obs(next_obs_list)
                done = terminated or truncated
                reward = float(np.mean(rewards))
                replay.push(state, action_index, reward, next_state, done)

                state = next_state
                team_return += reward
                impact_count += sum(1 for ev in info["collision_events"] if ev["impact"])
                global_step += 1

                if len(replay) >= cfg.warmup_steps:
                    states, actions_b, rewards_b, next_states, dones = replay.sample(
                        cfg.batch_size, device
                    )
                    q_values = policy(states).gather(1, actions_b)
                    with torch.no_grad():
                        next_q = target(next_states).max(dim=1, keepdim=True).values
                        targets = rewards_b + cfg.gamma * (1.0 - dones) * next_q
                    loss = F.smooth_l1_loss(q_values, targets)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()

                if global_step % cfg.target_update == 0:
                    target.load_state_dict(policy.state_dict())
                if done:
                    break

            landed = sum(info["done_flags"])
            history.append(
                {
                    "episode": episode,
                    "team_return": team_return,
                    "landed": landed,
                    "impacts": impact_count,
                    "epsilon": epsilon_by_step(global_step, cfg),
                    "steps": step,
                }
            )
            if episode == 1 or episode % 20 == 0:
                print(
                    f"[Joint-DQN] ep={episode:04d} return={team_return:8.1f} "
                    f"landed={landed}/3 impacts={impact_count} "
                    f"eps={epsilon_by_step(global_step, cfg):.3f}"
                )

            if episode % cfg.eval_every == 0:
                last_eval = evaluate(policy, device, cfg.eval_episodes, cfg.seed + 10000, cfg.max_steps)
                score = (
                    last_eval["mean_return"]
                    + 25.0 * last_eval["mean_landed"]
                    + 100.0 * last_eval["all_success_rate"]
                    - 30.0 * last_eval["mean_impacts"]
                )
                print(
                    "   eval: "
                    f"return={last_eval['mean_return']:.1f} "
                    f"landed={last_eval['mean_landed']:.1f}/3 "
                    f"success={last_eval['all_success_rate']:.2f} "
                    f"impacts={last_eval['mean_impacts']:.1f} "
                    f"score={score:.1f}"
                )
                if score > best_score:
                    best_score = score
                    torch.save(policy.state_dict(), output_dir / "best_policy.pt")
    finally:
        env.close()

    torch.save(policy.state_dict(), output_dir / "last_policy.pt")
    save_history(history, output_dir)
    save_json(
        {
            "algorithm": "Centralized-JointAction-DQN",
            "config": cfg,
            "dims": {
                "obs_dim": JOINT_OBS_DIM,
                "single_action_dim": ACTION_DIM_SINGLE,
                "joint_action_dim": JOINT_ACTION_DIM,
            },
            "best_score": best_score,
            "last_eval": last_eval,
        },
        output_dir / "metrics.json",
    )
    print(f"[Joint-DQN] done. last policy -> {output_dir / 'last_policy.pt'}")
    return {"best_score": best_score}


def parse_args() -> JointDQNConfig:
    parser = argparse.ArgumentParser(description="Centralized joint-action DQN for 3 landers.")
    parser.add_argument("--episodes", type=int, default=JointDQNConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=JointDQNConfig.max_steps)
    parser.add_argument("--eval-episodes", type=int, default=JointDQNConfig.eval_episodes)
    parser.add_argument("--eval-every", type=int, default=JointDQNConfig.eval_every)
    parser.add_argument("--seed", type=int, default=JointDQNConfig.seed)
    parser.add_argument("--device", default=JointDQNConfig.device)
    parser.add_argument("--output-dir", default=JointDQNConfig.output_dir)
    args = parser.parse_args()
    return JointDQNConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    train(parse_args())
