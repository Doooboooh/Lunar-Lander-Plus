from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .common import ensure_dir, evaluate_policy, get_device, make_env, save_history, save_json, set_seed, register_moving_pad

# 注册移动平台变体环境（幂等），供 --env-id 指定
register_moving_pad()


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.body(obs)
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


@dataclass
class PPOConfig:
    updates: int = 200
    rollout_steps: int = 1024
    max_steps: int = 1000
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    seed: int = 42
    eval_episodes: int = 5
    device: str = "auto"
    output_dir: str = "outputs/ppo"
    env_id: str = "LunarLander-v3"


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_values = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train(cfg: PPOConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    env = make_env(seed=cfg.seed, env_id=cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticNet(obs_dim, action_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []
    best_eval = float("-inf")
    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    completed_episodes = 0

    try:
        for update in range(1, cfg.updates + 1):
            obs_buf = []
            action_buf = []
            logprob_buf = []
            reward_buf = []
            done_buf = []
            value_buf = []

            for _ in range(cfg.rollout_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, logprob, value = model.act(obs_tensor)
                next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated

                obs_buf.append(obs)
                action_buf.append(int(action.item()))
                logprob_buf.append(float(logprob.item()))
                reward_buf.append(float(reward))
                done_buf.append(float(done))
                value_buf.append(float(value.item()))
                episode_return += float(reward)
                obs = next_obs

                if done:
                    completed_episodes += 1
                    history.append(
                        {
                            "episode": completed_episodes,
                            "return": episode_return,
                            "update": update,
                        }
                    )
                    episode_return = 0.0
                    obs, _ = env.reset(seed=cfg.seed + completed_episodes)

            with torch.no_grad():
                next_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, next_value = model(next_obs_tensor)

            obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
            actions = torch.tensor(action_buf, dtype=torch.long, device=device)
            old_logprobs = torch.tensor(logprob_buf, dtype=torch.float32, device=device)
            rewards = torch.tensor(reward_buf, dtype=torch.float32, device=device)
            dones = torch.tensor(done_buf, dtype=torch.float32, device=device)
            values = torch.tensor(value_buf, dtype=torch.float32, device=device)
            advantages, returns = compute_gae(rewards, dones, values, next_value.squeeze(0), cfg.gamma, cfg.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            batch_size = len(obs_buf)
            indices = np.arange(batch_size)
            for _ in range(cfg.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, cfg.minibatch_size):
                    mb_idx = indices[start : start + cfg.minibatch_size]
                    logits, new_values = model(obs_tensor[mb_idx])
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(actions[mb_idx])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logprobs - old_logprobs[mb_idx])
                    policy_loss = -torch.min(
                        advantages[mb_idx] * ratio,
                        advantages[mb_idx] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                    ).mean()
                    value_loss = F.mse_loss(new_values, returns[mb_idx])
                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            if update == 1 or update % 10 == 0:
                recent = np.mean([x["return"] for x in history[-10:]]) if history else episode_return
                print(f"[PPO] update={update:04d} episodes={completed_episodes:04d} recent_return={recent:8.1f}")
                if recent > best_eval:
                    best_eval = float(recent)
                    torch.save(model.state_dict(), output_dir / "best_policy.pt")
    finally:
        env.close()

    torch.save(model.state_dict(), output_dir / "last_policy.pt")
    if not (output_dir / "best_policy.pt").exists():
        torch.save(model.state_dict(), output_dir / "best_policy.pt")
    model.load_state_dict(torch.load(output_dir / "best_policy.pt", map_location=device))
    model.eval()

    def act(obs_np: np.ndarray) -> int:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(obs_tensor)
            return int(torch.argmax(logits, dim=1).item())

    metrics = evaluate_policy(act, cfg.eval_episodes, seed=cfg.seed + 10000, env_id=cfg.env_id)
    save_history(history, output_dir)
    save_json({"algorithm": "PPO", "config": cfg, "metrics": metrics}, output_dir / "metrics.json")
    print(f"[PPO] eval mean return: {metrics['mean_return']:.1f}")
    return metrics


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="PPO demo for LunarLander-v3.")
    parser.add_argument("--updates", type=int, default=PPOConfig.updates)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--eval-episodes", type=int, default=PPOConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--device", default=PPOConfig.device)
    parser.add_argument("--output-dir", default=PPOConfig.output_dir)
    parser.add_argument("--env-id", default=PPOConfig.env_id,
                        help="环境 id，默认 LunarLander-v3；移动平台用 MovingPadLunarLander-v0")
    args = parser.parse_args()
    return PPOConfig(
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        env_id=args.env_id,
    )


if __name__ == "__main__":
    train(parse_args())

