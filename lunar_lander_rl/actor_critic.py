from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .common import ensure_dir, evaluate_policy, get_device, make_env, save_history, save_json, set_seed, register_moving_pad

register_moving_pad()


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(obs)
        return self.actor(x), self.critic(x).squeeze(-1)


@dataclass
class ActorCriticConfig:
    episodes: int = 600
    max_steps: int = 1000
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    seed: int = 42
    eval_episodes: int = 5
    device: str = "auto"
    output_dir: str = "outputs/actor_critic"
    env_id: str = "LunarLander-v3"


def discounted_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


def train(cfg: ActorCriticConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    env = make_env(seed=cfg.seed, env_id=cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticNet(obs_dim, action_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []
    best_return = float("-inf")

    try:
        for episode in range(1, cfg.episodes + 1):
            obs, _ = env.reset(seed=cfg.seed + episode)
            log_probs = []
            values = []
            entropies = []
            rewards = []
            episode_return = 0.0

            for step in range(1, cfg.max_steps + 1):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, value = model(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))

                log_probs.append(dist.log_prob(action).squeeze(0))
                values.append(value.squeeze(0))
                entropies.append(dist.entropy().squeeze(0))
                rewards.append(float(reward))
                episode_return += float(reward)
                if terminated or truncated:
                    break

            returns = discounted_returns(rewards, cfg.gamma, device)
            values_tensor = torch.stack(values)
            log_probs_tensor = torch.stack(log_probs)
            entropies_tensor = torch.stack(entropies)
            advantages = returns - values_tensor.detach()

            policy_loss = -(log_probs_tensor * advantages).mean()
            value_loss = F.mse_loss(values_tensor, returns)
            entropy_bonus = entropies_tensor.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if episode_return > best_return:
                best_return = episode_return
                torch.save(model.state_dict(), output_dir / "best_policy.pt")

            history.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "best_return": best_return,
                    "steps": step,
                }
            )
            if episode == 1 or episode % 20 == 0:
                recent = np.mean([x["return"] for x in history[-20:]])
                print(
                    f"[Actor-Critic] episode={episode:04d} return={episode_return:8.1f} "
                    f"recent={recent:8.1f} best={best_return:8.1f}"
                )
    finally:
        env.close()

    model.load_state_dict(torch.load(output_dir / "best_policy.pt", map_location=device))
    model.eval()

    def act(obs_np: np.ndarray) -> int:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(obs_tensor)
            return int(torch.argmax(logits, dim=1).item())

    metrics = evaluate_policy(act, cfg.eval_episodes, seed=cfg.seed + 10000, env_id=cfg.env_id)
    save_history(history, output_dir)
    save_json({"algorithm": "Actor-Critic", "config": cfg, "metrics": metrics}, output_dir / "metrics.json")
    print(f"[Actor-Critic] eval mean return: {metrics['mean_return']:.1f}")
    return metrics


def parse_args() -> ActorCriticConfig:
    parser = argparse.ArgumentParser(description="Vanilla Actor-Critic demo for LunarLander-v3.")
    parser.add_argument("--episodes", type=int, default=ActorCriticConfig.episodes)
    parser.add_argument("--eval-episodes", type=int, default=ActorCriticConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=ActorCriticConfig.seed)
    parser.add_argument("--device", default=ActorCriticConfig.device)
    parser.add_argument("--output-dir", default=ActorCriticConfig.output_dir)
    parser.add_argument("--env-id", default=ActorCriticConfig.env_id,
                        help="环境 id，默认 LunarLander-v3；移动平台用 MovingPadLunarLander-v0")
    args = parser.parse_args()
    return ActorCriticConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        env_id=args.env_id,
    )


if __name__ == "__main__":
    train(parse_args())

