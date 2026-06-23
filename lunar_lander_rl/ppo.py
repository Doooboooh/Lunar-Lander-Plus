from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .common import EnvFactory, ensure_dir, evaluate_policy, get_device, make_env, save_history, save_json, set_seed


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], eps: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps)

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = float(batch.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, batch: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (np.asarray(batch, dtype=np.float32) - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8
        )
        return np.clip(normalized, -clip, clip)


class RewardNormalizer:
    """VecNormalize-style reward scaling using running discounted returns."""

    def __init__(self, gamma: float, clip: float = 10.0) -> None:
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.return_rms = RunningMeanStd(())
        self.running_return = 0.0

    def normalize(self, reward: float, done: bool) -> float:
        self.running_return = self.running_return * self.gamma * (1.0 - float(done)) + float(reward)
        self.return_rms.update(np.array([self.running_return], dtype=np.float64))
        scale = float(np.sqrt(self.return_rms.var + 1e-8))
        normalized = float(reward) / scale
        return float(np.clip(normalized, -self.clip, self.clip))

    def reset(self) -> None:
        self.running_return = 0.0


def save_obs_normalizer(path, obs_rms: RunningMeanStd | None) -> None:
    if obs_rms is None:
        return
    np.savez(
        path,
        mean=obs_rms.mean.astype(np.float32),
        var=obs_rms.var.astype(np.float32),
        count=np.array([obs_rms.count], dtype=np.float64),
    )


def load_obs_normalizer(path) -> RunningMeanStd | None:
    path = str(path)
    try:
        data = np.load(path)
    except FileNotFoundError:
        return None
    mean = np.asarray(data["mean"], dtype=np.float64)
    var = np.asarray(data["var"], dtype=np.float64)
    count = float(np.asarray(data["count"]).reshape(-1)[0])
    obs_rms = RunningMeanStd(mean.shape)
    obs_rms.mean = mean
    obs_rms.var = var
    obs_rms.count = count
    return obs_rms


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        if activation == "tanh":
            activation_layer: type[nn.Module] = nn.Tanh
        elif activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "elu":
            activation_layer = nn.ELU
        else:
            raise ValueError("activation must be one of: tanh, relu, elu")

        layers: list[nn.Module] = []
        input_dim = obs_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_layer())
            input_dim = hidden_dim
        self.body = nn.Sequential(*layers)
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
    hidden_layers: int = 2
    activation: str = "tanh"
    eval_interval: int = 10
    selection_eval_episodes: int = 3
    normalize_observations: bool = True
    normalize_rewards: bool = False
    reward_clip: float = 10.0
    seed: int = 42
    eval_episodes: int = 5
    device: str = "auto"
    output_dir: str = "outputs/ppo"


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


def train(
    cfg: PPOConfig,
    env_factory: EnvFactory = make_env,
    eval_env_factory: EnvFactory | None = None,
) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    env = env_factory(seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticNet(obs_dim, action_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    obs_rms = RunningMeanStd((obs_dim,)) if cfg.normalize_observations else None
    reward_normalizer = RewardNormalizer(cfg.gamma, cfg.reward_clip) if cfg.normalize_rewards else None
    history: list[dict[str, float]] = []
    best_eval = float("-inf")
    obs, _ = env.reset(seed=cfg.seed)
    if obs_rms is not None:
        obs_rms.update(obs)
    if reward_normalizer is not None:
        reward_normalizer.reset()
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
                model_obs = obs_rms.normalize(obs) if obs_rms is not None else obs
                obs_tensor = torch.tensor(model_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, logprob, value = model.act(obs_tensor)
                next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                reward_for_update = (
                    reward_normalizer.normalize(float(reward), done) if reward_normalizer is not None else float(reward)
                )

                obs_buf.append(obs)
                action_buf.append(int(action.item()))
                logprob_buf.append(float(logprob.item()))
                reward_buf.append(reward_for_update)
                done_buf.append(float(done))
                value_buf.append(float(value.item()))
                episode_return += float(reward)
                obs = next_obs
                if obs_rms is not None:
                    obs_rms.update(obs)

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
                    if reward_normalizer is not None:
                        reward_normalizer.reset()

            with torch.no_grad():
                next_model_obs = obs_rms.normalize(obs) if obs_rms is not None else obs
                next_obs_tensor = torch.tensor(next_model_obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, next_value = model(next_obs_tensor)

            rollout_obs = np.array(obs_buf, dtype=np.float32)
            if obs_rms is not None:
                rollout_obs = obs_rms.normalize(rollout_obs)
            obs_tensor = torch.tensor(rollout_obs, dtype=torch.float32, device=device)
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

            should_eval = update == 1 or (cfg.eval_interval > 0 and update % cfg.eval_interval == 0)
            if should_eval:
                model.eval()

                def current_act(obs_np: np.ndarray) -> int:
                    with torch.no_grad():
                        model_obs = obs_rms.normalize(obs_np) if obs_rms is not None else obs_np
                        obs_tensor = torch.tensor(model_obs, dtype=torch.float32, device=device).unsqueeze(0)
                        logits, _ = model(obs_tensor)
                        return int(torch.argmax(logits, dim=1).item())

                eval_metrics = evaluate_policy(
                    current_act,
                    cfg.selection_eval_episodes,
                    seed=cfg.seed + 30000 + update * 100,
                    env_factory=eval_env_factory or env_factory,
                )
                model.train()
                eval_mean = float(eval_metrics["mean_return"])
                if eval_mean > best_eval:
                    best_eval = eval_mean
                    torch.save(model.state_dict(), output_dir / "best_policy.pt")
                print(
                    f"[PPO] update={update:04d} eval_mean={eval_mean:8.1f} "
                    f"best_eval={best_eval:8.1f}"
                )
    finally:
        env.close()

    torch.save(model.state_dict(), output_dir / "last_policy.pt")
    save_obs_normalizer(output_dir / "obs_norm.npz", obs_rms)
    if not (output_dir / "best_policy.pt").exists():
        torch.save(model.state_dict(), output_dir / "best_policy.pt")
    model.load_state_dict(torch.load(output_dir / "best_policy.pt", map_location=device))
    model.eval()

    def act(obs_np: np.ndarray) -> int:
        with torch.no_grad():
            model_obs = obs_rms.normalize(obs_np) if obs_rms is not None else obs_np
            obs_tensor = torch.tensor(model_obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(obs_tensor)
            return int(torch.argmax(logits, dim=1).item())

    metrics = evaluate_policy(act, cfg.eval_episodes, seed=cfg.seed + 10000, env_factory=eval_env_factory or env_factory)
    save_history(history, output_dir)
    save_json({"algorithm": "PPO", "config": cfg, "metrics": metrics}, output_dir / "metrics.json")
    print(f"[PPO] eval mean return: {metrics['mean_return']:.1f}")
    return metrics


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="PPO demo for LunarLander-v3.")
    parser.add_argument("--updates", type=int, default=PPOConfig.updates)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--hidden-dim", type=int, default=PPOConfig.hidden_dim)
    parser.add_argument("--hidden-layers", type=int, default=PPOConfig.hidden_layers)
    parser.add_argument("--activation", choices=["tanh", "relu", "elu"], default=PPOConfig.activation)
    parser.add_argument("--lr", type=float, default=PPOConfig.lr)
    parser.add_argument("--entropy-coef", type=float, default=PPOConfig.entropy_coef)
    parser.add_argument("--eval-interval", type=int, default=PPOConfig.eval_interval)
    parser.add_argument("--selection-eval-episodes", type=int, default=PPOConfig.selection_eval_episodes)
    parser.add_argument("--normalize-observations", dest="normalize_observations", action="store_true")
    parser.add_argument("--no-normalize-observations", dest="normalize_observations", action="store_false")
    parser.add_argument("--normalize-rewards", dest="normalize_rewards", action="store_true")
    parser.add_argument("--no-normalize-rewards", dest="normalize_rewards", action="store_false")
    parser.add_argument("--reward-clip", type=float, default=PPOConfig.reward_clip)
    parser.add_argument("--eval-episodes", type=int, default=PPOConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--device", default=PPOConfig.device)
    parser.add_argument("--output-dir", default=PPOConfig.output_dir)
    parser.set_defaults(
        normalize_observations=PPOConfig.normalize_observations,
        normalize_rewards=PPOConfig.normalize_rewards,
    )
    args = parser.parse_args()
    return PPOConfig(
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        eval_interval=args.eval_interval,
        selection_eval_episodes=args.selection_eval_episodes,
        normalize_observations=args.normalize_observations,
        normalize_rewards=args.normalize_rewards,
        reward_clip=args.reward_clip,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    train(parse_args())
