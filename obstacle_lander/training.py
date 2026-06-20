"""Self-contained training loops for the obstacle LunarLander env.

Why this exists: ``lunar_lander_rl`` is left untouched (so the base project's
behavior is preserved). The obstacle extension needs two tweaks that the base
``train()`` functions don't support:

1. An ``env_factory`` so we can swap in :class:`ObstacleLanderEnv` (14-dim obs
   instead of 8-dim). Without this the in-training eval would create a plain
   8-dim env and crash when loading the 14-dim policy.
2. DQN/Actor-Critic: ``best_policy.pt`` saved by rolling-20 average instead of
   the noisier single-episode max. The single-episode criterion sometimes
   snapshot a lucky +249 episode that fails on 70% of test seeds.

The training loops otherwise mirror the base implementations closely so the
algorithm-level comparison in the report stays fair.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from lunar_lander_rl.actor_critic import (
    ActorCriticConfig,
    ActorCriticNet as ACNet,
    discounted_returns,
)
from lunar_lander_rl.common import (
    ensure_dir,
    get_device,
    mlp,
    save_history,
    save_json,
    set_seed,
)
from lunar_lander_rl.dqn import DQNConfig, ReplayBuffer, epsilon_by_step
from lunar_lander_rl.ppo import ActorCriticNet as PPONet, PPOConfig, compute_gae


EnvFactory = Callable[[int], Any]


def _evaluate_with_factory(
    policy_fn: Callable[[np.ndarray], int],
    env_factory: EnvFactory,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    """Eval helper that uses ``env_factory`` so the eval env matches training."""
    returns: list[float] = []
    for ep in range(episodes):
        env = env_factory(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        try:
            for _ in range(max_steps):
                action = int(policy_fn(obs))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
        finally:
            env.close()
        returns.append(total)
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
    }


def train_dqn(cfg: DQNConfig, env_factory: EnvFactory) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    env = env_factory(cfg.seed)
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
    import random as _random

    try:
        for episode in range(1, cfg.episodes + 1):
            state, _ = env.reset(seed=cfg.seed + episode)
            episode_return = 0.0

            for step in range(1, cfg.max_steps + 1):
                epsilon = epsilon_by_step(global_step, cfg)
                if _random.random() < epsilon:
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

            history.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "epsilon": epsilon_by_step(global_step, cfg),
                    "steps": step,
                }
            )
            recent_avg = float(np.mean([h["return"] for h in history[-20:]]))
            if recent_avg > best_return:
                best_return = recent_avg
                torch.save(policy.state_dict(), output_dir / "best_policy.pt")

            if episode == 1 or episode % 20 == 0:
                print(
                    f"[DQN-obstacle] episode={episode:04d} return={episode_return:8.1f} "
                    f"recent20={recent_avg:8.1f} best={best_return:8.1f} "
                    f"epsilon={epsilon_by_step(global_step, cfg):.3f}"
                )
    finally:
        env.close()

    policy.load_state_dict(torch.load(output_dir / "best_policy.pt", map_location=device))
    policy.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return int(policy(obs_tensor).argmax(dim=1).item())

    metrics = _evaluate_with_factory(act, env_factory, cfg.eval_episodes, cfg.max_steps, cfg.seed + 10000)
    save_history(history, output_dir)
    save_json({"algorithm": "DQN-obstacle", "config": cfg, "metrics": metrics}, output_dir / "metrics.json")
    print(f"[DQN-obstacle] eval mean return: {metrics['mean_return']:.1f}")
    return metrics


def train_ppo(cfg: PPOConfig, env_factory: EnvFactory) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    env = env_factory(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = PPONet(obs_dim, action_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []
    best_eval = float("-inf")
    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    completed_episodes = 0

    try:
        for update in range(1, cfg.updates + 1):
            obs_buf: list = []
            action_buf: list = []
            logprob_buf: list = []
            reward_buf: list = []
            done_buf: list = []
            value_buf: list = []

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
                print(f"[PPO-obstacle] update={update:04d} episodes={completed_episodes:04d} recent_return={recent:8.1f}")
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

    metrics = _evaluate_with_factory(act, env_factory, cfg.eval_episodes, cfg.max_steps, cfg.seed + 10000)
    save_history(history, output_dir)
    save_json({"algorithm": "PPO-obstacle", "config": cfg, "metrics": metrics}, output_dir / "metrics.json")
    print(f"[PPO-obstacle] eval mean return: {metrics['mean_return']:.1f}")
    return metrics


def train_actor_critic(cfg: ActorCriticConfig, env_factory: EnvFactory) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    env = env_factory(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ACNet(obs_dim, action_dim, cfg.hidden_dim).to(device)
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

            history.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "steps": step,
                }
            )
            recent_avg = float(np.mean([h["return"] for h in history[-20:]]))
            if recent_avg > best_return:
                best_return = recent_avg
                torch.save(model.state_dict(), output_dir / "best_policy.pt")

            if episode == 1 or episode % 20 == 0:
                print(
                    f"[Actor-Critic-obstacle] episode={episode:04d} return={episode_return:8.1f} "
                    f"recent={recent_avg:8.1f} best={best_return:8.1f}"
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

    metrics = _evaluate_with_factory(act, env_factory, cfg.eval_episodes, cfg.max_steps, cfg.seed + 10000)
    save_history(history, output_dir)
    save_json({"algorithm": "Actor-Critic-obstacle", "config": cfg, "metrics": metrics}, output_dir / "metrics.json")
    print(f"[Actor-Critic-obstacle] eval mean return: {metrics['mean_return']:.1f}")
    return metrics
