from __future__ import annotations

import argparse
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import ensure_dir, evaluate_policy, make_env, save_history, save_json, set_seed


@dataclass
class QLearningConfig:
    episodes: int = 400
    max_steps: int = 1000
    lr: float = 0.08
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42
    eval_episodes: int = 5
    output_dir: str = "outputs/q_learning"


OBS_LOW = np.array([-1.5, -0.1, -2.0, -2.0, -np.pi, -3.0, 0.0, 0.0])
OBS_HIGH = np.array([1.5, 1.5, 2.0, 2.0, np.pi, 3.0, 1.0, 1.0])
N_BINS = np.array([8, 8, 8, 8, 8, 8, 2, 2])


def discretize(obs: np.ndarray) -> tuple[int, ...]:
    clipped = np.clip(obs, OBS_LOW, OBS_HIGH)
    ratios = (clipped - OBS_LOW) / (OBS_HIGH - OBS_LOW)
    bins = np.floor(ratios * N_BINS).astype(np.int64)
    bins = np.clip(bins, 0, N_BINS - 1)
    return tuple(int(x) for x in bins)


def epsilon_for_episode(episode: int, cfg: QLearningConfig) -> float:
    return max(cfg.epsilon_end, cfg.epsilon_start * (cfg.epsilon_decay ** episode))


def train(cfg: QLearningConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)
    env = make_env(seed=cfg.seed)
    q_table = np.zeros((*N_BINS, env.action_space.n), dtype=np.float32)
    history: list[dict[str, float]] = []
    best_return = float("-inf")

    try:
        for episode in range(1, cfg.episodes + 1):
            obs, _ = env.reset(seed=cfg.seed + episode)
            state = discretize(obs)
            episode_return = 0.0
            epsilon = epsilon_for_episode(episode, cfg)

            for step in range(1, cfg.max_steps + 1):
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q_table[state]))

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = discretize(next_obs)
                done = terminated or truncated

                target = reward
                if not done:
                    target += cfg.gamma * float(np.max(q_table[next_state]))
                q_table[state + (action,)] += cfg.lr * (target - q_table[state + (action,)])

                state = next_state
                episode_return += float(reward)
                if done:
                    break

            best_return = max(best_return, episode_return)
            history.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "best_return": best_return,
                    "epsilon": epsilon,
                    "steps": step,
                }
            )
            if episode == 1 or episode % 20 == 0:
                print(
                    f"[Q-Learning] episode={episode:04d} return={episode_return:8.1f} "
                    f"best={best_return:8.1f} epsilon={epsilon:.3f}"
                )
    finally:
        env.close()

    with (output_dir / "q_table.pkl").open("wb") as f:
        pickle.dump(q_table, f)

    metrics = evaluate_policy(lambda obs: int(np.argmax(q_table[discretize(obs)])), cfg.eval_episodes, seed=cfg.seed + 10000)
    save_history(history, output_dir)
    save_json({"algorithm": "Q-Learning", "config": cfg, "metrics": metrics}, output_dir / "metrics.json")
    print(f"[Q-Learning] eval mean return: {metrics['mean_return']:.1f}")
    return metrics


def parse_args() -> QLearningConfig:
    parser = argparse.ArgumentParser(description="Tabular Q-Learning demo for LunarLander-v3.")
    parser.add_argument("--episodes", type=int, default=QLearningConfig.episodes)
    parser.add_argument("--eval-episodes", type=int, default=QLearningConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=QLearningConfig.seed)
    parser.add_argument("--output-dir", default=QLearningConfig.output_dir)
    args = parser.parse_args()
    return QLearningConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    train(parse_args())

