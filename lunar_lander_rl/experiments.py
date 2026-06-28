from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from lunar_lander_rl.config import dump_json
from lunar_lander_rl.envs import register_custom_lunar_envs


ALGORITHMS = {
    "a2c": A2C,
    "dqn": DQN,
    "ppo": PPO,
}


def _is_obstacle_env(env_id: str) -> bool:
    return env_id.startswith("Obstacle")


def make_raw_env(
    env_id: str,
    seed: int | None = None,
    render_mode: str | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    register_custom_lunar_envs()
    env = gym.make(env_id, render_mode=render_mode, **(env_kwargs or {}))
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_env(
    env_id: str,
    seed: int | None = None,
    render_mode: str | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    return Monitor(make_raw_env(env_id, seed=seed, render_mode=render_mode, env_kwargs=env_kwargs))


def make_monitored_env(
    env_id: str,
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    monitor_dir: str | Path | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    env = make_raw_env(env_id, seed=seed, render_mode=render_mode, env_kwargs=env_kwargs)
    filename = None
    if monitor_dir is not None:
        monitor_dir = Path(monitor_dir)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        filename = str(monitor_dir / "train")
    env = Monitor(env, filename=filename)
    return env


def make_vec_env(
    env_id: str,
    seed: int | None = None,
    *,
    n_envs: int = 1,
    render_mode: str | None = None,
    vec_normalize: bool = False,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    monitor_dir: str | Path | None = None,
    env_kwargs: dict[str, Any] | None = None,
):
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    def make_ranked_env(rank: int):
        def _init() -> gym.Env:
            env_seed = None if seed is None else seed + rank
            return make_raw_env(env_id, seed=env_seed, render_mode=render_mode, env_kwargs=env_kwargs)

        return _init

    env = DummyVecEnv([make_ranked_env(i) for i in range(n_envs)])
    monitor_filename = None
    if monitor_dir is not None:
        monitor_dir = Path(monitor_dir)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        monitor_filename = str(monitor_dir / "train")
    env = VecMonitor(env, filename=monitor_filename)
    if vec_normalize:
        env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs)
    return env


def train_from_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    algo_name = config["algorithm"].lower()
    if algo_name not in ALGORITHMS:
        supported = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm {algo_name!r}; choose one of: {supported}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "monitor"
    tensorboard_dir = output_dir / "tensorboard"

    seed = int(config.get("seed", 42))
    n_envs = int(config.get("n_envs", 1))
    env_kwargs = dict(config.get("env_kwargs") or {})
    vec_normalize_config = dict(config.get("vec_normalize", {}))
    use_vec_normalize = bool(vec_normalize_config.pop("enabled", False))
    env = make_vec_env(
        config["env_id"],
        seed=seed,
        n_envs=n_envs,
        vec_normalize=use_vec_normalize,
        monitor_dir=monitor_dir,
        env_kwargs=env_kwargs,
    )
    model_kwargs = dict(config.get("model_kwargs", {}))
    model_kwargs.setdefault("seed", seed)
    model_kwargs.setdefault("verbose", 1)
    model_kwargs.setdefault("tensorboard_log", str(tensorboard_dir))

    model_cls = ALGORITHMS[algo_name]
    model = model_cls("MlpPolicy", env, **model_kwargs)
    model.learn(
        total_timesteps=int(config["total_timesteps"]),
        tb_log_name=str(config.get("name", f"{algo_name}_{config['env_id']}")),
    )

    model_path = output_dir / f"{algo_name}_{config['env_id']}.zip"
    model.save(model_path)
    norm_path = None
    if use_vec_normalize:
        norm_path = output_dir / "vec_normalize.pkl"
        env.save(norm_path)

    eval_episodes = int(config.get("eval_episodes", 10))
    if use_vec_normalize:
        env.training = False
        env.norm_reward = False

    metrics: dict[str, Any] = {
        "algorithm": algo_name,
        "env_id": config["env_id"],
        "seed": seed,
        "n_envs": n_envs,
        "env_kwargs": env_kwargs,
        "total_timesteps": int(config["total_timesteps"]),
        "eval_episodes": eval_episodes,
        "model_path": str(model_path),
        "vec_normalize_path": str(norm_path) if norm_path is not None else None,
        "monitor_dir": str(monitor_dir),
        "tensorboard_dir": str(tensorboard_dir),
    }

    if _is_obstacle_env(config["env_id"]):
        obstacle_metrics = evaluate_obstacle_model(
            model=model,
            env_id=config["env_id"],
            episodes=eval_episodes,
            seed=seed + 1000,
            env_kwargs=env_kwargs,
            vec_normalize_path=norm_path,
        )
        metrics.update(obstacle_metrics)
    else:
        mean_reward, std_reward = evaluate_model(
            model=model,
            env_id=config["env_id"],
            episodes=eval_episodes,
            seed=seed + 1000,
            vec_normalize_path=norm_path,
            env_kwargs=env_kwargs,
        )
        metrics["mean_reward"] = mean_reward
        metrics["std_reward"] = std_reward

    dump_json(metrics, output_dir / "metrics.json")
    env.close()
    return model_path


def load_model(algorithm: str, model_path: str | Path) -> BaseAlgorithm:
    algo_name = algorithm.lower()
    if algo_name not in ALGORITHMS:
        supported = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm {algo_name!r}; choose one of: {supported}")
    return ALGORITHMS[algo_name].load(model_path)


def evaluate_model(
    *,
    model: BaseAlgorithm,
    env_id: str,
    episodes: int = 10,
    seed: int = 42,
    vec_normalize_path: str | Path | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[float, float]:
    if vec_normalize_path is None:
        env = make_env(env_id, seed=seed, env_kwargs=env_kwargs)
    else:
        raw_env = make_vec_env(env_id, seed=seed, env_kwargs=env_kwargs)
        env = VecNormalize.load(vec_normalize_path, raw_env)
        env.training = False
        env.norm_reward = False
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        deterministic=True,
    )
    env.close()
    return float(mean_reward), float(std_reward)


def evaluate_obstacle_model(
    *,
    model: BaseAlgorithm,
    env_id: str,
    episodes: int = 10,
    seed: int = 42,
    env_kwargs: dict[str, Any] | None = None,
    vec_normalize_path: str | Path | None = None,
) -> dict[str, float]:
    """Roll out a policy on an obstacle env and report return stats + collision_rate.

    Uses a hand-written vec rollout instead of ``sb3.evaluate_policy`` so each
    step's ``info["hit_obstacle"]`` can be read — evaluate_policy discards it.
    ``collision_rate`` is the fraction of finished episodes that hit an
    obstacle; ``success_rate`` is the fraction that terminated without a
    collision and with positive return.
    """
    venv = make_vec_env(env_id, seed=seed, n_envs=1, env_kwargs=env_kwargs)
    if vec_normalize_path is not None:
        venv = VecNormalize.load(vec_normalize_path, venv)
        venv.training = False
        venv.norm_reward = False

    returns: list[float] = []
    collisions = 0
    successes = 0

    obs = venv.reset()
    ep_returns = np.zeros(venv.num_envs, dtype=np.float64)
    ep_collided = np.zeros(venv.num_envs, dtype=bool)
    while len(returns) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        ep_returns += np.asarray(rewards, dtype=np.float64)
        for i, done in enumerate(dones):
            if infos[i].get("hit_obstacle"):
                ep_collided[i] = True
            if done:
                returns.append(float(ep_returns[i]))
                if ep_collided[i]:
                    collisions += 1
                elif ep_returns[i] > 0.0:
                    successes += 1
                ep_returns[i] = 0.0
                ep_collided[i] = False
    venv.close()

    arr = np.asarray(returns, dtype=np.float64)
    n = max(len(returns), 1)
    return {
        "episodes": float(len(returns)),
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "collision_rate": float(collisions / n),
        "success_rate": float(successes / n),
    }
