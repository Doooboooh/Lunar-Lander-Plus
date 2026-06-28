from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from lunar_lander_rl.config import dump_json
from lunar_lander_rl.envs import register_custom_lunar_envs


ALGORITHMS = {
    "a2c": A2C,
    "dqn": DQN,
    "ppo": PPO,
}


def make_raw_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    register_custom_lunar_envs()
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    return Monitor(make_raw_env(env_id, seed=seed, render_mode=render_mode))


def make_monitored_env(
    env_id: str,
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    monitor_dir: str | Path | None = None,
) -> gym.Env:
    env = make_raw_env(env_id, seed=seed, render_mode=render_mode)
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
):
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    def make_ranked_env(rank: int):
        def _init() -> gym.Env:
            env_seed = None if seed is None else seed + rank
            return make_raw_env(env_id, seed=env_seed, render_mode=render_mode)

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
    vec_normalize_config = dict(config.get("vec_normalize", {}))
    use_vec_normalize = bool(vec_normalize_config.pop("enabled", False))
    env = make_vec_env(
        config["env_id"],
        seed=seed,
        n_envs=n_envs,
        vec_normalize=use_vec_normalize,
        monitor_dir=monitor_dir,
        **vec_normalize_config,
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
    eval_metrics = evaluate_model_detailed(
        model=model,
        env_id=config["env_id"],
        episodes=eval_episodes,
        seed=seed + 1000,
        vec_normalize_path=norm_path,
    )
    dump_json(
        {
            "algorithm": algo_name,
            "env_id": config["env_id"],
            "seed": seed,
            "n_envs": n_envs,
            "total_timesteps": int(config["total_timesteps"]),
            "eval_episodes": eval_episodes,
            **eval_metrics,
            "model_path": str(model_path),
            "vec_normalize_path": str(norm_path) if norm_path is not None else None,
            "monitor_dir": str(monitor_dir),
            "tensorboard_dir": str(tensorboard_dir),
        },
        output_dir / "metrics.json",
    )
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
) -> tuple[float, float]:
    metrics = evaluate_model_detailed(
        model=model,
        env_id=env_id,
        episodes=episodes,
        seed=seed,
        vec_normalize_path=vec_normalize_path,
    )
    return metrics["mean_reward"], metrics["std_reward"]


def evaluate_model_detailed(
    *,
    model: BaseAlgorithm,
    env_id: str,
    episodes: int = 10,
    seed: int = 42,
    vec_normalize_path: str | Path | None = None,
) -> dict[str, float]:
    if vec_normalize_path is None:
        env = make_vec_env(env_id, seed=seed)
    else:
        raw_env = make_vec_env(env_id, seed=seed)
        env = VecNormalize.load(vec_normalize_path, raw_env)
        env.training = False
        env.norm_reward = False

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    completed_waypoints: list[int] = []
    completed_all_waypoints: list[bool] = []
    time_limit_truncated: list[bool] = []

    try:
        for _ in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            length = 0
            last_info: dict[str, Any] = {}
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_arr, infos = env.step(action)
                total_reward += float(reward[0])
                length += 1
                done = bool(done_arr[0])
                last_info = infos[0]

            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            completed_waypoints.append(int(last_info.get("completed_waypoints", last_info.get("active_waypoint", 0))))
            completed_all_waypoints.append(bool(last_info.get("completed_all_waypoints", False)))
            time_limit_truncated.append(bool(last_info.get("TimeLimit.truncated", False)))
    finally:
        env.close()

    rewards = list(episode_rewards)
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    reward_var = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards) if rewards else 0.0
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(reward_var**0.5),
        "mean_episode_length": float(sum(episode_lengths) / len(episode_lengths)) if episode_lengths else 0.0,
        "mean_completed_waypoints": float(sum(completed_waypoints) / len(completed_waypoints)) if completed_waypoints else 0.0,
        "waypoint_success_rate": float(sum(completed_all_waypoints) / len(completed_all_waypoints)) if completed_all_waypoints else 0.0,
        "time_limit_truncated_rate": float(sum(time_limit_truncated) / len(time_limit_truncated)) if time_limit_truncated else 0.0,
    }
