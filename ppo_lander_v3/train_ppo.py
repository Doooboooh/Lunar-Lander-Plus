"""LunarLander-v3 + PPO (stable-baselines3) 训练入口 — 改进版。

主要改动（对齐 sb3 zoo LunarLander-v2 tuned config）：
  - 加 VecNormalize（obs 归一化，对 LunarLander 最关键）
  - lr 3e-4 → 1e-4
  - gae_lambda 0.95 → 0.98
  - n_epochs 10 → 4
  - n_envs 8 → 16
  - total_timesteps 50k → 1M

跑法：
    python train_ppo.py
查看曲线：
    tensorboard --logdir tb
"""
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

HERE = Path(__file__).resolve().parent
TB_DIR = HERE / "tb"
MODEL_DIR = HERE / "models"
LOG_DIR = HERE / "logs"

ENV_ID = "LunarLander-v3"
SEED = 0
TOTAL_TS = 1_000_000
N_ENVS = 16


def make_env(rank: int, seed: int = SEED):
    def _init():
        env = gym.make(ENV_ID)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


class EpisodeRewardPrint(BaseCallback):
    """每个 episode 结束时把 reward 打到控制台。"""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                print(
                    f"[t={self.num_timesteps:>8}] "
                    f"ep_rew={ep['r']:+8.2f} ep_len={ep['l']:>4} "
                    f"ep_time={ep['t']:6.1f}s"
                )
        return True


def main() -> None:
    for d in (TB_DIR, MODEL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_venv = VecNormalize(
        VecMonitor(
            DummyVecEnv([make_env(i) for i in range(N_ENVS)]),
            filename=str(LOG_DIR / "monitor"),
        ),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    eval_vn = VecNormalize(
        VecMonitor(DummyVecEnv([make_env(rank=9_999)])),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    eval_cb = EvalCallback(
        eval_env=eval_vn,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(LOG_DIR),
        eval_freq=max(20_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(200_000 // N_ENVS, 1),
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_lander",
    )
    print_cb = EpisodeRewardPrint()

    model = PPO(
        policy="MlpPolicy",
        env=train_venv,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(TB_DIR),
        seed=SEED,
        device="auto",
    )

    model.learn(
        total_timesteps=TOTAL_TS,
        callback=[eval_cb, ckpt_cb, print_cb],
        progress_bar=False,
    )

    final = MODEL_DIR / "ppo_lander_final.zip"
    vn_path = MODEL_DIR / "vecnormalize.pkl"
    model.save(final)
    train_venv.save(str(vn_path))
    print(f"\nfinal model: {final}")
    print(f"vecnormalize: {vn_path}")
    print(f"best  model: {MODEL_DIR / 'best' / 'best_model.zip'}")
    print(f"tensorboard: tensorboard --logdir {TB_DIR}")


if __name__ == "__main__":
    main()
