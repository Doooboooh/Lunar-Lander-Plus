"""MAPPO (Multi-Agent PPO) for 三飞船 LunarLander。

架构：中心化训练 + 分散执行 (CTDE)
  - Actor（参数共享）：每艘飞船共用一个策略网络，输入各自的局部观测（23 维），
    输出 4 个离散动作的 logits。执行时每艘只看自己的局部观测。
  - Critic（中心化）：输入全局状态（3 艘观测拼接 = 69 维），输出全局价值 V。
    训练时用全局状态估 GAE，缓解多智能体的非平稳性（这是 MAPPO 优于独立 DQN 的关键）。

复用：环境 multi_agent_env.MultiAgentLunarLander（三体同框+碰撞物理已验证）。
复用 ppo.py 的 PPO-clip 目标与 GAE 思路。

用法：
    python -m lunar_lander_rl.multi_lander.mappo --updates 400
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..common import ensure_dir, get_device, save_history, save_json, set_seed
from .multi_agent_env import MultiAgentLunarLander, OBS_DIM

N_AGENTS = 3
LOCAL_DIM = OBS_DIM                 # 23，每艘局部观测
GLOBAL_DIM = OBS_DIM * N_AGENTS     # 69，全局状态（critic 输入）
ACTION_DIM = 4


class ActorNet(nn.Module):
    """分散 actor：输入局部观测，输出动作 logits。三艘参数共享。"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class CriticNet(nn.Module):
    """中心 critic：输入全局状态，输出 V。"""
    def __init__(self, global_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)


@dataclass
class MAPPOConfig:
    updates: int = 400
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
    output_dir: str = "outputs/multi_lander/mappo"


def compute_gae(rewards, values, last_value, gamma, lam):
    """GAE：rewards/values 形状 [T, N_agents]。返回 advantages [T,N], returns [T,N]。"""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    lastgaelam = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 0.0
            nextvalues = last_value
        else:
            nextnonterminal = 1.0
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def evaluate(actor, device, episodes: int, seed: int) -> dict:
    env = MultiAgentLunarLander(render_mode=None)
    rng = np.random.default_rng(seed)
    rets, landed_tot, impacts = [], [], []
    for ep in range(episodes):
        obs_list, _ = env.reset(seed=seed + ep)
        ep_r = [0.0] * N_AGENTS
        ep_impact = 0
        for _ in range(1000):
            with torch.no_grad():
                obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
                actions = actor(obs_t).argmax(dim=1).cpu().numpy().tolist()
            obs_list, rewards, term, trunc, info = env.step(actions)
            for i in range(N_AGENTS):
                ep_r[i] += rewards[i]
            for ev in info.get("collision_events", []):
                if ev.get("impact"):
                    ep_impact += 1
            if term or trunc:
                break
        rets.append(float(np.mean(ep_r)))
        landed_tot.append(sum(info.get("done_flags", [])))
        impacts.append(ep_impact)
    env.close()
    return {"mean_return": float(np.mean(rets)),
            "mean_landed": float(np.mean(landed_tot)),
            "mean_impacts": float(np.mean(impacts))}


def train(cfg: MAPPOConfig):
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[MAPPO] device={device}, local_dim={LOCAL_DIM}, global_dim={GLOBAL_DIM}")
    out = ensure_dir(cfg.output_dir)

    env = MultiAgentLunarLander(render_mode=None)
    actor = ActorNet(LOCAL_DIM, ACTION_DIM, cfg.hidden_dim).to(device)
    critic = CriticNet(GLOBAL_DIM, cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)

    history = []
    best_score = float("-inf")
    rng = np.random.default_rng(cfg.seed)
    global_step = 0

    obs_list, _ = env.reset(seed=cfg.seed)
    try:
        for update in range(1, cfg.updates + 1):
            # ---- 收集 rollout ----
            buf_obs = []        # [T, N, local_dim]
            buf_global = []     # [T, global_dim]
            buf_act = []        # [T, N]
            buf_logp = []       # [T, N]
            buf_val = []        # [T, N]
            buf_rew = []        # [T, N]
            buf_done = []       # [T] 全局 done
            ep_returns = [0.0] * N_AGENTS
            n_episodes = 0

            for _ in range(cfg.rollout_steps):
                obs_arr = np.array(obs_list, dtype=np.float32)             # [N, local]
                global_arr = obs_arr.reshape(-1)                           # [global]
                buf_obs.append(obs_arr)
                buf_global.append(global_arr)
                with torch.no_grad():
                    logits = actor(torch.tensor(obs_arr, device=device))   # [N, 4]
                    dist = Categorical(logits=logits)
                    act = dist.sample()
                    logp = dist.log_prob(act)
                    g = torch.tensor(global_arr, device=device).unsqueeze(0)
                    v = critic(g).squeeze(0)                               # [N]? 实为标量
                    # critic 输出全局单值，复制到每艘
                    v_each = v.expand(N_AGENTS)
                actions = act.cpu().numpy().tolist()
                buf_act.append(actions)
                buf_logp.append(logp.cpu().numpy())
                buf_val.append(v_each.cpu().numpy())
                next_obs, rewards, term, trunc, info = env.step(actions)
                buf_rew.append(np.array(rewards, dtype=np.float32))
                buf_done.append(bool(term or trunc))
                for i in range(N_AGENTS):
                    ep_returns[i] += rewards[i]
                global_step += 1
                obs_list = next_obs
                if term or trunc:
                    n_episodes += 1
                    obs_list, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))

            # last value（bootstrap）
            with torch.no_grad():
                g = torch.tensor(np.array(obs_list, dtype=np.float32).reshape(-1),
                                 device=device).unsqueeze(0)
                last_v = critic(g).squeeze(0).cpu().numpy()
                last_v_each = np.full(N_AGENTS, float(last_v))

            obs_b = torch.tensor(np.array(buf_obs), device=device)          # [T,N,local]
            glob_b = torch.tensor(np.array(buf_global), device=device)      # [T,global]
            act_b = torch.tensor(np.array(buf_act), device=device)          # [T,N]
            logp_b = torch.tensor(np.array(buf_logp), device=device)        # [T,N]
            val_b = torch.tensor(np.array(buf_val), device=device)          # [T,N]
            rew_b = torch.tensor(np.array(buf_rew), device=device)          # [T,N]
            adv_b, ret_b = compute_gae(np.array(buf_rew), np.array(buf_val),
                                       last_v_each, cfg.gamma, cfg.gae_lambda)
            adv_b = torch.tensor(adv_b, device=device)
            ret_b = torch.tensor(ret_b, device=device)
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            # ---- PPO 更新（flatten 时间×agent）----
            T, N = act_b.shape
            flat = lambda x: x.reshape(T * N, -1) if x.dim() == 3 else x.reshape(T * N)
            obs_f = obs_b.reshape(T * N, LOCAL_DIM)
            act_f = act_b.reshape(T * N)
            logp_f = logp_b.reshape(T * N)
            adv_f = adv_b.reshape(T * N)
            ret_f = ret_b.reshape(T * N)
            # critic 用每步的全局状态（按 agent 展开后，每个 agent 对应同一全局状态）
            glob_f = glob_b.repeat_interleave(N, dim=0)                     # [T*N, global]

            bsz = cfg.minibatch_size
            for _ in range(cfg.update_epochs):
                idx = torch.randperm(T * N, device=device)
                for s in range(0, T * N, bsz):
                    mb = idx[s:s + bsz]
                    logits = actor(obs_f[mb])
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(act_f[mb])
                    ent = dist.entropy()
                    v = critic(glob_f[mb])
                    # advantage 对应每艘
                    a = adv_f[mb]
                    logratio = new_logp - logp_f[mb]
                    ratio = logratio.exp()
                    s1 = -a * ratio
                    s2 = -a * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    actor_loss = torch.max(s1, s2).mean()
                    critic_loss = F.mse_loss(v, ret_f[mb])
                    loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * ent.mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                    opt.step()

            mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
            history.append({"update": update, "mean_return": mean_ret})
            if update == 1 or update % 20 == 0:
                print(f"[MAPPO] update={update:04d} mean_ret={mean_ret:7.1f} eps={n_episodes}")
            if update % 40 == 0:
                m = evaluate(actor, device, cfg.eval_episodes, cfg.seed + 10000)
                score = m["mean_return"] + 25 * m["mean_landed"] - 30 * m["mean_impacts"]
                print(f"   eval: ret={m['mean_return']:.1f} landed={m['mean_landed']:.1f}/3 "
                      f"impacts={m['mean_impacts']:.1f} score={score:.1f}")
                if score > best_score:
                    best_score = score
                    torch.save(actor.state_dict(), out / "best_actor.pt")
                    torch.save(critic.state_dict(), out / "best_critic.pt")
    finally:
        env.close()

    save_history(history, out)
    save_json({"algorithm": "MAPPO", "config": cfg, "best_score": best_score},
              out / "metrics.json")
    # 总是保存最后的 actor
    torch.save(actor.state_dict(), out / "last_actor.pt")
    print(f"[MAPPO] done. best_score={best_score:.1f} -> {out}/best_actor.pt")
    return {"best_score": best_score}


def parse_args() -> MAPPOConfig:
    p = argparse.ArgumentParser(description="MAPPO for 三飞船 LunarLander.")
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="outputs/multi_lander/mappo")
    a = p.parse_args()
    return MAPPOConfig(updates=a.updates, seed=a.seed, device=a.device, output_dir=a.output_dir)


if __name__ == "__main__":
    train(parse_args())
