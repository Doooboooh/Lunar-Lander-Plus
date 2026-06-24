"""拓展3 demo：跑多飞船顺序降落 episode，打印统计，可选保存 GIF。

用法（在项目根目录 Lunar-Lander-Plus/ 下，venv 已激活）：

    # 默认：3 架飞船，复用 DQN 单飞船策略，跑 5 个 episode 统计
    python -m lunar_lander_rl.multi_lander.demo

    # 换策略 / 飞船数 / episode 数
    python -m lunar_lander_rl.multi_lander.demo --algorithm ppo --num-landers 4 --episodes 10

    # 录一段多飞船 GIF（用渲染环境逐帧录制，rgb_array，不弹窗）
    python -m lunar_lander_rl.multi_lander.demo --gif outputs/multi_lander.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from .config import MultiLanderConfig
from .env import SequentialMultiLanderEnv, EpisodeReport
from .scheduler import PLATFORM_X_MIN, PLATFORM_X_MAX


def _print_episode(ep: int, report: EpisodeReport) -> None:
    print(f"\n=== Episode {ep} ({report.num_landers} 艘飞船) ===")
    for r in report.per_lander:
        status = []
        status.append("平稳落地" if r.landed else "未平稳")
        status.append("平台内" if r.in_platform else "出平台")
        if r.collided:
            status.append("碰撞!")
        flags = "/".join(status)
        print(f"  飞船{r.index}: 目标x={r.plan.target_x:+.2f} 起飞x={r.plan.init_x:+.2f} "
              f"落点x={r.final_x:+.2f} 原生reward={r.raw_return:7.1f} "
              f"奖惩={r.bonus:+6.1f} 步数={r.steps:4d} [{flags}] {r.plan.note}")
    print(f"  -> 合计reward={report.total_return:8.1f}  "
          f"平稳={report.n_landed}/{report.num_landers}  碰撞={report.n_collided}  "
          f"全部成功={'是' if report.all_success else '否'}")


def run_episodes(cfg: MultiLanderConfig, episodes: int,
                 verbose: bool = True) -> list[EpisodeReport]:
    """跑若干个多飞船 episode，返回报告列表。"""
    reports: list[EpisodeReport] = []
    for ep in range(episodes):
        env = SequentialMultiLanderEnv(cfg)
        report = env.run_episode(seed=cfg.seed + ep * 101)
        reports.append(report)
        if verbose:
            _print_episode(ep, report)
        env.close()
    return reports


def summarize(reports: list[EpisodeReport]) -> dict:
    n = len(reports)
    total = [r.total_return for r in reports]
    landed = [r.n_landed for r in reports]
    collided = [r.n_collided for r in reports]
    success_rate = sum(r.all_success for r in reports) / n if n else 0.0
    summary = {
        "episodes": n,
        "num_landers": reports[0].num_landers if reports else 0,
        "mean_total_return": float(np.mean(total)) if total else 0.0,
        "std_total_return": float(np.std(total)) if total else 0.0,
        "mean_landed": float(np.mean(landed)) if landed else 0.0,
        "mean_collided": float(np.mean(collided)) if collided else 0.0,
        "all_success_rate": float(success_rate),
    }
    print("\n========== 多飞船顺序降落 统计 ==========")
    print(f"  算法(单飞船策略): {reports[0].per_lander[0].plan if False else ''}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


# ---------------------------------------------------------------------------
# GIF：用一个带 rgb_array 渲染的环境，逐帧录制一次完整的多飞船 episode
# ---------------------------------------------------------------------------
def save_multi_gif(cfg: MultiLanderConfig, gif_path: str | Path,
                   seed: Optional[int] = None, fps: int = 30) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("未找到 imageio，请先 pip install imageio") from exc

    from ..common import import_gym
    gym = import_gym()
    from gymnasium.envs.box2d.lunar_lander import VIEWPORT_W, SCALE, LEG_DOWN
    _CX = VIEWPORT_W / SCALE / 2.0
    from .policy import SingleLanderPolicy
    from .scheduler import LanderScheduler

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    # 用一个独立的渲染环境录制
    rec_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    u = rec_env.unwrapped
    policy = SingleLanderPolicy(cfg.algorithm, cfg.model_dir, cfg.hidden_dim, cfg.device)
    sched = LanderScheduler(cfg.target_spacing, cfg.safe_dist,
                            cfg.init_x_jitter, cfg.init_y, seed=cfg.seed)

    ep_seed = seed if seed is not None else cfg.seed
    landed_xs: list[float] = []
    frames: list[np.ndarray] = []

    def _relocate(init_x: float, init_y: float = 0.0) -> None:
        cx = u.lander.position.x
        native_y = u.lander.position.y
        new_pos = (cx + init_x, native_y + init_y)
        u.lander.position = new_pos
        u.lander.linearVelocity = (0.0, 0.0)
        u.lander.angle = 0.0
        u.lander.angularVelocity = 0.0
        for leg in u.legs:
            leg.position = new_pos
            leg.linearVelocity = (0.0, 0.0)

    try:
        for k in range(cfg.num_landers):
            # 时序模式：不累积历史占用；空间模式：用已降落落点做避障
            active = [] if cfg.mode == "sequential" else landed_xs
            plan = sched.plan_next(k, cfg.num_landers, active, mode=cfg.mode)
            rec_env.reset(seed=ep_seed + k)
            _relocate(plan.init_x, plan.init_y)
            # 录起飞帧
            frames.append(rec_env.render())
            for _ in range(cfg.max_steps_per_lander):
                # 读观测（重定位后重新构造）
                pos = u.lander.position
                vel = u.lander.linearVelocity
                obs = np.array([
                    (pos.x - _CX) / SCALE,
                    (pos.y - (u.helipad_y + LEG_DOWN / SCALE)) / SCALE,
                    vel.x / SCALE, vel.y / SCALE,
                    u.lander.angle, u.lander.angularVelocity,
                    1.0 if u.legs[0].ground_contact else 0.0,
                    1.0 if u.legs[1].ground_contact else 0.0,
                ], dtype=np.float32)
                action = policy.act(obs)
                _, _, term, trunc, _ = rec_env.step(action)
                frames.append(rec_env.render())
                if term or trunc:
                    rel_x = (u.lander.position.x - _CX) / SCALE
                    both = u.legs[0].ground_contact and u.legs[1].ground_contact
                    speed = float(np.hypot(vel.x, vel.y) / SCALE)
                    if both and PLATFORM_X_MIN <= rel_x <= PLATFORM_X_MAX \
                            and speed < cfg.flat_vel_threshold:
                        landed_xs.append(rel_x)
                    # 留几帧停顿便于观察落点
                    for _ in range(15):
                        frames.append(rec_env.render())
                    break
    finally:
        rec_env.close()

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"\n已保存多飞船顺序降落 GIF -> {gif_path}（{len(frames)} 帧）")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="拓展3：多个飞船顺序降落 demo")
    p.add_argument("--num-landers", type=int, default=3)
    p.add_argument("--algorithm", default="dqn",
                   choices=["dqn", "ppo", "actor_critic", "q_learning"])
    p.add_argument("--model-dir", default=None,
                   help="单飞船策略目录，默认按 algorithm 选 outputs/<algo>")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gif", default=None, help="保存一次多飞船 episode 为 GIF")
    p.add_argument("--gif-fps", type=int, default=30)
    p.add_argument("--render", action="store_true", help="弹窗渲染（调试）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir or f"outputs/{args.algorithm}"
    cfg = MultiLanderConfig(
        num_landers=args.num_landers,
        algorithm=args.algorithm,
        model_dir=model_dir,
        seed=args.seed,
        device=args.device,
        render=args.render,
    )
    reports = run_episodes(cfg, args.episodes, verbose=True)
    summarize(reports)

    if args.gif:
        save_multi_gif(cfg, args.gif, seed=args.seed, fps=args.gif_fps)


if __name__ == "__main__":
    main()
