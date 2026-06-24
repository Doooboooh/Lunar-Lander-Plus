"""拓展3：多个飞船顺序降落 —— 配置。

把和"多飞船顺序降落"任务相关的参数集中在这里，方便实验和调参。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MultiLanderConfig:
    # --- 任务规模 ---
    num_landers: int = 3                # 一次 episode 控制的飞船数量
    max_steps_per_lander: int = 1000    # 每架飞船最多步数（沿用 LunarLander 默认）

    # --- 单飞船策略（复用基础实验训练好的模型）---
    algorithm: str = "dqn"              # dqn / ppo / actor_critic / q_learning
    model_dir: str = "outputs/dqn"      # 单飞船策略权重目录
    hidden_dim: int = 128               # 与基础实验网络一致
    device: str = "cpu"

    # --- 顺序降落 / 飞船间约束 ---
    # 重要发现：预训练单飞船策略是"回中器"——无论从哪起飞，落点都收敛到平台中心
    # ±0.1（实测 15 次 std≈0.027）。因此"空间错开落点"在不重训策略的前提下不可行。
    # 本模块采用"时序错开"：多架飞船依次起飞降落，前一架完成后下一架再进入，
    # 用时序顺序而非空间分离实现多飞船任务。target_* 仅作记录/展示用。
    mode: str = "sequential"            # sequential(时序错开,默认) / spatial(空间错开,需重训策略)
    target_spacing: float = 0.6         # 相邻飞船目标落点水平间距（spatial 模式用）
    safe_dist: float = 0.45             # 与已降落飞船的最小安全距离（米），小于则判碰撞
    init_x_jitter: float = 0.4          # 起飞 x 随机扰动幅度（模拟不同入射条件）
    init_y: float = 0.0                 # 起飞高度额外偏移（世界坐标米）；0=沿用原生高度

    # --- 任务级奖励权重（叠加到每架原生 reward 上）---
    collision_penalty: float = 80.0     # 落点离已降落飞船过近 / 碰撞
    order_bonus: float = 20.0           # 按预定顺序平稳降落的额外奖励
    all_success_bonus: float = 100.0    # 全部飞船平稳着陆的全局 bonus
    flat_vel_threshold: float = 3.0     # 判定"平稳"的速度阈值（沿用 LunarLander 着陆判据）

    # --- 杂项 ---
    seed: int = 0
    render: bool = False                # 渲染单飞船窗口（调试用）
    output_dir: str = "outputs/multi_lander"
