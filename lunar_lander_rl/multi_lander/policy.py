"""单飞船策略的统一封装。

拓展3 不重新训练飞船控制，而是复用基础实验里已经训练好的单飞船策略
（dqn / ppo / actor_critic / q_learning）。这里把"加载模型 -> act(obs)->action"
的细节封成一个可调用对象，供多飞船调度环境直接使用。

直接复用 lunar_lander_rl.evaluate.load_policy 的加载逻辑，保持和基础实验一致。
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from ..evaluate import load_policy as _base_load_policy


class SingleLanderPolicy:
    """一架飞船的控制器：吃 8 维观测，输出离散动作（0..3）。

    内部包装了基础实验训练好的网络/Q表，act() 即贪心执行。
    """

    def __init__(self, algorithm: str = "dqn", model_dir: str = "outputs/dqn",
                 hidden_dim: int = 128, device: str = "cpu") -> None:
        # 复用基础实验 evaluate.load_policy 的统一加载逻辑（四种算法通用）
        self._act_fn: Callable[[np.ndarray], int] = _base_load_policy(
            argparse.Namespace(
                algorithm=algorithm,
                model_dir=model_dir,
                hidden_dim=hidden_dim,
                device=device,
            )
        )
        self.algorithm = algorithm

    def act(self, obs: np.ndarray) -> int:
        """贪心选择动作。obs 为 LunarLander 的 8 维观测向量。"""
        return int(self._act_fn(obs))
