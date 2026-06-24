"""拓展3：多个飞船顺序降落模块。

公开接口：
  - SequentialMultiLanderEnv : 多飞船顺序降落环境
  - MultiLanderConfig        : 配置
  - LanderScheduler          : 调度层（目标落点 + 起飞点规划）
  - run_demo                 : 命令行入口（跑 episode + 统计 + 可选 GIF）
"""
from .config import MultiLanderConfig
from .env import SequentialMultiLanderEnv, EpisodeReport, LanderResult
from .scheduler import LanderScheduler, ShipPlan

__all__ = [
    "MultiLanderConfig",
    "SequentialMultiLanderEnv",
    "EpisodeReport",
    "LanderResult",
    "LanderScheduler",
    "ShipPlan",
    "run_demo",
]


def run_demo() -> None:
    from .demo import main
    main()
