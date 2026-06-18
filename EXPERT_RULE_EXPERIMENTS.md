# 手工专家系统实验记录

本文档记录不训练模型、仅使用规则/专家系统完成 LunarLander 基础降落与 waypoint 路线任务的实现和迭代结果。

## 改动清单

| 文件 | 内容 |
|---|---|
| `lunar_lander_rl/expert_policy.py` | 新增 `RuleBasedLanderPolicy` 和 `ExpertConfig`，支持基础降落、当前航点追踪、路线前瞻、路线完成后切回降落。 |
| `lunar_lander_rl/run_expert_suite.py` | 新增专家系统批量评测入口，自动输出 `summary.json`、`summary.md`，并追加 `outputs/expert_rules/iteration_log.jsonl`。 |
| `lunar_lander_rl/trajectory_eval.py` | 保留原 `landed_after_route_rate`，新增 `touchdown_after_route_rate` 与 `settled_after_route_rate`，用于区分严格双脚接触和低速稳定触地。 |

## 当前专家策略

策略分三段：

1. 基础任务：沿用启发式姿态/速度控制，目标为中心降落点。
2. waypoint 阶段：根据扩展观测里的当前目标相对位置生成期望水平/垂直速度，再用离散主发动机和侧向发动机追踪该速度。
3. 路线感知前瞻：评测脚本把完整 waypoint 列表传给专家策略；策略根据 `progress` 推断当前航点索引，在接近非最后航点时加入很小的下一段方向速度，减少长路线中反复悬停。

没有神经网络训练，也没有读取已有模型 checkpoint。

## 最终复现命令

```bash
python -m lunar_lander_rl.run_expert_suite \
  --episodes 20 \
  --seed 30000 \
  --tasks two_waypoint,orbit,figure_eight \
  --waypoints-file examples/drawn_diamond_path.json \
  --custom-label drawn_diamond \
  --output-dir outputs/expert_rules/final_current_20ep \
  --label final_current_20ep \
  --notes "final current default expert policy; post-route hover disabled"
```

完整结构化结果：

- `outputs/expert_rules/final_current_20ep/summary.json`
- `outputs/expert_rules/final_current_20ep/summary.md`
- `outputs/expert_rules/iteration_log.jsonl`

## 最终成绩

评测设置：20 episodes，seed 30000，waypoint radius 0.16，环境默认 1000 步限制。

| 任务 | 平均回报 | 完成点数 | 路线完成率 | 双脚接触率 | 触地率 | 稳定触地率 | 平均最终距离 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 基础平稳降落 | 275.49 | - | - | - | - | - | - |
| `two_waypoint` | 378.15 | 2.00 / 2 | 1.00 | 0.05 | 0.65 | 0.60 | 0.384 |
| `orbit` | 551.49 | 4.00 / 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.023 |
| `figure_eight` | 623.34 | 8.00 / 8 | 1.00 | 0.50 | 0.75 | 0.65 | 0.101 |
| `drawn_diamond` | 550.92 | 4.00 / 4 | 1.00 | 0.90 | 1.00 | 1.00 | 0.044 |

结论：规则专家已稳定完成 2 点、4 点、8 点和自定义路线的 waypoint 顺序任务，基础降落平均分也超过 Gym 常用 solved 标准 200。长路线的严格双脚接触率仍受 1000 步截断和最后一帧接触状态影响，因此报告中同时保留了触地率和稳定触地率。

## 关键迭代记录

| 迭代 | 主要调整 | 结果 |
|---|---|---|
| `iter_001` | 初版：当前航点追踪 + 完成路线后标准降落 | 2 点、4 点、手绘 4 点可完成；8 字只完成 7.2 / 8，路线完成率 0.20。 |
| `iter_003` | 过强横向控制 | 8 字退化到 4.4 / 8，说明不能靠暴力增益解决。 |
| `iter_004` | 加入路线前瞻但默认过强 | 多路线漏点，说明前瞻必须只在接近航点时小幅加入。 |
| `iter_006` | 中等目标速度 + 小范围前瞻 | 8 字路线完成率达到 1.00，但落地窗口仍偏紧。 |
| `iter_008` | 单脚接触时继续做姿态修正 | 4 点和手绘路线稳定触地改善，8 字路线保持 1.00 完成率。 |
| `iter_011_confirm` | 20 episode 确认 | 2 点、4 点、8 点、手绘路线均 1.00 路线完成率。 |
| `iter_012` | 尝试路线后保高返航 | 路线完成但触地率为 0，因保高过度导致截断；默认关闭。 |

