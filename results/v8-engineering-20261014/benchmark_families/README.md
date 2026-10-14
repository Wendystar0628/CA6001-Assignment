# v0.8 完整专项流水线性能

> 本目录为未固定 CPU 核的诊断数据。由于观察到同一 IDW 的跨轮约 2 倍波动，最终报告采用 [固定 logical core 2 的统一基准](../benchmark_pinned/README.md)。保留本目录用于追踪计时混杂，不跨批次推导最终加速比。

在第一组 17 条目完成后，继续串行测量 7 个完整流水线条目，没有与其他训练或测试并行。每条目仍为相同 12 状态、每态 1 次预热 + 3 次重新 `condition/predict`，合计 252 次计时。硬件、1 线程 CPU 与 CUDA 同步规则同 [主基准](../benchmark/README.md)。

所有条目都实际计算完整点预测和相同的 TabPFN / 图 Huber / 支持误差 CDF 原生经验风险头。没有把独立风险头免费当作缓存，也不包含外部验证校准拟合。

| 完整流水线 | 原实现中位 / p95 ms | 优化实现中位 / p95 ms | 12 状态配对加速比 |
|---|---:|---:|---:|
| Haar 厚度点预测 + 经验风险头 | 1255.89 / 1507.42 | 264.10 / 286.42 | 4.62× |
| 识别点预测 + 经验风险头 | 2192.43 / 2980.45 | 325.11 / 349.19 | 6.79× |

以上两对的完整点预测和原生概率均数值等价。三个改变质量决策的优化版本也实际计时，但不要求其点预测与旧算法相同：

| 完整质量优化版本 | 中位 ms | p95 ms |
|---|---:|---:|
| 识别 `localize` | 312.93 | 362.68 |
| 识别 `recall60` | 311.38 | 335.71 |
| 均衡厚度 `count_shrink` | 579.04 | 750.46 |

`localize` / `recall60` 与等价识别流水线处于相近成本量级；本次小幅计时差不是它们更快的证据。`count_shrink` 支持集内部选取计数收缩系数，需要额外计算；不能只用其决策阶段时间来代表完整代价。

全部 7 条目的原生风险概率均与既有 `v6_empirical_graph_union_local` 比对，最大绝对差 **8.88e-16**，0.2 阈值标签变化 **0**。因为本次质量变体只改变点预测，风险头仍相同；不能把点识别变化归因于风险头改善。

CUDA peak allocated 最大约 **197.00 MiB**。RSS 是整个进程工作集而不是各模型独占内存。初始化、逐次时延、p95 的样本边界和阶段计时口径沿用主基准。`stage_summary.csv` 与 `initialization_summary.csv` 是原始 CSV 的直接描述汇总，不是额外重跑。

```powershell
.venv/Scripts/python.exe -m cpt_budget.v8.benchmark --without-baselines --groups --registry cpt_budget.v8.runtime:family_benchmark_specs --output runs/v8-engineering-20261014/benchmark_families
```
