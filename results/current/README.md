# 当前13方案的证据（v0.9.1）

模型名单见[`configs/catalog.json`](../../configs/catalog.json)，状态见[IMPLEMENTATION_STATUS](../../docs/IMPLEMENTATION_STATUS.md)，本次变更见[`CHANGES_v0.9.1.md`](CHANGES_v0.9.1.md)。

## 精度与风险（可由代码重算）

`python -m cpt_budget evaluate` 从 `artifacts/evaluation/reference_predictions.parquet` 重算下列4个文件（约30秒；`--source live` 改为实时推理）：

- `all_model_ranking.csv`：13方案的4场地预算AUBC指标，`source_model_id`保留原实验身份。
- `metrics.csv` / `curves.csv` / `unique_site_metrics.csv`：60个状态、预算积分和场地指标。60个状态只含39个不同的已知CPT集合和8个不同的评价集合：Lower Hutt与Tauranga在fold 0和fold 1中于相同位置评分，Tauranga预算5的集合在两个种子下相同；状态之间并不独立。
- `evaluation_provenance.json`：命令、风险校准与聚合口径。

风险列的来源由 `risk_source` 写明：

| risk_source | 含义 | 方案 |
|---|---|---|
| `shared_empirical_head` | 共享经验风险头 `v6_empirical_graph_union_local` 的 p_low_1（由TabPFN图并集点预测和经验误差分布得到，与方案自身点路径无关），因此这些方案的风险指标完全相同 | balanced, thickness, recognition, count_shrink, thickness_guarded, tabpfn_joint |
| `native_gp` | GP 高斯预测分布的 p_low_1 | gp |
| `platt_on_point` | 无原生概率；评价时对 −ŷ 做 Platt 校准 | constant, idw, xgb, tabpfn_graph, simplex, cnp |

所有方案的 risk_probability 均为评价时按模型、按外层fold的“留一调查组” Platt 交叉校准，阈值 0.2 来自漏判:误报 = 4:1 的课程情景成本。simplex 等 `platt_on_point` 方案的风险数字不是原生风险头。

## 一致性证据

- `port_equivalence.csv`：`python -m cpt_budget port-check`。13方案×60状态的逐状态指标与 Git 标签 `v0.8-engineering` 中归档的逐状态指标对照（点指标对 `source_model_id`；概率/风险指标对归档风险头，thickness 对 `v6_empirical_graph_union_local`）。计数完全相同，实数差 ≤1e-9。
- `migration_checks.csv` / `migration_summary.json`：`python -m cpt_budget verify` 的副本。13×60状态实际推理与参考预测对照：点预测、0值标签、p_low_1、GP分位数列（列消失或多出即失败）。

## 运行时间（v0.9.1统一测量）

`python -m cpt_budget benchmark --cpu-core 2 --publish results/current`：全部13方案同一协议（12个状态、1次预热+3次计时、单核单线程、CUDA同步计时、权重加载单独计时）。`runtime_summary.csv` 每行写明 `path`（full/point_only）、`device` 与 `device_policy`；`runtime_output_checks.csv` 为计时后与参考预测的对照；`runtime_protocol.json` 记录硬件、GPU负载快照与执行顺序。CPU/GPU频率未锁定。

成绩使用冻结的验证场地与评价协议，来自用于模型选择的4个验证场地，不能当作独立确认或节省补勘根数的证据；独立的最终测试结果见[`results/final_test/README.md`](../final_test/README.md)。
