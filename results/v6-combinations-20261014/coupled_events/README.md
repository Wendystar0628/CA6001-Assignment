# 第三轮：支持集组合基座 × 事件层厚解码

这是**观察前两轮验证结果之后形成的探索性组合**。三个幅度基座由前两轮的连续预测与厚度结果提出，再复用此前已固定的两种事件机制。它不构成独立确认，也没有重新搜索混合系数或事件参数。

| 幅度基座 | Potts 候选 | Potts + 软格数候选 |
|---|---|---|
| `support_residual_stack_structural` | `v6_coupled_residual_stack_potts` | `v6_coupled_residual_stack_cardinality` |
| `support_stack_structural` | `v6_coupled_stack_potts` | `v6_coupled_stack_cardinality` |
| `v6_fpca_residual_w050` | `v6_coupled_fpca50_potts` | `v6_coupled_fpca50_cardinality` |

事件侧完全沿用 `v6.indicator`：最近三个已测 CPT 的 IDW² 指示概率、有效样本数与 Jeffreys 平滑、已测剖面低阻力格数的空间加权中位数。幅度尺度 `τ=0.5`、相邻转换代价 `0.25`、符号边界 `±0.05 logqc` 保持不变；两种机制的格数势能强度分别为 `0` 和 `0.25`。

每个基座都重新计算整根剖面的 MAP 标签、约束后的实际幅度及同一个 Gibbs 能量的精确边缘概率。**不复用基座的风险分数或概率**。H_point 仍来自实际预测 `qc < 1 MPa` 的厚度；期望厚度仅作独立诊断。

基座缓存只通过 Parquet 列投影读取 `fold_id, site_id, seed, budget_n, location_id, depth_index, pred_log_qc`，并校验完整且唯一的冻结位置／深度身份。查询真值、旧校准分数、原生概率与区间均未送入组合器。评价标签只由公共导出函数附加。全部六个候选均使用相同 60 个冻结状态。

```powershell
.venv/Scripts/python.exe -m cpt_budget.v6.coupled_events
.venv/Scripts/python.exe -m pytest tests/test_v6_coupled_events.py -q
```

两项局部检查通过：缓存只读取准许列且乱序正确对齐；错误或重复身份显式拒绝。事件推断精确性与支持／查询隔离复用已通过的 `test_v6_indicator.py` 检查。本轮没有改动该算法。

逐候选预测位于相应模型目录的 `predictions.parquet`，逐剖面解码变化位于 `decoding_diagnostics.csv`。统一排序、风险交叉校准与综合门槛由公共比较流程完成。
