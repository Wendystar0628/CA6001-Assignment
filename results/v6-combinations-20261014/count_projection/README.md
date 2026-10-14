# 第四轮：固定学习格数的最小改动投影

第三轮 source-only 厚度头的半计数候选取得厚度改善，但事件树给出的深度排序对连续点值有代价。本轮固定每根剖面的同一计数，只更换低值格子的排序来源，再接入独立的冻结 GP 风险概率头。三个基座、计数规则和 .05 翻转幅度均先冻结，没有增加权重扫描。

## 信息边界

从 `thickness_head/profile_decisions.csv` 只读取 profile keys 与 `half_count_decision`、`predicted_continuous_count`、`residual_point_count`。不读取任何真值列或其他列。程序逐根核验：

`k = floor((predicted_continuous_count + residual_point_count)/2 + .5)`。

同时核验原残差计数与冻结残差点预测一致、k 与第三轮 `v6_count_head_half_residual` 最终点计数一致。每个新候选最终点预测的低值格数也严格等于 k，因此其每根剖面的厚度误差与第三轮半计数候选完全相同，而非仅宏平均碰巧相同。

基座分别为 FPCA 符号保护混合、图低值并集、support_stack_structural。按基座 log(qc) 从小到大选择 k 格；平手按深度顺序。保持已符合目标标签的点值，仅将需要翻转的值改为 −.05 或 +.05。在固定翻转规则下，这一排序使总绝对点值变化最小，测试对小剖面的全部 k 子集进行枚举比较。

风险概率仅从冻结参考的 GP `p_low_1` 列按精确 cell keys 读取，源文件其他字段不进入生成器。这个概率头与点投影独立：它不是新点模型的原生 log(qc) 分布，不移植 GP 区间。统一风险评价另行执行。

## 全部点结果

| 候选 | log-MAE | 厚度误差 m | 原点 Recall | 原点 FPR |
|---|---:|---:|---:|---:|
| v6_count_project_fpca_guard_gp | 0.626419 | 0.552246 | 0.396891 | 0.108743 |
| v6_count_project_graph_union_gp | 0.629651 | 0.552246 | 0.397150 | 0.110009 |
| v6_count_project_support_stack_gp | **0.623176** | 0.552246 | **0.423306** | **0.106282** |

每候选 60 状态、8,640 行、360 个 profile-state 实例；四个不同验证场地按既定方法等权汇总。结构 stacking 基座比第三轮事件排序半计数候选的 MAE 0.631088、Recall 0.409630 更好，同时严格保留其厚度。相对 IDW 的连续误差改善约 2.42%，尚未单独达到本轮额外 5% 连续误差操作目标；完整门槛以统一评估为准。

这里的最小改动目标只是对冻结基座的距离，不保证更接近真值。保持计数也不保证低值格子处在正确深度，所以还必须报告点 Recall/FPR。三个结果全部保留，不按查询成绩修改计数、基座或翻转幅度。

## 复现

```powershell
.venv/Scripts/python.exe -m cpt_budget.v6.count_projection
```

3 项定向测试通过，覆盖固定规则下 L1 最小性、预测字段白名单及计数公式、任意行序恢复与最终点计数。`projection_diagnostics.csv` 保存每根的 frozen/actual/base count、翻转格数与绝对点改变量；`execution_summary.json` 确认 360 根 profile-state 在三个候选均保持第三轮计数。
