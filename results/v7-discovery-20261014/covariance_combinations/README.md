# v7 第二轮：协方差/图结构与连续预测基座组合

这 6 个方案在第一轮实测后确定，用于检验整剖面协方差和图双调和重建的厚度结构，能否补充更强的连续预测。它们是探索性追加组合，没有独立测试确认，也不称为事先注册的验证。

组合固定为两种基座 × 三种结构成员，每个候选都采用 **0.75 × 基座 log(qc) + 0.25 × 结构成员 log(qc)**，没有搜索权重：

- 基座：`v7_simplex_linear`；既有最佳残差 `residual_ft_crps_guarded_a050`。
- 结构成员：`v7_cokriging_full`、`v7_cokriging_lmc`、`v7_graph_biharmonic`。

六个模型名分别为 `v7_{simplex,residual}_{cokfull,coklmc,biharmonic}_b075_gp`。全部读取冻结的预测列，按 fold/site/seed/budget/location/depth 六列一对一对齐，组合器不读取查询真值。每个候选已经导出完整 60 状态、8640 行。

六个风险头均采用相同、冻结的 **GP 原生 `p_low_1`**；没有使用先前交叉校准后的风险分数。协同克里金第一轮的 90% 区间明显低覆盖，所以其概率及区间未被复制。点预测与风险头是独立组件，GP 概率不代表组合点预测的完整 CDF，也没有给组合模型添加借来的置信区间。统一风险校准和评价由 v7 汇总流程执行。

`src/cpt_budget/v7/covariance_combinations.py` 可重新导出；两项定向测试验证固定权重、凸包范围及成员不完整/错位时报错。没有修改其他家族的模型或输出。

```powershell
.venv/Scripts/python.exe -m cpt_budget.v7.covariance_combinations
.venv/Scripts/python.exe -m pytest tests/test_v7_covariance_combinations.py -q
```
