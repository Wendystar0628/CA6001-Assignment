# 当前12方案的证据

模型名单见[`configs/catalog.json`](../../configs/catalog.json)，状态见[IMPLEMENTATION_STATUS](../../docs/IMPLEMENTATION_STATUS.md)。

- `all_model_ranking.csv`：仅保留12方案的4场地预算AUBC指标，`source_model_id`保留原实验身份。
- `metrics.csv` / `curves.csv` / `unique_site_metrics.csv`：相应状态、预算积分和场地指标。
- `migration_checks.csv` / `migration_summary.json`：数值一致性验证，12×60状态的实际推理均通过参考预测与标签对照。
- `runtime_*.csv`：仅保留有关的v0.8固定核实测行。TabPFN底层残差组件的88ms不冒称两个完整TabPFN组合的时延，未单测的组合留空。

成绩使用冻结的验证场地与评价协议。结果来自反复使用的4个验证场地，不能当作独立确认或节省补勘根数的证据。
