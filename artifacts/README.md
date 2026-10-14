# 模型与评价工件

`manifest.json`记录当前13个模型运行需要的文件（含字节数；`python -m cpt_budget status`同时核对大小与准备好的数据）。大权重位于本地`models/fold_0..2/`，冻结验证清单位于`evaluation/`。包含测量值的参考预测及权重不上传Git。

每个fold包含训练后的TabPFN、匹配XGBoost、CNP权重，以及FPCA基和GP趋势参数。`manifest.json`中的来源字符串用于实验追溯，运行时使用当前工件路径。缺失时程序报错，不会自动替换为其他模型权重。

`evaluation/reference_predictions.parquet`（v0.9.1，13方案×60状态）：11个沿用方案为v0.8归档预测（v0.9.0的`thickness`行更名为`thickness_guarded`；删除`tabpfn_fpca`；GP补充q10/q25/q75/q90，其余GP列与归档差≤9e-11）；`thickness`与`tabpfn_joint`为v0.9.1实时推理，在60个状态的逐状态指标与Git标签`v0.8-engineering`归档指标完全一致后冻结（`results/current/port_equivalence.csv`）。原v0.9.0文件原样保留为`evaluation/reference_predictions.v0.9.0.parquet`，任何命令都不读取它。

完整训练流程见Git标签`v0.8-engineering`。当前CLI支持模型推断、冻结状态验证（`verify`）、指标重算（`evaluate`）、归档对照（`port-check`）与统一计时（`benchmark`）。
