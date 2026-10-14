# 模型与评价工件

`manifest.json`记录当前12个模型运行需要的文件。大权重位于本地`models/fold_0..2/`，冻结验证清单位于`evaluation/`。包含测量值的参考预测及权重不上传Git。

每个fold包含训练后的TabPFN、匹配XGBoost、CNP权重，以及FPCA基和GP趋势参数。`manifest.json`中的来源字符串用于实验追溯，运行时使用当前工件路径。请先运行`python -m cpt_budget status`检查文件；缺失时程序报错，不会自动替换为其他模型权重。

完整训练流程见Git标签`v0.8-engineering`。当前CLI支持模型推断与冻结评价状态验证。
