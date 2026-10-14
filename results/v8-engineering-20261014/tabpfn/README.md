# v8 TabPFN 残差推理工程优化

本分支保持 v4 最佳 `residual_ft_crps_guarded_a050` 的权重、模型结构、一个 estimator、官方预处理、精度与 8 根查询 batch 不变。只优化实际 condition/predict 热路径；没有新增训练、搜索 alpha 或缓存最终预测。

## 实际改动

1. 一根完整已测剖面的邻居距离与排序在 24 个深度格之间共享，避免逐格建立 Pandas 子表。
2. 24 维空间特征中的 IDW 列同时用于支持集整位置 LOO 残差和查询基座，去除三次重复构造。
3. guarded 点方案只需要残差中位数，调用官方 inverse-CDF 仅解码 q=0.5，避免计算随后丢弃的 6 个其他分位数、均值、众数、CDF。
4. 原路径与优化路径都按折复用权重；冷启动初始化单列，不将 import/权重加载省略算作算法加速。

矩阵乘法的归约顺序必须谨慎处理。初稿把 kernel 加权和做成 GEMM，虽然特征差仅约 1e-16，却破坏低阻力比例的浮点并列值，官方 rank/quantile 预处理会放大该变化。本次最终实现保留每个深度的一维 NumPy 归约顺序，只共享几何、索引与数组，保证原语义。

## 全量实际推理验证

- RTX 4080，torch 2.11.0+cu128；每次计时边界使用 CUDA synchronize。
- 所有 60 个冻结状态都重新执行原路径和优化路径，共 8,640 格；两路径交替先后。
- 原/新 `pred_log_qc` **逐格精确一致，最大差 0，阈值标签变化 0**。
- 与已归档 v4 最佳残差预测逐格对比也精确一致。
- 五项定向测试覆盖同坐标排除、距离并列、行序、整位置 LOO、查询标签隔离、缺格 fallback 与错误深度。

连续 log-MAE 仍为 **0.620528**；厚度误差 **0.630371 m**，点召回 **37.3767%**、点误报 **7.2766%**。这是等价工程实现，不能声称质量得到改善。原 IDW 符号保护仍会限制低阻力识别的进一步收益。

开发期全 60 状态的 condition+predict 总耗时由 48.176 秒降至 8.802 秒，约 **5.47 倍**。该数值是在其他 CPU 分支并行工作的环境下测得，仅作诊断；最终正式速度比较以主代理统一串行 benchmark 为准。

`execution_timings.csv` 单列初始化、condition、predict。`equivalence.csv` 记录每个状态的数值与标签差；`profile.txt` 说明原瓶颈主要是重复特征构建，而不是 Transformer 前向。`point_by_site.csv` 保留场地异质性。风险校准由统一主流程执行，输出没有冒充原生概率。

复现：`.venv/Scripts/python.exe -m cpt_budget.v8.tabpfn`。

运行接口：`old_factory(fold_id, config=None)` 与 `new_factory(fold_id, config=None)`；对象的 `initialize()` 可单列真实权重加载，然后正常 `condition(observed)` / `predict(query)`。路径为 `src/cpt_budget/v8/tabpfn.py`。优化模型输出目录为 `v8_tabpfn_guarded_fast/`。
