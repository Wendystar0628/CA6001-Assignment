# v7 空间度量与成对剖面迁移

完成 4 个新机制候选与 1 个匹配控制，每个均为冻结验证矩阵的 60 状态、8640 深度行。以下为预算曲线积分后、先合并重复 fold 场地、再对 4 个独立场地等权的点指标；风险头与综合比较由 v7 总评统一处理。

| model_id | log-MAE | 厚度 MAE m | 点召回 | 点误报 |
|---|---:|---:|---:|---:|
| v7_metric_pair_transport | 0.625479 | 0.703776 | 0.243985 | 0.033868 |
| v7_source_pair_transport | 0.625618 | 0.701335 | 0.241962 | 0.036006 |
| v7_euclidean_all_control | 0.625653 | 0.669922 | 0.270319 | 0.047754 |
| v7_support_psd_metric | 0.625694 | 0.659180 | 0.271648 | 0.045991 |
| v7_source_pair_variance | 0.763065 | 0.813802 | 0.176086 | 0.027968 |

关键结论：**本组未发现数据驱动空间度量或源域剖面迁移的显著独立收益。** 项目原 IDW 使用最近 3 邻居；本组使用全部已测点，因此必须保留 Euclidean-all 匹配控制。几种方法相对原 IDW 的连续误差改善主要已被全邻居控制解释，不能归功于度量学习。PSD 度量较匹配控制仅小幅改善厚度；source transport 增加了厚度误差并降低召回；仅学习半方差再反比加权的方案明显较差。保留这些负结果供组合比较，不作新冠军声明。

## 实现机制与信息边界

- `source_pair_variance`：对应 fold 的 source train 场地内所有有向 CPT 对构成训练样本。按场地等总权重拟合 3 个深度带、每带 4 个非负径向半方差系数。查询与已测 anchor 的几何距离映射为预期差异，反比后归一化加权。权重非负不意味着该矩阵是 GP 正定协方差；本实现没有这种声明。
- `support_psd_metric`：仅当前 revealed 全剖面两两平方差作为监督，拟合二维二次型的 3 个系数，向欧氏矩阵岭收缩，投影为 SPD、限制条件数 ≤ 4、行列式归一为 1，然后全邻居 IDW。query 只提供几何，不提供目标剖面。
- `source_pair_transport`：source train 有向 CPT 对监督“anchor 剖面→邻近目标剖面”的修正。输入归一化距离、深度、已测 anchor 当前值/上下相邻差/全剖面均值与下四分位数/最小值。80 棵、最多 3 层和 7 叶的绝对误差 HGB，最小叶 80，source 场地等总权重。修正截断在 ±0.75 log(qc)，乘 0.5，且已测位置距离趋零时修正趋零。
- `metric_pair_transport`：使用支持集 PSD 距离加权上述 source pair 修正。
- `euclidean_all_control`：全部已测点的普通欧氏 IDW，作为同邻居数的匹配控制；不是第五个新算法。

成对数据相互依赖，pair 数量不是独立样本量。source 训练完全限于对应 fold 的 train_sites；当前目标场地的学习只使用 revealed CPT。评估真值由统一导出器在预测完成后加入；查询标签污染测试证实不会影响预测。没有读取外层 test。

## 复现与检查

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv/Scripts/python.exe -m cpt_budget.v7.metric_kernel
.venv/Scripts/python.exe -m pytest -q tests/test_v7_metric_kernel.py
```

7 项检查通过：SPD 正定、条件数上限、单位行列式和平移不变；径向项非负单调；五种方法的观测点精确还原及 query 标签污染不变。每 fold 的 `source_fold_*.json` 保留源场地、训练数量、半方差参数；`contexts.json` 保留每状态的支持集与度量矩阵。模型树没有存入 Git，种子与全部训练参数在源码中固定。

## 原始与官方依据

- [Xing 等，Distance Metric Learning with Application to Clustering with Side-Information，NeurIPS 2002](https://papers.nips.cc/paper_files/paper/2002/hash/c3e4035af2a1cde9f21e1ae1951ac80b-Abstract.html)：正半定空间度量学习的机制参考。本实现是 2D 监督岭拟合加 PSD 投影，并非原文凸优化复现。
- [scikit-learn HistGradientBoostingRegressor 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)：成对迁移模型的估计器依据。这里的 source pair 构造、输入限制和组合方式是本项目适配。
