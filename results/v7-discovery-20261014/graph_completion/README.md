# v7 图低秩与矩阵重建探索

本组测试四个新机制，不修改 v5 Huber 求解器，也不基于评价标签搜索参数。全部使用同一 60 状态、8640 单元；不训练源场地模型、不使用外层测试。

## 机制与边界

矩阵行为全部计划位置、列为 24 个深度单元。输入图只使用允许的计划 `xy`：4 邻域对称并集加欧氏最小生成树，高斯边权，平均加权度归一化为 1。整根缺失不能单靠低秩条件识别，这里的外推依据明确来自空间图。

令 `Y` 为已测完整剖面，`c` 为已测剖面的逐深度中位数，`X` 为中心化待重建矩阵，`L` 为位置图 Laplacian，`D` 为邻深度单元差分矩阵。所有方法满足 `X[known] = Y-c`。

- `v7_graph_nuclear`：最小化 `0.5 tr(Xᵀ L X) + 0.025 ||XD||²_F + 0.25 ||X||_*`。整根剖面的变化共享低秩结构。
- `v7_graph_nuclear_gradient`：最小化 `0.5 tr(Xᵀ L X) + 0.25 ||XD||_*`。低秩施加于层间变化，避免直接压缩绝对阻力幅值。
- `v7_graph_biharmonic`：在固定已测边界下求解 `L² + 0.05L` 的延拓；可跨越简单凸组合插值范围，也可能过冲。
- `v7_graph_lowrank_local`：对仅已测矩阵做 rank≤3 SVD，低秩部分用双调和图延拓；被截断的局部残差用最近 3 根、逆距离平方插值。它保留局部异常的可能性，也可能传播局部噪声。

两个核范数问题用 ADMM：rho=1，核范数权重0.25，最大1000步，绝对容差1e-6、相对容差1e-5。行、列小矩阵特征分解解精确边界 Sylvester 子问题，SVD 软阈值解核范数近端步。primal/dual 诊断仅适用于核范数 ADMM；另两个方法为直接线性求解，无迭代收敛过程。

所有参数和四个机制在实测前确定，并保留本轮负结果。方法本身无原生事件概率，统一风险校准由总评估器完成。

## 实测结果

下表使用预算 5/7/9/11/13 的归一化梯形 AUBC，先平均种子与同场地重复折，再平均 4 个不同场地。它不是 60 状态的简单平均。

| 方法 | log-MAE | 厚度 MAE/m | 点召回 | 点误报率 |
|---|---:|---:|---:|---:|
| 图核范数 | 0.632115 | 0.651693 | 0.307575 | 0.059175 |
| 图深度差分核范数 | 0.634549 | 0.655762 | 0.304599 | 0.064892 |
| 图双调和 | 0.676841 | 0.562012 | 0.442642 | 0.102896 |
| 图低秩＋局部残差 | 0.665382 | 0.566569 | 0.453786 | 0.096848 |

图核范数保持相对较低连续误差，但未同时改善厚度与低阻力召回。双调和和低秩局部残差更有利于厚度与召回，但连续预测与误报的代价明显。因此本组没有宣称独立晋升或综合获胜；后两者可以作为后续组合的互补成员。

两个核范数候选均 60/60 状态满足预先规定的 ADMM primal/dual 停止条件，最多分别162和214步；全部240状态已测边界误差为0。6项针对性检查通过：恒定层状剖面、已测边界、顺序不变、非平凡求解收敛、奇异值软阈值与几何输入拒绝标签。

## 复现与证据

```powershell
.venv/Scripts/python.exe -m cpt_budget.v7.graph_completion
.venv/Scripts/python.exe -m pytest tests/test_v7_graph_completion.py -q
```

各模型目录保存完整 `predictions.parquet`、`metrics.csv`、60状态缓存及 `solver_diagnostics.csv`。汇总见 `point_summary.csv`、`point_by_site.csv`；配置见 `protocol.json`。所有 Consolidated 和逐状态文件已标为 `0.7-discovery` / `v7-discovery-20261014`。

方法依据：[Matrix Completion on Graphs](https://arxiv.org/abs/1408.1717)、[A Singular Value Thresholding Algorithm for Matrix Completion](https://arxiv.org/abs/0810.3286)、[ADMM 算法与停止条件](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)。实现为针对 CPT 的适配，不声称复现原论文全部实验或其保证条件。
