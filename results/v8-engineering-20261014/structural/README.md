# v0.8 结构组合：真实推理路径的工程优化

本族保留现有最佳均衡方案的结构点基底，并对其实际推理过程优化。该基底是 `0.5 × simplex + 0.5 × support_stack_structural`。它不包含 GP／经验风险头，也不包含后续整曲线厚度决策；因此这里的 point-only 指标不能直接替代最终均衡系统的完整指标。

## 结果与采用意见

| 实现 | 预算曲线 log-MAE | 厚度 MAE/m | 点召回 | 点误报 |
|---|---:|---:|---:|---:|
| 数值等价优化 `v8_structural_exact` | 0.592896 | 0.615723 | 0.317633 | 0.059480 |
| LOO 可靠性门控 `v8_structural_reliability` | 0.595472 | 0.627604 | 0.320999 | 0.058818 |
| 五成员 LOO stacking `v8_structural_loo_simplex` | 0.604937 | 0.628092 | 0.319969 | 0.061999 |

采用数值等价的快速实现；两个质量变体未改善连续误差与厚度，保留为负结果。全部指标使用项目统一口径：每折／场地／种子的 5 个预算点归一化梯形积分，再平均种子、同场地重复折，最后 4 个唯一场地等权平均。不是预算点的算术平均。

完整 60 状态、8,640 单元与 `v7_simplex_structural_s050_gp` 点预测逐格比较，最大绝对差 `1.5543122344752192e-15`，低阻力阈值标签变化 **0**。共享三候选执行和仅执行快速原方案的 `new_factory` 路径均通过全矩阵等价比较。

## 真实工程变化

- 每次 `condition` 仅提取一次位置、完整剖面和深度数组，避免各 LOO 子集反复执行 Pandas pivot、groupby 与逐深度特征构造。
- 一次计算已测位置间距离和标准化形状距离，复用至所有 whole-location LOO 子问题。每个子问题仍排除相同 near group 与共址剖面。
- FPCA 的源基完全不变；按支持集大小把核矩阵求解打包为批量 `numpy.linalg.solve`，保留每个 LOO 子集各自的截距、距离尺度及系数。
- 对单调配准动态规划的同层节点向量化，严格保留原状态网格、代价加法顺序和优先选择 identity 的平局规则。标准化形状每根独立计算，因此有向 `(reference, source)` 配准可在同一已测上下文内跨邻居三元组和 LOO 子集复用。
- 每个支持子集只建立一次 Delaunay 三角网和凸包，多个查询复用。凸包外投影、共址平均与共线退化规则均保持原样。
- `new_factory` 的 exact 模式跳过质量变体才需要的 LOO-simplex 和五成员权重拟合。
- 每次 `condition` 清空几何／配准缓存，已测标签改变后绝不沿用旧配准；跨状态重放预测没有进入计时路径。

## 质量变体的边界

五成员版本把 simplex 加入原 IDW、FPCA 核岭、形状配准、加权中位数，以 0.25 步长的非负单纯形权重网格，在当前已测剖面的 whole-location LOO 预测上拟合。目标为 `L1 + 0.5 × soft thickness MAE + 0.5 × weighted event Brier + 0.05 × weight L2`，先验为五成员均匀权重。可靠性版本按原固定组合与五成员组合的训练目标差，线性裁剪出 `[0,1]` 门控系数。

这两个方案用相同已测标签选择并评估权重，因此记录的是拟合目标，不能解释为独立验证。测试集与未揭示查询标签从未进入这些路径；没有根据场地名称选择不同参数。

## 调用与计时

```python
from cpt_budget.v8.structural import old_factory, new_factory, StructuralPredictor

old = old_factory(fold_id)  # 原实现真实推理，不读冻结预测
fast = new_factory(fold_id)
fast.condition(observed)
prediction = fast.predict(query_geometry)

# 需要质量变体时，单独构造对应算法：
candidate = StructuralPredictor(fold_id, variant="loo_simplex")
candidate.condition(observed)
prediction = candidate.predict(query_geometry)
```

`initialization_seconds` 记录源基载入；`stage_timings` 记录 condition、数组／FPCA 预计算、LOO 成员、权重拟合、predict。当前并行实验中源基载入中位数约 1.2 ms，仅为诊断值。正式速度比较由主流程在所有并行实验结束后，以单线程、串行配对和相同冷／热上下文口径执行，不将这里的并发时间用于跨家族速度排名。

启动三个候选完整矩阵：

```powershell
.venv/Scripts/python.exe -m cpt_budget.v8.structural
```

3 项定向检查通过：动态规划路径等价、实时原／新点头及查询行序／标签隔离、near-group 与共址 LOO 排除。共享源基文件仍位于 `runs/v5-exploration-20261014/functional/bases/fold_{0,1,2}/`；没有重新训练源基或更改旧模型。
