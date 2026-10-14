# 第二轮：Delaunay 与现有强方案组合

第一轮 Delaunay 分片线性插值取得连续误差 0.607248、厚度误差 0.585286 m 后，固定本组 10 个候选。它们是第一轮结果驱动的探索，未独立预注册，不修改原 Delaunay、GP 或既有成员的参数。

- Delaunay 与四个成员（v4 最佳残差、FPCA 残差 0.5、图残差 low-union、支持集自适应结构组合）分别取 Delaunay 权重 0.50/0.75，共 8 个线性混合。
- Delaunay、FPCA 残差 0.5、图残差 low-union 逐单元取中位数，形成 1 个稳健混合。
- 保持 Delaunay 点预测不变，只加 GP 风险头，形成 1 个对照。

全部采用**既有 GP 原生低阻力概率作为独立风险头**。该概率不是组合点预测的 CDF，不代表组合模型具有 GP 的完整后验；组合阶段没有读取 OOF 校准概率、query 真值或外部测试标签。主流程统一做风险校准与评价。未经风险头的 Delaunay 已在第一轮保留，可用于隔离风险头改变的作用。

`protocol.json` 在生成候选之前写入。每个候选均输出完整 60 个冻结状态、8,640 行预测及单独协议；共 10 个候选。

```powershell
.venv/Scripts/python.exe -m cpt_budget.v7.simplex_combinations
.venv/Scripts/python.exe -m pytest tests/test_v7_simplex_combinations.py -q
```

两项关键测试检查候选数、混合公式、中位数、原点对照不变与输入边界。完整效果由统一结果表报告，不能因组合数量增加而视为有效改进。
