# v7 局部几何回归探索

本组增加三个结构不同的空间估计器，不调整旧 GP 或 IDW。所有参数在运行前写入 `frozen_settings.json`；使用全部 60 个既定验证状态，每个模型 8,640 个预测单元。预测器只接收当前已测 CPT 和 query 坐标/深度；没有 query 标签选参、逐场地模型选择或额外测试集访问。

## 方法与预先固定参数

- **`v7_mls_affine`：移动最小二乘仿射面。** 对每个 query 和深度拟合局部截距及 x/y 斜率，允许空间趋势而非只平均剖面。局部尺度为第 min(4,n) 个已测位置的距离（至少 1 米），高斯权归一化，斜率 L2 惩罚 0.1；结果限制到当前已测该深度的 min/max。稀疏数据上的平滑可能抹掉低阻力薄层；这是正则化应用，不直接声称原论文的理想逼近定理适用。[MLS 原论文](https://www.ams.org/mcom/1998-67-224/S0025-5718-98-00974-0/S0025-5718-98-00974-0.pdf)，[作者机构记录](https://cris.tau.ac.il/en/publications/the-approximation-power-of-moving-least-squares/)。
- **`v7_lad_affine`：稳健局部绝对误差仿射面。** 使用相同几何权，最小化加权绝对误差与 0.05 倍斜率 L1 惩罚，由单线程 HiGHS 求解线性规划。截距不惩罚，同样按已测范围截断；比平方损失不易被异常剖面主导，但逐深度的中位数回归可能生成不够平滑的层界。此为标准 LAD 的显式 LP 实现，[SciPy linprog 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)。
- **`v7_simplex_linear`：Delaunay 分片线性剖面插值。** 在已测位置上三角剖分，凸包内按三角形重心坐标混合三根完整剖面；凸包外投影到最近凸包边并做线性插值。共线支持退化为主轴线段插值；同坐标支持先平均，不制造虚假几何。非负、和为 1 的权重保持值域，却无法推断凸包外的新变化或出现于所有支持之外的新薄层。[SciPy LinearNDInterpolator 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html)；凸包外规则是本项目新增的明确适配。

## 完整验证的点预测结果

按预算 5/7/9/11/13 梯形积分，先平均种子，再折内场地，最后对 4 个不同场地等权；重复验证场地不被重复加权。下表没有经过风险校准，不等同于校准后的风险召回/误报。

| 模型 | log-MAE | 厚度 MAE (m) | 点召回 | 点误报率 |
|---|---:|---:|---:|---:|
| Delaunay 分片线性 | 0.607248 | 0.585286 | 0.340035 | 0.070404 |
| LAD 仿射面 | 0.639314 | 0.577148 | 0.332968 | 0.068932 |
| MLS 仿射面 | 0.639079 | 0.697754 | 0.261505 | 0.037990 |

Delaunay 的 360 个查询剖面状态中，205 个落在支持凸包内，155 个用最近凸包边投影。相对既有 IDW 均值 0.638586 / 0.630371 m，Delaunay 连续误差减少约 4.91%、厚度误差减少约 7.15%；点召回低于 IDW 0.373767，因此不能称全面替代强基线。LAD 的厚度收益值得与更强连续预测/风险头组合。MLS 低误报同时伴随低召回，不应单独解读为识别质量高。

四场地结果详见 `point_by_site.csv`；`point_summary.csv` 为等权汇总；每个候选有 `point_metrics.csv` 与 `geometry_diagnostics.csv`。这些结果只用于后续组合设计，由主流程统一进行风险校准、比较和选择。

## 检查与重跑

4 项定向测试通过：三角形内的仿射再现及外侧投影、共线/单点退化、常数保持及 LAD 抗异常、预测不读取 query 标签并拒绝其他场地。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_v7_local_geometry.py -q
.venv/Scripts/python.exe -m cpt_budget.v7.local_geometry
```

要求原有 frozen validation 数据及 v6 最终预测矩阵存在。该构造没有原生概率或置信区间，不制造其不确定性质量声明。
