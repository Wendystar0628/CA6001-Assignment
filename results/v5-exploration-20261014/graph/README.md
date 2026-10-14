# 计划位置图上的 CPT 剖面重建

这两个候选检验另一种结构假设：把所有允许的计划位置与深度单元组成图，把已揭示整根 CPT 固定为边界，联合重建剩余图信号。它们没有训练源场地网络，也没有修改 IDW 的三近邻权重。

## 冻结的两个候选

- `graph_harmonic`：最小化 `sum_edges w_ij * (f_i-f_j)^2 / 2`，在已测节点保持 `f_L=y_L`。未知节点由 `L_UU f_U = -L_UL y_L` 解出。
- `graph_huber`：相同图和边界，最小化平滑图总变差 `sum_edges w_ij * delta^2 * (sqrt(1+((f_i-f_j)/delta)^2)-1)`，`delta=0.35` log(qc)。固定 40 次以内 IRLS，变化阈值 `1e-7 * max(1,max(abs(f)))`。这是 graph TV 的项目适配，不能写成复现文献原算法。

平面连接采用对称 4 近邻图与欧氏最小生成树的并集，避免纯 kNN 产生无已测边界的孤立块。水平边权为高斯距离相似度，尺度取该图非零边长中位数；边权下限 `1e-8`，再归一化到平均加权度 1。每条平面边复制到每个深度；同一位置相邻深度之间另加权重 0.15 的垂向边（非等距时按间距反比缩放）。所有参数在运行前冻结，没有验证集调参。

平方边差偏好平滑场；pseudo-Huber 对大幅跳变的惩罚接近线性，因而允许较陡的垂向层界或水平边界。它并未识别真实地层：仅有几何不能确定未知断层或透镜体在哪里。完整已测剖面作为边界，才提供层界信息。

## 信息边界与公平口径

构造函数显式接收 `planned_grid`，只允许 `PreparedDataset.query()` 的 geometry 列。该表包含验证场地全部可用计划位置，也包含固定评价位置的无标签几何；这些几何在勘察计划中允许公开，基线也可获取。模型不接收任何隐藏 qc。每次 `condition()` 仅以当前已揭示 profile 建立边界，并按 site 隔离图。`predict()` 同样拒绝带标签的输入。

这是一种使用计划点布局的 transductive 候选，其与 IDW/GP 的信息使用差别必须披露。它输出点预测，不虚构原生概率、CDF 或区间。下列 Recall/FPR 是点预测低于 1 MPa 的原始分类指标，不能和统一校准后的风险 Recall/FPR 混用。

## 相同 60 状态的实际结果

仍是已有冻结 Maximin 轨迹：三折、每折两个验证场地、两个 seed、5/7/9/11/13 根预算。聚合先对预算积分，再平均 seed 和同一 site 的重复 fold，最后对四个不同场地等权。没有新增独立外层测试。

| 候选 | 预算平均 log-MAE | H_point 厚度误差 m | 点 Recall | 点 FPR |
|---|---:|---:|---:|---:|
| graph_harmonic | 0.627926 | 0.723796 | 0.263358 | 0.056553 |
| graph_huber | 0.625616 | 0.694824 | 0.313999 | 0.058023 |

Huber 略优于 harmonic；两者的剖面误差均优于现有 IDW 汇总误差约 0.6386，但厚度误差弱于 IDW 约 0.6304m，不能宣称整体超过强基线。最终风险指标和综合门槛由主实验统一校准与计算。

实际每场地图有 480–936 个节点、1540–3129 条边，具体见 `graph_sizes.csv`。所有已测边界吻合误差均为 0。harmonic 60/60 线性求解完成，最大未知节点驻点残差约 `4.22e-15`。Huber 在固定 40 轮以内有 31/60 状态达到严格变化阈值，其余 29/60 保留 `graph_converged=false`；最大驻点残差 `9.06e-6`，各状态优化目标均下降。因此 Huber 结果应称为固定预算数值近似，不应宣称全部收敛。

定向检查 5 项通过：常数场重建、已测边界精确保持、geometry 标签隔离、查询排列不变、简单尖锐层界保留与几何连通。

## 复现

```powershell
.venv\Scripts\python.exe -m cpt_budget.v5.graph
.venv\Scripts\python.exe -m pytest tests/test_v5_graph.py -q
```

每个候选目录包含完整逐状态预测、60 行原始指标、geometry 上下文与求解诊断。根目录包含 `macro_summary.csv`、`site_summary.csv`、`curves.csv` 和 `graph_protocol.json`。

## 一手依据

- Zhu, Ghahramani, Lafferty (ICML 2003), *Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions*：[作者论文列表与原文](https://mlg.eng.cam.ac.uk/zoubin/ssl.html)。图拉普拉斯调和延拓的理论来源。
- Jung et al. (IEEE TSP 2019), *Semi-supervised Learning in Network-Structured Data via Total Variation Minimization*：[作者论文](https://arxiv.org/abs/1901.09838)。图 TV 倾向保留分簇跳变；其恢复条件不自动适用于本 CPT 几何图。
- [SciPy sparse spsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html)：实际边界线性系统求解器。
