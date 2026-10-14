# 方案选择依据

当前比较包括4个基线（1弱3强）、2个TabPFN组合、3个主方案和4个讨论方案，共13个入口，名单见[`configs/catalog.json`](../configs/catalog.json)。

## 选择规则

候选集合是同一评价协议下记录过的全部固定配置：截至v0.8共203个（Git标签`v0.8-engineering`中的`results/v8-engineering-20261014/comparison/all_model_ranking.csv`；v0.3–v0.7各表与它在共有行上逐值相同）。协议为60个冻结状态、4个验证场地、预算5–13的AUBC，定义见[当前方案对比](../CA6001_CPT_当前方案对比.md#指标总表)。

规则：**每个保留入口要么是该角色在这一协议下的历史最优配置，要么明确写出它不是最优的原因和换来的取舍。** 不同协议的结果不参与比较，包括v0.1的8个测试场地结果、v0.2的按fold选择系统、v0.6的跨fold选择系统。

| 入口 | 英文显示名 | 角色 | 依据 |
|---|---|---|---|
| `constant` | Constant median | 弱基线 | 设计上的弱参照；存在略强的弱基线`target_depth_median`（log-MAE 0.775819），但弱参照的作用只是给出下限 |
| `idw` | IDW (3NN) | 强基线 | 唯一的IDW；实际上是最强的基线 |
| `gp` | GP | 强基线 | 该协议下唯一的普通GP。协同克里金`v7_cokriging_short`（0.637330 / 0.608236 m）使用源场地协方差，属于新方法，不作基线 |
| `xgb` | XGBoost residual | 强基线 | XGBoost各变体中最优 |
| `tabpfn_joint` | TabPFN joint | TabPFN组合 | v0.7晋级主线，位于全部配置的log-MAE–厚度Pareto前沿。它是取舍而非单项最优：含TabPFN的配置中，`v7_simplex_fpca50_s050_gp`的log-MAE更低（0.594194），但厚度0.627604 m、点召回30.36%；`v7_pair_transfer_*`厚度更低（0.504557 m），但log-MAE为0.606581 |
| `tabpfn_graph` | TabPFN + graph low-union | TabPFN组合 | 在点误报≤9.14%的含TabPFN配置中点召回最高（42.47%）；更高召回只出现在点误报≥10.6%的配置中 |
| `balanced` | Balanced | 主方案 | 位于log-MAE–厚度前沿，也位于四指标（log-MAE、厚度、点平衡错误、风险平衡错误）前沿；没有配置支配它 |
| `thickness` | Thickness (Haar, nearest) | 主方案 | 记录过的最低厚度MAE（0.472493 m）。另有两个配置厚度相同，但log-MAE更差（0.620549、0.689833）。取舍：点误报10.05%，超出v0.7晋级规则的护栏（≤IDW+2pp＝9.28%） |
| `recognition` | Recognition (recall-60) | 主方案 | 记录过的最高点召回（48.77%），点平衡错误也最低；取舍是点误报9.14%（IDW 7.28%） |
| `simplex` | Simplex | 讨论方案 | 示例：简单局部几何插值已有竞争力（log-MAE 0.607248），不是某角色的最优 |
| `count_shrink` | Count-shrink | 讨论方案 | 记录过的最低log-MAE（0.591791）；但厚度0.546712 m差于均衡方案，完整链路耗时约为均衡方案的1.89倍 |
| `thickness_guarded` | Thickness (FPR-guarded) | 讨论方案 | 满足v0.7点误报护栏的最低厚度MAE（0.487630 m，点误报8.99%）；厚度MAE低于它的配置全部超出护栏 |
| `cnp` | CNP | 讨论方案 | 负结果示例，不是某角色的最优 |

共享经验风险头`v6_empirical_graph_union_local`位于风险召回–风险误报Pareto前沿上。在满足v0.7风险召回护栏（≥IDW−2pp＝67.55%）的风险头中，它的风险误报最低。Brier或风险平衡错误更好的风险头，要多付出3–4pp风险误报才能多换来3–5pp风险召回，所以这是取舍而非劣选。

## v0.9.1的调整

- `thickness`由`v8_haar_fast`（0.487630 m）改为历史最优厚度`v7_hcount_simplex_nearest_a100_gp`的点路径（0.472493 m）。两者的唯一区别在于Haar细节系数：新版取几何上最近的已测剖面，旧版取3个近邻的加权中位数。归档ID末尾的`_gp`表示v0.7时它配的是GP风险头（风险召回/误报62.20%/24.87%）；v0.9.1改配共享经验风险头，点指标不变。
- `v8_haar_fast`改为讨论方案`thickness_guarded`，保留满足护栏的厚度选项。
- `tabpfn_fpca`（`v6_fpca_residual_sign_guard`）被移除，原因是被支配。它在log-MAE、厚度、点平衡错误、风险平衡错误四项上同时被`v6_coupled_residual_stack_potts`支配，在log-MAE–厚度平面上被64个配置支配。它的低阻力判定沿用IDW残差符号，所以厚度MAE、点召回、点误报都与IDW逐单元相同（0.630371 m / 37.38% / 7.28%），改进只体现在log-MAE。替换入口`tabpfn_joint`：log-MAE 0.598844（−0.014602）、厚度0.525228 m（−0.105143 m）。代价是点误报8.19%（+0.91pp）；风险指标改由共享风险头给出。

## 各组说明

- **基线**：常数中位数、3NN IDW、空间GP、同轨迹XGBoost残差。IDW实际上是最强的基线：GP（log-MAE +1.81%、厚度+5.84%）与XGBoost（log-MAE −0.67%、厚度+2.81%）都未在两项上同时超过IDW。GP是唯一输出预测区间的入口，名义90%区间的实测覆盖率只有79.76%（各预算78.1%–81.2%），明显欠覆盖。
- **TabPFN组合**：`tabpfn_joint`的幅值底座由受保护的TabPFN残差与三角网插值各占一半，低阻力标签由整曲线经验误差决策决定，所以TabPFN只贡献一半幅值底座，三角网和整曲线决策承担了大部分作用。`tabpfn_graph`由TabPFN残差与图Huber平均，取两者低值判定的并集，换取较高召回。
- **主方案**：均衡、厚度、识别分别对应整体误差、厚度预测与低阻力识别。三者的点预测路径都不含TabPFN，但它们的风险列来自共享风险头，而共享风险头会调用TabPFN残差与图Huber组件。
- **讨论方案**：见上表。CNP在全部60个状态中从未预测出低于1 MPa的值（点召回、点误报均为0，厚度MAE与常数基线相同），log-MAE仅比常数中位数低3.9%。

命名提示：`recognition`对应v0.8的`v8_recognition_recall60`（可靠性定位＋60%计数）。v0.7 README中点召回41.89%的“识别代表”是另一个模型，不是当前的`recognition`。

TabPFN残差、FPCA、形状配准、图Huber、Haar和整曲线决策都是组合内部共享的计算组件，不另计为模型入口。

以上取舍都基于4个验证场地，选择规则也是在看过这些结果之后确定的；独立的最终测试结果见[results/final_test/README.md](../results/final_test/README.md)。完整指标、耗时与各场地的局部代价见[当前方案对比](../CA6001_CPT_当前方案对比.md)。
