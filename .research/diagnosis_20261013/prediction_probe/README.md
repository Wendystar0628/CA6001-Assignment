# 预测方向的有限验证探针

## 结论

这两个低成本改动都未显示“超过强基线约 10%，且不明显增加漏判”的可用信号。IDW 残差学习明显优于当前直接预测 qc 的 XGBoost，但与 IDW 本身相比增益小、预算 13 反而退步，阈值任务还明显变差；仅用目标场地已测深度均值替代 GP 的源场地趋势也不稳定。说明现有空间插值已经是强信号，机械增加学习器或替换趋势不足以解决目标。

这是一次选择后续研究方向的小型探索，不能证明原方向永远不可行，也不能宣称正式的泛化性能或补勘节约。

## 固定比较口径

- 只使用外层 fold 0 的 train 和 validation，未运行 test 预测、未用 test 标签选择超参。
- 复用冻结的 validation、seed 17、maximin 状态；每种方法在完全相同的 5/9/13 根已测集合上预测同一固定评价集。
- 两个可回放验证场地：Lower Hutt Town Hall、Tauranga Farmers Building。第三个验证场地 Kaiapoi Near River 只参加源 episode 验证，因没有满足主回放条件的冻结状态而不参加本表。
- 当前 IDW 与 XGB 从现有类及 fold 0 artifact 重算；原 GP 取冻结回放指标，并用其中一个状态重新计算确认 log-MAE 差为 -1.11e-16。
- 候选 A：沿用 16 维邻域特征、源 episode 和 site/episode/location/depth 等权，只将 XGB 标签改为 `truth_log_qc - idw_log_qc`，预测时加回 IDW。超参数与原 XGB 相同，validation early stopping，第 10 棵树为最佳（best_iteration=9）。无模型网格、无残差缩放系数搜索。
- 候选 B：沿用当前 GP，唯一变更是深度趋势改为“本状态已揭示目标场地 CPT 的逐深度 log_qc 均值”。目标场地未测标签从不进入拟合。其均值估计不确定性未传递进区间，故区间数值不构成可靠性验证。
- qc 单位 MPa；主阈值 1 MPa；厚度单位 m；误差按既有评价器先位置、再场地等权。此探针不评估主动选址或成本。

## 两场地宏平均（全部预算）

| model_id | budget_n | log_mae | qc_mae_mpa | h_point_mae_m | recall |
| --- | --- | --- | --- | --- | --- |
| gp | 5 | 0.6782 | 4.7730 | 0.7500 | 0.5559 |
| gp | 9 | 0.7602 | 5.1752 | 0.8542 | 0.3176 |
| gp | 13 | 0.6836 | 4.9456 | 0.7917 | 0.2784 |
| gp_local_depth_mean | 5 | 0.7026 | 5.0799 | 1.1667 | 0.3196 |
| gp_local_depth_mean | 9 | 0.6972 | 5.0514 | 1.1250 | 0.2294 |
| gp_local_depth_mean | 13 | 0.6841 | 4.9885 | 1.2083 | 0.1500 |
| idw | 5 | 0.7055 | 4.9941 | 0.7917 | 0.4480 |
| idw | 9 | 0.6608 | 4.4570 | 0.7917 | 0.3088 |
| idw | 13 | 0.6169 | 4.3111 | 0.8333 | 0.2990 |
| residual_xgb | 5 | 0.6870 | 5.0078 | 1.1458 | 0.0598 |
| residual_xgb | 9 | 0.6592 | 4.4616 | 1.1458 | 0.0696 |
| residual_xgb | 13 | 0.6381 | 4.3428 | 1.2083 | 0.0500 |
| xgb | 5 | 0.8198 | 5.6819 | 1.2708 | 0.0000 |
| xgb | 9 | 0.8735 | 5.7519 | 1.2708 | 0.0000 |
| xgb | 13 | 0.8611 | 5.7097 | 1.2708 | 0.0000 |

相对 IDW，残差 XGB 在预算 5/9/13 的 log-MAE 改善为 **2.62% / 0.24% / -3.43%**；对应厚度 MAE 从 0.792/0.792/0.833 m 增至 1.146/1.146/1.208 m，recall 从 0.448/0.309/0.299 降至 0.060/0.070/0.050。

目标深度均值 GP 相比源趋势 GP 在预算 9 的 log-MAE 改善约 **8.29%**，预算 5 恶化 **3.59%**，预算 13 恶化 **0.07%**；三预算厚度误差都更大。不能只报告预算 9。

## 每个验证场地（不隐藏失败预算）

| site_id | model_id | budget_n | log_mae | h_point_mae_m | recall |
| --- | --- | --- | --- | --- | --- |
| lower_hutt_town_hall | gp | 5 | 0.7944 | 0.5000 | 0.7000 |
| lower_hutt_town_hall | gp | 9 | 0.8108 | 0.5000 | 0.4000 |
| lower_hutt_town_hall | gp | 13 | 0.7600 | 0.3333 | 0.4000 |
| lower_hutt_town_hall | gp_local_depth_mean | 5 | 0.7932 | 0.4583 | 0.6000 |
| lower_hutt_town_hall | gp_local_depth_mean | 9 | 0.7751 | 0.3750 | 0.4000 |
| lower_hutt_town_hall | gp_local_depth_mean | 13 | 0.7228 | 0.2917 | 0.3000 |
| lower_hutt_town_hall | idw | 5 | 0.7933 | 0.4167 | 0.7000 |
| lower_hutt_town_hall | idw | 9 | 0.6362 | 0.0833 | 0.5000 |
| lower_hutt_town_hall | idw | 13 | 0.6195 | 0.0833 | 0.5000 |
| lower_hutt_town_hall | residual_xgb | 5 | 0.7466 | 0.3333 | 0.1000 |
| lower_hutt_town_hall | residual_xgb | 9 | 0.6217 | 0.3333 | 0.1000 |
| lower_hutt_town_hall | residual_xgb | 13 | 0.6037 | 0.3750 | 0.1000 |
| lower_hutt_town_hall | xgb | 5 | 0.8332 | 0.4167 | 0.0000 |
| lower_hutt_town_hall | xgb | 9 | 0.8765 | 0.4167 | 0.0000 |
| lower_hutt_town_hall | xgb | 13 | 0.8470 | 0.4167 | 0.0000 |
| tauranga_farmers_building | gp | 5 | 0.5621 | 1.0000 | 0.4118 |
| tauranga_farmers_building | gp | 9 | 0.7096 | 1.2083 | 0.2353 |
| tauranga_farmers_building | gp | 13 | 0.6072 | 1.2500 | 0.1569 |
| tauranga_farmers_building | gp_local_depth_mean | 5 | 0.6120 | 1.8750 | 0.0392 |
| tauranga_farmers_building | gp_local_depth_mean | 9 | 0.6193 | 1.8750 | 0.0588 |
| tauranga_farmers_building | gp_local_depth_mean | 13 | 0.6453 | 2.1250 | 0.0000 |
| tauranga_farmers_building | idw | 5 | 0.6177 | 1.1667 | 0.1961 |
| tauranga_farmers_building | idw | 9 | 0.6855 | 1.5000 | 0.1176 |
| tauranga_farmers_building | idw | 13 | 0.6144 | 1.5833 | 0.0980 |
| tauranga_farmers_building | residual_xgb | 5 | 0.6273 | 1.9583 | 0.0196 |
| tauranga_farmers_building | residual_xgb | 9 | 0.6968 | 1.9583 | 0.0392 |
| tauranga_farmers_building | residual_xgb | 13 | 0.6724 | 2.0417 | 0.0000 |
| tauranga_farmers_building | xgb | 5 | 0.8064 | 2.1250 | 0.0000 |
| tauranga_farmers_building | xgb | 9 | 0.8704 | 2.1250 | 0.0000 |
| tauranga_farmers_building | xgb | 13 | 0.8751 | 2.1250 | 0.0000 |

## 源 episode 验证

| model_id | site_id | log_mae |
| --- | --- | --- |
| idw | kaiapoi_near_river | 1.1350 |
| idw | lower_hutt_town_hall | 0.7050 |
| idw | tauranga_farmers_building | 0.6328 |
| residual_xgb | kaiapoi_near_river | 1.0648 |
| residual_xgb | lower_hutt_town_hall | 0.6959 |
| residual_xgb | tauranga_farmers_building | 0.6538 |

源 episode 上 Kaiapoi 和 Lower Hutt 的残差修正有收益，Tauranga 则退步，表明存在明显场地差异。回放时残差模型对 Tauranga 的 log_qc 平均上移 0.248–0.284，所有评价单元的修正均为正；这能解释阈值附近真实低 qc 被抬高并导致漏判增加。这是预测分布现象，不足以仅凭一次验证判定训练源的地质原因。

## 后续取舍

1. **不建议直接把以上任一候选提升为主方案或开启完整三折 GPU 重训。** 两者没有足够的“超过 IDW 且守住 recall”信号。
2. 若沿残差学习继续，先修改任务目标：保留 IDW 作为锚点，以低 qc 概率/厚度或风险加权损失约束修正，再在当前已测点的整位置留一条件下判断是否启用。必须使用整根 CPT 留一而非随机深度行，否则会泄漏同位置信息。
3. 更有信息的下一步是将“剖面误差目标”和“低 qc 厚度/漏判目标”明确拆开，并检查由源域训练来的正向修正在什么场地失效；没有这一步，更长训练或更大的模型并不保证靠近经济目标。
4. 任何新的选择规则都要冻结后再进入多种子、三折比较。由于正式 test 结果此前已经被观察，后续 test 重跑需标为探索性或安排新的外部/封存评价，不能当作全新未触碰测试。

## 耗时与产物

源 episode 特征构造 5.87 秒，残差 XGB 训练 0.15 秒，本轮完整探针 8.50 秒（CPU，4 线程）。不含源 GP 单状态复核。

脚本 `probe.py`，逐状态全指标 `metrics.csv`，宏平均 `macro_by_budget.csv`，预测 `predictions.parquet`，配置/训练时间 `probe_metadata.json`，残差位移 `residual_shift.csv`。全部独立保存于本目录，未修改正式源码、配置、runs 或 results。
