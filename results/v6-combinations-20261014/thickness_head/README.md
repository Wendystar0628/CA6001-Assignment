# 第三轮：源训练厚度／事件双头

前两轮组合没有建立明显的综合优势后，继续尝试直接学习目标量，而不只组合 log(qc) 均值。四个候选、树参数、特征与点翻转幅度先在分工消息冻结；本轮仍是反复使用验证场地的探索，不是独立确认性实验。

## 两个真正训练的头

每折使用冻结的 v4 源训练轨迹。一个 ExtraTreesRegressor 学习整根 24 格中低阻力格数，另一个 ExtraTreesClassifier 学习每个深度 qc<1 MPa 的事件。源验证轨迹仅用于报告，没有早停、择权或扫参。

两头均固定 256 棵树、最大深度 12、max_features=.75、random_state=20261014、单线程。计数回归采用 squared_error、叶子最少 8 行；事件分类采用 gini、叶子最少 24 行，不进行类别再平衡。各源场地训练权重总和相等；重复预算与轨迹不被宣称为新增独立勘察。

Profile 的 85 维特征由 support-only 的 24 格 IDW、24 格加权低值比例、24 格 support 深度中位数，加上距离、预算和已测低值格数统计组成。事件头加深度、该格 IDW/加权低值比例/中位数、垂向梯度和局部三格均值，总计 91 维。没有场地 ID 特征，也没有未测 query 的形状或测量值特征。源 query 真值只用于训练标签和后续评价。

原始算法接口与参数依据：[ExtraTreesRegressor 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)、[ExtraTreesClassifier 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)。这里的双头及剖面决策是项目适配，不是论文的完整复现。

## 固定四个候选

- `v6_count_head_residual`：学习格数四舍五入后，按事件概率选 top-k 格；以最佳残差为点值基础。
- `v6_count_head_fpca50`：同一格数／排序机制，以 50% FPCA 与最佳残差混合为点值基础。
- `v6_count_head_half_residual`：学习格数与残差现有格数各半，再按概率 top-k。
- `v6_event_head_residual`：直接以事件头概率≥.5 决定标签，以最佳残差为点值基础。

保持与目标标签相符的点值，仅将需要翻转的点设为 −.05 或 +.05 log(qc)。这不是完整连续分布重建，报告时须承认部分值聚集在物理阈值附近。厚度始终从最终点预测 `log(qc)<0` 重新计算，不能把回归头预测的厚度直接替代点厚度指标。

四个候选附带相同 source-only 事件头的概率，用于共同风险评价。这个分类概率不是 log(qc) 的原生 CDF，也不是目标验证查询校准概率；top-k/计数决策亦不等价于概率≥.5。

## 文件与复现

```powershell
.venv/Scripts/python.exe -m cpt_budget.v6.thickness_head
```

`method_protocol.json` 为完整方法约定；每折 `heads.joblib` 保存真实训练模型，`training_summary.json` 保存训练规模与耗时；`profile_decisions.csv` 保留连续计数、四舍五入计数和最终点计数。所有四候选最终完整矩阵各 60 状态、8,640 行。最终门槛结论以统一比较为准。

4 项定向检查覆盖 query 标签隔离、任意输入行序、完整剖面与源角色边界、目标计数和最终点厚度一致性。未读取 outer-test。

## 完成的训练与点指标

| fold | profile-episode 行 | depth-event 行 | 去重训练 query CPT | 特征构建秒 | 两头训练秒 |
|---|---:|---:|---:|---:|---:|
| 0 | 1600 | 38400 | 200 | 26.8 | 49.9 |
| 1 | 1576 | 37824 | 190 | 33.2 | 50.7 |
| 2 | 1688 | 40512 | 210 | 19.3 | 52.0 |

每折两头各 256 棵树，共 1,536 棵；合计模型训练 152.58 秒，不含特征构建与评价。精确时间保存在各折 summary；跨 fold CPT 可能重复，不能相加宣称独立 CPT 总数。

| 候选 | log-MAE | 厚度误差 m | 原点 Recall | 原点 FPR |
|---|---:|---:|---:|---:|
| count_head_residual | 0.646214 | 0.646159 | 0.432510 | 0.146578 |
| count_head_fpca50 | 0.644501 | 0.646159 | 0.432510 | 0.146578 |
| count_head_half_residual | 0.631088 | **0.552246** | 0.409630 | 0.108661 |
| event_head_residual | **0.615733** | 0.785970 | 0.118995 | 0.035272 |

融合计数候选相对 IDW 厚度误差减少约 12.4%，原点召回增加约 3.59 个百分点，代价是原点 FPR 同时增加约 3.59 个百分点；连续误差仅改善约 1.17%。这是明确的厚度机制收益，但不是全面胜出。原始事件概率≥.5 的候选过于保守，较低 FPR 伴随召回下降，不应选作识别冠军。

源训练集的计数误差很低，源验证仍明显偏高，提示跨场地迁移困难没有消失。该观察只做诊断，没有据此重训、改树参数、选阈值或调整 count 融合权重。最终共同风险评价应对四候选得到相同的 risk 指标，因为它们共享同一个事件概率头。
