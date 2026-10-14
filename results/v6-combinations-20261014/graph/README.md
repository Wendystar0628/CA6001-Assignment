# v0.6 图模型与最佳残差的冻结组合

6 个候选全部输出 60 个原冻结验证状态、每模型 8640 个深度单元。数值评价由主代理统一校准和运行综合门槛，本目录不自行选胜者。对应代码：`src/cpt_budget/v6/graph_combinations.py`。

设 `G` 为 v0.5 的 `graph_huber` 点预测，`R` 为 v0.4 的 `residual_ft_crps_guarded_a050` 点预测；均在自然对数阻力域。

| 候选 | 固定机制 |
|---|---|
| v6_graph_residual_a025 | 0.25 G + 0.75 R |
| v6_graph_residual_a050 | 0.50 G + 0.50 R |
| v6_graph_residual_a075 | 0.75 G + 0.25 R |
| v6_graph_residual_sign_guard | 0.5 混合后投影回 R 的低阻力／非低阻力侧 |
| v6_graph_residual_low_union | 0.5 混合后投影到 G 或 R 任一判为低阻力的并集侧 |
| v6_graph_loo_bias_a050 | R + 0.5 × 图 Huber 对已测 LOO-IDW 残差的延拓 |

全部组合权重和规则在生成新候选分数前，先在并行设计消息中冻结。低阻力阈值固定为 log(1 MPa)=0，严格负侧 margin 与共同旧残差口径一致，为 1e-9。所有候选只输出点预测，不把图或混合值伪装成原生概率；风险概率由公共评价器统一校准。

## 结构意义与边界

直接混合检验图的全局空间约束能否补充残差模型。符号保护是明确的机制对照：它严格保持最佳残差的逐单元点标签，因此点 Recall、点 FPR 和点厚度必然保持不变，只可能改变连续误差及后续校准。低阻力并集则保证点 Recall 不低于两个组成模型，但点 FPR 也不会低于两个组成模型；不能只宣传召回增加。

LOO 图偏差候选从每个当前已测 CPT 的真值减去“排除整根及所有同坐标 CPT”的 IDW 预测，得到 support 残差。原图根据这些已测误差构成 Dirichlet 边界，只用计划位置几何将其延拓到未测位置。相比原图，它改变了被重建的目标场，不改图构造或求解超参数。它也可能重复加入 R 已补偿的偏差，实验会保留这种失败可能。

图参数完全沿用 v0.5：4NN＋几何 MST，垂向边权 0.15，pseudo-Huber delta 0.35，最多 40 次 IRLS。新的 LOO 偏差图 **40/60** 状态达到严格收敛，20/60 未达到；边界误差为 0，最大驻点残差约 2.34e-4。完整标记位于 `solver_diagnostics.csv`，未为追分增加迭代。前五种候选沿用 v0.5 原图求解结果，其旧收敛边界仍适用。

查询真值不进入组合权重、符号规则、图边界或 LOO 残差。无 outer-test 查询。原始机制来源沿用 [Gaussian fields / harmonic functions](https://mlg.eng.cam.ac.uk/zoubin/ssl.html) 与 [graph total variation](https://arxiv.org/abs/1901.09838)；本项目的 CPT 组合为适配实现，并非论文的逐项复现。

## 运行与检查

```powershell
.venv/Scripts/python.exe -m cpt_budget.v6.graph_combinations
.venv/Scripts/python.exe -m pytest tests/test_v6_graph_combinations.py -q
```

3 项针对性检查通过：凸组合与严格符号保护；整根 LOO 误差及输入不可变性；常数场的零误差经图延拓仍为零。所有候选都已覆盖完整矩阵，后续主代理负责统一评分与报告。
