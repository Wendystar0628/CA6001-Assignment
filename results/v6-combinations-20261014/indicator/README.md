# v0.6：低阻力事件与层厚联合解码

这组实验改变的是预测对象和结构约束：先根据已测 CPT 直接估计 `qc < 1 MPa` 的事件，再将整根剖面的事件位置、总厚度和连续幅度联合起来。它不是简单对多个回归结果取平均，也不是调整评价门槛。

## 四个预先固定的候选

| 模型 | 幅度基座 | 结构约束 |
|---|---|---|
| `v6_event_potts_residual` | v0.4 最佳 guarded CRPS 残差 | 相邻深度的 Potts 转换代价 |
| `v6_event_cardinality_residual` | 同上 | Potts + 低阻力格数的软约束 |
| `v6_event_adaptive_residual` | 同上 | 只用已测位置的留一验证，选择幅度尺度与格数约束强度 |
| `v6_event_cardinality_idw` | 原始 IDW | 与第二项相同，用于识别残差基座的额外作用 |

概率证据来自最近三个已测位置的 IDW² 二元指示值。用有效样本数 `1 / sum(w²)` 和每类 0.5 的 Jeffreys 伪计数平滑，避免把少量邻居当作确定证据。目标格数是这些已测剖面低阻力格数的空间加权中位数；它没有使用未测位置的真实厚度。

对一个长度为 24 的二元事件剖面 `s`，能量为：

```text
E(s) = Σ[-log P_indicator(s_z) + (project(base_z, s_z) - base_z)² / (2 τ²)]
       + 0.25 × 相邻事件状态改变次数
       + γ × (Σ s_z - target_count)²
```

投影是在保留事件符号的前提下最小改动幅度：非低阻力约束为 `log_qc >= 0.05`，低阻力约束为 `log_qc <= -0.05`。默认 `τ=0.5`、`γ=0.25`；Potts 候选固定 `γ=0`。软格数约束可被强幅度证据抵消，并不强制预测厚度等于目标厚度。

动态规划以“深度、累计低阻力格数、当前状态”为状态，准确计算 MAP 事件剖面。前向后向计算同一有限 Gibbs 模型的精确边缘概率，写入 `p_low_1`。这是该模型的内部概率；仍需要外部交叉校准，不声称已经是真实地质风险概率。

`pred_log_qc` 是 MAP 解码后真正报告的连续幅度，`pred_qc_mpa=exp(pred_log_qc)`。因此 **H_point 始终从实际预测值是否低于 1 MPa 来计算**，不会换成概率求和。期望格数仅在诊断表单列记录。模型没有输出预测区间。

## 已测点留一选择的边界

自适应候选仅在当前已揭示 CPT 上留一位置，同时排除同 `near_group_id` 和共坐标位置。当前 60 个支持状态均没有重复近邻组或共坐标位置，所以加强排除规则不改变本次数值。固定搜索 `τ ∈ {0.25,0.5,1}`、`γ ∈ {0,0.25,1}`，固定目标为：

```text
log_MAE + 0.25 × H_point_MAE_m / 0.25m + 0.5 × balanced_event_error
```

正负类都存在时 balanced error 是漏报率、误报率的均值；仅一类存在时使用该类错误率。并列时先取较小 `τ`，再取较小 `γ`。

**留一幅度使用 IDW 代理**，因为固定缓存里没有每个已测点留一状态的残差模型预测。因此选择目标与最终残差基座存在模型差异；该局限明确保留在协议中。留一的 CPT 标签只属于当前已测支持集，未测 query 标签完全不进入模型或参数选择。

## 方法依据和复现

- [Tolosana-Delgado 等，Indicator Kriging without Order Relation Violations (2008)](https://link.springer.com/article/10.1007/s11004-008-9146-8) 支持通过二元指示变量建模位置上的条件事件概率。本实现采用非负空间核权重，不声称复现其对数比协克里金方法。
- [Unsupervised and Simultaneous Stratigraphic Interpretation of CPT Soundings at Site Scale (2021)](https://ascelibrary.org/doi/10.1061/%28ASCE%29GM.1943-5622.0002113) 的摘要说明了 CPT 的空间统计结构与隐马尔可夫随机场结合的方向。本实现使用可精确求解的小型二元链与格数势能，不声称复现其贝叶斯地层识别方法。

```powershell
.venv/Scripts/python.exe -m cpt_budget.v6.indicator
.venv/Scripts/python.exe -m pytest tests/test_v6_indicator.py -q
```

所有参数在生成评价输出前固定。四个候选使用同一 60 个冻结验证状态；评价真值由公共导出函数附加，生成器不接收它。`decoding_diagnostics.csv` 保存逐查询剖面的目标格数、实际格数、期望格数和改动事件数；`adaptive_contexts/` 保存支持集留一选择依据。

针对性检查通过 4 项：精确 DP 与全部二元枚举一致；查询标签不影响输出且行序正确；软厚度势能不会替换实际点预测标签；支持集留一排除整近邻组与共坐标位置。
