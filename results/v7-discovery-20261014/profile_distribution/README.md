# v7：整根条件剖面分布与厚度决策

本轮实现 5 个候选，均完成相同 60 个冻结状态、每候选 8640 个深度单元。所有输出位于同名子目录的 `predictions.parquet`。本页点指标经过预算曲线积分、先按折内场地聚合，再让 4 个不同场地等权；风险校准与晋升比较由统一 v7 evaluator 处理。

## 新机制

每个混合原子都是完整 24 深度曲线，同一个原子索引贯穿所有深度，因此其低阻力厚度分布来自完整曲线的真实枚举。不会将边缘概率相乘来伪造联合分布。每个候选的 `p_low_1` 都是所属有限混合的精确边缘事件概率。

- Analog 原子：`base + 0.5 × (support_curve − spatial_weighted_mean_support_curve)`。混合均值严格等于 base。
- Error 原子：`base + 0.5 × whole_profile_LOO_IDW_error`。LOO 排除完整 near_group、同一位置及同坐标点，参考为另外 3 个最近的独立位置。
- 空间权重：Gaussian kernel，尺度取已测坐标非零成对距离的中位数；除以 near_group 内成员数以避免重复调查组简单增加质量。所有参数在看到本轮得分前已发给主代理约定。
- `median` 取逐深度加权中位数；`hmedian` 先求完整混合厚度的中位数，再约束预测低阻力单元数为该值。在此计数下，选择最大边缘概率的单元，最小化混合模型下的预期 Hamming 识别损失；概率相同时按 base 大小排序以最小化修改。只把与最终事件标签冲突的 base 单元改成 ±0.05 logqc。

这一区别直接产生可观测结果：1800 个候选剖面决策中，898 次完整厚度中位数不同于逐深度中位数所形成的厚度。厚度决策的最终点预测计数全部等于其完整厚度中位数。

## 点指标初步结果

| 模型 | log-MAE | 厚度 MAE/m | 点召回 | 点误报 |
|---|---:|---:|---:|---:|
| error hmedian + 50% residual/FPCA | 0.613969 | 0.525716 | 0.358910 | 0.078301 |
| analog hmedian | 0.621401 | 0.601725 | 0.389393 | 0.072954 |
| analog point median | 0.621987 | 0.662272 | 0.344514 | 0.061961 |
| error hmedian | 0.622674 | 0.529622 | 0.369934 | 0.085579 |
| error point median | 0.622705 | 0.625488 | 0.320246 | 0.070275 |

第一行相对 IDW 的连续误差下降约 3.86%，厚度误差下降约 16.60%。这并不意味着所有低阻力定位指标或所有场地均改善：Marshland 的点召回只有 0.0559，Tauranga 的点误报为 0.1818。应优先把整个曲线分布看作厚度决策候选，保留按场地的定位代价。表格不包含统一风险校准得分，不能据此宣称通过综合门槛。

## 边界与可复算证据

这是一组明确写出的预测分布假设，不是已校准的后验。LOO-IDW 的误差也不是残差 TabPFN 或 FPCA 基底自身的校准误差；跨模型误差迁移是需要验证的假设。当前 5–13 根已测曲线很少，空间异质性会破坏迁移。没有独立测试、没有覆盖率保证、没有通过风险概率标签替换改变点预测的做法。

- `joint_atoms.csv`：9720 个实际原子权重、来源位置、低阻力计数和 24 位事件 mask。
- `profile_decisions.csv`：1800 个完整厚度中位数、边缘中位数计数、期望计数、最终计数和修改量。
- `point_metrics.csv` / `point_site_metrics.csv` / `point_ranking.csv`：完整点指标及 4 场地汇总。
- `method_protocol.json` / `execution_summary.json`：公式、适用范围和运行完成性。
- `tests/test_v7_profile_distribution.py`：4 项测试通过，包括 joint 与 marginal 决策不同、固定计数的 Hamming 最优性、geometry-only 边界、完整 near_group LOO 排除。

运行：`.venv/Scripts/python.exe -m cpt_budget.v7.profile_distribution`。线性代数限制为单线程，无随机抽样误差。

## 原始研究来源

函数式预测及其整曲线 bootstrap 的思路参考 [Hyndman 与 Shang 的作者论文页](https://fpp.robjhyndman.com/publications/forecasting-functional-time-series-2/)；保持函数依赖结构的重要性也见 [Paparoditis 与 Shang 的原论文](https://arxiv.org/abs/2004.03971)。本实现是空间有限混合适配，不复现这些时间序列估计器，也不继承其渐近覆盖保证。
