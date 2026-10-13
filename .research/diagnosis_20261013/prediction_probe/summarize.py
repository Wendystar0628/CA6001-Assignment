from pathlib import Path
import json
import pandas as pd

out = Path(__file__).resolve().parent
metrics = pd.read_csv(out / 'metrics.csv')
macro = pd.read_csv(out / 'macro_by_budget.csv')
meta = json.loads((out / 'probe_metadata.json').read_text(encoding='utf-8'))

def table(frame, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |']
    for row in frame[cols].itertuples(index=False, name=None):
        values = [('—' if pd.isna(v) else f'{v:.4f}' if isinstance(v, float) else str(v)) for v in row]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)

text = '''# 预测方向的有限验证探针

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

'''
text += table(macro, ['model_id','budget_n','log_mae','qc_mae_mpa','h_point_mae_m','recall'])
text += '''

相对 IDW，残差 XGB 在预算 5/9/13 的 log-MAE 改善为 **2.62% / 0.24% / -3.43%**；对应厚度 MAE 从 0.792/0.792/0.833 m 增至 1.146/1.146/1.208 m，recall 从 0.448/0.309/0.299 降至 0.060/0.070/0.050。

目标深度均值 GP 相比源趋势 GP 在预算 9 的 log-MAE 改善约 **8.29%**，预算 5 恶化 **3.59%**，预算 13 恶化 **0.07%**；三预算厚度误差都更大。不能只报告预算 9。

## 每个验证场地（不隐藏失败预算）

'''
text += table(metrics.sort_values(['site_id','model_id','budget_n']), ['site_id','model_id','budget_n','log_mae','h_point_mae_m','recall'])
text += '\n\n## 源 episode 验证\n\n'
text += table(pd.read_csv(out / 'source_episode_validation.csv'), ['model_id','site_id','log_mae'])
text += '''

源 episode 上 Kaiapoi 和 Lower Hutt 的残差修正有收益，Tauranga 则退步，表明存在明显场地差异。回放时残差模型对 Tauranga 的 log_qc 平均上移 0.248–0.284，所有评价单元的修正均为正；这能解释阈值附近真实低 qc 被抬高并导致漏判增加。这是预测分布现象，不足以仅凭一次验证判定训练源的地质原因。

## 后续取舍

1. **不建议直接把以上任一候选提升为主方案或开启完整三折 GPU 重训。** 两者没有足够的“超过 IDW 且守住 recall”信号。
2. 若沿残差学习继续，先修改任务目标：保留 IDW 作为锚点，以低 qc 概率/厚度或风险加权损失约束修正，再在当前已测点的整位置留一条件下判断是否启用。必须使用整根 CPT 留一而非随机深度行，否则会泄漏同位置信息。
3. 更有信息的下一步是将“剖面误差目标”和“低 qc 厚度/漏判目标”明确拆开，并检查由源域训练来的正向修正在什么场地失效；没有这一步，更长训练或更大的模型并不保证靠近经济目标。
4. 任何新的选择规则都要冻结后再进入多种子、三折比较。由于正式 test 结果此前已经被观察，后续 test 重跑需标为探索性或安排新的外部/封存评价，不能当作全新未触碰测试。

## 耗时与产物

'''
text += f"源 episode 特征构造 {meta['source_feature_seconds']:.2f} 秒，残差 XGB 训练 {meta['residual_training_seconds']:.2f} 秒，本轮完整探针 {meta['total_seconds']:.2f} 秒（CPU，4 线程）。不含源 GP 单状态复核。\n\n"
text += '脚本 `probe.py`，逐状态全指标 `metrics.csv`，宏平均 `macro_by_budget.csv`，预测 `predictions.parquet`，配置/训练时间 `probe_metadata.json`，残差位移 `residual_shift.csv`。全部独立保存于本目录，未修改正式源码、配置、runs 或 results。\n'
(out / 'README.md').write_text(text, encoding='utf-8')
print(str(out / 'README.md'))
