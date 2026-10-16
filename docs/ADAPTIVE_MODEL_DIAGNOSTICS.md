# 自适应补勘：模型接口、训练证据与必要消融

更新：2026-10-17。本文对应 `src/cpt_budget/adaptive_models.py`。新补勘轨迹的模型选择以开发场地结果为准；以下单状态诊断用于确认实现可用和解释模型组成，不是新的综合排行。

## 五个主模型及统一口径

| 入口 | 源数据学习与当前上下文 | 点预测构成 | 选点辅助角色 |
|---|---|---|---|
| `gp` | 固定源场地深度趋势；每轮只对公开 CPT 估计目标场地 GP | 各向异性 Matérn GP | 原生预测标准差 |
| `xgb` | 源场地训练并冻结残差树；邻孔特征随公开 CPT 更新 | 3 邻孔 IDW＋按折选择的残差幅值 | 主模型与其低值标签分歧 |
| `cnp` | 冻结源场地训练的 26,817 参数网络；公开整剖面作为集合上下文 | 查询相对坐标的整剖面 DeepSets 编码与深度解码 | 神经网络比较，不生成虚构概率 |
| `tabpfn_joint` | 冻结实际梯度微调过的 v2 权重；每轮更新当前场地残差上下文 | 0.5×Simplex＋0.5×符号保护残差，再做整剖面低值格数决策 | 主预测候选 |
| `tabpfn_graph` | 同一微调残差；graph-Huber 仅以公开 CPT 为边界 | 残差与 graph 均值，再取两者低值标签并集 | 另一种融合与识别取舍 |

五者统一从同一公开支持集获得信息，统一对同一条预测曲线计算 `ln(qc)` 误差和 `qc<1 MPa` 累计厚度。主比较不调用旧的共享经验风险头。GP 标准差与模型分歧先作为排序信号，不解释为工程失效概率。

## 回放接口与固定网格

```python
from cpt_budget.adaptive_models import AdaptiveModelSuite, make_predictor

suite = AdaptiveModelSuite(
    fold_id=0,
    model_ids=('gp', 'xgb', 'cnp', 'tabpfn_joint', 'tabpfn_graph'),
    role='development',       # 正式外层评价用 evaluation
    device='cuda',
    geometry=planned_query,   # 标签剥离后的完整24深度计划表
).initialize()
suite.condition(revealed_rows)
predictions = suite.predict(query_rows, model_ids=('gp', 'xgb', 'tabpfn_joint'))
point_frame = predictions['tabpfn_joint'].frame
candidate_signals = suite.predict_auxiliary(query_rows, primary_id='tabpfn_joint')

# 单模型轻量入口：预测直接返回 DataFrame。
predictor = make_predictor('gp', 0, role='development').initialize()
predictor.condition(revealed_rows)
frame = predictor.predict(query_rows)
```

- `condition`、`initialize` 返回自身，可缓存同一实例。每轮 `condition` 会重置残差上下文；不会继续使用上一轮的上下文拟合状态。
- 同一次 `suite.predict` 请求 joint、graph 和已微调残差消融时，共用一次 TabPFN 残差前向；未微调同代权重另做一次前向。
- 所有 query 在进入预测器前只保留坐标、位置／近邻组标识及预设深度，剥离潜在的 qc、厚度等测量列。
- GP 的统一列为 `std_log_qc`；它从原生高斯分位数还原。`predict_auxiliary` 输出 `location_id, gp_std_log_qc, disagreement`。主模型为 XGBoost 时以 joint 作为分歧对照，否则以 XGBoost 为对照。
- 固定查询网格可以包含后续会公开的计划位置。已公开的同位置／同深度行返回已知测量，`is_revealed=True`，GP 的已知行 `std_log_qc=0`。预测准确度只在独立隐藏审计点计算，不把这些已知行计为模型测试成绩。
- Graph 只能查询事先冻结的计划节点。若增加任意显示网格，须在整个回放开始前并入 `geometry`；它会改变图拓扑。本版采用冻结计划候选坐标，避免逐轮改变几何。

## 源场地与目标场地隔离

`fold_source_report` 导出 GP 趋势、XGBoost、CNP 归一化与网络、TabPFN 的训练和 checkpoint 选择场地。`audit_fold_sources` 在 `condition` 前核对：

- `evaluation`：目标必须属于该折 test，不能在训练列表或早停／幅值选择的 validation 列表内。
- `development`：允许目标是该折已有 validation 场地，但仍拒绝源训练场地。这里的结果属于开发分析，不能称独立盲测。

三折实际列表见 [model_diagnostics_sources.json](../results/adaptive_v1/model_diagnostics_sources.json)。例如 Christchurch Stadium 只可用 fold 0 的外层测试工件；New Lock 只可用 fold 1；Hollywood 只可用 fold 2。使用另一折已见过目标场地的权重，不会因更换支持集而变成无泄漏评价。

这些检查控制当前代码的训练／选择来源，不能消除研究者已经看过旧测试结果的历史。本轮应表述为有开发历史数据上的新回放重评估。

## 训练真实性与 CNP 合理性

| 折 | CNP 实际训练步 | CNP 选中步 | 初始化验证 log-MAE | 选中验证 log-MAE | 最佳权重已见源 episode／全部 episode |
|---|---:|---:|---:|---:|---:|
| 0 | 600 | 200 | 1.2276 | 0.9212 | 194／408 |
| 1 | 500 | 100 | 1.4398 | 0.8702 | 100／472 |
| 2 | 1200 | 800 | 1.2265 | 0.9417 | 400／400 |

三折 CNP 的全部 14 个学习张量都与对应随机种子的初始化不同，权重有限，checkpoint 步数与训练摘要一致。归一化仅使用源训练场地，推理按同一 `target_mean/target_scale` 反变换，网络只接收公开支持剖面及查询坐标／预设深度。因此不能把它写成未训练占位模型，也不能据其结果否定全部 CNP。

当前版本也有明确局限：网络较小，使用全局集合池化和共享深度解码器；L1 目标优化整体条件中位数，没有专门的低值召回目标；前两折选中 checkpoint 对源 episode 的覆盖尚不完整。这些是低值模式被平滑或回归中心的可能原因，尚不能从检查中识别各因素的因果贡献。本轮保留真实版本比较，不故意破坏它来制造弱基线，也不启动新的大规模超参数搜索。

旧开发参考表的 60 状态中，三折 CNP 输出最小值分别为 0.693、0.166、0.296，全部高于 `ln(1 MPa)=0`；零召回不是四舍五入造成的。按各折源训练场地、每场地等权统计，准备数据中的低值深度单元约占 6.9%、7.1%、9.8%。这支持检查低值少数模式与中位数损失的关系，但不是实际 episode 抽样权重的完整复原，不能单凭比例宣称找到了唯一原因。分布记录见 [model_diagnostics_cnp_distribution.json](../results/adaptive_v1/model_diagnostics_cnp_distribution.json)。

三折 TabPFN 训练分别有 600、600、500 次成功优化器更新，选中步骤为 500、300、200。通过已安装 `tabpfn==9.2.0` 的加载器把本地 v2 基座转换到同一架构布局后，可匹配的 129 个学习权重张量均发生变化；全部学习参数有限。`criterion.losses_per_bucket` 是未使用的损失统计缓冲，其 NaN 不属于预测参数；本次实测预测均有限。仅调用上下文 `.fit()` 与这些源场地梯度更新分开记录。

详细数值、checkpoint 指纹及初始化比较见 [model_diagnostics_training.json](../results/adaptive_v1/model_diagnostics_training.json)。原始同代权重来自已有本地 v2 文件，本轮没有下载或升级基础模型。

## 最小必要消融

| 入口 | 残差权重 | 残差缩放／符号保护 | 额外后处理 |
|---|---|---|---|
| `tabpfn_residual_unfinetuned` | 原始 v2 | IDW＋0.5×残差，不做符号保护 | 无 |
| `tabpfn_residual_raw` | 已微调 v2 | IDW＋0.5×残差，不做符号保护 | 无 |
| `tabpfn_residual_guarded` | 已微调 v2 | 同幅值，再保留 IDW 的原低值标签 | 无 |
| `tabpfn_joint_unfinetuned` | 原始 v2 | 同 joint 的符号保护残差 | 与 joint 完全相同的空间混合和整剖面决策 |
| `tabpfn_joint` | 已微调 v2 | 同上 | 空间混合和整剖面决策 |

raw 与 unfinetuned 比较源场地微调的作用；guarded 与 raw 比较符号规则；joint 与 guarded 比较空间混合／整剖面决策的联合贡献；joint 与 joint_unfinetuned 在同样后处理中比较权重更新。不能将所有收益笼统归因于 Transformer。

## 一次真实端点运行诊断

开发样例：Lower Hutt Town Hall、fold 0、seed 17，固定 6 根隐藏审计 CPT。起步 5 根公开 CPT，完整计划为审计近邻组以外的 23 根 CPT。这里只用于接口与计算可行性，不据此选最终主模型。

| 方法 | 5 根 log-MAE | 23 根 log-MAE | 23 根厚度 MAE / m |
|---|---:|---:|---:|
| GP | 0.7944 | 0.7937 | 0.3333 |
| XGBoost | 0.7888 | 0.7024 | 0.1250 |
| CNP | 0.9227 | 0.9330 | 0.4167 |
| TabPFN joint | 0.7883 | 0.6303 | 0.1667 |
| TabPFN graph | 0.7313 | 0.6898 | 0.1250 |

此样例中，未微调 raw／微调 raw 的起步误差为 0.8332／0.7590，完整计划端点为 0.6435／0.6595，改善方向并不一致。因此微调是否有效仍需跨开发场地完整比较。CNP 两端均无 `ln(qc)<0` 预测，召回为零；它没有因加入更多公开 CPT 自动解决低值识别问题。

两个 TabPFN 权重加其余组件的初始化约 12.7 秒。五主模型及消融全部更新上下文约 0.23 秒（5 根）与 1.25 秒（23 根）；首次 joint 查询约 1.60 秒，完整端点约 0.19 秒。为顺序诊断实测，未固定 CPU 核和重复计时，不能与固定核正式 benchmark 混作性能排名。

记录：[model_diagnostics_probe.csv](../results/adaptive_v1/model_diagnostics_probe.csv)。新轨迹结论应从统一的 E1 和补勘回放表汇总，按场地等权，以开发 log-MAE 主排序，同时显示厚度、召回与误报取舍；旧测试成绩不参与本轮主模型选择。
