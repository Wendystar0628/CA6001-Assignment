# CPT v0.3：训练链路诊断与合法适配结果

完成日期：2026-10-14；分支`feat/v0.3-diagnostics`，结果标签`v0.3-diagnostics`。本轮由训练链路、残差适配、上下文适配、独立树模型四个子任务并行推进，GPU统一串行运行。

**主要结论：TabPFN能够近乎精确拟合固定真实目标和IDW教师，普通XGBoost使用相同特征也未胜过空间基线。本轮证据不支持把失败专门归咎于Transformer结构。最有利的点预测适配是保留IDW判断的小幅TabPFN残差修正，但统一风险校准后的误报率上升，仍未达到综合晋级条件。**

这次完成了诊断和可运行适配，未证明稳定节省CPT根数。所有候选和失败条件保留；原v0.1/v0.2代码、结果及既有v0.2候选选择未被覆盖。

## 1. 训练链路：能学会，但不等于能泛化

固定fold 0源训练场地Avondale的一个真实情景：5根support、2根完整query、48个深度单元。按预登记字典序选择；两个探针从官方TabPFN v2权重独立重启，各200次更新、学习率1e-5、CRPS＋MSE。没有选择验证／测试场地进行拟合。

| 诊断目标 | 初始log-MAE | 200步后log-MAE | 官方训练损失：初始→最终 |
|---|---:|---:|---:|
| 拟合固定真实query标签 | 0.355669 | 0.002752 | 1.367081→0.004768 |
| 拟合固定IDW教师输出 | 0.305152 | 0.000556 | 1.037795→0.001081 |

模型为官方完整11,081,864参数结构。两次均发生200次真实参数更新，没有AMP溢出；保存重载后的点预测、分位数、概率最大差均为0。教师拟合后对真实标签的MAE为0.338001，接近教师IDW的0.337780，符合“学会模仿但没有新增信息”。

这些结果反对“参数没有更新”“正式推断没有使用训练结果”“模型连简单插值关系都不能拟合”的解释。**它们使用了固定query标签训练，不能作为未见场地效果。所选48格没有低于1MPa的事件，也不能据此证明低值分类能力。**

证据：[训练诊断摘要](training_chain/summary.json)、每步历史及初末逐行输出位于同目录。

## 2. 对适配方式的真实验证

复用v0.2全部相同的maximin支持集、固定评价位置、三折、种子17/29、预算5/7/9/11/13。每个模型60个预算状态、8640个评价深度行；每个模型只有12条曲线、6个fold/site组合、**4个不同验证场地**。15个模型／控制ID共900个预算行；其中本轮新增8个方法候选和1个alpha=0恢复控制，旧版6个模型为冻结参考。

宏平均先对预算作归一化梯形积分，再平均种子、同一场地跨fold重复，最后等场地平均。所有候选使用相同位置，不筛掉失败场地。以下risk Recall/FPR为验证勘察组留一校准结果，不能与原点分类指标混用。

| 模型／适配 | 预算平均log-MAE ↓ | H_point误差(m) ↓ | 校准risk Recall ↑ | 校准risk FPR ↓ |
|---|---:|---:|---:|---:|
| IDW | 0.638586 | 0.6304 | 69.55% | 38.35% |
| GP | 0.650156 | 0.6672 | 62.20% | 24.87% |
| 原始TabPFN＋24特征 | 0.753408 | 0.8758 | 54.60% | 37.02% |
| v0.2 CRPS微调TabPFN | 0.708437 | 0.8678 | 62.30% | 44.12% |
| 新独立XGBoost＋24特征 | 0.747789 | 0.9344 | 69.59% | 38.64% |
| 加源训练场地上下文 | 0.736257 | 0.8654 | 59.87% | 39.38% |
| anchor一致构造上下文 | 0.756672 | 0.8639 | 51.17% | 34.58% |
| IDW＋0.25×TabPFN残差 | 0.627386 | 0.6818 | 59.73% | 35.36% |
| IDW＋0.25×残差＋点阈值保护 | **0.627008** | **0.6304** | 71.30% | **46.76%** |

所有alpha、全部原点指标、区间覆盖率及逐场地结果见[完整宏平均](comparison/unique_site_macro.csv)、[逐场地结果](comparison/unique_site_metrics.csv)和[逐预算指标](comparison/metrics.csv)。表中均为探索性验证，不是新的独立测试。

### 残差适配的收益与代价

上下文标签是“已测真实log(qc)−整位置及同坐标留一IDW”。推断时使用“当前IDW＋alpha×原生残差分布”，alpha预登记0/.25/.5/1。正alpha的物理事件概率按F_R((log阈值−IDW)/alpha)重建，alpha=0精确恢复IDW点预测，不虚构区间。

alpha=.25改善了连续误差，但前两折H_point由.6875升至.790365m、原点Recall由.3633降至.2828。完整残差alpha=1则明显变差，说明修正幅度和原任务风险存在取舍。缩小整个残差分布还会导致区间过窄：alpha=.25的90%原生区间实际宏覆盖率约39.77%，不能把这些区间说成可靠。

看到这个结果之后，我们**透明追加**了点阈值保护候选：alpha只试.25/.5，将修正后的预测限制在IDW原来相对1MPa的同一侧。它不看隐藏真值，不改阈值；只输出点预测，不沿用被非线性裁剪破坏的CDF或区间。全部8640格的原点标签与IDW相同，因此H_point、原点Recall/FPR保持不变；这不能描述为提高了低值识别能力。

最佳点保护alpha=.25相对纯IDW的连续误差改善**1.81%**；但经相同的留组Platt校准后，risk FPR由38.35%升至46.76%。原点标签相同，不意味着校准概率和0.2概率阈值产生的风险标签相同。这个结果说明保持点分类尚未解决概率迁移问题。

### 历史上下文和构造一致性

source-context每个预算从各源训练场地固定选一根完整真实query；三折分别336/360/360个源深度行，再拼接目标已测context。选择仅依赖ID、支持数量和固定seed，不看验证表现。它略好于原始TabPFN总体，但未超过微调模型或空间基线。

honest-context将目标已测CPT按几何固定分成约60% anchor和40%标签context，两边特征都只从同一anchor构造。它消除了原始支持数m−1对m的差异，但总体没有改善。该实验同时减少了标签context和邻域信息，不能将负结果解释为“所有构造偏移均无影响”。

六个先前诊断状态的原始特征统计发现：预算5时两个场地query的最近邻距离全部比context对应范围更近。这是maximin支持点彼此较疏造成的构造偏移证据；support_count常量还可能被官方预处理删除。因此原始特征差异是机制线索，尚不是模型失败的独立因果证明。详见[上下文方法与边界](context_adaptation/method_summary.json)。

### 独立树模型对照

XGBoost使用同样24维特征、同样源训练episode、场地等权，深度4、学习率.05、最多500树、验证早停30。三个fold的预算平均误差均未胜过对应IDW／GP；总体CRPS微调TabPFN反而优于这个直接XGBoost。由此不能将失败专门归咎Transformer，但也不能证明任何一种架构都无改进空间。

## 3. 综合门槛与停止位置

每折参考仍是在验证集上选出的IDW／GP较强者，三折分别IDW、IDW、GP。要求主要误差改善，H_point≤基线+.05m，risk Recall≥基线−.02，risk FPR≤基线+.02；全局还要求严格多数不同验证场地改善。

**本轮新增候选全部未晋级。** 点保护残差前两折未过风险FPR护栏；第三折预测误差和FPR都未过。其他候选同样有明确失败项。若把最佳点保护候选在全部折直接使用，相对逐折强参考的宏平均误差只下降约0.44%，且风险护栏失败，不能称为综合胜出。

本轮没有扩大剩余种子或外层测试，没有更改旧质量目标来生成成本节约。已有v0.2的IDW残差是比较参考，不在本轮新增候选晋级池；这里的“全部回退”不代表取消旧v0.2的选择。完整记录见[候选门槛](comparison/candidate_gates.csv)与[本轮选择](comparison/selection.json)。

对后续工作的判断：保留强空间基线的结构化残差适配，比直接替换为更大Transformer更有当前证据支持；但收益仍小，且风险概率迁移没有解决。下一轮若继续，需单独研究空间残差的可迁移性和风险概率校准，并获取更独立的场地证据；不能把再次利用这4个验证场地的调参当作确认性证明。

## 4. 检查、代码与复现

18项新定向测试通过；独立只读复算确认15个模型的60状态完整、逐cell评价位置相同、重复场地权重一致，alpha=0精确恢复IDW。检查没有发现残差CDF变换或点／风险指标混用。固定情景拟合检查仅覆盖该小情景，不能据此声称所有潜在实现问题都已排除。

代码在`src/cpt_budget/diagnostics/`，协议为[诊断与适配共识](../../docs/V3_DIAGNOSTIC_PROTOCOL.md)。依赖沿用项目环境；统一状态读取本机`runs/v2/v2-validation-20261013-r2`，需先完成该v0.2运行或恢复完整本机快照，不能只靠本目录的精简表替代逐深度预测。

从Project根目录运行，下列GPU命令应依次执行：

```powershell
.venv/Scripts/python.exe -m cpt_budget.diagnostics.training_chain --steps 200
.venv/Scripts/python.exe -m cpt_budget.diagnostics.xgb24
.venv/Scripts/python.exe -m cpt_budget.diagnostics.run_adaptations xgb_validation
.venv/Scripts/python.exe -m cpt_budget.diagnostics.run_adaptations prepare_contexts
.venv/Scripts/python.exe -m cpt_budget.diagnostics.run_adaptations residual
.venv/Scripts/python.exe -m cpt_budget.diagnostics.run_adaptations source_bank
.venv/Scripts/python.exe -m cpt_budget.diagnostics.run_adaptations honest
.venv/Scripts/python.exe -m cpt_budget.diagnostics.run_adaptations point_guarded
.venv/Scripts/python.exe -m cpt_budget.diagnostics.analyze
```

固定诊断目录会复用已完成的逐状态预测；改变方法参数必须使用新目录／新协议，不能混入旧缓存。训练诊断checkpoint不属于可部署候选；不要把用query真值过拟合的权重用于验证回放。

本目录保留摘要、源bank来源、完整数值指标与门槛；大权重、逐深度候选预测和状态context清单保留本机`runs/diagnostics-20261014/`。原始数据和模型许可沿用项目已有说明。

关键Git节点：`fcef55a`冻结诊断 → `c626734`训练链路与残差评估 → `b2b2b69`上下文与比较 → `12d4c2e`点阈值保护 → `v0.3-diagnostics`结果冻结。

官方依据：[TabPFN v2模型卡](https://huggingface.co/Prior-Labs/TabPFN-v2-reg)、[9.2.0微调实现](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/finetuning/finetuned_regressor.py)、[9.2.0集成实现](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/preprocessing/ensemble.py)。本轮锁定9.2.0和v2权重，不使用当前3.x文档默认值替代实际运行设置。
