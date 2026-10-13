# 模型实现与训练诊断

本次只读取冻结的主实验与源码；新建两个 CPU 小诊断，未用 GPU，未改 src/config/runs/results，未重训。数字来自 `diagnostic.json` 与 `gp_probe.json`。

判断：没有发现“没有真实梯度微调”“阈值重复标准化”“查询标签进入输入”等足以一键解释结果的已证实错误。损失调用与官方 v9.2.0 一致、支持/查询转换 support-only、推断替换本场地上下文，与蓝图大体吻合。现有差距更像**特征/任务定义与评价目标的适配不足，加上探索预算很浅**；不能归结为单纯加速或延长训练。

## 1. 已证实：物理几何尺度在 TabPFN 预处理中被消掉

- `src/cpt_budget/features.py:69` 添加每场地恒定 `log_geometry_scale`，邻居距离在第 91 行按此尺度归一。蓝图想同时保留相对距离与物理尺度。
- 官方默认 `POLYNOMIAL_FEATURES="no"`，预处理无条件经过 `RemoveConstantFeaturesStep`。每个 context 来自单场地，所以物理尺度、support_count 和三个 valid 列恒定并被删除。
- 用真实 New Lock 的 5 根支持集调用安装包真实预处理步骤，16 列只剩 11 列。把坐标和尺度同时放大 100 倍，预处理后 support/query 特征最大差异仅 2.9e-13/4.6e-13。这说明现有配置实际无法区分相同相对几何但绝对间距大幅不同的两个任务。未做神经网络推断，因此结论精确限于进入后续处理的数据。
- 性质：明确的设计意图—实际数据流偏差；对结果影响大小未验证。IDW 本身也尺度不变，因此不能说修复必胜 IDW。
- 最小下一步：在验证场地比较原特征、保留有效物理几何/邻居方向/邻深层摘要的表示、IDW 残差学习；先证明新增信息穿过全部预处理，再少量训练。仅重复添加一个常量列不会修复。
- 置信：特征消失高；解释性能差距中低。

官方来源：[v9.2.0 pipeline_factory](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/preprocessing/pipeline_factory.py)、[RemoveConstantFeaturesStep](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/preprocessing/steps/remove_constant_features_step.py)。

## 2. 已证实训练探索很浅，但不能称为“只要多训就会好”

- `tabpfn_train.py:164–169` 每场地每 epoch 只做一次更新，在三个固定 mask 中轮换，每次最多八根 query。
- 三折实际总更新 140/150/150，计时约 130.5/129.3/122.9 秒；并非碰到蓝图两小时训练上限。
- 最佳 checkpoint 分别 epoch 7/8/9。当时覆盖固定 episode-query 组合 62.0%/92.3%/66.6%。fold0 New Lock 213 个可用组合只见过 56 个，fold2 只见过 72 个；这是维持场地均衡的实现后果，不能直接把大场地所有行堆进去改变权重。
- 所有梯度范数均超过 1、确实裁剪，中位原范数约 9.45/8.60/9.97；这不是梯度消失或优化器没更新的证据。
- fold0 已早停，fold1 在 epoch8 最优后变差，fold2 epoch9/10 几乎持平；简单重复固定任务可能继续过拟合。训练 NLL 与验证最佳并不一致。
- 最小下一步：按总更新次数/真实查询曝光而非“10 epoch”定义小试验；先在源训练/验证中做更多整位置遮蔽与完整 query 轮换，保持场地总权重相同。建议只试 150 vs 450 步、学习率 1e-5 vs 3e-6；可保留预训练 checkpoint 为合法候选。验证曲线无改善即停止，不盲目烧 GPU。
- 置信：预算/覆盖高；更多任务带来提升中，延长原训练有效性低。

## 3. 训练、选模型与核心目标存在真实的目标差别

- `tabpfn_adapter.py:243` 只优化分布 NLL；`tabpfn_train.py:77–98,196–198` 只按逐场地 log-RMSE 选 checkpoint；共同点预测是分布中位数。核心预期却要求 MAE 或同质量少补勘，并限制厚度误差与漏判。
- 这符合原蓝图，是算法设计选择，不是实现漏写。NLL可以通过更宽、较可靠的分布改善；RMSE偏好压低大误差；中位数更贴近MAE，三者与阈值分类/厚度并非同一目标。
- 官方实现支持 CRPS、CRLS、MSE/MAE 辅助项，但其中 MAE辅助针对**分布均值**，不能误称它直接优化当前中位数 MAE；也不能只增加分类权重牺牲 qc 剖面。
- 更接近主目标的最小路线：先采用 `prediction = IDW + residual`，把简单空间插值作为结构化先验，学习需要修正的位置；用源验证 MAE 主指标并设厚度/Recall护栏选强度。保留 NLL 或 CRPS 训练分布，评估可靠性，避免概率带只会变宽。先看验证集是否优于IDW，达不到便不进正式回放。
- 与主代理其他证据合并时，重点看 IDW残差、混合模型是否已有验证/探索结果。本项仅给结构可行性，未训练这些新模型。
- 置信：目标差别高；它是当前落后主因中，改法是否达到10%待实验。

官方来源：[v9.2.0 回归微调损失](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/finetuning/finetuned_regressor.py)。

## 4. 最小 GP 实验证据不支持“优化器少重启就是主因”

- `gp.py:61–74` 使用固定源深度趋势、初始尺度 [20,20,1]m、ARD Matérn+WhiteKernel、默认 0 次重启。主配置未做 GP 的验证超参数搜索，称作结构合理基线更准确，尚非充分优化的专业上限。
- 在 fold0 的两个**验证场地** seed17、预算5/13四个冻结状态做CPU试验：改成2次重启，log-MAE最大变化仅 3.9e-5，没有发现明显坏局部最优。
- 再加入只依赖已揭示支持的局部残差均值：Lower Hutt13根 MAE 0.7600→0.7305；Tauranga13根 MAE 0.6072→0.6004，但是 Recall 0.1569→0，厚度 MAE 1.25→2.125m。更高的边际似然、更好的单一整体误差不保证核心任务更好。
- 最小下一步：只用训练/验证比较源趋势、局部趋势及核配置，带 Recall/厚度护栏。不要通过削弱/不调 GP 来让主模型显得胜出，也不要从四状态试验泛化为“所有GP优化都无效”。
- 置信：此四状态重启无效高；全项目是否存在其它GP提升空间仍开放。

官方来源：[scikit-learn Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)。
