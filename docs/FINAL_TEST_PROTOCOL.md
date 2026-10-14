# 最终独立外层测试协议（预注册，v0.9.1）

本文件在读取任何测试场地真值之前写成，并与测试状态、模型目录、风险校准器、代码树、模型权重和盲跑预测一起以SHA-256冻结在`results/final_test/frozen.json`。冻结后本文件的任何改动都会使`final-test --unblind`拒绝运行。揭盲（第2阶段）只能由项目负责人单独发起。

修订记录：第1阶段冻结后、揭盲前，按负责人决定把测试范围限定为v0.9.1当前设计（删除旧版本相关内容与对应的代码路径），测试场地、状态、校准器、问题与判定规则均未改变，随后以`--force-refreeze`重新冻结。

## 1. 目的

在v0.9.1用于模型选择与验证的4个验证场地之外，用没有参与v0.9.1选择与验证的场地，对v0.9.1冻结的13个入口做一次独立外层测试，回答第6节预先固定的5个问题。本测试只回答"在这些场地上相对IDW是否改进"，不重新选模型、不调参数、不改阈值。

## 2. 测试场地

### 2.1 候选范围与判定方法

候选范围是全部可回放场地（`sites.parquet`中`replay_eligible`，即≥20个位置）。共8个。每个场地在3个折中的角色取自`artifacts/evaluation/folds.json`，是否进入60个冻结验证状态取自`artifacts/evaluation/states.parquet`（代码：`final_test.site_roles()`，结果：`results/final_test/site_roles.csv`）。

| 场地（site_id） | 位置/近邻组 | fold 0 | fold 1 | fold 2 | 60个冻结验证状态 | 分组核实 | 决定 |
|---|---|---|---|---|---|---|---|
| Chch Stadium（`christchurch_stadium`） | 23/20 | **test** | train | train | 否 | 是 | **纳入，fold 0权重** |
| Wellington Airport（`wellington_airport`） | 27/27 | **test** | train | train | 否 | 是 | **纳入，fold 0权重** |
| New Lock（`newlock`） | 80/74 | train | **test** | train | 否 | 否 | **纳入，fold 1权重** |
| Hollywood（`hollywood`） | 25/13 | train | train | **test** | 否 | 否 | **纳入，fold 2权重** |
| Lower Hutt（`lower_hutt_town_hall`） | 29/29 | validation | validation | test | fold 0、1 | 是 | 排除 |
| Tauranga（`tauranga_farmers_building`） | 28/28 | validation | validation | test | fold 0、1 | 是 | 排除 |
| Chch Metro Sports（`christchurch_metro_sports_facility`） | 39/35 | test | train | validation | fold 2 | 是 | 排除 |
| Marshland（`marshland`） | 20/20 | train | test | validation | fold 2 | 是 | 排除 |

排除理由：后4个场地的标签构成了全部60个冻结验证状态，是v0.9.1模型选择与验证所依据的场地，已不独立。代码断言"未进入冻结验证状态的可回放场地"恰为上表4个纳入场地，且每个纳入场地只在协议所列的那一折是test组。

场地身份核对（避免同名混淆）：
- `marshland`（20个位置，可回放，fold 2验证）与`marshland_road`（2个位置，不可回放，也是fold 2验证组）是两个不同的勘察组；两者都不是测试场地。
- `christchurch_metro_sports_facility`（39个位置）与`christchurch_stadium`（30条剖面、23个位置）是两个不同勘察组，分属fold 2验证与fold 0测试；在fold 0中Metro Sports同为test组，不进入fold 0任何训练或验证。
- `newlock`即New Lock（Terneuzen，荷兰）；`hollywood`即Hollywood。

### 2.2 分组未核实（Hollywood、New Lock）

`folds.json`中`exclude_unverified_groups=false`、`unverified_grouping_sites=[avonside, baytown, hollywood, newlock]`。含义：这4个场地没有经纬度或已声明的坐标系，只有场地内局部米制坐标，因此无法检验它们与其他场地是否相距不足250 m（同一勘察区）。每个场地被暂时单独成组；"跨场地独立性未核实"。若某个未核实场地实际与某训练场地属于同一勘察区，外层测试会因空间相关而偏乐观。

决定与理由：
- **New Lock纳入。** 下载清单把它识别为Terneuzen（荷兰）New Lock项目，其余场地都在新西兰（含有坐标的27个）；与它同为未核实的Avonside（基督城郊区）、Baytown、Hollywood来自不同的原始场地文件夹。它与fold 1训练场地同属一个勘察区的可能性很低，但无法用坐标证明。
- **Hollywood纳入。** 原始包中是独立命名的场地文件夹（P1-1…P5-9网格编号），无坐标。另外两个未核实场地Avonside、Baytown在fold 2同为test组，不进入fold 2训练或验证；fold 2的训练与验证场地除New Lock外都是有坐标的新西兰场地。剩余风险是Hollywood实际位于某个fold 2训练/验证场地250 m以内，无法排除但无证据支持。
- 由于不能证明独立，**预注册敏感性分析**：除4场地等权主结果外，同时报告仅含分组已核实的2个场地（Chch Stadium、Wellington Airport）的等权结果（`test_ranking_verified_sites.csv`）。若两者结论方向不同，按第6.6节如实报告，不以任一方为准。

### 2.3 测试场地在其他折中的角色

测试场地在自己的测试折之外是训练源（例如Hollywood、New Lock是fold 0的训练场地）。这些折的权重参与了在验证场地上的v0.9.1选择，因此存在交叉折设计固有的弱间接通道；但没有任何选择或验证步骤使用过测试场地在其测试折中的标签或指标。

## 3. 每个场地使用的权重与泄漏审计

每个测试场地只用其为test组的那一折的全部推理工件评分：Chch Stadium、Wellington Airport用`artifacts/models/fold_0/`，New Lock用`fold_1/`，Hollywood用`fold_2/`。`final-test --prepare`逐项检查（`results/final_test/leakage_audit.json`，4/4通过）：

| 工件 | 该折的拟合来源 | 检查 |
|---|---|---|
| TabPFN残差检查点`tabpfn.ckpt` | `tabpfn_training.json`的train_sites、validation_sites（早停）、exposure.per_site | 测试场地不在其中 |
| XGBoost残差`xgb.json`/`residual_trees.ubj` | train_sites、validation_sites | 同上 |
| CNP`cnp.pt` | train_sites、validation_sites、source_normalizer_sites | 同上 |
| FPCA基`basis.npz` | `basis_manifest.json`的source_site_ids（运行时另有断言"Source FPCA contains target site"） | 同上 |
| GP深度趋势`gp_trend.json` | 用该折`folds.json`的train_sites重拟合`fit_gp_trend`，与存档逐值比较 | 最大绝对差0（3折） |
| 风险校准器 | 见第5节 | 训练场地⊂该折验证场地，与测试场地无交集 |
| 运行时拟合部分 | 三角网/Haar、结构融合堆叠权重、整曲线误差原子、图Huber、IDW、GP超参数 | 每个状态只用该状态已揭示的CPT |
| 图Huber规划几何 | 该场地全部位置的坐标（无测量值），与验证协议相同 | 为使测试场地可用，`experiment.planned_geometry()`扩展到全部可回放场地；图按场地单独构建，`verify`复核13×60＝780项仍全部通过 |

所有清单还须落在该折train/validation场地之内（检查通过）。

## 4. 测试状态

- 生成器：`data.splits.make_replay_splits`（评价集＝按种子随机选取的完整近邻组，目标≈20%位置，在(场地,种子)内固定；初始5个点＝在剩余池中从随机起点做最远点采样）＋`acquisition.select_batch(policy='maximin', predictions=None)`，每批2个，预算5、7、9、11、13。只读取坐标、位置ID和近邻组ID：生成时使用的数据视图`coordinate_dataset()`的网格只有`site_id`、`location_id`两列，物理上不含qc。
- 等价性：同一代码以种子17、29重现全部60个冻结验证状态（已知集顺序与评价集逐一相同，60/60）。
- 种子：17、29、43、71、101（全部5个配置种子）。
- 规模：4个场地×5个种子×5个预算＝**100个状态**；100个不同的已知集，19个不同的评价集（20个(场地,种子)中有一对评价集相同）。每状态评价位置数：Chch Stadium 5、Wellington Airport 6、New Lock 16、Hollywood 5（×24个深度单元）。
- 文件：`artifacts/evaluation/test_states.parquet`（已登记于`artifacts/manifest.json`）。

## 5. 风险校准（只用非测试标签）

对每个入口、每个用到的折（0、1、2）：取该折**全部**20个冻结验证状态（2个验证场地×2种子×5预算）的参考预测（`artifacts/evaluation/reference_predictions.parquet`，与实时推理780/780等价），拟合v0.9.1同一个单调Platt校准器（`risk.fit_risk_calibrator`，场地等权，斜率非负）；输入与验证一致：共享经验风险头与GP用logit(`p_low_1`)，其余用−`pred_log_qc`。阈值P≥0.2（漏报:误报成本4:1）。校准器原样应用于该折的测试状态，不再修改（`results/final_test/calibrators.json`）。fold 0、1的校准场地为Lower Hutt与Tauranga，fold 2为Chch Metro Sports与Marshland。

说明：验证结果中的风险列是"留一勘察组"交叉校准；测试时用该折全部验证标签拟合的最终映射，等同于`crossfit_risk`返回的final校准器（测试已核对系数一致）。

## 6. 冻结的入口、指标与预注册问题

### 6.1 入口（冻结于v0.9.1，揭盲后不作任何修改）

`configs/catalog.json`的13个入口：基线`constant`、`idw`、`gp`、`xgb`；TabPFN`tabpfn_joint`、`tabpfn_graph`；主方案`balanced`、`thickness`、`recognition`；讨论`simplex`、`count_shrink`、`thickness_guarded`、`cnp`。`source_model_id`、`risk_source`、设备以目录为准；目录标为cuda的7个入口的TabPFN部分在RTX 4080上运行（盲跑显式要求CUDA，不允许静默回退CPU）。

### 6.2 指标与聚合（与v0.9.1完全相同）

每个(入口,状态)：在评价单元上合并计算`evaluate`与`evaluate_risk`（log-MAE、点厚度MAE `h_point_mae_m`、点召回、点FPR、风险召回、风险FPR、风险Brier、GP的90%覆盖率与宽度）→ 对每个(fold,场地,种子)在预算5–13上做归一化梯形AUBC（`analysis.curve_metrics`；任一预算未定义则该曲线为NaN）→ 每个场地对5个种子取均值（pandas默认跳过NaN）→ 测试场地等权平均（跳过NaN；每个指标同时报告参与的场地数）。单元测试已核对该聚合函数在验证指标上逐值重现`results/current/all_model_ranking.csv`。

召回在评价集没有低qc单元（qc<1 MPa）时无定义；此时该(场地,种子)不参与召回均值，如某场地5个种子全部无定义，则召回的等权平均只覆盖其余场地，并在表中注明场地数。

### 6.3 与IDW的配对比较

对每个入口和指标：场地级差值（入口−IDW）、场地胜/平/负（|差|≤1e-12记平；方向按指标优劣）、(场地,种子)级20对差值的胜负数与均值；场地级精确双侧符号检验p值仅作描述（4个场地时最小可能p＝0.125，不可能达到0.05）。另对`thickness_guarded`做同样的配对（用于Q2）。

单项判定（按等权均值差Δ与场地胜负，n为该指标有定义的场地数）：
- **一致改进**：Δ优于IDW，且至少⌈0.75n⌉个场地优于IDW（n＝4时为≥3个）；
- **仅均值改进**：Δ优于IDW，但场地胜数不足；
- **一致变差**：Δ劣于IDW，且至少⌈0.75n⌉个场地劣于IDW；
- **无改进**：其他情况（含Δ＝0）。

### 6.4 预注册问题与判定措辞

- **Q1 Balanced是否在log-MAE和点厚度MAE上优于IDW？** 两项都"一致改进"→"Balanced在独立测试上一致优于IDW"；至少一项为"一致改进"或"仅均值改进"→"Balanced对IDW只有部分改进（列出指标与胜负）"；两项都"一致变差"→"Balanced在两项指标上一致劣于IDW"；否则→"Balanced相对IDW无改进"。
- **Q2 Thickness是否在点厚度MAE上优于IDW与thickness_guarded，其点FPR是多少？** 分别给出对IDW、对thickness_guarded的单项判定；报告Thickness、thickness_guarded、IDW的测试点FPR，以及Thickness是否满足护栏"点FPR≤测试IDW点FPR+2个百分点"。护栏不满足时写明"厚度改进以更高误报为代价"。
- **Q3 Recognition是否提高点召回，代价是多少FPR？** 点召回对IDW的单项判定（越高越好）；点FPR对IDW的单项判定、FPR增加的百分点数，及是否满足IDW+2个百分点护栏。召回提高但FPR超护栏时写明"召回提高伴随超出护栏的误报"。
- **Q4 TabPFN入口是否优于IDW？** `tabpfn_joint`与`tabpfn_graph`各自在log-MAE与点厚度MAE上的单项判定，措辞同Q1。须同时说明`tabpfn_joint`中TabPFN只提供一半幅值底座。
- **Q5 排名是否稳定？** 13个入口在验证（`results/current/all_model_ranking.csv`）与测试上的等权均值，分别对log-MAE与点厚度MAE计算Spearman ρ：ρ≥0.7"稳定"，0.4≤ρ<0.7"中等"，ρ<0.4"不稳定"；并报告验证前3与测试前3的重合数、Balanced的测试名次。

"无改进"是合法结论，按原措辞写入报告；不得改用其他指标、子集、预算或阈值来改写结论。

### 6.5 无论结果如何都报告

13个入口的全部指标：4场地等权总表、逐场地表、逐预算表（逐场地与等权）、(场地,种子)曲线表、逐状态表；对IDW与对thickness_guarded的配对表与胜/平/负；仅核实场地的敏感性总表；GP的90%名义覆盖率；盲跑运行时间（`results/final_test/blind/runtime.csv`）。

### 6.6 解释规则

- 主结果是4场地等权；仅核实场地的结果是敏感性分析。两者方向不一致时报告为"结论依赖未核实分组的场地"。
- 只有4个测试场地，结果是描述性的；"一致改进"不等于统计显著。
- 揭盲后的任何额外分析（新指标、新子集、诊断）必须标为"揭盲后探索"，不得改变本协议的判定。

## 7. 已知局限

1. 只有4个测试场地（其中2个分组未核实），每个场地的评价集只有5–16个位置，种子间评价集固定、只是已知集不同，种子不是独立重复。
2. 测试场地在其他折中作为训练源，影响过那些折的权重，而那些权重参与了验证场地上的选择（弱间接通道，见2.3）。
3. 风险校准只用2个验证场地拟合（每折），对新场地的可迁移性有限；风险指标共享同一经验风险头，6个入口的风险列相同。
4. GP是唯一给出区间的入口；验证上90%名义覆盖率实际约80%，测试上同样只报告不修正。
5. 场地坐标为各场地局部坐标；New Lock、Hollywood无全球坐标。
6. 本测试评价的是"给定冻结的maximin揭示序列下的预测质量"，不评价选点策略，也不证明能节省CPT数量。
7. 盲跑在GPU上运行；TabPFN在CPU上与参考不等价，复现须用CUDA。

## 8. 执行顺序与防护

```powershell
.venv/Scripts/python.exe -m cpt_budget final-test --prepare   # 状态、泄漏审计、校准器（冻结后拒绝运行）
.venv/Scripts/python.exe -m cpt_budget final-test --blind     # 13×100盲跑，写results/final_test/blind/
.venv/Scripts/python.exe -m cpt_budget final-test --freeze    # 写frozen.json（揭盲后拒绝重新冻结）
.venv/Scripts/python.exe -m cpt_budget final-test --unblind   # 第2阶段，仅一次
```

- 盲跑的结构性防护：每个状态的修订网格先删除全部评价位置的行，再取已揭示剖面；查询来自不含任何测量列的几何网格；运行期间`PreparedDataset.truth`被替换为抛错；输出帧出现任何测量/真值列即报错。测试中把真值读取函数替换为抛错后盲跑仍然通过。
- `--unblind`在读取真值前：重新计算协议、测试状态、目录、校准器、代码树（`src/cpt_budget/**/*.py`、`configs/main.yaml`、`pyproject.toml`）、模型与数据工件、验证排名、盲跑预测的SHA-256，任一与`frozen.json`不同即拒绝；先写入运行日志`results/final_test/unblind_log.jsonl`再读真值；已有运行记录时拒绝再次运行，除非`--force-rerun`，此时输出写到`unblind_rerun_<n>/`并在日志中标记为rerun。
- 揭盲只对冻结的盲跑预测评分，不重新推理。
- 读取测试真值的唯一函数是`final_test.load_test_truth`，只被`unblind`调用。
