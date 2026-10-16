# CA6001 CPT v0.9.1 历史预测方案对比与取舍

本页保存v0.9.1的13个固定入口比较。新的五AI、自适应停止与经济评价见[自适应实验协议](docs/ADAPTIVE_PROTOCOL.md)、[当前实现状态](docs/IMPLEMENTATION_STATUS.md)及`results/adaptive_v1/`。原始对比数据和本页指标保持不变，历史训练与探索见Git标签`v0.8-engineering`。

## 指标总表

数据来自[`results/current/all_model_ranking.csv`](results/current/all_model_ranking.csv)，可用`python -m cpt_budget evaluate`重算。↓表示越低越好。

| 类别 | 入口 | 方案 | log-MAE ↓ | 厚度MAE(m) ↓ | 点召回 | 点误报 | 风险召回 | 风险误报 | 风险Brier ↓ | risk_source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 基线（弱） | `constant` | 常数中位数 | 0.889189 | 0.966146 | 0.00% | 0.00% | 25.00%* | 25.00%* | 0.158444 | platt_on_point |
| 基线（强） | `idw` | 3NN IDW | 0.638586 | 0.630371 | 37.38% | 7.28% | 69.55% | 38.35% | 0.130449 | platt_on_point |
| 基线（强） | `gp` | 空间GP | 0.650156 | 0.667155 | 33.43% | 7.37% | 62.20% | 24.87% | 0.125157 | native_gp |
| 基线（强） | `xgb` | 同轨迹XGBoost残差 | 0.634278 | 0.648112 | 34.26% | 6.76% | 68.96% | 39.30% | 0.130882 | platt_on_point |
| TabPFN组合 | `tabpfn_joint` | TabPFN残差与三角网各半作底座＋整曲线决策 | 0.598844 | 0.525228 | 37.68% | 8.19% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| TabPFN组合 | `tabpfn_graph` | TabPFN＋图Huber低值并集 | 0.616500 | 0.618978 | 42.47% | 8.60% | 74.40% | 29.18% | 0.121324 | platt_on_point |
| 主方案 | `balanced` | 均衡：结构融合＋整曲线决策 | 0.592435 | 0.536947 | 38.52% | 8.31% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| 主方案 | `thickness` | 厚度：Haar计数决策（最近剖面细节） | 0.614517 | **0.472493** | 39.38% | 10.05% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| 主方案 | `recognition` | 识别：可靠性定位＋60%计数 | 0.613079 | 0.526855 | **48.77%** | 9.14% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| 讨论方案 | `simplex` | 三角网单独插值 | 0.607248 | 0.585286 | 34.00% | 7.04% | 66.65% | 25.03% | 0.121464 | platt_on_point |
| 讨论方案 | `count_shrink` | 支持集计数收缩 | **0.591791** | 0.546712 | 38.20% | 7.98% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| 讨论方案 | `thickness_guarded` | Haar厚度（点误报护栏内最优） | 0.611418 | 0.487630 | 38.36% | 8.99% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| 讨论方案 | `cnp` | 条件神经过程（负结果） | 0.854427 | 0.966146 | 0.00% | 0.00% | 37.55% | 38.73% | 0.192664 | platt_on_point |

\* 常数基线的风险值是退化结果：校准后只在Lower Hutt全判为阳性（该场地风险召回与风险误报均为100%），其余三个场地全判为阴性，不代表任何风险识别能力。

相对IDW：均衡log-MAE −7.23%、厚度MAE −14.82%；厚度方案厚度MAE −25.05%，点误报+2.77pp；识别方案点召回+11.39pp、点误报+1.86pp；`tabpfn_joint`log-MAE −6.22%、厚度MAE −16.68%。

**指标定义。** 每个状态先在评价剖面上计算：log-MAE为ln(qc/MPa)绝对误差先在每根剖面内平均、再对剖面平均；厚度MAE为2–8 m窗口内每根剖面预测与实测“qc<1 MPa单元数×0.25 m”之差的绝对值，再对剖面平均；点召回与点误报是点预测qc<1 MPa在全部深度单元上的Recall与FPR（FPR＝FP/(FP+TN)）；风险召回/误报以校准后风险概率≥0.2判定；风险Brier为校准概率与实测低阻力标签的平方误差。随后每条曲线（fold×场地×种子）按预算5、7、9、11、13做梯形积分并除以8，得到AUBC；同一场地的曲线先平均（Lower Hutt、Tauranga为2个fold×2个种子，另两场地为2个种子），再对4个场地等权平均。表中6位小数只为便于与CSV逐值核对；厚度误差以0.25 m为量化单位，又只来自少量评价剖面，有效精度远低于小数位数。

## 为什么这样保留

选择规则是：每个保留入口要么是同一评价协议下该角色的历史最优配置，要么写明它为何不是、换来了什么。逐项依据见[方案选择依据](docs/MODEL_SELECTION.md)。v0.9.1相对v0.9.0有三处变化：

- `thickness`改为历史最优厚度配置`v7_hcount_simplex_nearest_a100_gp`的点路径（0.472493 m，原为0.487630 m）。代价是点误报10.05%，超出v0.7晋级规则的护栏（≤IDW+2pp＝9.28%）。
- 原`thickness`（`v8_haar_fast`）改为讨论方案`thickness_guarded`，是满足点误报护栏的最优厚度配置。
- `tabpfn_fpca`（`v6_fpca_residual_sign_guard`，0.613446 / 0.630371 m）被`tabpfn_joint`（`v7_pair_joint_simplex_empirical_graph`，0.598844 / 0.525228 m）替换。原因是原入口在log-MAE、厚度、点平衡错误、风险平衡错误四项上同时被另一配置支配。它的低阻力点判定按构造与IDW逐单元相同，所以厚度MAE、点召回、点误报都与IDW完全一样。新入口是v0.7晋级主线，位于全部配置的log-MAE–厚度Pareto前沿上。

TabPFN组合中，`tabpfn_joint`的幅值底座由“受保护的TabPFN残差”和“三角网插值”各占一半，低阻力标签由整曲线经验误差决策决定，所以TabPFN只贡献一半幅值底座；三角网单独使用（`simplex`）已达log-MAE 0.607248。`tabpfn_graph`代表较高的低阻力召回：在点误报不超过9.14%的TabPFN配置中，它的召回最高。

三个主方案分别面向均衡、厚度和识别。识别方案采用v0.8的recall60，是记录过的最高点召回，但点误报9.14%高于IDW的7.28%。讨论方案中，`simplex`说明局部几何本身已经很强；`count_shrink`的log-MAE最低，但厚度更差，完整链路耗时约为均衡方案的1.89倍；`thickness_guarded`给出满足误报护栏的厚度选项；`cnp`是负结果（见下文）。

## 风险列怎么读

没有任何入口在预测时直接输出风险标签。表中风险列都在评价时由`evaluate`计算：对每个模型、每个外层fold分别做单调Platt校准；每折只有2个可回放验证场地，按“留一调查组”交叉拟合，所以每个场地的校准器只用同折另一个验证场地的标签拟合。校准后以风险概率≥0.2判为低阻力，0.2来自“漏判:误报＝4:1”的课程情景成本假设，未做阈值敏感性分析。这一校准用到了验证标签，因此风险指标也属于验证集上的探索性结果。

`risk_source`说明校准前的分数从哪里来：

- `shared_empirical_head`：共享经验风险头`v6_empirical_graph_union_local`的低阻力概率。它由`tabpfn_graph`的点路径加经验误差分布得到，与入口自己的点路径无关。因此`tabpfn_joint`、`balanced`、`thickness`、`recognition`、`count_shrink`、`thickness_guarded`这6个入口的风险指标完全相同（68.21% / 23.90% / 0.121124），不能用来区分这6个入口。
- `native_gp`：GP高斯预测分布给出的概率。
- `platt_on_point`：没有原生概率，对点预测−log qc做Platt校准。`tabpfn_graph`的风险数字（74.40% / 29.18%）也来自这条路径，而不是共享风险头。`simplex`、`constant`、`cnp`等同样没有风险头。

点识别（点预测qc<1 MPa）与风险识别（校准概率≥0.2）是两套不同的判定流程，两者的召回与误报不能直接对比。

## 运行时间（v0.9.1统一测量）

来源：[`runtime_summary.csv`](results/current/runtime_summary.csv)、[`runtime_protocol.json`](results/current/runtime_protocol.json)，命令`python -m cpt_budget benchmark --cpu-core 2 --publish results/current`。

协议：全部13个入口使用同一协议。共12个状态：3个fold×2个场地×seed 17×预算5/13。其中fold 0与fold 1在Lower Hutt、Tauranga上的输入相同，实为8种不同输入。每个状态1次预热、3次计时。进程绑定逻辑核2，BLAS/OpenMP单线程，计时用`perf_counter`，CUDA路径在计时边界同步。权重加载单独计时。不含数据读取、与参考结果的对照、指标评价和风险校准。CPU/GPU频率未锁定，入口按固定顺序串行运行，可能存在顺序效应。硬件为i7-14700KF＋RTX 4080，运行前GPU利用率约2%。

| 入口 | 路径 | 设备 | 中位ms | p95 ms | 条件化中位 | 预测中位 | 初始化中位 | 与参考一致 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `constant` | 完整 | CPU | 2.6 | 2.7 | 0.8 | 1.8 | 2.9 | 是 |
| `idw` | 完整 | CPU | 6.8 | 7.2 | 3.1 | 3.7 | 2.9 | 是 |
| `simplex` | 完整 | CPU | 8.9 | 9.7 | 5.3 | 3.6 | 2.9 | 是 |
| `cnp` | 完整 | CPU | 15.5 | 19.1 | 8.0 | 7.1 | 5.0 | 是 |
| `gp` | 完整 | CPU | 74.5 | 197.1 | 69.3 | 5.1 | 3.4 | 是 |
| `xgb` | 完整 | CPU | 83.4 | 110.1 | 1.4 | 82.0 | 4.4 | 是 |
| `tabpfn_graph` | 完整 | CUDA | 136.4 | 162.1 | 18.5 | 117.6 | 67.1 | 是 |
| `thickness` | 完整 | CUDA | 151.2 | 176.7 | 28.7 | 122.0 | 66.9 | 是 |
| `thickness_guarded` | 完整 | CUDA | 165.8 | 193.4 | 33.2 | 132.6 | 72.9 | 是 |
| `tabpfn_joint` | 完整 | CUDA | 169.1 | 202.6 | 35.3 | 134.0 | 75.5 | 是 |
| `balanced` | 完整 | CUDA | 183.2 | 202.4 | 47.9 | 130.5 | 70.2 | 是 |
| `recognition` | 完整 | CUDA | 186.2 | 197.9 | 50.2 | 128.1 | 64.1 | 是 |
| `count_shrink` | 完整 | CUDA | 345.8 | 432.3 | 50.2 | 298.7 | 66.2 | 是 |
| `thickness` | 仅点预测 | CPU | 9.9 | 10.9 | 6.0 | 4.0 | 3.3 | 是 |
| `thickness_guarded` | 仅点预测 | CPU | 10.3 | 11.5 | 6.0 | 4.2 | 2.9 | 是 |
| `balanced` | 仅点预测 | CPU | 37.1 | 44.5 | 27.1 | 10.0 | 4.0 | 是 |
| `recognition` | 仅点预测 | CPU | 39.6 | 48.5 | 29.8 | 9.9 | 4.0 | 是 |
| `count_shrink` | 仅点预测 | CPU | 197.4 | 268.8 | 26.0 | 171.3 | 3.7 | 是 |
| `tabpfn_joint` | 仅点预测 | CPU | 1857.2 | 2751.0 | 28.8 | 1829.7 | 41.7 | 否（最大点差0.33，1个单元标签变化） |

读表时注意：

- **设备不同，不是同条件比较。** “CUDA”行只表示TabPFN残差组件在RTX 4080上运行（没有CUDA时直接报错），其余数值计算仍在绑定的单个CPU核上；“CPU”行全部在该CPU核上。CPU行与CUDA行之间的倍数不能当作算法复杂度之比。
- 完整路径包含共享风险头。风险头每次都要重新计算`tabpfn_graph`，主方案的大部分耗时来自它：例如`thickness`完整路径151.2 ms，仅点预测9.9 ms。`tabpfn_joint`的点路径与风险头共用同一次TabPFN前向计算。
- “仅点预测”（`--point-only`）只适用于有独立风险头的6个入口，全部在CPU上运行。其中`tabpfn_joint`要在CPU上运行TabPFN，每个状态约1.9 s，且结果与CUDA参考不完全一致（12个状态中最大点差0.33，1个单元标签变化）。TabPFN的CPU与CUDA输出在数值上不等价，正式结果以CUDA为准。
- 计时时XGBoost使用n_jobs=1；常规运行沿用`configs/main.yaml`的4线程，两者输出相同。CNP固定在torch CPU上运行。
- 每个中位数和p95只来自12个状态×3次计时，属于描述性统计，不是稳定的尾部估计；GP的p95（197.1 ms）约为其中位数的2.6倍。
- 比值：`count_shrink`/`balanced`＝345.8/183.2≈1.89；`recognition`/`balanced`≈1.02；`balanced`/`idw`≈27.0（CUDA对CPU，仅作参考）。

## 结论边界

**评价数据量。** 60个冻结状态＝3个fold×2个场地×2个种子（17、29）×5个预算，覆盖4个不同验证场地，但不是60次独立评价。fold 0与fold 1在Lower Hutt、Tauranga上使用完全相同的已知集与评价集；Tauranga预算5的已知集对两个种子也相同。因此60个状态只含39个不同已知集、8个不同评价集（4场地×2种子）。全部成绩来自39根留出CPT（936个深度单元），每个场地每个种子只有4–8根评价剖面（Marshland 4根、Lower Hutt与Tauranga各6根、Chch Metro Sports 8根）。每个评价集的实测低阻力单元数差别也很大：Lower Hutt 8–10/144，Marshland 5或17/96，Chch Metro Sports 26–32/192，Tauranga 40–51/144。

**验证与最终测试。** 上表全部来自4个验证场地（Lower Hutt、Tauranga、Chch Metro Sports、Marshland），它们自v0.2起在每一轮模型选择中反复使用，截至v0.8在同一60状态上累计比较了203个固定配置，存在选择偏差，相对IDW的差距应视为乐观估计。最终测试（4个从未参与选择的场地：Chch Stadium、Wellington Airport、New Lock、Hollywood；盲测预测、哈希冻结、只揭盲一次）中，log-MAE前三与验证相同：count_shrink 0.5049、balanced 0.5052、tabpfn_joint 0.5061（验证→测试Spearman ρ＝0.912）；厚度最优为thickness 0.4383 m（较IDW −0.065 m，3胜1平0负），厚度排名ρ＝0.510。仅4个测试场地，结论为描述性；Wellington Airport的评价集没有低阻力单元。协议见[docs/FINAL_TEST_PROTOCOL.md](docs/FINAL_TEST_PROTOCOL.md)，结果见[results/final_test/README.md](results/final_test/README.md)。目前没有补勘根数节约的证明。

**局部代价。**

- 均衡方案：Lower Hutt厚度MAE从IDW的0.133 m升至0.254 m；Marshland点召回从22.79%降至5.15%。
- 厚度方案：Lower Hutt厚度MAE 0.297 m、识别方案0.367 m，同样差于IDW；该场地最好的是XGBoost（0.124 m）。厚度方案在Tauranga的点误报为22.73%（IDW 13.07%）。
- 识别方案：Marshland点召回30.00%（IDW 22.79%，均衡5.15%）；该场地点误报3.12%，高于均衡的0.93%，但低于IDW的4.49%。它的四场地平均点误报9.14%仍高于IDW的7.28%。
- Tauranga的log-MAE：常数中位数为0.5739，低于IDW（0.6313）及除`tabpfn_joint`（0.5714）以外的全部入口（如均衡0.5803）。

**预算趋势。** 预算从5增加到13，所有入口的log-MAE都下降，只有CNP基本不变（0.8512→0.8515）。厚度MAE却多数上升（场地等权平均）：IDW 0.612→0.699 m（中间预算9时最低，为0.589 m）、GP 0.594→0.717 m、均衡0.458→0.564 m、simplex 0.564→0.655 m、`tabpfn_joint` 0.496→0.589 m。变化较小的是厚度方案（0.469→0.499 m）、识别方案（0.530→0.555 m）与`thickness_guarded`（0.488→0.491 m）。均衡方案的点召回从45.0%降至35.7%。AUBC是对预算的平均，会掩盖这些非单调趋势。

**区间。** GP是唯一输出预测区间的入口：名义90%区间的实测覆盖率为79.76%（各预算78.1%–81.2%），平均log宽度2.136，明显欠覆盖。

**CNP。** CNP在全部60个状态中从未预测出低于1 MPa的值，因此点召回与点误报均为0，厚度MAE与常数基线相同（0.966146 m）。它的log-MAE仅比常数中位数低3.9%。

来源：[保留结果](results/current/README.md)；[模型来源映射](configs/catalog.json)；[当前实现状态](docs/IMPLEMENTATION_STATUS.md)。
