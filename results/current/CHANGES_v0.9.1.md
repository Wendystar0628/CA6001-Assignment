# CHANGES v0.9.1（结果与代码变更摘要，供文档代理使用）

生成日期 2026-10-14。全部数字来自本目录文件，可用 `python -m cpt_budget evaluate|port-check|verify|benchmark` 重算。尚未提交 Git。

## 1. 目录变更（12 → 13 方案）

| 变更 | v0.9.0 | v0.9.1 |
|---|---|---|
| `thickness`（selected） | `v8_haar_fast`（3NN加权中位细节） | `v7_hcount_simplex_nearest_a100_gp` 的点路径（最近剖面细节）＋共享经验风险头 |
| 新增 `thickness_guarded`（discussion） | — | `v8_haar_fast`，即原 thickness；v0.7 点FPR护栏（≤IDW+2pp=9.28%）内厚度最优 |
| `tabpfn_fpca` → `tabpfn_joint`（tabpfn） | `v6_fpca_residual_sign_guard` | `v7_pair_joint_simplex_empirical_graph`（v0.7 晋级主线；底座 = 0.5·TabPFN残差 + 0.5·simplex，TabPFN 只提供一半幅值底座） |
| 其余10个 | 不变 | 不变（数值复现 ≤1e-12） |

分组：baseline 4 / tabpfn 2 / selected 3 / discussion 4。目录新增字段：`study_version`、`risk_source`、`device`、`point_only_supported`、`note`。

注意：thickness 归档 ID 的后缀 `_gp` 表示 v0.7 时配 GP 风险头（风险召回/FPR 62.20%/24.87%，Brier 0.125157）；v0.9.1 配共享经验风险头。两种配对的点指标完全相同。thickness 的点FPR 10.05% 超出 v0.7 晋级护栏。

## 2. 最终目录指标（4场地预算AUBC，`all_model_ranking.csv`）

| id | source_model_id | log-MAE | 厚度MAE (m) | 召回 | 点FPR | 风险召回 | 风险FPR | 风险Brier | risk_source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| constant | target_constant_median | 0.889189 | 0.966146 | 0.00% | 0.00% | 25.00% | 25.00% | 0.158444 | platt_on_point |
| idw | idw | 0.638586 | 0.630371 | 37.38% | 7.28% | 69.55% | 38.35% | 0.130449 | platt_on_point |
| gp | gp | 0.650156 | 0.667155 | 33.43% | 7.37% | 62.20% | 24.87% | 0.125157 | native_gp |
| xgb | xgb_residual_matched | 0.634278 | 0.648112 | 34.26% | 6.76% | 68.96% | 39.30% | 0.130882 | platt_on_point |
| tabpfn_joint | v7_pair_joint_simplex_empirical_graph | 0.598844 | 0.525228 | 37.68% | 8.19% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| tabpfn_graph | v6_graph_residual_low_union | 0.616500 | 0.618978 | 42.47% | 8.60% | 74.40% | 29.18% | 0.121324 | platt_on_point |
| balanced | v8_balanced_fast | 0.592435 | 0.536947 | 38.52% | 8.31% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| thickness | v7_hcount_simplex_nearest_a100_gp | 0.614517 | 0.472493 | 39.38% | 10.05% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| recognition | v8_recognition_recall60 | 0.613079 | 0.526855 | 48.77% | 9.14% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| simplex | v7_simplex_linear | 0.607248 | 0.585286 | 34.00% | 7.04% | 66.65% | 25.03% | 0.121464 | platt_on_point |
| count_shrink | v8_balanced_count_shrink | 0.591791 | 0.546712 | 38.20% | 7.98% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| thickness_guarded | v8_haar_fast | 0.611418 | 0.487630 | 38.36% | 8.99% | 68.21% | 23.90% | 0.121124 | shared_empirical_head |
| cnp | cnp_profile | 0.854427 | 0.966146 | 0.00% | 0.00% | 37.55% | 38.73% | 0.192664 | platt_on_point |

GP 另有 coverage_90 = 0.797607，width_90_log = 2.135718（唯一的原生区间）。

`risk_source` 的含义：`shared_empirical_head` = 共享经验风险头 v6_empirical_graph_union_local 的 p_low_1。它由 tabpfn_graph 点路径加经验误差分布得到，与方案自身点路径无关，因此 6 个方案的风险指标完全相同。`native_gp` = GP 预测分布。`platt_on_point` = 无原生概率，评价时对 −ŷ 做 Platt。全部风险概率都按“模型 × 外层fold”做留一调查组 Platt 交叉校准，阈值 0.2（漏判:误报 = 4:1）。

## 3. 新旧数字（变化的槽位）

| 槽位 | 旧（v0.9.0） | 新（v0.9.1） | 差值 |
|---|---|---|---|
| thickness | v8_haar_fast: E 0.611418, H 0.487630, R 38.36%, FPR 8.99%, 风险 68.21%/23.90%/0.121124 | v7_hcount_simplex_nearest_a100_gp: E 0.614517, H 0.472493, R 39.38%, FPR 10.05%, 风险 68.21%/23.90%/0.121124 | ΔE +0.003099, ΔH -0.015137, ΔR +1.02pp, ΔFPR +1.06pp |
| tabpfn_fpca → tabpfn_joint | v6_fpca_residual_sign_guard: E 0.613446, H 0.630371, R 37.38%, FPR 7.28%, 风险 72.25%/28.60%/0.121448 | v7_pair_joint_simplex_empirical_graph: E 0.598844, H 0.525228, R 37.68%, FPR 8.19%, 风险 68.21%/23.90%/0.121124 | ΔE -0.014602, ΔH -0.105143, ΔR +0.31pp, ΔFPR +0.91pp |
| thickness_guarded（新增） | 原 thickness 行 | 与原 thickness 相同（E 0.611418, H 0.487630） | 0 |

从目录删除：`tabpfn_fpca`（v6_fpca_residual_sign_guard）。它在 E/H/pBER/rBER 上被 v6_coupled_residual_stack_potts 支配，且点标签按构造等于 IDW。其组件 `FunctionalProfilePredictor` 仍留在 components/functional.py，但目录已不再使用。

## 4. 等价性证据

- `port_equivalence.csv`（`port-check`）：把 13 方案 × 60 状态的逐状态指标，与 Git 标签 v0.8-engineering 中 `results/v8-engineering-20261014/comparison/metrics.csv` 对照。26/26 组等价；tp/fp/fn/tn 与 risk_tp/fp/fn/tn 的差为 0，实数指标最大差 1.78e-15。thickness 的点指标对 v7_hcount_simplex_nearest_a100_gp，概率/风险指标对 v6_empirical_graph_union_local；tabpfn_joint 全部对 v7_pair_joint_simplex_empirical_graph。
- 参考预测：`artifacts/evaluation/reference_predictions.parquet` 重建为 13 × 8640 = 112,320 行。thickness 与 tabpfn_joint 为实时推理，通过等价门后冻结。GP 补充 q10/q25/q75/q90（实时GP；与归档 pred/p_low_1/q05/q95 的差 ≤8.8e-11）。原文件备份为 `reference_predictions.v0.9.0.parquet`，已在 manifest 登记（含 sha256）。
- `evaluate` 复现：10 个不变方案加 thickness_guarded（= 原 thickness），在 metrics/curves/unique_site/ranking 中与 v0.9.0 文件的最大差 ≤8.9e-16。例外是 GP 的 width_50/80_log，差 ≤5.9e-12，来自实时补充的分位数；coverage 完全相同。
- `verify`（runs/current/verification，副本为 migration_*）：780 次检查（13 × 60）全部通过，0 个标签变化。最大点差 CNP 4.77e-07、GP 8.27e-11，其余 ≤3.2e-15。p_low_1 与 GP 的 6 个分位数列都会检查，列消失或多出即失败。注意：thickness 与 tabpfn_joint 的参考由同一代码生成，它们的独立证据是 port_equivalence。
- `pytest -q`：33 passed。

## 5. 统一运行时间（`runtime_summary.csv`，v0.9.1 实测）

协议：12 个状态（3 fold × 2 场地 × seed 17 × 预算 5/13）。每个状态 1 次预热 + 3 次计时。进程固定到逻辑核 2（mask 0x4），1 线程。CUDA 路径在计时边界同步；权重加载单独计时；CPU/GPU 频率未锁定。运行前 GPU 利用率约 2%，没有其他计算任务。硬件 Intel(R) Core(TM) i7-14700KF + RTX 4080，整轮 188 s。

| id | path | device | 中位 ms | p95 ms | 条件化中位 | 预测中位 | 初始化中位 ms | 与参考一致 |
|---|---|---|---:|---:|---:|---:|---:|---|
| constant | full | cpu | 2.6 | 2.7 | 0.8 | 1.8 | 2.9 | 是 |
| idw | full | cpu | 6.8 | 7.2 | 3.1 | 3.7 | 2.9 | 是 |
| gp | full | cpu | 74.5 | 197.1 | 69.3 | 5.1 | 3.4 | 是 |
| xgb | full | cpu | 83.4 | 110.1 | 1.4 | 82.0 | 4.4 | 是 |
| tabpfn_joint | full | cuda | 169.1 | 202.6 | 35.3 | 134.0 | 75.5 | 是 |
| tabpfn_graph | full | cuda | 136.4 | 162.1 | 18.5 | 117.6 | 67.1 | 是 |
| balanced | full | cuda | 183.2 | 202.4 | 47.9 | 130.5 | 70.2 | 是 |
| thickness | full | cuda | 151.2 | 176.7 | 28.7 | 122.0 | 66.9 | 是 |
| recognition | full | cuda | 186.2 | 197.9 | 50.2 | 128.1 | 64.1 | 是 |
| simplex | full | cpu | 8.9 | 9.7 | 5.3 | 3.6 | 2.9 | 是 |
| count_shrink | full | cuda | 345.8 | 432.3 | 50.2 | 298.7 | 66.2 | 是 |
| thickness_guarded | full | cuda | 165.8 | 193.4 | 33.2 | 132.6 | 72.9 | 是 |
| cnp | full | cpu | 15.5 | 19.1 | 8.0 | 7.1 | 5.0 | 是 |
| tabpfn_joint | point_only | cpu | 1857.2 | 2751.0 | 28.8 | 1829.7 | 41.7 | 否（最大点差 0.330，1 个标签变化） |
| balanced | point_only | cpu | 37.1 | 44.5 | 27.1 | 10.0 | 4.0 | 是 |
| thickness | point_only | cpu | 9.9 | 10.9 | 6.0 | 4.0 | 3.3 | 是 |
| recognition | point_only | cpu | 39.6 | 48.5 | 29.8 | 9.9 | 4.0 | 是 |
| count_shrink | point_only | cpu | 197.4 | 268.8 | 26.0 | 171.3 | 3.7 | 是 |
| thickness_guarded | point_only | cpu | 10.3 | 11.5 | 6.0 | 4.2 | 2.9 | 是 |

说明：
- full = 目录完整流程（含原生或共享风险头）。`device=cuda` 指 TabPFN 组件在 CUDA 上（无 CUDA 直接报错），其余数值计算仍在固定的 CPU 核上。
- point_only = `--point-only`，全部在 CPU。tabpfn_joint 的 point_only 在 CPU 上运行 TabPFN，约 1.9 s/状态，且与 CUDA 参考不完全一致（最大点差 0.33，12 个状态中 1 个格点标签变化）。TabPFN 的 CPU 与 CUDA 结果数值上不等价。
- XGBoost 计时时 n_jobs=1；常规运行仍为 4，输出相同。
- 共享风险头每次都重算 tabpfn_graph，主方案完整延迟主要来自它；tabpfn_joint 与风险头共用一次 TabPFN 前向。

对照 v0.8 固定核批次（同一协议，旧名称）：constant 2.1→2.6，idw 6.4→6.8，gp 73.5→74.5，balanced 181.6→183.2，thickness_guarded（haar_full）152.6→165.8，recognition 185.9→186.2，count_shrink 317.4→345.8 ms。首次测量的有 xgb、tabpfn_graph、tabpfn_joint、thickness（nearest）、simplex、cnp。

## 6. 代码变更

- `components/haar.py`：`_FastCore(detail='median'|'nearest')`。nearest = 几何最近揭示剖面的 21 个细节系数。
- `runtime.py`：
  - 构建 13 个方案；`thickness` 用 nearest，`thickness_guarded` 用 median。
  - 新增 `TabPFNJoint`；完整路径与共享风险头共用一次 TabPFN 残差。`TabCombination` 只保留 graph。
  - simplex 改用缓存三角网 `_SimplexGeometry`（verify 最大差 8.9e-16，0 个标签变化）。
  - `--point-only` 只对有独立风险头的 6 个方案有效，其余方案直接报错。
  - `make_model(..., device=)` 显式指定 TabPFN 设备；新增 `RetainedModel.initialize()`。
- `models/tabpfn_adapter.py`：device=auto 回退到 CPU 时发出警告；device=cuda 而没有 CUDA 时报错。
- `report.py` + CLI `evaluate`：从参考预测（或 `--source live`）重算 4 个结果文件，含交叉风险校准与 `risk_source` 列，并写 `evaluation_provenance.json`。CLI `port-check`：对照 v0.8 归档的逐状态指标。
- `benchmark.py` + CLI `benchmark`：移植 v0.8 协议，由目录驱动。逐行记录 device 与 device_policy，计时后与参考对照，`--publish results/current`。
- `cli.py`：`verify` 在原生列（p_low_1、GP q05/q10/q25/q75/q90/q95）消失或多出时失败，检查 source_model_id 与 risk_source 一致，并新增 `--output`。`status` 核对字节数与 data_processed_v2。
- `components/stacking.py` 改用 `model_asset`；functional、residual_point、constant、xgb_residual 补上 `import pandas as pd`。
- `tests/test_current_models.py` 新增覆盖：13 方案与分组、移植身份、point-only 拒绝、缓存 simplex 等价、nearest 细节、verify 列检查、evaluate 复现排名。共 33 个测试通过。
- `.gitignore`：改为 `artifacts/evaluation/reference_predictions*.parquet`，备份同样不进 Git。

## 7. 文档代理需要更新的说法

- “12方案”改为“13方案”；discussion 组为 4 个。
- thickness 改为历史最优厚度 0.472493 m（点FPR 10.05%，超出 v0.7 护栏）。原来 0.487630 m 的方案现在是 thickness_guarded。
- TabPFN 槽位由 tabpfn_fpca 改为 tabpfn_joint（E 0.598844 / H 0.525228）。需说明 TabPFN 只提供一半幅值底座。
- 共享风险头覆盖 6 个方案（不是“三者”），这些方案的风险数字相同。simplex 等方案的风险数字来自评价时的 Platt 校准，不是原生风险头。
- 运行时间改为引用 v0.9.1 统一测量：覆盖全部 13 个方案并带 device 列，不再使用 v0.8 名称。
- 重算命令：`python -m cpt_budget evaluate`、`port-check`、`verify`、`benchmark --cpu-core 2 --publish results/current`。

## 8. 机器可读摘要

```json
{
 "version": "0.9.1",
 "catalog_entries": 13,
 "groups": {
  "baseline": 4,
  "tabpfn": 2,
  "selected": 3,
  "discussion": 4
 },
 "ranking": {
  "constant": {
   "source_model_id": "target_constant_median",
   "risk_source": "platt_on_point",
   "log_mae": 0.8891893571450752,
   "h_point_mae_m": 0.9661458333333334,
   "recall": 0.0,
   "false_positive_rate": 0.0,
   "risk_recall": 0.25,
   "risk_fpr": 0.25,
   "risk_brier": 0.1584442000269693
  },
  "idw": {
   "source_model_id": "idw",
   "risk_source": "platt_on_point",
   "log_mae": 0.6385857548912574,
   "h_point_mae_m": 0.63037109375,
   "recall": 0.3737671450791855,
   "false_positive_rate": 0.0727663395372843,
   "risk_recall": 0.6954917574000754,
   "risk_fpr": 0.383525087543083,
   "risk_brier": 0.1304488269298607
  },
  "gp": {
   "source_model_id": "gp",
   "risk_source": "native_gp",
   "log_mae": 0.6501557847425754,
   "h_point_mae_m": 0.6671549479166666,
   "recall": 0.3343481924019608,
   "false_positive_rate": 0.0737446482250769,
   "risk_recall": 0.6220454197303922,
   "risk_fpr": 0.2487103800646363,
   "risk_brier": 0.1251570489639994
  },
  "xgb": {
   "source_model_id": "xgb_residual_matched",
   "risk_source": "platt_on_point",
   "log_mae": 0.6342783716585573,
   "h_point_mae_m": 0.6481119791666666,
   "recall": 0.3425759155825791,
   "false_positive_rate": 0.06757220584672,
   "risk_recall": 0.6895649215214932,
   "risk_fpr": 0.3930410609618615,
   "risk_brier": 0.1308824341774372
  },
  "tabpfn_joint": {
   "source_model_id": "v7_pair_joint_simplex_empirical_graph",
   "risk_source": "shared_empirical_head",
   "log_mae": 0.5988439109937171,
   "h_point_mae_m": 0.5252278645833334,
   "recall": 0.3768270409125188,
   "false_positive_rate": 0.0818771447648194,
   "risk_recall": 0.6821433411104827,
   "risk_fpr": 0.2390387758533572,
   "risk_brier": 0.1211237727081628
  },
  "tabpfn_graph": {
   "source_model_id": "v6_graph_residual_low_union",
   "risk_source": "platt_on_point",
   "log_mae": 0.6164997698625062,
   "h_point_mae_m": 0.6189778645833333,
   "recall": 0.424671091393288,
   "false_positive_rate": 0.0860223080069651,
   "risk_recall": 0.7440130679675716,
   "risk_fpr": 0.2917828241738992,
   "risk_brier": 0.1213235258650939
  },
  "balanced": {
   "source_model_id": "v8_balanced_fast",
   "risk_source": "shared_empirical_head",
   "log_mae": 0.5924346804689666,
   "h_point_mae_m": 0.5369466145833333,
   "recall": 0.3852390295060332,
   "false_positive_rate": 0.0830801202637311,
   "risk_recall": 0.6821433411104827,
   "risk_fpr": 0.2390387758533572,
   "risk_brier": 0.1211237727081628
  },
  "thickness": {
   "source_model_id": "v7_hcount_simplex_nearest_a100_gp",
   "risk_source": "shared_empirical_head",
   "log_mae": 0.6145172485211584,
   "h_point_mae_m": 0.4724934895833333,
   "recall": 0.3937769548925339,
   "false_positive_rate": 0.1004803870784883,
   "risk_recall": 0.6821433411104827,
   "risk_fpr": 0.2390387758533572,
   "risk_brier": 0.1211237727081628
  },
  "recognition": {
   "source_model_id": "v8_recognition_recall60",
   "risk_source": "shared_empirical_head",
   "log_mae": 0.6130791361045966,
   "h_point_mae_m": 0.52685546875,
   "recall": 0.4876551011029412,
   "false_positive_rate": 0.0913638488011575,
   "risk_recall": 0.6821433411104827,
   "risk_fpr": 0.2390387758533572,
   "risk_brier": 0.1211237727081628
  },
  "simplex": {
   "source_model_id": "v7_simplex_linear",
   "risk_source": "platt_on_point",
   "log_mae": 0.6072477540431518,
   "h_point_mae_m": 0.5852864583333334,
   "recall": 0.340034790959653,
   "false_positive_rate": 0.0704037084053406,
   "risk_recall": 0.666517604638009,
   "risk_fpr": 0.2502689615123116,
   "risk_brier": 0.1214642294057804
  },
  "count_shrink": {
   "source_model_id": "v8_balanced_count_shrink",
   "risk_source": "shared_empirical_head",
   "log_mae": 0.5917905573557339,
   "h_point_mae_m": 0.5467122395833333,
   "recall": 0.3820106287707391,
   "false_positive_rate": 0.0798326839210179,
   "risk_recall": 0.6821433411104827,
   "risk_fpr": 0.2390387758533572,
   "risk_brier": 0.1211237727081628
  },
  "thickness_guarded": {
   "source_model_id": "v8_haar_fast",
   "risk_source": "shared_empirical_head",
   "log_mae": 0.6114177806567319,
   "h_point_mae_m": 0.4876302083333333,
   "recall": 0.3836128982843137,
   "false_positive_rate": 0.0898519323929229,
   "risk_recall": 0.6821433411104827,
   "risk_fpr": 0.2390387758533572,
   "risk_brier": 0.1211237727081628
  },
  "cnp": {
   "source_model_id": "cnp_profile",
   "risk_source": "platt_on_point",
   "log_mae": 0.8544270889968748,
   "h_point_mae_m": 0.9661458333333334,
   "recall": 0.0,
   "false_positive_rate": 0.0,
   "risk_recall": 0.37548828125,
   "risk_fpr": 0.3873449508363472,
   "risk_brier": 0.1926638193136644
  }
 },
 "changed": {
  "thickness": {
   "old_source": "v8_haar_fast",
   "new_source": "v7_hcount_simplex_nearest_a100_gp"
  },
  "thickness_guarded": {
   "new_entry": true,
   "source": "v8_haar_fast"
  },
  "tabpfn_joint": {
   "replaces": "tabpfn_fpca",
   "old_source": "v6_fpca_residual_sign_guard",
   "new_source": "v7_pair_joint_simplex_empirical_graph"
  }
 },
 "verification": {
  "checks": 780,
  "passed": true,
  "label_changes": 0
 },
 "port_equivalence": {
  "groups": 26,
  "equivalent": 26
 },
 "pytest": "33 passed",
 "runtime_ms": {
  "constant/full": {
   "device": "cpu",
   "median": 2.594,
   "p95": 2.711,
   "matches_reference": true
  },
  "idw/full": {
   "device": "cpu",
   "median": 6.783,
   "p95": 7.213,
   "matches_reference": true
  },
  "gp/full": {
   "device": "cpu",
   "median": 74.479,
   "p95": 197.066,
   "matches_reference": true
  },
  "xgb/full": {
   "device": "cpu",
   "median": 83.41,
   "p95": 110.057,
   "matches_reference": true
  },
  "tabpfn_joint/full": {
   "device": "cuda",
   "median": 169.102,
   "p95": 202.649,
   "matches_reference": true
  },
  "tabpfn_graph/full": {
   "device": "cuda",
   "median": 136.439,
   "p95": 162.147,
   "matches_reference": true
  },
  "balanced/full": {
   "device": "cuda",
   "median": 183.247,
   "p95": 202.37,
   "matches_reference": true
  },
  "thickness/full": {
   "device": "cuda",
   "median": 151.216,
   "p95": 176.703,
   "matches_reference": true
  },
  "recognition/full": {
   "device": "cuda",
   "median": 186.151,
   "p95": 197.919,
   "matches_reference": true
  },
  "simplex/full": {
   "device": "cpu",
   "median": 8.922,
   "p95": 9.657,
   "matches_reference": true
  },
  "count_shrink/full": {
   "device": "cuda",
   "median": 345.765,
   "p95": 432.347,
   "matches_reference": true
  },
  "thickness_guarded/full": {
   "device": "cuda",
   "median": 165.838,
   "p95": 193.392,
   "matches_reference": true
  },
  "cnp/full": {
   "device": "cpu",
   "median": 15.541,
   "p95": 19.05,
   "matches_reference": true
  },
  "tabpfn_joint/point_only": {
   "device": "cpu",
   "median": 1857.171,
   "p95": 2750.971,
   "matches_reference": false
  },
  "balanced/point_only": {
   "device": "cpu",
   "median": 37.122,
   "p95": 44.488,
   "matches_reference": true
  },
  "thickness/point_only": {
   "device": "cpu",
   "median": 9.939,
   "p95": 10.909,
   "matches_reference": true
  },
  "recognition/point_only": {
   "device": "cpu",
   "median": 39.572,
   "p95": 48.536,
   "matches_reference": true
  },
  "count_shrink/point_only": {
   "device": "cpu",
   "median": 197.449,
   "p95": 268.845,
   "matches_reference": true
  },
  "thickness_guarded/point_only": {
   "device": "cpu",
   "median": 10.274,
   "p95": 11.534,
   "matches_reference": true
  }
 }
}
```
