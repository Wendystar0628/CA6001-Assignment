# 实现共识与模块接口

本文件将项目蓝图落实为当前开发接口；研究范围、阈值和预算继续以蓝图为准。未测真值只由数据仓库和评价器持有，不能传入预测器或选址器。

## 数据与配置

- 配置为普通 `dict`，`config['_root']` 是 Project 绝对路径；其余分组字段沿用蓝图 YAML。
- 以 `site_id / location_id / depth_index` 为稳定连接键；24 个深度中心，qc 为 MPa，自然对数。主阈值 1 MPa。
- `PreparedDataset(profiles, grid, sites, root)` 包含三个 DataFrame；`root` 为 Project 路径。`profiles` 包含全部剖面登记与 `is_representative / usable`，`grid` 只含可用代表位置。
- `dataset.locations(site_id=None)` 返回每个可用位置一行，带 site_id、profile_id、near_group_id、x_m/y_m、geometry_scale_m；`dataset.observed(ids)` 返回每位置24行含 log_qc/qc_mpa；`dataset.query(ids)` 只返回上述标识、坐标、深度、geometry_scale_m；`dataset.truth(ids)` 返回网格真值。
- `prepare_data(config) -> PreparedDataset`（data/normalize.py），写 profiles/grid/sites.parquet 与 preparation.json；`load_dataset(config)`（types.py）读取它们。
- `make_splits(dataset, config) -> dict`（data/splits.py），写 folds.json（根对象包含 folds 列表）、episodes.parquet、replay_splits.json（记录列表）。
- fold 字段：fold_id, train_groups, validation_groups, test_groups, train_sites, validation_sites, test_sites。
- episode 字段：fold_id, role（train/validation/test）, site_id, episode_id, support_location_ids, query_location_ids, seed。
- replay split 字段：site_id, seed, evaluation_ids, pool_ids, initial_ids, budgets, min_spacing_m；不能形成主预算的场地另记原因，不伪造空的可运行划分。

## 模型

- `build_features(observed: DataFrame, query_grid: DataFrame, *, leave_location_out=False) -> DataFrame`：固定16列；行顺序与 query_grid 相同。共享特征列名导出为 FEATURE_COLUMNS。
- 模型类 `IDWPredictor`, `GPPredictor`, `XGBPredictor`, `TabPFNPredictor`；构造接受 `config=None` 和各自可选 artifact 参数。
- 所有模型 `condition(observed: DataFrame) -> None`；`predict(query_grid: DataFrame) -> PredictionBundle`。
- `PredictionBundle(frame: DataFrame, model_id: str)`；frame 带查询标识与坐标，pred_log_qc / pred_qc_mpa，概率模型再给 q05_log/q10_log/q25_log/q50_log/q75_log/q90_log/q95_log、p_low_0_5/p_low_1/p_low_2。点模型不要伪造概率。
- model_id 固定 idw / gp / xgb / tabpfn / tabpfn_finetuned。
- GP 源趋势按训练场地等权生成，GPPredictor(config, trend=深度序列)；未给 trend 只允许 pilot 明确使用零趋势，正式运行需提供训练趋势。
- XGB 源训练入口 `fit_xgb(dataset, episodes, config, output_dir) -> Path`；返回持久化模型路径。`XGBPredictor(config, artifact=path)`。
- TabPFN `TabPFNPredictor(config, artifact=None)`；artifact=None 为原始v2，指定路径为微调模型。`fit_tabpfn(dataset, train_episodes, validation_episodes, config, output_dir) -> dict`，返回 checkpoint_path 与训练摘要。只在单独GPU进程执行。

## 回放、评价与输出

- `select_batch(observed_coordinates, candidate_coordinates, predictions, config, *, policy, min_spacing_m, batch_size=2) -> tuple[list[str], DataFrame]`；predictions 为 PredictionBundle 或 None。评分表带 location_id、base_score、threshold_bonus、score、selected、selection_order、spacing_fallback。第二点前不能读取第一点真值。
- `evaluate(predictions, evaluation_truth, config) -> dict`，主阈值字段沿用蓝图；缺失分类指标为 None/NaN，并有原因。
- `run_replay(dataset, split, predictor, config, *, policy, fold_id=0) -> dict[str, DataFrame]`，返回 predictions、metrics、recommendations、acquisition_scores、states。调用者负责落盘，公共键由回放器加入。
- 公共键：fold_id, site_id, seed, model_id, policy_id, budget_n；run_id 由调用者添加。predictions 包含 evaluation 真值以供离线出图（仅在预测之后连接），保留 role='evaluation'/'candidate'。states 每轮含 observed_ids/remaining_ids/evaluation_ids。
- `freeze_quality_targets(metrics, config) -> dict`；`summarize_results(metrics, quality_targets, config) -> dict[str, DataFrame]`（economics.py），只使用验证结果确定门槛，统计保留未达标与缺失原因。
- `export_figures(run_dir, dataset, config, *, preset='both', figure=None) -> list[Path]`（figures.py）：读 predictions.parquet / metrics.parquet / recommendations.parquet / acquisition_scores.parquet / states.parquet 及 summary/，不训练。报告/幻灯片版及帧复用同一真实结果。

## 并行责任与实施顺序

主代理维护 types.py、config.py、包配置、CLI、pipeline、README与集成。数据代理维护 data/、configs/sites.yaml 和数据测试；基线代理维护 features.py、models/base.py/idw.py/gp.py/xgb.py/distribution.py；TabPFN代理维护 models/tabpfn_adapter.py/tabpfn_train.py；回放代理维护 acquisition.py/replay.py/evaluate.py/economics.py；图表代理维护 plot_style.py/figures.py/configs/presentation.yaml。各人仅修改所属文件；改接口先通知主代理。正式结果不以占位数值补齐；运行不足明确记录。
