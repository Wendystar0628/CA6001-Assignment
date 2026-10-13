# CPT v0.2 实现共识

本版落实结果诊断中的表示、风险目标与整根选址改进。v0.1的源码默认行为、配置、run及结果不覆盖。新增代码位于 `src/cpt_budget/v2/`，独立命令为 `python -m cpt_budget.v2`；采用独立 `configs/v2.yaml`、`data_processed_v2/`、`runs/v2/<run_id>`。当前外层结果已被查看，新版属于探索性第二轮，不声称全新独立确认性测试。

## 固定研究范围

保留2–8m、24单元、1MPa、三折勘察组、5/7/9/11/13根、整位置与批次揭示约束。完整划分仍固定5种子；开发与候选选择先用验证种子17/29，正式扩大须通过验证门槛。数据来源不变，4个全球分组未核实场地继续明确标注。

模型ID：`idw`、`gp`、`idw_residual`、`tabpfn_spatial`（新特征、原始v2权重）、`tabpfn_crps`（新特征、CRPS+MSE）、`tabpfn_task`（再加低值CDF Brier目标）。推断仍1 estimator，集成不作为本轮主要改进。风险模块为所有模型提供同样的校准流程，不把校准概率冒充点模型的原生分布。

主要点指标沿用v0.1；新增字段独立为 `risk_probability / risk_label`，以及 `risk_recall / risk_precision / risk_fpr / risk_brier / h_risk_expected_mae_m`。成本主要条件仍用log-MAE和公共H_point；新风险Recall/FPR作为明确的新版质量条件。原点Recall、H_point和原门槛结果继续可查，不靠替换指标宣称赢过v0.1。

## 共享接口

继续使用 `types.PreparedDataset`、`PredictionBundle`、原始观察/查询DataFrame与公共键。所有新增模块接受普通config dict和绝对Project根 `_root`。

### 表示与残差模块

`v2/features.py` 导出 `FEATURE_COLUMNS_V2`、`build_features_v2(observed, query_grid, *, leave_location_out=False)`。保留16列作为基础，加入能通过完整预处理的固定物理长度核统计（例如10/50m的已测qc加权均值）及相邻深度摘要；不只重复增加场地常量。仍严格整位置留一，不读取query真值。至少用真实数据确认100倍水平缩放在官方完整预处理之后产生可辨别输入差异。

`v2/residual.py`：`fit_residual(dataset, train_episodes, validation_episodes, config, output_dir)->dict`，返回artifact_path、alpha及选择证据；`ResidualPredictor(config, artifact=Path)` 具有condition/predict。学习源episode的IDW残差，alpha候选[0,0.25,0.5,1]，验证选择，包含纯IDW回退。验证门槛：H_point不超过IDW+0.05m，点Recall不低于IDW−0.02，FPR不超过IDW+0.02，满足者选择最小log-MAE。不得从零残差推导零观测方差；此模型输出点预测，风险由统一校准器提供。

### TabPFN模块

`v2/tabpfn.py`：`SpatialTabPFNPredictor(config, artifact=None, model_id='tabpfn_spatial')`，condition/predict沿用共享接口；使用新特征和原生CDF/七分位数，适配代码只复用官方或v1已验证路径。

`fit_spatial_tabpfn(dataset, train_episodes, validation_episodes, config, output_dir, *, variant='crps')->dict`，返回checkpoint_path、model_id、history_path、实际步数/曝光率/选择指标。variant='task'在CRPS+MSE上加入0.5×原输入标签空间1MPa事件的Brier损失；阈值必须正确经过当前support目标变换，不能误把ln(1)=0当作所有标准化空间中的阈值。保持原生分布，不另造独立假区间。

采用明确步数预算：最多450次更新/折/候选，按场地轮转并完整轮换query块；每个场地权重相同，有限批次<=512行，支持全部保留。每50步验证，patience=3，允许在覆盖全部固定query组合后按验证早停；原始权重的验证结果单列。选择score为场地宏平均 `log_mae + 0.5 * native_event_brier`，同时输出H_point/Recall/FPR；最终是否扩展由主流水线门槛判断。GPU只允许一个进程；作者先实现并做单步真实验证，正式三折由主代理调度。

### 风险与经济模块

`v2/risk.py`：`fit_risk_calibrator(predictions, *, group_column='investigation_group')->dict`，只接收显式split_role='validation'的evaluation行，含truth_qc_mpa；原生p_low_1用logit分数，点模型用−pred_log_qc。按场地等权拟合一维单调Platt校准（斜率非负），单类情况使用明确标记的常量先验。字典可JSON保存。

`apply_risk(frame, calibrator, *, threshold=0.2)->DataFrame` 添加risk_probability/risk_label，保留原概率和点值。0.2来自课程情景漏判/误报代价4:1，不是工程规范。

`crossfit_risk(predictions)->tuple[DataFrame,dict]`：按验证勘察组留一拟合与预测，返回无本组校准泄漏的validation预测，及在全部validation上拟合的最终calibrator。每model/fold由调用者单独传入。未揭示真值不能通过apply传回模型或策略。

`evaluate_risk(prediction_frame, truth, config)->dict`：新增风险指标，阴性/无预测阳性保留缺失口径；原评价仍由v1 evaluate单独计算。

`v2/quality.py`：`freeze_joint_targets(validation_metrics,config)->dict`。参考GP+maximin、预算13。按场地/种子已定义的权重，对同一个q依次取E/H/FPR的q分位与Recall的1−q分位，q候选[0.5,0.6,0.7,0.8,0.9,1.0]；选择使参考实例联合通过率至少0.5的最小q，保存实际通过率和每场地通过率。正/负类缺失条件仅在适用实例检查；不能用test。保留v1边际中位数目标供对照。新版成本表须明确target_version，不与旧目标下根数节约混算。

### 整根选址与回放

`v2/acquisition.py`：`select_gp_batch(proxy, observed_coordinates, candidate_grid, config, *, policy, min_spacing_m, probability_transform=None)->tuple[list[str],DataFrame]`，proxy为已condition的v1 GPPredictor，候选query无标签。策略ID固定 `gp_threshold / gp_alc / gp_sur`；maximin复用原函数。

ALC/SUR使用公开行动池未测位置作为参考几何，计算潜在函数联合后验，观测块加白噪声；整根24点。SUR风险为标准化全局潜在方差与低值Bayes风险的等权组合，代价FN=4/FP=1；使用固定随机种子的相关整根fantasy，先ALC筛8个候选，每候选16条fantasy；候选评分时不重新拟合核。概率变换可接受冻结的GP校准器。第二根需更新协方差/条件fantasy，不读取第一根真实观测，保留相同批次间距回退。记录profile_gain/risk_gain/total_score及第二根边际收益，区分第一点初始分数与第二点条件分数。

`v2/replay.py`：`run_replay_v2(dataset,split,predictor,config,*,policy,fold_id=0,proxy=None,calibrator=None,proxy_calibrator=None)->dict[str,DataFrame]`。返回与v1一样五表，新增selection_model_id、风险指标、target study标记。proxy可与predictor同对象(GP)，否则每轮用相同已知数据condition。阶段一仅maximin，阶段二固定选出的预测器，比较maximin/gp_threshold/gp_alc/gp_sur；双模型系统明确标出GP负责选址。

### 主流水线与图表

主代理实现 `v2/pipeline.py`、`__main__.py`、配置、结果存储、校准应用及门槛判定。所有新run metadata记录validation选择、参数、来源commit与exploratory_v2。

`v2/figures.py`：`export_v2(run_dir,dataset,config)->list[Path]`。重点是同场地同预算的预测与风险取舍、验证门槛通过情况、策略全预算曲线、原目标/新目标的区别。图中区分OOF validation与第二轮test，不能混入校准拟合内指标，也不能把新版门槛带来的通过率提升算成算法收益。优先复用现有绘图样式，报告与slides重排。

## 推进门槛与Git节点

1. N0：研究证据与本协议冻结，保存v0.1 tag，建立v0.2分支。
2. N1：新版特征/模型/校准/联合质量模块可用，关键语义检查及真实微调单步通过。
3. N2：GP整根ALC/SUR与两点批量回放接通，同一验证场地真实5→7运行与图形通过。
4. N3：三折源训练与两种子全预算validation完成，按OOF风险指标选候选。模型晋级须log-MAE曲线面积优于同一最强基线，并且H_point<=基线+0.05m、riskRecall>=基线−0.02、riskFPR<=基线+0.02；基线按validation log-MAE曲线面积在IDW/GP中选定。无候选通过时明确回退，并继续在固定基线上评估选址，不宣称模型提升。
5. N4：策略validation完毕；只有同向改善且风险护栏满足时才扩大五种子第二轮测试。否则交付可运行v0.2与验证负结果，不为凑成功再改门槛。保留全部比较、失败条件与后续未完成项。

分工：表示/残差、TabPFN适配、风险/质量、选址/回放、图表各独占相应v2文件；主代理负责CLI/集成/版本提交。不得修改其他人的文件或覆盖v0.1产物；接口变化先同步。
