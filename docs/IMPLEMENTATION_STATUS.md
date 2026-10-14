# 当前项目实现状态

更新时间：2026-10-14；版本v0.9.1。v0.9.1的代码、结果与文档改动均在工作区，尚未提交Git；上一个标签为`v0.9-curated`（`1044802`）。变更摘要见[`CHANGES_v0.9.1.md`](../results/current/CHANGES_v0.9.1.md)。

## 当前模型

| 类别 | 入口 | 方案 | log-MAE ↓ | 厚度MAE(m) ↓ | 点召回 | 点误报 | risk_source |
|---|---|---|---:|---:|---:|---:|---|
| 基线（弱） | `constant` | 常数中位数 | 0.889189 | 0.966146 | 0.00% | 0.00% | platt_on_point |
| 基线（强） | `idw` | 3NN IDW | 0.638586 | 0.630371 | 37.38% | 7.28% | platt_on_point |
| 基线（强） | `gp` | 空间GP | 0.650156 | 0.667155 | 33.43% | 7.37% | native_gp |
| 基线（强） | `xgb` | 同轨迹XGBoost残差 | 0.634278 | 0.648112 | 34.26% | 6.76% | platt_on_point |
| TabPFN组合 | `tabpfn_joint` | TabPFN残差与三角网各半作底座＋整曲线决策 | 0.598844 | 0.525228 | 37.68% | 8.19% | shared_empirical_head |
| TabPFN组合 | `tabpfn_graph` | TabPFN＋图Huber低值并集 | 0.616500 | 0.618978 | 42.47% | 8.60% | platt_on_point |
| 主方案 | `balanced` | 均衡：结构融合＋整曲线决策 | 0.592435 | 0.536947 | 38.52% | 8.31% | shared_empirical_head |
| 主方案 | `thickness` | 厚度：Haar计数决策（最近剖面细节） | 0.614517 | 0.472493 | 39.38% | 10.05% | shared_empirical_head |
| 主方案 | `recognition` | 识别：可靠性定位＋60%计数 | 0.613079 | 0.526855 | 48.77% | 9.14% | shared_empirical_head |
| 讨论方案 | `simplex` | 三角网单独插值 | 0.607248 | 0.585286 | 34.00% | 7.04% | platt_on_point |
| 讨论方案 | `count_shrink` | 支持集计数收缩 | 0.591791 | 0.546712 | 38.20% | 7.98% | shared_empirical_head |
| 讨论方案 | `thickness_guarded` | Haar厚度（点误报护栏内最优） | 0.611418 | 0.487630 | 38.36% | 8.99% | shared_empirical_head |
| 讨论方案 | `cnp` | 条件神经过程（负结果） | 0.854427 | 0.966146 | 0.00% | 0.00% | platt_on_point |

分组：基线4个（1弱3强）、TabPFN组合2个、主方案3个、讨论方案4个，名单唯一来源为`configs/catalog.json`。指标为4个验证场地、预算5–13的AUBC，定义及风险列见[当前方案对比](../CA6001_CPT_当前方案对比.md)。`thickness`的点误报10.05%超出v0.7晋级规则的护栏（≤IDW+2pp＝9.28%），满足护栏的厚度选项是`thickness_guarded`。选择规则见[方案选择依据](MODEL_SELECTION.md)。

## 已完成与已验证

- 13个入口都支持实际的condition/predict推理；`make_model(..., device=)`可显式指定TabPFN设备，`RetainedModel.initialize()`可单独加载权重。
- `artifacts/manifest.json`登记31个文件：每折9个模型文件×3折，加上4个评价文件（冻结状态、fold信息、参考预测及其v0.9.0备份）。`status`核对文件存在与字节数，并检查`data_processed_v2/`。
- `verify`：13个入口×60个状态共780项全部通过，低阻力标签变化为0。最大点差：CNP 4.77e-7（浮点运算），GP 8.27e-11，其余≤3.2e-15。p_low_1与GP的6个分位数列都纳入检查。参考预测是本地存档（未入Git），新clone无法直接复核。`thickness`与`tabpfn_joint`的参考预测由当前代码生成，它们的独立证据是下面的`port-check`。
- `port-check`：13个入口的逐状态指标与`v0.8-engineering`标签归档的逐状态指标对照，26/26组等价（计数差为0，实数指标最大差1.78e-15）。
- `evaluate`：从参考预测重算`results/current`的指标、曲线、场地指标与排名，包括交叉风险校准和`risk_source`列；未变化的入口与v0.9.0文件相比最大差≤8.9e-16。
- `benchmark`：全部13个入口及6条`--point-only`路径按统一协议测量运行时间，逐行记录设备，计时后与参考预测对照；只有在CPU上运行TabPFN的`tabpfn_joint`仅点预测路径与参考不一致。结果见[当前方案对比](../CA6001_CPT_当前方案对比.md#运行时间v091统一测量)。
- `pytest -q`：46项测试通过（含最终测试盲跑不读真值、哈希校验与只揭盲一次的测试）。
- 每个入口都可通过`source_model_id`追溯到原实验身份。

## 当前结构

```text
configs/catalog.json             13个入口的唯一名单（含risk_source、device、point_only_supported）
configs/main.yaml, sites.yaml    数据与运行配置（main.yaml中部分键仅为旧流程保留，见文件注释）
src/cpt_budget/runtime.py        统一模型工厂与共享风险头
src/cpt_budget/components/       共享计算组件
src/cpt_budget/models/           基线、TabPFN残差组件及CNP
src/cpt_budget/experiment.py     冻结评价协议（60个状态）
src/cpt_budget/report.py         evaluate与port-check
src/cpt_budget/benchmark.py      统一运行时间测量
src/cpt_budget/data/             数据恢复和分组
artifacts/                       本地模型与评价工件（不入Git）
results/current/                 当前13个入口的证据
runs/current/                    本地运行输出
notebooks/                       最终报告底稿（本地，尚未提交）
```

## 仍未完成的事项

- **最终报告**：课程最终报告以Jupyter Notebook提交。当前底稿为`notebooks/CA6001_report_figures.ipynb`，仍是草稿，尚未提交Git；项目尚未定稿。
- **幻灯片与录制**：最多12张内容幻灯片，录制不超过15分钟，均未完成。
- **代码与提交**：v0.9.1改动尚未提交和打标签。版本号已为0.9.1。
- **最终测试（已完成）与证据边界**：验证成绩来自用于模型选择的4个验证场地，存在选择偏差；60个状态只含39个不同已知集和8个不同评价集。最终测试（4个从未参与选择的场地：Chch Stadium、Wellington Airport、New Lock、Hollywood；盲测预测、哈希冻结、只揭盲一次）中，log-MAE前三与验证相同：count_shrink 0.5049、balanced 0.5052、tabpfn_joint 0.5061（验证→测试Spearman ρ＝0.912）；厚度最优为thickness 0.4383 m（较IDW −0.065 m，3胜1平0负），厚度排名ρ＝0.510。仅4个测试场地，结论为描述性；Wellington Airport的评价集没有低阻力单元。协议见[docs/FINAL_TEST_PROTOCOL.md](FINAL_TEST_PROTOCOL.md)，结果见[results/final_test/README.md](../results/final_test/README.md)。目前没有补勘根数节约的证明。
- **未接入的功能**：选址、回放与成本函数（`acquisition.py`、`replay.py`、`economics.py`）只作为库函数保留，CLI未暴露；`predict`不输出风险标签，风险标签只在`evaluate`中由交叉校准生成，校准器未保存为工件。完整训练与探索流程见`v0.8-engineering`标签。

## 入口

[运行说明](RUN_GUIDE.md) · [方案选择依据](MODEL_SELECTION.md) · [当前性能对比](../CA6001_CPT_当前方案对比.md) · [结果证据](../results/current/README.md)
