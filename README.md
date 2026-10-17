# CA6001 · AI自适应CPT补勘与经济效益评价

本项目研究有限已测CPT条件下的2–8 m锥阻剖面重建、累计低锥阻厚度预测、自适应补勘与停止决策，并将实际回放工作量接入经济情景。当前阶段为`adaptive_v1`，在v0.9.1工程上扩展。

- [阶段前后对比](CA6001_项目阶段对比.html) · [自适应交互展示](CA6001_项目状态.html) · [当前实现状态](docs/IMPLEMENTATION_STATUS.md)
- [课程技术报告（Word）](report/CA6001_Adaptive_CPT_Report.docx) · [PDF预览](report/CA6001_Adaptive_CPT_Report.pdf) · [实验与绘图Notebook](notebooks/CA6001_report_figures.ipynb)
- [自适应实验协议](docs/ADAPTIVE_PROTOCOL.md) · [运行说明](docs/RUN_GUIDE.md)
- [结果与复现文件](results/adaptive_v1/README.md) · [经济参数来源](docs/ECONOMIC_CASE_SOURCES.md)

主比较为 **GP、XGBoost残差、CNP、TabPFN joint、TabPFN＋graph**。开发场地按预定log-MAE规则选出graph；完整计划`B_full`、同模型maximin顺序早停`B_seq`和自适应早停`P_adapt`使用同一初始集和隐藏审计集。初始至少5根，每批目标2根，近邻组始终完整揭示，实际根数按组记录。

## 预测比较

下表为4个评价场地、3种子、63个登记的完整组状态（62个不同已知集）的场地等权均值。观察节点覆盖5–13根附近及每个完整计划端点；它与旧版5–13 AUBC口径不同。

| 方法 | log-MAE ↓ | 厚度MAE / m ↓ | 点召回 ↑ | 点误报率 ↓ |
|---|---:|---:|---:|---:|
| GP | 0.509278 | 0.400318 | 14.03% | 1.88% |
| XGBoost残差 | 0.511805 | 0.485069 | 31.23% | 6.89% |
| CNP | 0.656595 | 0.430469 | 0.00% | 0.00% |
| TabPFN joint | 0.497141 | 0.491450 | 33.30% | 7.80% |
| TabPFN＋graph（冻结主模型） | 0.502526 | 0.594727 | 36.28% | 9.65% |

joint的剖面误差较XGBoost低约2.87%，graph低约1.81%；GP厚度误差更低。消融保留同代未微调权重及训练残差路径，微调收益因组合方式而异。

## 选点与经济结果

Airport三个种子均从21根降至11根，约省47.62%；B_seq和P_adapt停止数量相同。四场地等权省点率为顺序早停13.57%、自适应早停11.90%，自适应本轮未增加额外节约。两策略均无错误早停；New Lock仍需64根，完整计划下也未达到研究质量目标。

主α=1、EUR2025情景中，Airport净收益为€23,300；New Lock约为−€55,933（三种子均值）。数值来自真实停止根数与明确的成本/工期假设，不能视为实际合同收益。完整正负结果、来源与敏感性见结果目录和交互展示。

## 运行

在Project根目录使用现有`.venv`。全新环境按`requirements-lock.txt`和`pyproject.toml`安装；Notebook与绘图依赖可用`.[notebook]`，模型依赖见`.[tabpfn]`。

```powershell
.venv/Scripts/python.exe -m cpt_budget status
.venv/Scripts/python.exe -m cpt_budget adaptive --stage pilot
.venv/Scripts/python.exe -m cpt_budget adaptive --stage publish
.venv/Scripts/python.exe -m cpt_budget.adaptive_display --notebook
```

完整重跑应使用新的配置输出目录；已有评价结果不会被重新用于选择阈值。大权重、原始数据和逐单元预测保存在本地，获取方式见[Data说明](Data/README.md)。新实验的配置、位置划分、权重指纹、轨迹、事后审核和经济输入均可追溯。

## 证据范围与历史

492个可用代表位置来自31个场地，每孔24个深度单元；深度行和种子不是独立工地。8个回放场地已有开发或揭盲历史，本轮使用对应外层折权重进行有限数据重评估。`qc < 1 MPa`是研究指标，停止审核对应剖面与累计厚度研究目标，低值识别另行检查。经济金额是有来源或明确假设的反事实情景，主情景α=1。

v0.9.1的13入口及旧成绩保存在[历史方案对比](CA6001_CPT_当前方案对比.md)、`results/current/`与`results/final_test/`；旧Notebook和HTML位于`notebooks/archive/`。项目背景见[原始设计](CA6001_CPT_高分项目设计.md)与[实现蓝图](CA6001_CPT_项目实现蓝图.md)。当前交付为实验数据、七张报告图（`python -m cpt_budget.report_figures`）、课程技术报告（`python report/build_report.py`，按课程模板、封面外9页）、Notebook和交互展示。
