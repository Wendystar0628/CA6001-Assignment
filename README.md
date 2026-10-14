# CA6001 · CPT重建与预算研究

本项目研究有限勘探预算下的CPT曲线重建、低阻力层厚度预测与低阻力识别。当前版本v0.9.1提供 **4个基线（1弱3强）＋2个TabPFN组合＋3个主方案＋4个讨论方案**，共13个模型入口，名单唯一来源为[configs/catalog.json](configs/catalog.json)。

- [当前实现状态](docs/IMPLEMENTATION_STATUS.md)
- [13方案性能与取舍](CA6001_CPT_当前方案对比.md)
- [运行说明](docs/RUN_GUIDE.md)
- [方案选择依据](docs/MODEL_SELECTION.md)
- [结果证据](results/current/README.md) · [v0.9.1变更](results/current/CHANGES_v0.9.1.md)

```powershell
.venv/Scripts/python.exe -m cpt_budget models
.venv/Scripts/python.exe -m cpt_budget status
.venv/Scripts/python.exe -m cpt_budget predict --model balanced --point-only
.venv/Scripts/python.exe -m cpt_budget verify       # 13×60状态与参考预测对照，780/780
.venv/Scripts/python.exe -m cpt_budget evaluate     # 从参考预测重算results/current的4个指标文件
.venv/Scripts/python.exe -m cpt_budget port-check   # 与v0.8归档逐状态指标对照，26/26
.venv/Scripts/python.exe -m cpt_budget benchmark --cpu-core 2
```

沿用已有`.venv`即可；新环境按`requirements-lock.txt`及`pyproject.toml`安装（TabPFN组合、主方案风险头与CNP还需可选依赖`.[tabpfn]`，即tabpfn==9.2.0与torch）。大权重与参考预测位于本地`artifacts/`，不提交Git；缺失时`status`明确报错。数据获取见[Data/README.md](Data/README.md)。

主要结果（4个验证场地、预算5–13的AUBC，与IDW强基线比较；IDW为log-MAE 0.638586、厚度MAE 0.630371 m、点召回37.38%、点误报7.28%）：

- 均衡方案`balanced`：log-MAE 0.592435（−7.23%），厚度MAE 0.536947 m（−14.82%）。
- 厚度方案`thickness`：厚度MAE 0.472493 m（−25.05%），是同一协议下记录过的最低厚度误差；但点误报10.05%，超出v0.7晋级规则的点误报护栏（≤IDW+2pp＝9.28%）。护栏内的最优厚度配置保留为讨论方案`thickness_guarded`（0.487630 m，点误报8.99%）。
- 识别方案`recognition`：点召回48.77%（+11.39pp），代价是点误报9.14%（+1.86pp）。
- 讨论方案`count_shrink`的log-MAE最低（0.591791），但厚度MAE较均衡方案差，完整链路耗时约为均衡方案的1.89倍。

上述成绩来自用于模型选择的4个验证场地（60个冻结状态只含39个不同已知集和8个不同评价集），应视为乐观估计。最终测试（4个从未参与选择的场地：Chch Stadium、Wellington Airport、New Lock、Hollywood；盲测预测、哈希冻结、只揭盲一次）中，log-MAE前三与验证相同：count_shrink 0.5049、balanced 0.5052、tabpfn_joint 0.5061（验证→测试Spearman ρ＝0.912）；厚度最优为thickness 0.4383 m（较IDW −0.065 m，3胜1平0负），厚度排名ρ＝0.510。仅4个测试场地，结论为描述性；Wellington Airport的评价集没有低阻力单元。协议见[docs/FINAL_TEST_PROTOCOL.md](docs/FINAL_TEST_PROTOCOL.md)，结果见[results/final_test/README.md](results/final_test/README.md)。不能证明能节省补勘根数。

项目背景见[高分设计](CA6001_CPT_高分项目设计.md)与[实现蓝图](CA6001_CPT_项目实现蓝图.md)。两份均为2026-10-13的早期设计，与当前实现不一致之处列在各自文首。课程最终报告以Jupyter Notebook提交，当前底稿为`notebooks/CA6001_report_figures.ipynb`（本地，尚未提交Git）。
