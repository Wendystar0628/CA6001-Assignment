# v0.4残差训练运行说明

新版代码和关键实验节点直接提交到`main`。研究方案见[冻结协议](V4_RESIDUAL_TRAINING_PROTOCOL.md)。本轮训练目标是空间IDW的剩余误差，GP保持强对照；不将能够运行等同于性能目标达成。

使用既有Python环境、`data_processed_v2/`、v0.2的冻结回放结果及v0.3诊断对照。大数据和权重仍保存在本地，仓库保留代码与精简证据。从Project根目录运行：

```powershell
# 准备多情景源训练/验证轨迹
.venv/Scripts/python.exe -m cpt_budget.v4 prepare --run v4-residual-20261014

# 三折×两种损失训练，并完成全部同状态评价
.venv/Scripts/python.exe -m cpt_budget.v4 run --run v4-residual-20261014

# 分阶段操作和查看进度
.venv/Scripts/python.exe -m cpt_budget.v4 fit --run v4-residual-20261014
.venv/Scripts/python.exe -m cpt_budget.v4 evaluate --run v4-residual-20261014
.venv/Scripts/python.exe -m cpt_budget.v4 status --run v4-residual-20261014
```

`fit/evaluate`支持`--fold 0`、`--variant crps`或`--variant task`，但最终选择要求两种损失、三折、全部10候选和60状态完整。GPU同一时间只运行一个进程。默认配置`configs/v4.yaml`已冻结；改变配置时必须换`--run`名称，不能复用旧状态缓存。

训练完成的checkpoint和验证完成的逐状态预测可续跑复用。训练中断不会把中途最佳checkpoint自动当成完成训练；失败会记录到`run.json`。如发生中断，应保留失败运行，使用新run重启或明确实施恢复，不声称已执行未完成的更新。

结果位于`runs/<run>/`：`episodes/`为固定轨迹及跳过说明，`checkpoints/fold_*/residual_*/summary.json`为训练步数/最佳权重/曝光率，`validation/`为候选预测缓存，`comparison/`为最终同预算比较、同alpha微调收益和门槛。`run.json`记录阶段状态及错误。

评价还要求在同一批v0.4轨迹上训练的XGBoost残差对照，保存在`matched_xgb/`。缺失时`evaluate`自动生成；也可提前用`python -m cpt_budget.v4.matched_baseline --run <run>`在CPU运行。其可用源数据相同，实际训练曝光方式不同（树模型使用全部行，TabPFN受更新与块预算限制），报告中明确保留这个区别。

`crps`使用CRPS＋MSE；`task`增加物理qc<1MPa事件的Brier目标。残差事件阈值逐查询点计算。评价同时列出原点指标和留勘察组校准风险指标；点保护不保证校准后的风险标签也保持不变。

本轮不会自动运行外层测试。未通过门槛时保留完整负结果；最佳checkpoint只由源验证选择，其对应数据曝光率与训练结束时曝光率分开报告。
