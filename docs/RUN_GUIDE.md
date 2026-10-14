# 当前精简版运行说明

在Project仓库根目录执行。模型名单唯一来源为`configs/catalog.json`，仅12个入口。

```powershell
.venv/Scripts/python.exe -m cpt_budget models
.venv/Scripts/python.exe -m cpt_budget status
.venv/Scripts/python.exe -m cpt_budget predict --model balanced --fold 0 --seed 17 --budget 5
# CPU点预测，不计算独立TabPFN/图风险头
.venv/Scripts/python.exe -m cpt_budget predict --model balanced --point-only
.venv/Scripts/python.exe -m cpt_budget verify
.venv/Scripts/python.exe -m pytest -q
```

`predict`使用一个已登记的验证状态，默认选择指定fold的第一个场地；可用`--site`指定场地。输出默认`runs/current/prediction.csv`，预测函数不接收查询真值。`verify`重新执行所有选定模型并与迁移前快照比较；可加`--models idw gp`、`--folds 0`、`--seeds 17`、`--budgets 5 13`。

## 必需数据和权重

- 原始数据与恢复方法见[Data/README.md](../Data/README.md)。当前准备数据在`data_processed_v2/`。
- `artifacts/evaluation/`保存唯一冻结60状态、fold信息及本地参考预测。
- `artifacts/models/fold_0..2/`保存各折TabPFN CRPS权重、FPCA基、GP趋势、匹配XGBoost及CNP讨论权重。
- `artifacts/manifest.json`登记必要文件及来源；`status`检查本地是否齐全。

大权重和包含原始测量的参考预测不提交Git。当前机器已迁移并验证。全新clone需要取得这些保留工件，或从`v0.8-engineering`标签恢复旧训练流程重建；不能把仅有代码的clone说成已经附带训练权重。

CPU可运行常数、IDW、GP、XGBoost、CNP、simplex，以及`--point-only`的其他点模型。TabPFN组合及主方案的完整经验风险头需要现有TabPFN/CUDA环境；共享风险头会调用训练后的TabPFN残差与图Huber。

## 数据与通用回放

```powershell
.venv/Scripts/python.exe -m cpt_budget restore-inputs
.venv/Scripts/python.exe -m cpt_budget prepare
.venv/Scripts/python.exe -m cpt_budget split
```

上述准备命令不替换`artifacts/evaluation/`的冻结验证清单。更新数据后，应重新建立独立的协议与工件，不能悄悄沿用旧成绩。

通用两点选择、固定评价集回放和成本统计函数仍保留在`acquisition.py`、`replay.py`、`economics.py`。旧版跨大量模型的训练扫描、动画/幻灯片导出和多轮组合搜索入口已从当前树删除。需要研究历史流程时，在独立checkout中查看旧标签，不将它们重新混入当前12模型入口。
