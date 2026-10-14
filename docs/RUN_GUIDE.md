# 运行说明

在Project仓库根目录执行（版本v0.9.1）。模型名单唯一来源为`configs/catalog.json`，共13个入口；`models`会列出每个入口的id、分组、`risk_source`和名称。

## 常用命令

```powershell
.venv/Scripts/python.exe -m cpt_budget models
.venv/Scripts/python.exe -m cpt_budget status
.venv/Scripts/python.exe -m cpt_budget predict --model balanced --fold 0 --seed 17 --budget 5
.venv/Scripts/python.exe -m cpt_budget predict --model balanced --point-only   # 只做点预测，跳过共享风险头
.venv/Scripts/python.exe -m cpt_budget verify                                   # 13×60＝780项，应全部通过
.venv/Scripts/python.exe -m cpt_budget evaluate                                 # 重算results/current的指标文件
.venv/Scripts/python.exe -m cpt_budget port-check                               # 与v0.8归档逐状态指标对照，26/26
.venv/Scripts/python.exe -m cpt_budget benchmark --cpu-core 2                   # 统一运行时间测量
.venv/Scripts/python.exe -m pytest -q                                           # 46项测试
```

- **`status`**：核对`artifacts/manifest.json`登记的31个文件是否存在、字节数是否一致，并检查`data_processed_v2/`中的profiles、grid、sites三个表；缺失或不一致时以非零状态退出。
- **`predict`**：取一个已登记的验证状态（`--fold`默认0，`--seed`默认17，`--budget`默认5），默认使用该fold按site_id字母序的第一个场地（fold 0/1为Lower Hutt，fold 2为Chch Metro Sports），可用`--site`指定。输出默认写到`runs/current/prediction.csv`。预测函数不接收查询位置的真值。输出只有点预测，有原生概率的入口另带`p_low_1`（GP另带分位数）；没有风险标签，风险标签只在`evaluate`中由交叉校准生成。
- **`--point-only`**：只对有独立共享风险头的6个入口有效：`tabpfn_joint`、`balanced`、`thickness`、`recognition`、`count_shrink`、`thickness_guarded`。其余7个入口的输出本来就是点预测（或GP原生分布），传入该参数会直接报错。`tabpfn_joint`的仅点预测路径仍要运行TabPFN。
- **`verify`**：在全部60个冻结状态上重新推理，与`artifacts/evaluation/reference_predictions.parquet`对照，检查点预测（GP容差1e-5，其余1e-6）、0值低阻力标签，以及原生列`p_low_1`和GP的q05/q10/q25/q75/q90/q95。原生列消失或多出都判为失败，并核对`source_model_id`与`risk_source`。可加`--models idw gp`、`--folds 0`、`--seeds 17`、`--budgets 5 13`。输出默认写入`runs/current/verification/`，做子集验证时请用`--output`指定其他目录，以免覆盖完整的780项记录。正式证据的副本是`results/current/migration_checks.csv`和`migration_summary.json`。在RTX 4080上整轮约2–3分钟。
- **`evaluate`**：从参考预测重算`metrics.csv`、`curves.csv`、`unique_site_metrics.csv`、`all_model_ranking.csv`和`evaluation_provenance.json`，包括按模型、按fold的交叉风险校准，约30秒。`--source live`改为先实时推理再评价。`--models`只评价指定入口，但仍会写入`--output`目录（默认`results/current`），所以做子集评价时必须另指定`--output`。
- **`port-check`**：把13个入口×60个状态的逐状态指标，与Git标签`v0.8-engineering`中归档的逐状态指标对照（用`git show`读取，需要本地存在该标签），结果写入`results/current/port_equivalence.csv`；当前为26/26组等价。
- **`final-test`**：预注册的最终外层测试，协议见[FINAL_TEST_PROTOCOL.md](FINAL_TEST_PROTOCOL.md)。依次为`--prepare`（测试状态、泄漏审计、只用验证标签的风险校准器）、`--blind`（13个入口×100个测试状态，不读取测试真值，写`results/final_test/blind/`）、`--freeze`（写`results/final_test/frozen.json`）、`--unblind`（只能运行一次；任一冻结哈希不符即拒绝，重跑须`--force-rerun`并被记为rerun）。
- **`benchmark`**：运行时间测量协议见[当前方案对比](../CA6001_CPT_当前方案对比.md#运行时间v091统一测量)。默认每个状态1次预热、3次计时（`--warmups`、`--repeats`），同时测量6个入口的`--point-only`路径（可用`--no-point-only`关闭）。输出写到`runs/current/benchmark/`；`--publish results/current`会把结果复制为`results/current/runtime_*`。正式结果使用`--cpu-core 2 --publish results/current`，整轮约3分钟。

## 必需数据、权重与依赖

- 原始数据与恢复方法见[Data/README.md](../Data/README.md)。当前准备数据在`data_processed_v2/`。
- `artifacts/evaluation/`：冻结的60个状态（`states.parquet`）、fold信息、本地参考预测（13×8640＝112,320行），以及v0.9.0参考预测的备份。
- `artifacts/models/fold_0..2/`：各折的TabPFN CRPS权重、FPCA基、GP趋势、匹配XGBoost及CNP权重，每折9个文件。
- `artifacts/manifest.json`登记以上31个文件的路径、字节数与来源。

大权重和包含原始测量的参考预测不提交Git。全新clone需要另行取得这些工件，或在单独checkout中用`v0.8-engineering`标签的训练流程重建；运行前先用`status`检查。

依赖：基础依赖见`pyproject.toml`。TabPFN与torch属于可选依赖`.[tabpfn]`（tabpfn==9.2.0、torch>=2.5）；`requirements-lock.txt`已锁定tabpfn 9.2.0和torch 2.11.0+cu128。

- 不需要TabPFN与torch即可运行：`constant`、`idw`、`gp`、`xgb`、`simplex`，以及`balanced`、`thickness`、`recognition`、`count_shrink`、`thickness_guarded`的`--point-only`点预测。
- `cnp`需要torch，固定在CPU上运行。
- 需要tabpfn==9.2.0与torch：`tabpfn_joint`（含`--point-only`）、`tabpfn_graph`，以及6个共享风险头入口的完整路径。共享风险头会调用训练后的TabPFN残差与图Huber组件。

**CUDA。** `predict`、`verify`、`evaluate --source live`沿用`configs/main.yaml`，其中未设置`tabpfn.device`，即自动选择：有CUDA时用GPU；没有CUDA时TabPFN回退到CPU，并发出`RuntimeWarning`。CPU上的TabPFN输出与CUDA参考在数值上不等价：v0.9.1的统一测量中，`tabpfn_joint`仅点预测在CPU上的最大点差为0.33，12个状态中有1个单元标签变化，因此无CUDA时`verify`可能不通过。参考一致性只在CUDA上验证过。`benchmark`对目录中`device=cuda`的入口要求实际CUDA，没有CUDA时直接报错，不会回退。

## 数据准备

```powershell
.venv/Scripts/python.exe -m cpt_budget restore-inputs
.venv/Scripts/python.exe -m cpt_budget prepare
.venv/Scripts/python.exe -m cpt_budget split
```

上述命令不替换`artifacts/evaluation/`中冻结的验证状态。`split`会按`configs/main.yaml`重新生成fold、episode和回放划分，但当前成绩使用的60个冻结状态来自v0.2运行（seed 17/29、maximin），不随`split`改变。更新数据后，应重新建立评价协议与工件，并重新计算成绩。

通用的两点选址、固定评价集回放和成本统计函数分别位于`acquisition.py`、`replay.py`、`economics.py`，只作为库函数保留，CLI未暴露，当前结果也不使用它们。完整训练与探索流程可在独立checkout中查看`v0.8-engineering`标签。
