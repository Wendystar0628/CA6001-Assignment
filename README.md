# CA6001 · Budget-limited CPT investigation

用已有 CPT 测量重建未测剖面，并在固定预算下推荐下一批两个位置。项目比较 IDW、空间 GP、XGBoost、原始 TabPFN v2 和按场地 episode 微调的 TabPFN；通过历史数据离线回放，区分预测模型与选址规则的影响。

设计依据：[项目实现蓝图](CA6001_CPT_项目实现蓝图.md) · [研究设计](CA6001_CPT_高分项目设计.md) · [模块接口共识](docs/IMPLEMENTATION_CONTRACT.md)。

## v0.5 新方向探索

已并行实现并实验功能型剖面、图调和／Huber重建、单调剖面配准、条件神经过程，并加入常数／深度中位数低信息基线。九个新候选各完成60状态，保留原23个强参照／候选。图Huber误差较IDW改善约2.03%，形状配准厚度误差改善约5.58%，但新增方法均未通过完整综合门槛；常数弱参照误差比IDW高约39.24%。[完整研究与实测结果](results/v5-exploration-20261014/README.md) · [实验协议](docs/V5_NEW_DIRECTIONS_PROTOCOL.md)。标签`v0.5-exploration`，代码直接位于main。

## v0.4 主线实现

已将既有v0.2/v0.3合入`main`，后续直接在主线上提交。新增与真实补勘轨迹对齐的源训练情景、IDW剩余误差的TabPFN梯度微调、逐查询物理事件损失，以及与同alpha未微调残差的公平比较。运行入口为`python -m cpt_budget.v4`，详见[运行说明](docs/V4_RUN_GUIDE.md)和[训练协议](docs/V4_RESIDUAL_TRAINING_PROTOCOL.md)。完整训练与回放的最终效果以归档结果为准。

**v0.4已完成六个模型训练和完整验证。** 同alpha=1的任务残差微调相对未微调降低12.18%误差；最优点保护候选相对IDW改善2.83%。只有一折通过综合护栏，全局多数场地条件未通过，未扩大外层测试。[完整结果与限制](results/v4-residual-20261014/README.md)，标签`v0.4-residual`。

## v0.3 诊断结果

已并行完成训练链路、IDW教师、残差与上下文适配、独立24特征XGBoost诊断。TabPFN可近乎精确拟合固定训练目标；最好的点保护残差适配较IDW降低约1.81%平均误差，但校准风险误报率上升，新增候选均未通过综合门槛。[完整结果与复现](results/diagnostics-20261014/README.md) · [预登记协议及追加记录](docs/V3_DIAGNOSTIC_PROTOCOL.md)。分支`feat/v0.3-diagnostics`，标签`v0.3-diagnostics`。

## v0.2 入口

v0.2已在 `feat/v0.2-risk-aware` 分支实现，使用独立的 `python -m cpt_budget.v2` 命令、`configs/v2.yaml` 和 `runs/v2/`。新增有效物理尺度与垂向特征、任务微调、有护栏的IDW残差、按组留一的风险校准，以及整根CPT的ALC/SUR选址。候选须通过完整源验证门槛才扩大第二轮测试。

查看 [v0.2运行说明](docs/V2_RUN_GUIDE.md) 和 [v0.2实现共识](docs/V2_IMPLEMENTATION_CONTRACT.md)。下文保留v0.1复现入口及已冻结结果；新增能力是否改善性能以各run的验证与晋级记录为准。

**v0.2三折验证已完成：平均误差指标改善1.37%，但仅2/4不同验证场地改善，未通过全局扩大测试门槛。** 新TabPFN候选均未晋级，部分场地风险误报率仍高；没有证据宣称稳定节省CPT根数。查看[完整实现与验证结果](results/v2-validation-20261013/README.md)，结果标签为`v0.2-validation`。

## 当前可运行内容

- 数据恢复、规范化、整位置与勘察组划分、三折及 episode 生成。
- 五种真实预测模型；三种选址规则；固定评价集上的批量揭示与评价。
- TabPFN 真实梯度更新、验证早停、权重保存和重载；目标场地只替换上下文。
- 验证场地质量目标、离散预算达标、配对误差与条件成本汇总。
- 报告 SVG/PDF/PNG、16:9 演示图与满足输入条件的 New Lock 帧序列。
- 分阶段命令、结果来源记录及断点续跑。

**正式三折训练与440条测试回放已完成。** 查看 [首次完整实验结果](results/main-20261013/README.md) 和 [实现状态](docs/IMPLEMENTATION_STATUS.md)。当前结果没有证明微调＋阈值选址可稳定节约补勘；原始表格、负结果与未达标情况均保留。

开发 pilot 只验证程序链路。单场地、单种子或单步微调不能代替正式结果。最终课程报告、12张内容幻灯片与团队录制尚未制作。

## 安装（Windows PowerShell）

在 `Project` 仓库根目录运行。路径有空格或 `&` 时，用单引号包住完整路径。

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe -e .
```

锁定环境已经在本机 RTX 4080 上执行过真实微调和推断。其他显卡／驱动应先检查 CUDA 是否可用；CPU 可运行 IDW、GP 和 XGBoost。权重首次加载可能需要网络，模型缓存在 Hugging Face 缓存目录；不要将凭据或模型权重提交 Git。

## 数据准备

原始下载包和大型数组不放入 Git。按 [数据说明](Data/README.md) 获取 `Global_CPT1196.zip`，放到 `Data/downloads/`。仓库中的 screened 清单固定了源文件和数组键，可从原始包恢复数值，无需重新选择研究样本。

```powershell
.venv/Scripts/python.exe -m cpt_budget restore-inputs
.venv/Scripts/python.exe -m cpt_budget prepare
.venv/Scripts/python.exe -m cpt_budget split
```

当前输入共有 503 个登记剖面、495 个不同计划位置。按完整的 24 单元网格及最大 0.5 m 插值缺口筛选后，可用 **492 个位置、11,808 个网格单元、31 个场地**。3 个位置的深度缺口不允许填补；8 个同坐标非代表剖面不重复计费。

4 个场地（Avonside、Baytown、Hollywood、New Lock）的全球位置不能从源坐标表确认。它们保留明确的原始项目分组与 `grouping_verified=false`，不宣称已验证全球近邻独立性。严格排除未核实勘察组时，在单独配置中设置 `split.exclude_unverified_groups: true` 并使用独立 `prepared_dir` 和新 run，不按模型表现决定排除。

## 最短真实演示

```powershell
# 快速 CPU 闭环：源验证场地、seed 17、5→7 根、真实基线与选址
.venv/Scripts/python.exe -m cpt_budget pilot --models idw,gp,xgb --run my-cpu-pilot

# 五模型闭环：含一次真实微调、三策略以及两套图形
.venv/Scripts/python.exe -m cpt_budget pilot --models idw,gp,xgb,tabpfn,tabpfn_finetuned --run my-five-model-pilot
```

Pilot 自动选择一个能完整回放的源验证场地，使用共同的初始观测与评价集。它不会查看外层测试表现选择参数。`--site` 可指定该类场地的稳定 ID。默认 pilot 将微调限制为一次梯度更新；它不等于完成一折正式训练。

## 正式实验

```powershell
# 先训练一个外层折、冻结验证质量，随后执行该折正式回放
.venv/Scripts/python.exe -m cpt_budget fit --fold 0 --run main-experiment
.venv/Scripts/python.exe -m cpt_budget freeze-quality --fold 0 --run main-experiment
.venv/Scripts/python.exe -m cpt_budget evaluate-static --fold 0 --run main-experiment
.venv/Scripts/python.exe -m cpt_budget replay --fold 0 --run main-experiment
.venv/Scripts/python.exe -m cpt_budget summarize --run main-experiment
.venv/Scripts/python.exe -m cpt_budget export --run main-experiment

# 或按 main.yaml 连续运行三折；--resume 复用配置及输入一致的完成任务
.venv/Scripts/python.exe -m cpt_budget run --run full-main --resume
```

GPU 同时只跑一个模型进程。正式默认最多 10 epoch、验证早停，每折训练约两小时上限。只想检查部分模型时显式传入 `--models`，这个选择会写入 run 配置；不能把子集结果称为完整实验矩阵。

## 看结果与续跑

每次结果位于 `runs/<run_id>/`。先看 `run.json` 中的模式、实际配置、代码版本、已完成任务与失败原因，再打开 `figures/report/` 或 `figures/slides/`。数值表在 `metrics.parquet` 与 `summary/*.csv`，轨迹在 `states.parquet`，训练记录在 `checkpoints/fold_<k>/`。`inputs/` 保留实际划分和来源快照。

```powershell
.venv/Scripts/python.exe -m cpt_budget status --run my-five-model-pilot
.venv/Scripts/python.exe -m cpt_budget export --run my-five-model-pilot --preset slides --figure profile
```

画图和汇总重新读取保存结果，不训练模型。配置或输入分组变化必须建立新 run；已有完整训练产物会复用，同一回放任务的结果按任务键替换，避免中断后重复追加。冻结结果后的正确性修复应保留旧版本并重新生成受影响结果。

## 结果解释边界

- `log(qc)` 是自然对数；深度为各来源记录深度，并非统一地层标高。
- 低锥阻主阈值为 1 MPa；它是实验任务定义，不能直接解释成真实土类或工程安全标准。
- 所有方法共享位置预算，评价集永不被选中；同一批两点确定后才揭示真值。
- 没有真实正例的场地，Recall/AP 缺失；没有有效质量目标或完整预算曲线时，不生成稳定达标与精确成本节省。
- 无报价时只报告根数和条件分析，不生成“0元节约”。逐深度区间不等于整条剖面或厚度的联合置信区间。

## 关键检查与版本节点

```powershell
.venv/Scripts/python.exe -m pytest -q
```

检查聚焦数据连接、组隔离、整位置留一、真实梯度／上下文边界、两点批量揭示、指标缺失和预算截断。Git 阶段节点按蓝图第 12.5 节执行；原始包、缓存、完整 runs 和权重不直接推送。
