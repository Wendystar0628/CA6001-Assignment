# CPT v0.2 运行说明

v0.2位于 `feat/v0.2-risk-aware` 分支。原 `v0.1-results` 标签、`configs/main.yaml`、旧命令与结果仍保留；新版以独立配置和目录运行。

## 新版增加的能力

- 24维空间与垂向特征，固定物理长度核统计经过完整TabPFN预处理后仍保留尺度信息。
- CRPS＋MSE和增加低值CDF Brier目标的微调候选；每场地等权、查询块完整轮换，记录最佳checkpoint与训练结束时各自的数据曝光率。
- 有厚度、Recall、误报护栏的IDW残差模型，允许明确回退到IDW。
- 按验证勘察组留一的风险校准；点预测指标保留，风险Recall/FPR单列。这里的OOF限定为校准映射，模型本身仍按源验证选择。
- GP整根CPT的ALC和风险SUR，第二根考虑第一根的信息，两个点确定后才揭示真实数据。
- 联合可达的验证质量参考及原边际规则对照，明确这是质量定义变更，不是算法收益。

详细接口和预定参数见 [实现共识](V2_IMPLEMENTATION_CONTRACT.md)，研究依据见 [结果诊断](CA6001_CPT_结果诊断与改进路线.md)。

## 安装与准备

依赖沿用仓库README的Python3.11环境，无需另安装机器学习框架。在Project根目录运行：

```powershell
.venv/Scripts/python.exe -m cpt_budget.v2 prepare
```

输出位于 `data_processed_v2/`，不会覆盖旧版准备数据。原始包缺失时先按旧README执行 `restore-inputs`。

## 分阶段运行

```powershell
.venv/Scripts/python.exe -m cpt_budget.v2 fit --run my-v2-run
.venv/Scripts/python.exe -m cpt_budget.v2 validate-models --run my-v2-run
.venv/Scripts/python.exe -m cpt_budget.v2 calibrate --run my-v2-run
.venv/Scripts/python.exe -m cpt_budget.v2 select-models --run my-v2-run
.venv/Scripts/python.exe -m cpt_budget.v2 validate-policies --run my-v2-run
.venv/Scripts/python.exe -m cpt_budget.v2 select-policies --run my-v2-run
.venv/Scripts/python.exe -m cpt_budget.v2 summarize --run my-v2-run
.venv/Scripts/python.exe -m cpt_budget.v2 export --run my-v2-run
```

完整流程可用 `run --run my-v2-run --resume`。模型训练完成的产物会复用，回放按任务保存；同名run的配置或源数据分组变化会被拒绝。训练失败保留失败记录和已完成产物，不将失败更新计作有效训练。

GPU同时只运行一个训练／推断进程。CPU开发可选 `--models idw,gp,idw_residual --fold 0`，但同一个run的模型清单要保持一致。仅有一个折的开发结果不能触发完整第二轮测试。

## 结果与门槛

结果目录为 `runs/v2/<run_id>/`：

| 入口 | 内容 |
|---|---|
| `run.json` | 配置、来源版本、完成阶段与失败记录 |
| `checkpoints/fold_*` | 微调、残差和GP源趋势；最佳权重及实际曝光记录 |
| `metrics.parquet` | 固定maximin下的原始点指标 |
| `predictions_oof.parquet`、`metrics_oof.parquet` | 留勘察组校准后的验证预测与独立风险指标 |
| `model_selection.json` | 每折强基线、候选护栏及回退理由 |
| `policy_metrics.parquet`、`policy_selection.json` | 固定预测器的策略验证、按场地配对及全局扩展门槛 |
| `quality_targets.json` | 同一验证参考下的新联合规则与旧边际规则 |
| `figures_v2` | 报告／幻灯片图及其源表、作用域标记 |

晋级要求使用全部预定场地、种子和5/7/9/11/13预算，缺失任何键会拒绝选择。四个护栏统计量都先按预算作归一化梯形积分，再平均种子、场地。未通过的折保留基线／maximin回退；不能只测试验证中成功的折。只有完整三折的全局规则通过，`test-selected`才运行五种子第二轮测试，否则明确返回未通过原因。

风险决策阈值0.2对应本次研究情景的漏判／误报代价4:1，不是工程标准。跨场地校准可能失准；应同时查看绝对误报率、Recall、Brier与原点误差，不能把高误报换来的高Recall说成可靠提升。SUR中的校准风险是代理目标，不具有严格非负信息价值保证。

当前外层结果已被用于构思v0.2，所有新版结果标为探索性第二轮。若没有候选满足原核心效果，应保存负结果，不通过放宽门槛、只报告成功场地或改变主指标制造提升。
