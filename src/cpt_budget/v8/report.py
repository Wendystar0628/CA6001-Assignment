"""Generate the requested Markdown report directly from saved measurements."""
import json
import pandas as pd
from cpt_budget.config import PROJECT_ROOT
from .common import RUN

REPORT_NAME='CA6001_CPT_深度工程优化与性能对比报告_v0.8.md'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def generate():
    quality=pd.read_csv(RUN/'comparison/all_model_ranking.csv').set_index('model_id')
    timing=pd.read_csv(RUN/'benchmark_pinned/summary.csv').set_index('model_id')
    pairs=pd.read_csv(RUN/'benchmark_pinned/paired_summary.csv').set_index('pair_id')
    protocol=json.loads((RUN/'benchmark_pinned/protocol.json').read_text(encoding='utf-8'))
    selected=[]
    def qt(items):
        rows=[]
        for name,model in items:
            r=quality.loc[model];selected.append({'name':name,'model_id':model,**r.to_dict()})
            rows.append([name,f'{r.log_mae:.6f}',f'{r.h_point_mae_m:.6f}',
                         f'{100*r.recall:.2f}%',f'{100*r.false_positive_rate:.2f}%'])
        return table(['方案','log-MAE ↓','厚度MAE(m) ↓','点Recall ↑','点FPR ↓'],rows)
    baselines=qt([('常数中位数弱基线','target_constant_median'),('仅深度中位数','target_depth_median'),
                  ('IDW强基线（优化版预测相同）','idw'),('GP强基线','gp'),('同轨迹XGBoost残差','xgb_residual_matched'),
                  ('v0.7三角网分片线性','v7_simplex_linear')])
    tabpfn=qt([('历史TabPFN空间特征','tabpfn_spatial'),('历史TabPFN CRPS微调','tabpfn_crps'),
               ('历史TabPFN任务损失微调','tabpfn_task'),('未微调TabPFN残差α=1','tabpfn_residual_a100'),
               ('源训练任务损失残差α=1','residual_ft_task_a100'),
               ('源训练CRPS残差α=.5＋符号保护','residual_ft_crps_guarded_a050'),
               ('FPCA＋TabPFN残差符号保护','v6_fpca_residual_sign_guard'),
               ('图Huber＋TabPFN残差低值并集','v6_graph_residual_low_union'),
               ('本轮优化TabPFN残差（输出不变）','v8_tabpfn_guarded_fast')])
    optimized=qt([('均衡主线：等价快版','v8_balanced_fast'),('均衡：支持集计数收缩','v8_balanced_count_shrink'),
                  ('厚度主线：Haar等价快版','v8_haar_fast'),('Haar支持集混合计数','v8_haar_loo_blend'),
                  ('识别：原方案等价快版','v8_recognition_fast'),('识别：可靠性定位','v8_recognition_localize'),
                  ('识别：60%计数分位','v8_recognition_recall60')])
    speed_rows=[]
    for name,pair,old,new in [
      ('同公式IDW点预测','idw_optimized','idw_optimized_old','idw_optimized_new'),
      ('结构组合点基底','structural_point','structural_old','structural_new'),
      ('TabPFN残差点路径（CUDA）','tabpfn_residual','tabpfn_residual_old','tabpfn_residual_new'),
      ('均衡完整点路径（CPU）','balanced_point','balanced_point_old','balanced_point_new'),
      ('均衡点＋原生风险完整路径','balanced_full','balanced_full_old','balanced_full_new'),
      ('Haar点＋原生风险完整路径','haar_full','haar_full_old','haar_full_new'),
      ('识别点＋原生风险完整路径','recognition_full','recognition_full_old','recognition_full_new')]:
        a,b=timing.loc[old],timing.loc[new]
        speed_rows.append([name,f'{a.total_ms_median:.2f}',f'{b.total_ms_median:.2f}',
                           f'{b.total_ms_p95:.2f}',f"{pairs.loc[pair,'total_speedup_paired_state_median']:.2f}×"])
    speed=table(['计时范围','旧median ms','新median ms','新p95 ms','逐状态配对加速中位数'],speed_rows)
    runtime_rows=[]
    for name,key in [('弱常数基线','target_constant_median'),('IDW同结果快版','idw_optimized_new'),('GP条件化＋分布预测','gp'),
                     ('均衡完整系统','balanced_full_new'),('Haar完整系统','haar_full_new'),
                     ('识别等价完整系统','recognition_full_new'),('可靠性定位完整系统','recognition_localize_full'),
                     ('高召回完整系统','recognition_recall60_full'),('计数收缩完整系统','balanced_count_shrink_full')]:
        r=timing.loc[key];runtime_rows.append([name,f'{r.total_ms_median:.2f}',f'{r.total_ms_p95:.2f}'])
    practical=table(['当前可用路径','median ms','p95 ms'],runtime_rows)
    component_rows=[]
    for name,old,new,pair in [('联合误差曲线算子','joint_operator_old','joint_operator_new','joint_operator'),
                            ('analog识别算子','recognition_operator_old','recognition_operator_new','recognition_operator'),
                            ('Haar完整point但不含风险','haar_old','haar_new','haar_point')]:
        component_rows.append([name,f'{timing.loc[old,"total_ms_median"]:.2f}',f'{timing.loc[new,"total_ms_median"]:.2f}',
                               f'{pairs.loc[pair,"total_speedup_paired_state_median"]:.2f}×'])
    components=table(['局部计时范围','旧median ms','新median ms','配对加速'],component_rows)
    r=quality.loc['v8_balanced_fast'];b=quality.loc['idw']
    profile_gain=100*(1-r.log_mae/b.log_mae);thickness_gain=100*(1-r.h_point_mae_m/b.h_point_mae_m)
    quality['point_balanced_error']=.5*(1-quality.recall+quality.false_positive_rate)
    pd.DataFrame(selected).drop_duplicates('model_id').to_csv(RUN/'report_selected_models.csv',index=False)
    text=f'''# CA6001 CPT：多方案深度工程优化与性能对比报告（v0.8）

日期：2026-10-14。范围：整合v0.4–v0.7领先方案，针对性并行优化，再统一串行测量。所有主表都是同一个固定模型在全部评价状态上的结果，没有跨fold拼接不同冠军。

## 1. 结论与建议

本轮主要成果是**在保持预测质量的前提下显著降低真实运行时间**，同时获得两个有价值的识别适配选项。没有继续堆叠大规模模型搜索，也没有重新训练TabPFN。

- **默认均衡方案**：采用`v8_balanced_fast`。相对IDW，连续误差下降{profile_gain:.2f}%、厚度误差下降{thickness_gain:.2f}%；它与v0.7均衡主线数值等价，收益来自工程执行效率。
- **厚度优先**：采用`v8_haar_fast`。Haar的支持集校正没有进一步降低平均厚度误差，保留原厚度决策的快实现更合适。
- **识别优先**：`v8_recognition_localize`在厚度不变时小幅改善召回和误报；`v8_recognition_recall60`提供更高召回，但必须接受更多误报。
- **TabPFN保留为可复用残差组件**：相同权重、estimator、精度与查询批量下，60状态GPU新旧预测逐位相同，优化其CPU特征准备和解码链路。
- **最小连续误差不是唯一默认选择**：`v8_balanced_count_shrink`的MAE只小幅降低，厚度变差且支持集适配增加时延，见后文实测。

## 2. 统一比较口径

准确度使用原60个冻结状态：3fold、种子17/29、预算5/7/9/11/13；每次评价6根完整剖面、24个2–8m深度单元，事件阈值1MPa。全模型每次使用相同已测集合和查询位置。每个模型8640深度行，但只有**4个不同验证场地**，深度格和重复预算不是独立样本。

统计顺序是预算归一化积分AUBC→种子均值→同场地重复fold均值→不同场地等权。本次203模型统一表包含183个历史参照与20个新配置；附加IDW同结果工程控制单独验证和测速，准确度仍对应IDW行。

**log-MAE**评价log(qc)的连续误差；**厚度MAE**由最终预测qc<1MPa的格数×.25m计算；点Recall/FPR来自同一阈值。它们与外部校准概率≥.2得到的risk Recall/FPR是不同决策，不能混写。

新质量适配只使用当前已测CPT的整近邻组LOO；源FPCA基和TabPFN权重冻结，query真值不进入拟合。误差收缩使用LOO-IDW幅度代理，不能称为完整TabPFN／结构组合的嵌套LOO。本轮仍是反复使用验证集的探索，没有新outer-test或节省补勘根数的证明。

## 3. 预测质量：基线、历史TabPFN及领先组合

### 3.1 强弱基线

{baselines}

IDW使用同深度3近邻、距离平方倒数权重；快版没有改邻居数、公式或零距离处理。GP沿用原源训练深度趋势、三维Matérn残差核与优化配置。XGBoost使用既有匹配轨迹，不削弱强基线。常数模型的零误报同时伴随零召回。

### 3.2 历史TabPFN及其组合

{tabpfn}

原空间／早期微调TabPFN误差较高；针对补勘轨迹训练的残差方案已优于IDW的连续误差。符号保护方案保留IDW原点标签，因此其厚度、召回和点误报不会因幅度修正而改善。FPCA与图组合进一步提供互补作用。

**本轮没有改变Transformer架构或重新做梯度微调。** TabPFN的condition/fit是现有权重的上下文条件化；不能把工程加速描述为新模型能力，也不能据这张表断言某种架构普遍优劣。

### 3.3 本轮优化后的主要方案

{optimized}

均衡、Haar和识别的表中方案均匹配相同独立经验图风险头，以隔离点预测的变化。它们的risk Recall/FPR均为**68.21%／23.90%**，而IDW为69.55%／38.35%。这是风险误报降低、风险召回略降的取舍；点召回提高不会自动改变独立风险头。TabPFN残差自己的风险为73.23%／42.16%，不能只比较其中一个数值。

另保留joint原生概率的对照：原生有限曲线概率在当前公共校准后误报仍偏高，不能把它与经验图风险头混为一谈。完整风险指标见统一CSV。

## 4. 工程改动、等价性与质量取舍

| 对象 | 主要工程改动 | 全60状态等价证据 |
|---|---|---|
| 结构点基底 | 完整剖面数组、LOO距离复用、FPCA批量求解、配准/三角网缓存 | 最大差1.55e-15，低值标签变化0 |
| 整曲线决策 | 一次构建整组LOO，批量原子、稳定加权分位数和计数投影 | 点差0，native概率差≤2.22e-16 |
| Haar厚度 | 缓存正交基、多输出Cholesky、邻近系数排序与批量投影 | 最大差8.88e-16，标签变化0 |
| analog识别 | 复用批量曲线引擎、几何与局部证据 | 原识别快版点值和标签完全相同 |
| TabPFN残差 | 整剖面共享邻接、复用LOO-IDW／query基座、只解码所需中位数 | 8640格GPU预测逐位相同 |
| 公平IDW对照 | 共享查询邻接，仍调用原idw_value的一维归约 | 8640格逐位相同 |

TabPFN优化曾发现一个实质数值风险：直接矩阵归约产生约1e-16特征舍入变化，会拆开rank/quantile预处理的并列值，最终预测可改变约.07 log单位。最终实现保留原一维统计归约顺序，只共享几何和索引，消除了该变化；没有把数值扰动当质量增益。

真实完整均衡point路径也已从observed/query重算全部60状态，与旧主线最大差1.55e-15、标签差0；经验CDF独立全60核对最大差8.88e-16。正式benchmark再核对完整CUDA风险路径，不通过读取最终预测文件伪装成推理。

质量适配结果并非全部成功：

- 结构五成员拟合和可靠性门控未改善主基底，保留负结果，默认不替换。
- 均衡计数收缩：E从.592435降到.591791，约0.11%；H从.536947增到.546712，召回与误报均略降。是否使用应结合额外时延。
- Haar计数混合略降E和误报，却使厚度误差上升；厚度优先继续使用等价快版。偏差修正的召回46.92%伴随误报11.59%，不作为默认。
- 可靠性定位：点Recall从41.89%到42.38%、FPR从7.85%到7.81%，厚度不变；幅度重投影使E变差，弃作默认。
- 60%计数分位：Recall到48.77%、FPR到9.14%，H降到.526855m，属于显式高召回选项。

## 5. 正式速度测量：固定CPU核、同状态真实推理

硬件为Intel Core i7-14700KF、RTX4080。正式报告使用`benchmark_pinned`这**同一进程、同一逻辑CPU核2**的重测；验证并回读进程affinity mask=0x4，数值库单线程、CUDA边界同步。不臆断该逻辑核属于哪一种核心，也未锁CPU/GPU频率。

此前未固定核的批次中，同一旧IDW中位时延曾约109ms和55ms，属于明显计时混杂，因此没有混入本节，也不跨那些批次推导加速。旧数据保留为诊断记录。固定核复测首尾同一旧IDW中位56.105→56.557ms，变化0.80%；逐状态变化范围约−0.41%至+2.76%，波动明显缓解。

固定12状态为3fold×2场地×seed17×预算5/13；每状态预热1次、真实condition/predict重复3次。各路径36个测量样本；共26条测量路径，包含旧/新、基线和质量变体，并非26种新算法。数据读取、输入复制、准确度评价与外部校准拟合不计入在线推理；工厂与权重初始化另记。这里“完整系统”指点预测加原生风险模型链路，计时截止native概率输出；不包含产品I/O、应用最终Platt映射或界面展示。

### 5.1 同口径前后加速

{speed}

加速列是**先计算每个相同状态的旧/新中位时延比，再取12状态中位数**，所以不必等于表中两个总体中位数直接相除。p95来自36个异质热态样本，只描述本次样本，不是稳定尾延迟或最大时延保证。

### 5.2 当前实际使用路径的时延

{practical}

IDW也做了等价工程优化，避免把旧逐行Pandas开销当作复杂算法更高效的证据。GP时延包含按当前支持集拟合及原生分布推断；IDW只有点输出；后面几个完整系统包含CUDA经验风险头，能力、设备与范围均需一起看。

质量变体的时延是单独真实测量，不套用等价快版速度。localize／recall60与exact在不同顺序的几毫秒差别可能来自剩余运行波动，不据此声称质量适配本身加速。计数收缩的LOO适配成本明显更高，应与仅0.11%的MAE变化一起考虑。

### 5.3 算子速度不等于系统速度

{components}

joint和recognition算子得到的是在计时外现场计算的同一IDW基底，不读取冻结最终输出；该准备耗时单列。Haar纯point不包含独立风险头。**不能将约十毫秒的算子时延写成含TabPFN／图风险头的系统时延。** 完整均衡point路径可只在CPU运行；选择当前完整风险功能则必须计入其CUDA链路。

## 6. 初始化、内存与可复现边界

初始化记录分别给出factory构造、显式权重加载和总耗时。首个模型存在框架懒加载、OS文件缓存与CUDA上下文影响，样本不足以比较冷启动加速；主速度结论仅为统一预热后的新旧推理。

本次GPU路径的peak allocated中位约125.54MiB、全表最大约197MiB，没有以此宣称显存下降。GPU表中的peak allocated包含当次存活权重及张量，不等于驱动占用或reserved显存。RSS是整个Python进程工作集，含已导入库、历史模型和评价参考，不能拿来作模型专属内存排行榜。完整硬件、库版本、初始化和内存数字见性能CSV／protocol。

每次condition重新建立当前支持状态的必要上下文；缓存只复用几何/剖面中间结果，没有跨标签上下文复用最终预测。新API仍针对本项目完整24格剖面及源fold工件；工程等价不代表已完成任意外部场地部署。

## 7. 场地代价与当前选择

均衡快版保持v0.7全部优点和局限：Lower Hutt厚度误差.1328→.2539m，Marshland点召回22.79%→5.15%，风险召回也偏低。加速不会修复这些模型问题。

识别recall60在Marshland相对原识别分支改善：点Recall19.56%→30.00%，FPR2.60%→3.12%，H误差.4922→.4570m；但Lower Hutt和Metro Sports厚度变差。原点召回在4/4场地都升高，误报也在4/4都升高。它适合愿意承担复核误报成本的研究情景，不能称全面替代默认方案。

建议保留以下可选路径，而不是只按一个数字排唯一冠军：均衡默认快版；Haar厚度快版；识别保守localize；高召回recall60；TabPFN残差快组件。计数收缩和其他适配保留作取舍，不因极小MAE变化自动替换主线。

## 8. 交付与复现

源码在`src/cpt_budget/v8/`。报告对应的质量快照、数值等价与计时证据在[results/v8-engineering-20261014](results/v8-engineering-20261014/README.md)。

- [全部203模型准确度](results/v8-engineering-20261014/comparison/all_model_ranking.csv)
- [逐候选优化前后差值](results/v8-engineering-20261014/comparison/optimization_pairs.csv)
- [正式固定核速度表](results/v8-engineering-20261014/benchmark_pinned/summary.csv)
- [同状态配对加速](results/v8-engineering-20261014/benchmark_pinned/paired_summary.csv)
- [性能实验协议](results/v8-engineering-20261014/benchmark_pinned/protocol.json)
- [工程优化协议](docs/V8_ENGINEERING_PROTOCOL.md)

```powershell
.venv/Scripts/python.exe -m cpt_budget.v8.evaluate
.venv/Scripts/python.exe -m cpt_budget.v8.benchmark --cpu-core 2 --registry cpt_budget.v8.runtime:benchmark_specs --registry cpt_budget.v8.runtime:family_benchmark_specs --registry cpt_budget.v8.baselines:benchmark_specs --output runs/v8-engineering-20261014/benchmark_pinned
.venv/Scripts/python.exe -m cpt_budget.v8.report
```

前置数据、源模型、原冻结运行须按仓库流程准备；逐深度大预测及模型权重留本地runs，Git归档为精简证据。算法各模块可独立重跑，`evaluate`只汇总已完成候选，不负责训练原模型。

26项定向检查通过；另有完整60状态等价、12状态串行配对和原生风险参考检查。模型精度与运行时间均为当前机器和验证数据范围内的结果；反复使用验证集仍有选择乐观偏差，没有新增独立测试或经济收益证明。
'''
    path=PROJECT_ROOT/REPORT_NAME
    path.write_text(text,encoding='utf-8')
    return path

if __name__=='__main__':
    print(generate())
