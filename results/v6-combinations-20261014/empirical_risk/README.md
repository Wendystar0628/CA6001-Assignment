# v0.6：已测数据条件下的经验误差分布

本轮冻结并实现 **6 个候选**：3 个既有点预测基座 × 2 种经验误差加权。所有候选均完整输出 60 个冻结状态、每模型 8640 个深度单元。原始点预测、原始点标签及 `H_point` 保持不变；仅新增 `p_low_1`，之后由主代理按公共流程进行跨勘察组校准与综合评价。

## 固定机制

3 个点基座分别是 `residual_ft_crps_guarded_a050`、`v6_fpca_residual_sign_guard`、`v6_graph_residual_low_union`。代码从 `load_frozen()` 提供的无真值点列重建其已冻结公式，不读取旧 OOF 概率，也不依赖候选输出文件中的真值。

对当前已测 CPT，逐根排除同 location、同 near_group 和所有同坐标 CPT，用剩余已测剖面的 3NN、距离幂 2 的 IDW 预测相同深度，得到 `e = 已测 log(qc) − LOO-IDW`。每个查询单元的事件概率为：

`P(qc < 1 MPa) = Σ w_i Φ((-point − e_i) / 0.15)`。

其中 Φ 是标准正态 CDF。0.15 为冻结的 log(qc) 核平滑带宽，不是已经验证的测量噪声大小。两种权重为：

- **depth**：深度 Gaussian 核带宽 0.5 m，各已测剖面内部归一后，剖面间等权。
- **local**：同一深度核，再乘水平 Gaussian 核；水平长度为当前已测位置两两非零距离中位数。

模型 ID 为 `v6_empirical_{best|fpca_guard|graph_union}_{depth|local}`。全部设置在新候选评分前通过并行设计消息冻结。

## 已知边界

这些误差来自 **IDW**，将它们用于残差模型或组合模型需要“误差分布可近似迁移”的假设；它们不是各基座自己的校准误差。原点预测与所构造经验分布的中位数不一定一致，本实验明确保留原点预测作为决策输出，用分布估计事件风险，不强制二者相等。

每状态仅有 **5–13 根 CPT／5–13 个 near group**。深度单元高度相关，不能把 24×CPT 数量写成独立样本量。实际所有 LOO 参考均有 3 个独立位置；depth 模式的剖面权重有效样本数为 5–13，local 模式查询处最小有效剖面数约 **3.10**，显示局部尾部估计仍可能不稳定。完整记录见 `support_diagnostics.csv`。若没有独立 LOO 参考，代码报错，不以零残差伪装成已校准误差。

[Vovk 等 2017 原始论文](https://proceedings.mlr.press/v60/vovk17a.html)给出了特定 conformal predictive system 在 IID 条件下的有效性；[2019 的 split/cross-conformal 工作](https://arxiv.org/abs/1911.00941)也区分严格保证与经验表现。这里采用的是空间加权、核平滑、替代模型误差迁移的经验分布，没有运行其完整 conformal 构造，**不继承覆盖率保证，不称为 conformal 区间**。此引用用于说明概率分布构造与理论边界，而非宣称论文复现。

查询真值不进入误差计算、权重、带宽或点基座构造；无 outer-test 访问。公共外部风险校准与这里的原生经验概率需分开报告。

## 复现

```powershell
.venv/Scripts/python.exe -m cpt_budget.v6.empirical_risk
.venv/Scripts/python.exe -m pytest tests/test_v6_empirical_risk.py -q
```

3 项定向测试通过：整根 near-group 排除及输入保留；概率范围、随点预测的单调性、查询顺序不变性与真值列拒收；零误差对称分布及缺少独立 support 时明确失败。
