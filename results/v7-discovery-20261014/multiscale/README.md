# v7 垂向多尺度探索

本轮在评价前确定 5 个候选，将整根 24 点 CPT 剖面变换到固定 Haar 或 DCT 基。粗尺度系数采用空间核岭回归，细尺度采用更局部的空间聚合。没有训练新的源场地表示，也没有改动既有 FPCA、配准或 TabPFN。

## 机制与固定参数

- Haar 使用 3 级正交分解，24 系数为 `3 个粗系数 + 3/6/12 个由粗到细的细节系数`，无补零、无截断。
- DCT 使用正交 DCT-II，前 4 系数视为粗尺度，保留其余全部 20 系数。
- 粗尺度均使用已测剖面系数均值作为截距，中心化 RBF 核岭回归；ridge=0.1，核尺度为已测位置非零两两距离中位数。
- `haar_krr_nearest` 与 `dct_krr_nearest`：高频系数来自几何最近的一个完整已测剖面。
- `haar_krr_median` 与 `dct_krr_detail`：高频各系数采用最近 3 剖面、距离平方倒数加权中位数。
- `haar_multiband`：3/6/12 细节系数分别用距离倒数幂 1.5/2.5/4 聚合，越细的垂向结构使用越局部的水平邻域权重。

候选只接收已揭露 CPT 标签及查询几何；没有使用查询标签选择基、分解级数、频率边界、参数或邻居，没有支持集 LOO 调参。支持与评价集边界沿用共同冻结状态的 near-group 排除规则。所有输出仅为点预测，风险概率由主比较流程统一处理。

## 初步结果（未做统一风险校准）

每个候选均为 60 个冻结状态、8,640 个深度评价单元。预算按 5/7/9/11/13 积分，随后先对种子和重复场地折取均值，再对 4 个不同场地取均值。表中召回与误报为 `qc < 1 MPa` 点识别结果，不能当作校准风险头的结果。

| 方案 | log-MAE | 厚度 MAE / m | 点召回 | 点误报率 |
|---|---:|---:|---:|---:|
| IDW（既有参照） | 0.638586 | 0.630371 | 0.373767 | 0.072766 |
| Haar 多尺度距离权重 | 0.651374 | 0.557454 | 0.336780 | 0.061657 |
| Haar 粗 KRR＋细节加权中位数 | 0.675224 | 0.487630 | 0.404423 | 0.091431 |
| DCT 粗 KRR＋细节加权中位数 | 0.679214 | 0.526693 | 0.379388 | 0.081842 |
| Haar 粗 KRR＋最近细节 | 0.689833 | 0.472493 | 0.392422 | 0.104418 |
| DCT 粗 KRR＋最近细节 | 0.698075 | 0.513184 | 0.353671 | 0.094961 |

五个新候选连续预测误差均高于 IDW，但厚度误差均降低。Haar 最近细节的厚度误差最低，相对 IDW 降低约 25%；加权中位数在厚度、点召回、点误报之间更温和。五个候选均在 3/4 场地改善厚度，在 Lower Hutt 的厚度反而变差，不能写成所有场地一致改善。Tauranga 的厚度改善常伴随明显增加的误报。

因此本家族适合作为组合中的薄层结构来源，连续误差仍应由更强的低频/残差成员约束。这是探索线索，暂不宣称独立替代强基线。

## 解释边界

变换可逆只证明给定的原剖面信息没有被变换丢弃，并不证明可以预测未观测位置的每个薄层。最近剖面的细节既可能补回层结构，也会带入该剖面的噪声与位置不匹配。Haar 对深度平移敏感，DCT 在锐利边界附近可产生振铃。没有地质层位标签，也没有声称 0.25 m 以下的真实层界分辨率。

全部负结果保留在 `point_metrics.csv`、`point_site_metrics.csv`、`point_macro.csv`，完整预测在各候选文件夹。`method_protocol.json` 和 `contexts.json` 留有固定参数与已揭露位置记录。

## 执行与检查

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv/Scripts/python.exe -m cpt_budget.v7.multiscale
.venv/Scripts/python.exe -m pytest tests/test_v7_multiscale.py -q
```

单线程 CPU 完成全部实验。7 项定向检查通过，覆盖完整变换可逆性/能量保持、单薄层信息保留、已测坐标复现、各候选忽略查询标签和保持重排一致性，以及不完整剖面拒绝。

数学表示与 API 参考：[SciPy DCT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html)、[SciPy IDCT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idct.html)、[PyWavelets 多级 DWT 文档](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html)。这里的空间系数组合是本项目探索实现，不是对某篇完整地质模型的复现。
