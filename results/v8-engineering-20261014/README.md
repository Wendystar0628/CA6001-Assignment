# v0.8 工程优化结果与证据

主报告：[CA6001 CPT 深度工程优化与性能对比报告](../../CA6001_CPT_深度工程优化与性能对比报告_v0.8.md)。

本轮优化结构组合、整曲线厚度决策、Haar、低阻力识别与TabPFN残差，并补充同公式IDW工程控制。准确度比较覆盖183个历史参照与20个新配置；IDW快控制逐值等价于原基线，单独保存验证与性能证据。

**正式速度结论仅使用`benchmark_pinned/`**：固定同一逻辑CPU核2、单线程、CUDA同步，26条路径、936次计时、312次预热、78次初始化。其余`benchmark/`、`benchmark_families/`和`benchmark_baselines/`是未固定核诊断，不混入主速度表。首尾重复的旧IDW中位时延只差0.80%，此前跨批次约两倍差异已明显缓解。

主要配对加速：IDW8.90×、TabPFN残差点链路4.76×、均衡完整point17.09×、均衡point＋原生风险6.76×、Haar完整链路4.14×、识别完整链路6.59×。等价快版保持预测结果；质量适配另列明确取舍，不将算子时延当完整系统时延。

- [准确度总表](comparison/all_model_ranking.csv)与[优化前后差值](comparison/optimization_pairs.csv)
- [正式速度总表](benchmark_pinned/summary.csv)与[同状态配对加速](benchmark_pinned/paired_summary.csv)
- [计时协议与硬件](benchmark_pinned/protocol.json)、[初始化](benchmark_pinned/initialization_summary.csv)、[输出等价](benchmark_pinned/paired_output_checks.csv)
- [完整系统原生风险参考核对](benchmark_pinned/native_reference_checks.csv)
- [研究范围摘要](run_summary.json)

各family目录保留实现说明、完整质量候选、等价性、适配记录及开发期诊断。诊断时延不用于正式排序。大权重与逐深度预测保留在本地`runs/v8-engineering-20261014/`，本目录为精简证据；源码及复现命令见主报告。

26项针对性检查通过；10对运行路径数值等价。所有准确度仍来自同一批被多轮使用的验证场地，没有新增独立测试；工程加速也不会消除原有场地识别缺陷。
