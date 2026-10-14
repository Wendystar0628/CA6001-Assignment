# CA6001 · CPT重建与预算研究

本项目研究有限勘探预算下的CPT曲线重建、低阻力层厚度预测与低阻力识别。当前提供 **4个强弱基线＋2个TabPFN组合＋3个其他主方案＋3个讨论方案**，共12个模型入口，名单见[configs/catalog.json](configs/catalog.json)。

- [当前实现状态](docs/IMPLEMENTATION_STATUS.md)
- [12方案性能与取舍](CA6001_CPT_当前方案对比.md)
- [运行说明](docs/RUN_GUIDE.md)
- [方案选择依据](docs/MODEL_SELECTION.md)

```powershell
.venv/Scripts/python.exe -m cpt_budget models
.venv/Scripts/python.exe -m cpt_budget status
.venv/Scripts/python.exe -m cpt_budget predict --model balanced --point-only
.venv/Scripts/python.exe -m cpt_budget verify
```

沿用已有`.venv`即可；新环境按`requirements-lock.txt`及`pyproject.toml`安装。大权重位于本地`artifacts/models/`，不提交Git；缺失时`status`明确报错。数据获取见[Data/README.md](Data/README.md)。

均衡方案的log-MAE为0.592435、厚度MAE为0.536947m；IDW强基线分别为0.638586和0.630371m。厚度方案达到0.487630m，高召回方案的点召回率为48.77%，对应点误报率9.14%。完整比较与取舍见上述方案对比文档。

项目背景见[高分设计](CA6001_CPT_高分项目设计.md)与[实现蓝图](CA6001_CPT_项目实现蓝图.md)。

所有结果仍来自多轮使用的4个验证场地，不是独立泛化或节约补勘根数的证明。
