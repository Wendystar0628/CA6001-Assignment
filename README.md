# CA6001 · 精选CPT重建与预算研究

当前仅保留 **4个强弱基线＋2个TabPFN组合＋3个其他主方案＋3个讨论方案**，共12个入口。模型名单以[configs/catalog.json](configs/catalog.json)为准，旧探索只在Git历史中追溯。

- [当前实现状态](docs/IMPLEMENTATION_STATUS.md)
- [12方案性能与取舍](CA6001_CPT_当前方案对比.md)
- [运行说明](docs/RUN_GUIDE.md)
- [清理记录及本地剩余事项](docs/CLEANUP_DECISIONS.md)

```powershell
.venv/Scripts/python.exe -m cpt_budget models
.venv/Scripts/python.exe -m cpt_budget status
.venv/Scripts/python.exe -m cpt_budget predict --model balanced --point-only
.venv/Scripts/python.exe -m cpt_budget verify
```

沿用已有`.venv`即可；新环境按`requirements-lock.txt`及`pyproject.toml`安装。大权重位于本地`artifacts/models/`，不提交Git；缺失时`status`明确报错。数据获取见[Data/README.md](Data/README.md)。

当前代码和Git内结果/文档已精简，保留模型720项迁移对照及21项测试通过。**约1.92GiB未跟踪旧runs及缓存因自动安全审查拒绝删除，仍需本机手动清理**；保留范围及准确目录已列明，不将该部分记作已完成。

原始课程材料、[高分设计背景](CA6001_CPT_高分项目设计.md)、[原实现蓝图](CA6001_CPT_项目实现蓝图.md)、原始数据、虚拟环境及用户notebooks保留。原v0.8报告/训练/探索代码可用Git标签`v0.8-engineering`查看；未重写远程历史。

所有结果仍来自多轮使用的4个验证场地，不是独立泛化或节约补勘根数的证明。
