# CPT 原始输入与复现

仓库保留筛选清单和解析程序，不上传原始数据 ZIP、数值 NPZ 或生成的 Parquet。复现需要先取得原始包，再按已登记的源文件恢复数值输入。

## 所需文件

| 文件 | 获取方式与作用 |
|---|---|
| `cpt_screened_spatial_profiles.json` | 仓库内的 503 根候选清单；`file` 精确定位 ZIP 成员，`array_key` 精确连接数值数组。不要重新编号。 |
| `downloads/Global_CPT1196.zip` | 从 [TC304 官方 Global-CPT/3/1196 原始包](https://tc304.org/attachments/database/cpt/global_cpt/Global_CPT_3_1196.zip) 下载，保存为此文件名。原下载名为 `Global_CPT_3_1196.zip`。 |
| `cpt_screened_depth_qc.npz` | 用下面的恢复命令生成。已有此文件时，命令直接保留，不覆盖。 |

数据入口：[ISSMGE TC304 CPT databases](https://tc304.org/cpt_databases.html)。现有下载记录为 `downloads/geotech_download_manifest.json`；官方原始 ZIP 的本地记录大小为 49,638,862 字节。主任务只需要这一份原始包；New Lock 已在包内，不需要追加 Terneuzen 独立包，也不需要 Delft 数据。

## 在项目根目录运行

先按项目主 README 安装依赖，然后运行：

```powershell
python -m cpt_budget restore-inputs --config configs/main.yaml
python -m cpt_budget prepare --config configs/main.yaml
python -m cpt_budget split --config configs/main.yaml
```

`restore-inputs` 仅解析清单指定的 503 个 `.xlsx` / `.xls` 文件：取记录深度和 qc，保留有限、深度非负、qc 正值的样本，按深度稳定排序，在重复深度保留首条有效记录，转为 `float32` 后按原 `array_key` 写入压缩 NPZ。这与已有筛选审计程序的数值解析口径一致。它不重新筛选整库、不修改清单，也不改动原始 ZIP；缺少指定成员或单位无法确认为 qc/MPa 时会明确报错。

`prepare` 再执行同坐标代表选择和 2–8 m 的 24 个中点网格检查；503 根候选不是最终可用数量。当前得到 495 个不同原始位置（503 根中 8 根为同坐标的非代表剖面）、492 个完整可用位置；3 根候选因主窗口内大采样缺口被排除。正式数量以本次运行的 `data_processed_v2/preparation.json` 为准。

如配置另一个原始包位置，可设置 `data.source_archive`，路径相对于 `project.input_dir`；默认是 `downloads/Global_CPT1196.zip`。

## 数据来源与许可记录

原包 `readme.txt` 要求引用：

> Ching, J., Uzielli, M., Phoon, K. K., and Xu, X. J. (2023). Characterization of autocovariance parameters of detrended cone tip resistance from a global CPT database. *ASCE Journal of Geotechnical and Geoenvironmental Engineering*, 149(10), 04023090.

论文：[doi:10.1061/JGGEFK.GTENG-11214](https://doi.org/10.1061/JGGEFK.GTENG-11214)。当前保存的原包 README 给出引用要求，未包含明确的数据再分发许可证。公开下载和论文引用不能替代数据许可；本项目记录的是已取得文件中的许可信息，未声称已获开放再分发授权，因此原始包及其数值副本不随代码上传。后续使用或对外共享数据应查阅来源方当时的条款。

## 尚未核实的全球坐标

27 个场地可从原包位置工作簿的 `X(degree)` / `Y(degree)` 补取经纬度；原表未声明地理基准。以下 4 个场地只有米制坐标，没有经纬度或可用的 CRS 说明，不能根据坐标数值猜测投影，也不能把不同场地的局部原点相比较：

- **Avonside、Baytown、Hollywood**：仅保留原始独立场地文件夹作为临时项目分组，跨场地独立性未核实。
- **New Lock**：官方 Terneuzen 下载路径及本地下载清单可识别其项目身份，但现有工作簿仍不能确定全球坐标和 CRS。

`configs/sites.yaml` 与生成的场地表对这 4 组保留 `grouping_verified=false`；每折划分会列出涉及的未核实组。默认可用于带限制说明的 pilot，不能声称它们已通过 250 m 的全球邻近关系核验。需要严格排除时，显式设置 `split.exclude_unverified_groups: true`；完整勘察组会一起排除。

其中 Hollywood 与 New Lock 也是 v0.1 后未再参与比较的 4 个外层测试候选场地中的两个（另两个为 Chch Stadium、Wellington Airport），未来若在这些场地做独立测试，其独立性同样受上述分组未核实的限制。当前成绩只来自 4 个验证场地（Lower Hutt、Tauranga、Chch Metro Sports、Marshland），见[当前方案对比](../CA6001_CPT_当前方案对比.md#结论边界)。
