# New Lock Terneuzen 经济案例：口径、来源与参数

更新：2026-10-17。配置为 [economics_case.yaml](../configs/economics_case.yaml)，计算入口为 `cpt_budget.adaptive_economics.evaluate_economic_case`。

本案例把**实验测得的补勘数量和审计结论**与**公开工程背景、明确的经济情景参数**相连。金额属于历史项目背景下的反事实估算，不是 New Lock 的实际合同节约。执行方案中的“50→30 根、€165,000”演算没有作为输入。

## 主表采用什么口径

主表采用业主财务净收益，币种 EUR，情景价格水平 2025。可避免的勘察支出、项目现场日费用和运营方增量净现金流分别记账。港口使用者船时效益放入单独的 `social_user` 表，不加到业主净收益；该表也不自称完整社会成本效益评价。

金额统一采用不另加税费的情景口径，未核定项目税额或税收可抵扣性。若将同一配置应用到其他场地，得到的是统一荷兰成本场景下的算法比较，不能称为那些场地各自的当地报价或历史项目收益。

主情景 `α=1`：勘察净提前的日历日全部传递至项目完工。`α=0.5/0.75/1` 为同一轨迹的敏感性。它们是执行方案指定的工期假设，并非由该工程历史施工网络验证的因果关系。成本与工期估算可分别参考 [GAO 成本指南](https://www.gao.gov/products/gao-20-195g) 和 [GAO 工期指南](https://www.gao.gov/products/gao-16-89g)。

## 已核实的一手背景与价格线索

| 来源 | 核实内容 | 在本案例中的用途 |
|---|---|---|
| [ISSMGE TC304 CPT 数据页](https://tc304.org/cpt_databases.html) | NL-CPTU/98/2km² 对应 New Lock Terneuzen；原始深度约 3–70m | 确认数据背景。98 根是原始库数，不能代替清洗后候选数或每次回放的 `n_full`。 |
| [Rijkswaterstaat，2025-09-01](https://www.rijkswaterstaat.nl/en/news/archive/2025/09/new-lock-terneuzen-fully-operational-from-1-august-2025) | 2025-08-01 全面运营；增加船闸通航能力 | 区分正式运行与仪式日期，支持“提前服务”这一情景的工程背景；没有据此推算每日现金流。 |
| [MOS Grondmechanica 报价 A2501409，2025-05-01，PDF 第 1–3 页](https://www.spieghelhoek.nl/wp-content/uploads/2025/08/offerte-Mos-Realisatie-appartementencomplex-De-Spiegelhoek-010525-003.pdf) | Bussum 的住宅项目：7 根最多 15m 的 CPT、其他现场工作和咨询合计 €2,850；额外阶段现场进场 €560；加深附加价 €5/m；CPT 队等待 €225/h；顾问 €150/h | 荷兰公开、可直接读到的服务方报价。€560 为返场类比锚点；€150/h 为假设 AI 复核工时的锚点。地域和任务差异保留。总包价不能除以 7 当作 CPT 单价；€5/m 是加深附加价，不能当全部贯入单价。 |
| [荷兰 KiM，2023-03-30](https://www.kimnet.nl/documenten/2023/03/30/kostenkengetallen-voor-het-goederenvervoer) | 官方成本库区分运输方式和船型，并单列装卸及等待小时成本；公开更新覆盖 2019–2021 数据 | 为另列的船时分析提供适用方法与可进一步细化的来源。尚无本案例代表船型与流量组合，未把某一船型成本冒充 New Lock 全船队平均值。 |

访问日期均为 2026-10-17。未找到公开的 New Lock 补勘 CPT 合同单价、逐日可避免现场费用、运营方增量净现金流或实际 CPT 设备生产率。因此下面这些量明确设为情景假设。引用真实工程名称并不会让假设自动成为历史事实。

## 主情景输入和单因素范围

| 参数 | 主值 | 下界—上界 | 证据性质 |
|---|---:|---:|---|
| 全包 CPT 单价，EUR/根 | 750 | 350—1,500 | 明确假设，含设备、人工、定位、贯入与常规单孔处理；不含共同及新增进场。 |
| 完整计划贯入深度，m/根 | 20 | 10—30 | 明确假设。用于描述完整补勘孔，绝非用 2–8m 的 6m 研究区间代替完整孔深。全包计价时不再乘一遍米价。 |
| 进场，EUR/次 | 560 | 300—1,500 | MOS 荷兰项目公开数值的情景移用，不是 Terneuzen 合同价。共同一次进场相消。 |
| 作业效率，根/设备/工作日 | 5 | 3—8 | 可达陆域补勘的明确假设，未实测。 |
| 并行设备 | 1 | 1—1 | 明确假设。 |
| 批次间额外等待，日历日 | 0.25 | 0—1 | 明确假设，逐批释放决定的额外耗时。 |
| 另增复核作业日，工作日/批 | 0 | 0—0.25 | 主情景复核耗时已包含在批次等待中；仅有额外耗时才另计。 |
| 可避免项目费用，EUR/日历日 | 5,000 | 0—15,000 | 明确假设；不含 CPT 队、AI 复核、运营费用及融资，不由总投资除工期反推。 |
| 运营方增量净现金流，EUR/日历日 | 3,000 | 0—10,000 | 明确假设，有需求且在共同截止日前增加服务日；无业主现金流实测。包含零收益敏感性。 |
| AI 单项目增量，EUR | 600 | 300—2,000 | 假设 4 个顾问小时 × €150/h 的类比预算，合计涵盖部署、计算、复核与额外核验，不能再次重复扣费。 |
| 错误早停后修复等待，日历日 | 2 | 0—5 | 明确假设，另加补满剩余 CPT 和一次返场。 |

完整参数表含单位、地区、价格年份、来源类型、URL、范围和重叠说明，可由 `parameter_table(config)` 导出；`sensitivity` 表包含 `α` 与上述主要经济量的逐项单因素变化。它不会把多个不相容的有利极值同时组合成一个主情景。

另提供标为 `alpha_opening_grid` 的 3×3 展示网格：`α=0.5/0.75/1` × 运营方日净收益 `€0/3,000/10,000`。每格复用同一条真实回放的根数、审计与修复结果，仅改变两个明确的经济假设；不用于选择模型、策略或停止门槛。

单独的使用者情景暂取 20 船/日、每船减少等待 0.5h、€200/船小时，三者均为明确假设，故每日使用者效益 €2,000 也是情景值。没有把港口 GDP、全部货值、总港口吞吐或 KiM 的某一船型成本当作本项目已实现的效益。

## 数量与质量输入

`decisions` 每条回放轨迹一行，保留 `run_id/fold_id/role/site_id/seed/model_id/policy_id`。必填 `n_full/n_stop/audit_pass`，并由回放提供 `n_initial/n_batches`；`batch_sizes` 可为包含初始批次的 JSON 数组。`n_stop` 计入初始孔，不能只计新增孔。

- `B_full`：传统完整计划，完成本条轨迹全部计划位置；相对自己收益为零。
- `B_seq`：固定顺序逐步补勘并执行同一停止规则。
- `P_adapt`：AI 自适应选点并执行停止规则。

`n_full` 和 `n_stop` 必须来自本次实际回放。固定隐藏审计 CPT 不参加在线候选；它们如作为两方案共同已存在的核验资源则相消。若真实部署需要额外新打审计孔，应把这些孔加入各方案实际数量和额外费用，不能假设现场核验免费。

仅在**本次补勘点确实服务于已验证的浅层任务，能够被替代**这一情景范围内，才把整孔计为可省。用于深层设计、法定要求或其他未验证用途的点是两方案共同必测。默认输出的 `scope_note` 明示浅层任务替代假设；传入 `scope_replaceable=False` 时即使审计通过，也补满全部计划，不给整孔节约。

统一 20m 为本阶段情景值；如取得实际补勘计划，传入 `full_depths_m` 与所选点对应的 `measured_depths_m`。若切换到 `itemized_per_metre`，只收定位、完整贯入米数、处理和进场，不再加全包孔价。此时孔深差异明显时还需要独立校核现场效率；当前主情景采用统一孔深和每根效率。

## 工期、费用与错误早停

`field_workdays = ceil(n / (CPT_per_rig_workday × rigs))`。主情景为设备连续驻场，多个决策批次的贯入工作合并取整，批次间额外等待单列；批次不自动等同于一次返场。工作日按 `7/5` 平均换算日历日，已经用日历日定义的等待不再乘 `7/5`。这是平均换算，并非精确恢复历史周末或节假日。

`ΔT_project = α × (T_full,calendar − T_method,calendar)`；负数保留为延误。主表：

`B_net = (C_full − C_method) + ΔT_project × (c_project,day + b_owner,net,day) − C_AI,incremental`。

提前服务按共同评价截止日内增加服务天数计算。若只是固定经营寿命整体向前平移，应改用两条现金流的折现差，不能继续沿用此增量服务日公式又加一整段经营寿命。本版本没有另加折现现金流、融资收益或使用者效益。

若 `audit_pass=False` 或审计结论缺失，而 `n_stop<n_full`：保留 `claimed_saved_cpt` 说明当时声称的根数，但经济修复情景补满剩余 CPT，使 `saved_cpt=0`；另收一次返场，加入修复等待，补测施工单独向上取整。它描述“后来发现问题并修复”的反事实成本，并非声称实验真实实施了返场。`net_benefit`、工期变化与使用者效益均保留负数。没有把错误早停继续计作成功节约。

## 引用信息（APA 格式基础）

- ISSMGE Technical Committee 304. (n.d.). *CPT databases*. https://tc304.org/cpt_databases.html
- Rijkswaterstaat. (2025, September 1). *New Lock Terneuzen fully operational from 1 August 2025*. https://www.rijkswaterstaat.nl/en/news/archive/2025/09/new-lock-terneuzen-fully-operational-from-1-august-2025
- MOS Grondmechanica B.V. (2025, May 1). *Offerte A2501409: Realisatie appartementencomplex De Spiegelhoek/Bussum/2501236* [Quotation, pp. 1–3]. https://www.spieghelhoek.nl/wp-content/uploads/2025/08/offerte-Mos-Realisatie-appartementencomplex-De-Spiegelhoek-010525-003.pdf
- Kennisinstituut voor Mobiliteitsbeleid. (2023, March 30). *Kostenkengetallen voor het goederenvervoer*. Ministry of Infrastructure and Water Management. https://www.kimnet.nl/documenten/2023/03/30/kostenkengetallen-voor-het-goederenvervoer
- U.S. Government Accountability Office. (2015). *GAO schedule assessment guide: Best practices for project schedules* (GAO-16-89G). https://www.gao.gov/products/gao-16-89g
- U.S. Government Accountability Office. (2020). *Cost estimating and assessment guide: Best practices for developing and managing program costs* (GAO-20-195G). https://www.gao.gov/products/gao-20-195g
