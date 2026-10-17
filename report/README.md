# CA6001 课程技术报告（LaTeX）

封面 + 正文10页（课程上限为封面外10页），按课程模板的十个章节组织，参考文献为APA格式（biblatex-apa + biber）。

## 编译

需要 TeX 发行版（MiKTeX / TeX Live，含 `latexmk`、`biber`、`newtx`、`biblatex-apa`）。在本目录运行：

```powershell
latexmk          # 生成 CA6001_Adaptive_CPT_Report.pdf，中间文件在 build/
latexmk -c       # 清理中间文件
```

也可直接上传到 Overleaf：需同时上传 `../results/adaptive_v1/figures/*.pdf`，并把 `main.tex` 中的 `\graphicspath` 改为对应路径；编译器选 pdfLaTeX，参考文献后端为 Biber。

## 修改哪里

| 内容 | 位置 |
|---|---|
| 正文（每章一个文件） | `sections/00_cover.tex` … `sections/09_conclusion.tex` |
| 封面小组成员与分工表 | `sections/00_cover.tex`（提交前填写） |
| 参考文献 | `references.bib` |
| 版式、字体、宏（如 `\qc`、`\Bfull`、`\eur{}`） | `main.tex` |
| 表3（预测质量＋TabPFN消融）、表4（停止与经济结果） | 由 `make_tables.py` 从 `results/` 生成到 `tables/`，**不要手改** |
| 表1（数据层级）、表2（超参数） | 直接写在 `sections/03_dataset.tex`、`sections/05_methodology.tex` |
| 图1–7 | 由 `python -m cpt_budget.report_figures` 从结果CSV生成到 `results/adaptive_v1/figures/` |

结果更新后的完整流程（在项目根目录）：

```powershell
.venv/Scripts/python.exe -m cpt_budget.report_figures
.venv/Scripts/python.exe report/make_tables.py
cd report; latexmk
```

修改正文后请确认页数仍不超过11页（封面＋10页）。
