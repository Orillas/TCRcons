#!/usr/bin/env python3
"""Generate notes.html for the Donor1 stress test (honest report).

Reads stress_results.tsv (single methods + balanced consensus) and
stress_consensus_sweep.tsv (threshold sweep), embeds the figure, and
writes an HTML report with the protocol, results, and interpretation.
"""
from pathlib import Path
from collections import defaultdict

import pandas as pd

OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test")
MAIN = OUT / "stress_results.tsv"
SWEEP = OUT / "stress_consensus_sweep.tsv"
HTML = OUT / "notes.html"

METHOD_ORDER = ["hd_baseline", "giana", "tcrmatch", "clustcr", "gliph2", "tcrdist3",
                "deeptcr", "consensus"]


def mean_table(df, mode_col="method"):
    keys = ["subset", "noise_frac", mode_col]
    metrics = ["retention_specific", "purity_epitope", "specific_purity", "retention_all"]
    g = df.groupby(keys, sort=False)[metrics].mean().reset_index()
    return g


def fmt_table(df, label_col, highlight=None):
    highlight = highlight or set()
    cols = ["subset", "noise_frac", label_col, "retention_specific",
            "purity_epitope", "specific_purity"]
    rows = ["<table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"]
    for _, r in df[cols].iterrows():
        cls = ' class="hl"' if r[label_col] in highlight else ""
        cells = (f"<td>{int(r['subset'])}</td><td>{r['noise_frac']:.2f}</td>"
                 f"<td>{r[label_col]}</td><td>{r['retention_specific']:.3f}</td>"
                 f"<td>{r['purity_epitope']:.3f}</td><td>{r['specific_purity']:.3f}</td>")
        rows.append(f"<tr{cls}>{cells}</tr>")
    rows.append("</table>")
    return "\n".join(rows)


def main():
    main_df = pd.read_csv(MAIN, sep="\t")
    main_tbl = mean_table(main_df, "method")
    main_tbl["order"] = main_tbl["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    main_tbl = main_tbl.sort_values(["subset", "order"])

    sweep_html = ""
    if SWEEP.exists():
        sw = pd.read_csv(SWEEP, sep="\t")
        sw_tbl = sw.groupby(["subset", "noise_frac", "mode"], sort=False)[
            ["retention_specific", "specific_purity", "purity_epitope"]].mean().reset_index()
        sweep_html = "<h3>Consensus threshold sweep (mean over reps)</h3>"
        sweep_html += fmt_table(sw_tbl, "mode", highlight={"balanced_t0.6"})

    subsets_done = sorted(main_df["subset"].unique())
    noise_by_sub = main_df.groupby("subset")["noise_frac"].first().to_dict()

    # consensus vs best-single purity per subset (the honest headline)
    headline = ["<h3>Consensus (balanced t0.3) vs best single — specific purity</h3><table>"
                "<tr><th>subset</th><th>noise</th><th>consensus</th><th>best single</th>"
                "<th>(which)</th></tr>"]
    singles = main_df[main_df["kind"] == "single"]
    cons = main_df[main_df["method"] == "consensus"]
    for s in subsets_done:
        nf = noise_by_sub[s]
        cval = cons[cons["subset"] == s]["specific_purity"].mean()
        sing = singles[singles["subset"] == s].groupby("method")["specific_purity"].mean()
        best_m = sing.idxmax(); best_v = sing.max()
        flag = "⚠" if cval < best_v else "✓"
        headline.append(f"<tr><td>{s}</td><td>{nf:.2f}</td><td>{cval:.3f}</td>"
                        f"<td>{best_v:.3f}</td><td>{best_m} {flag}</td></tr>")
    headline.append("</table>")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Donor1 noise stress test — tcrconsensus</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#222;line-height:1.5}}
h1{{font-size:1.4em}}h2{{font-size:1.15em;border-bottom:1px solid #ccc;padding-bottom:.2em;margin-top:1.5em}}
h3{{font-size:1.02em;margin-top:1.2em}}
table{{border-collapse:collapse;font-size:.9em;margin:.5em 0}}
th,td{{border:1px solid #ddd;padding:3px 8px;text-align:right}}
th{{background:#f4f4f4}}td:nth-child(3){{text-align:left}}
tr.hl{{background:#fff3cd;font-weight:bold}}
code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
.warn{{background:#fff3cd;border-left:4px solid #f0ad4e;padding:.6em 1em;margin:.8em 0}}
.note{{background:#eef;border-left:4px solid #36c;padding:.6em 1em;margin:.8em 0}}
img{{max-width:100%;border:1px solid #ccc;margin:1em 0}}
</style></head><body>
<h1>Donor1 噪声压力测试（i3-unit 10X, STRESS_TEST.md）</h1>

<div class="note"><b>协议</b>：固定 2876 条抗原特异性 CDR3β + 逐步掺入无关噪声 → 6 子集
(2876/7613/12350/17087/21824/26561，噪声分数 0→0.89)；每方法独立 3 次。
指标：<code>retention_specific</code>（特异性序列保留率, §3.1）、
<code>specific_purity</code>（特异性簇内特异性占比 = 1−噪声污染, §3.2）、
<code>purity_epitope</code>（epitope 同质性, 复现 i3-unit purity_function）。
已完成子集：{subsets_done}。</div>

<h2>1. 头条：consensus vs 最佳单方法（specific purity）</h2>
{"".join(headline)}
<p><b>诚实结论</b>：balanced 共识（阈值 0.3，≈3 方法同意）在含噪声子集上 specific purity
<b>低于</b>最佳单方法（⚠ 行）。噪声分数越高差距越大。</p>

<h2>2. 全方法明细（mean over reps）</h2>
<p>balanced 共识行高亮。</p>
{fmt_table(main_tbl, "method", highlight={"consensus"})}

<h2>3. 阈值扫描</h2>
<p>提高共识阈值（要求更多方法同意）单调提升噪声下纯度，但即便 t0.6 仍低于最佳单方法。</p>
{sweep_html}

<div class="warn"><b>根因（诚实诊断）</b>：共识的<b>连通分量</b>步骤在重噪声下经
<b>噪声桥</b>把特异性簇链式合并 → 放大污染（CC chaining）。这<b>不是</b> tcrconsensus
的胜利场景；它诊断了 <b>F3 置换 FDR 自适应阈值</b>与 <b>Tier-2 符号排斥</b>（切断噪声桥）
要解决的弱点。本测试数据为 CDR3β-only（无 V/J、无配对链），也削弱了 consensus 赖以互补
的多信号优势（对比：带 V/J+配对链的 v3_cd8 benchmark 上 consensus 在 BCubed 胜出 +26%）。</div>

<h2>4. 解读与对论文的意义</h2>
<ul>
<li>低噪声（subset 1, 0%）：所有方法 purity=1.000，consensus 亦然。</li>
<li>中高噪声（62–89%）：CC 共识纯度退化<b>快于</b>内部自带噪声抑制的方法（deeptcr、tcrdist3）。</li>
<li>这是固定阈值 CC 共识的<b>已知局限</b>，正是项目 F3（FDR 阈值）+ Tier-2（符号排斥）创新的动机。</li>
<li>建议：噪声场景应启用 conservative/FDR 模式，而非 balanced 默认（scenario-adaptive）。</li>
</ul>

<h2>5. 图</h2>
<img src="stress_figure.png" alt="retention and purity vs noise">

<p><small>生成自 stress_results.tsv / stress_consensus_sweep.tsv。reps=3，确定性方法
(hd/giana/tcrmatch/gliph2/tcrdist3) 复用 rep1，stochastic (clustcr/deeptcr) 跑 3 次。
cluster size≥2 计入。</small></p>
</body></html>
"""
    HTML.write_text(html, encoding="utf-8")
    print(f"wrote {HTML} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
