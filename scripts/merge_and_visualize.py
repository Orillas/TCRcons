#!/usr/bin/env python3
"""合并三组实验结果并生成可视化。

三组:
1. Improved: CC + Empirical + merge 0.6  (from reproducibility.log)
2. Control 1: Leiden + Equal + merge 0.4   (from reproducibility_old.log / reproducibility_results.json)
3. Control 2: CC + Equal + merge 0.6       (from reproducibility_ctrl2.log / ctrl2_cc_equal_results.json)
"""
import json
import sys
import re
import os

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy import stats as sp_stats

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

LOG_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility")
FIG_DIR = Path("/home/jilin/DeepTCR/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============ 提取结果 ============
def parse_log_results(log_path):
    """从日志文件提取 >>> ARI=... 结果。"""
    results = []
    pattern = r'ARI=([0-9.]+),\s*Purity=([0-9.]+),\s*NMI=([0-9.]+),\s*Ret(?:ention)?=([0-9.]+),\s*Clusters=(\d+)'
    current_seed = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'seed=(\d+)', line)
            if m:
                current_seed = int(m.group(1))
            m = re.search(pattern, line)
            if m and current_seed:
                results.append({
                    "seed": current_seed,
                    "ari": float(m.group(1)),
                    "purity": float(m.group(2)),
                    "nmi": float(m.group(3)),
                    "retention": float(m.group(4)),
                    "n_clusters": int(m.group(5)),
                })
                current_seed = None
    return results


print("=" * 70, flush=True)
print("合并Three-Group Results Merge", flush=True)
print("=" * 70, flush=True)

# Improved (从日志解析)
new_log = LOG_DIR / "reproducibility.log"
new_data = parse_log_results(new_log) if new_log.exists() else []
for r in new_data:
    r["config"] = "new_cc_empirical"
print(f"Improved (CC+Empirical+merge0.6): {len(new_data)} runs", flush=True)

# Control 1 (从 JSON)
old_json = LOG_DIR / "reproducibility_results.json"
old_data = []
if old_json.exists():
    with open(old_json) as f:
        d = json.load(f)
    old_data = [r for r in d["results"] if r["config"] == "old_leiden_equal"]
print(f"Control 1 (Leiden+Equal+merge0.4): {len(old_data)} runs", flush=True)

# Control 2 (从 JSON)
ctrl2_json = LOG_DIR / "ctrl2_cc_equal_results.json"
ctrl2_data = []
if ctrl2_json.exists():
    with open(ctrl2_json) as f:
        d = json.load(f)
    ctrl2_data = d.get("results", [])
    for r in ctrl2_data:
        r["config"] = "cc_equal_weights"
print(f"Control 2 (CC+Equal+merge0.6): {len(ctrl2_data)} runs", flush=True)

if not new_data or not old_data or not ctrl2_data:
    print("ERROR: 数据不完整!", flush=True)
    sys.exit(1)

# 保存合并结果
all_results = new_data + old_data + ctrl2_data
merged_path = LOG_DIR / "merged_three_groups.json"
with open(merged_path, "w") as f:
    json.dump({
        "meta": {
            "groups": {
                "improved": "CC + empirical_weights + merge 0.6",
                "control1": "Leiden + equal_weights + merge 0.4",
                "control2": "CC + equal_weights + merge 0.6",
            },
            "seeds": [42, 123, 456, 789, 2024],
        },
        "results": all_results,
    }, f, indent=2)
print(f"合并结果: {merged_path}", flush=True)

# ============ 配色 ============
COLORS = {
    "new": "#2171b5",       # 蓝 - Improved
    "old": "#cb181d",       # 红 - Control 1 (Leiden)
    "ctrl2": "#238b45",     # 绿 - Control 2 (CCEqual)
}
COLORS_LIGHT = {
    "new": "#6baed6",
    "old": "#fb6a4a",
    "ctrl2": "#74c476",
}

GROUPS = [
    ("new_cc_empirical", "Improved\n(CC+Empirical)", "new"),
    ("cc_equal_weights", "Control 2\n(CC+Equal)", "ctrl2"),
    ("old_leiden_equal", "Control 1\n(Leiden+Equal)", "old"),
]

group_data = {}
for config, label, key in GROUPS:
    group_data[config] = [r for r in all_results if r["config"] == config]

# ============ Figure 1: 三组 ARI Box Plot ============
fig, ax = plt.subplots(figsize=(7, 5.5))

positions = [1, 2, 3]
box_data = []
box_colors = []
box_edge_colors = []
box_labels = []
scatter_x = []
scatter_y = []
scatter_c = []

for i, (config, label, key) in enumerate(GROUPS):
    data = group_data[config]
    aris = [r["ari"] for r in data]
    box_data.append(aris)
    box_colors.append(COLORS_LIGHT[key])
    box_edge_colors.append(COLORS[key])
    box_labels.append(label)
    # scatter
    jitter = np.random.uniform(-0.06, 0.06, len(aris))
    for j, a in enumerate(aris):
        scatter_x.append(i + 1 + jitter[j])
        scatter_y.append(a)
        scatter_c.append(COLORS[key])

bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.45,
                showmeans=True, meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 5})
for i, (patch, ec, fc) in enumerate(zip(bp["boxes"], box_edge_colors, box_colors)):
    patch.set_facecolor(fc)
    patch.set_edgecolor(ec)

ax.scatter(scatter_x, scatter_y, c=scatter_c, s=45, zorder=5, alpha=0.8, edgecolors="white", linewidths=0.5)

ax.set_xticklabels(box_labels)
ax.set_ylabel("Adjusted Rand Index (ARI)")
ax.set_title("Reproducibility: Three-Group ARI Comparison (n=5 seeds)")
ax.grid(axis="y", alpha=0.3)

# 均值标注
for i, (config, label, key) in enumerate(GROUPS):
    aris = [r["ari"] for r in group_data[config]]
    ax.text(i + 1, max(aris) + 0.015, f"μ={np.mean(aris):.3f}",
            ha="center", color=COLORS[key], fontweight="bold", fontsize=11)

# 提升箭头
ari_new = np.mean([r["ari"] for r in group_data["new_cc_empirical"]])
ari_old = np.mean([r["ari"] for r in group_data["old_leiden_equal"]])
pct = (ari_new - ari_old) / ari_old * 100
ax.annotate(f"+{pct:.0f}%", xy=(2, (ari_new + ari_old) / 2),
            fontsize=14, fontweight="bold", color="#2ca02c", ha="center")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_ari_three_groups.png")
print(f"Saved: fig1_ari_three_groups.png", flush=True)
plt.close()

# ============ Figure 2: 多指标对比 ============
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

metrics = [("ari", "ARI"), ("purity", "Purity"), ("nmi", "NMI"), ("retention", "Retention")]

for ax, (key, ylabel) in zip(axes, metrics):
    x = np.arange(len(GROUPS))
    means = [np.mean([r[key] for r in group_data[c]]) for c, _, _ in GROUPS]
    stds = [np.std([r[key] for r in group_data[c]]) for c, _, _ in GROUPS]
    colors = [COLORS_LIGHT[k] for _, _, k in GROUPS]
    edges = [COLORS[k] for _, _, k in GROUPS]

    bars = ax.bar(x, means, yerr=stds, width=0.55, color=colors, edgecolor=edges,
                  linewidth=1.5, capsize=5, error_kw={"linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for _, l, _ in GROUPS], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.008, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Three-Group Multi-Metric Comparison (n=5 seeds)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_multi_metrics_three_groups.png")
print(f"Saved: fig2_multi_metrics_three_groups.png", flush=True)
plt.close()

# ============ Figure 3: 配对散点图 (Improved vs 两个对照) ============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

seeds = [42, 123, 456, 789, 2024]

# Improved vs Control 1
ax = axes[0]
for s in seeds:
    n = next(r for r in group_data["new_cc_empirical"] if r["seed"] == s)
    o = next(r for r in group_data["old_leiden_equal"] if r["seed"] == s)
    ax.scatter(o["ari"], n["ari"], color=COLORS["new"], s=80, zorder=5, edgecolors="white")
    ax.text(o["ari"] + 0.003, n["ari"] + 0.005, f"s={s}", fontsize=8, color="gray")
    ax.plot([o["ari"], o["ari"]], [o["ari"], n["ari"]], color="#2ca02c", alpha=0.4, linewidth=1, linestyle="--")

ari_new_arr = [r["ari"] for r in group_data["new_cc_empirical"]]
ari_old_arr = [r["ari"] for r in group_data["old_leiden_equal"]]
lim = [min(min(ari_old_arr), min(ari_new_arr)) - 0.03, max(max(ari_old_arr), max(ari_new_arr)) + 0.03]
ax.plot(lim, lim, "k--", alpha=0.3, label="No change")
ax.set_xlabel("ARI - Control 1 (Leiden+Equal)")
ax.set_ylabel("ARI - Improved (CC+Empirical)")
ax.set_title("Improved vs Control 1 (Leiden)")
ax.legend()
ax.grid(alpha=0.3)

# Improved vs Control 2
ax = axes[1]
for s in seeds:
    n = next(r for r in group_data["new_cc_empirical"] if r["seed"] == s)
    c = next(r for r in group_data["cc_equal_weights"] if r["seed"] == s)
    ax.scatter(c["ari"], n["ari"], color=COLORS["ctrl2"], s=80, zorder=5, edgecolors="white")
    ax.text(c["ari"] + 0.003, n["ari"] + 0.005, f"s={s}", fontsize=8, color="gray")

ari_ctrl2_arr = [r["ari"] for r in group_data["cc_equal_weights"]]
lim = [min(min(ari_ctrl2_arr), min(ari_new_arr)) - 0.03, max(max(ari_ctrl2_arr), max(ari_new_arr)) + 0.03]
ax.plot(lim, lim, "k--", alpha=0.3, label="No change")
ax.set_xlabel("ARI - Control 2 (CC+Equal)")
ax.set_ylabel("ARI - Improved (CC+Empirical)")
ax.set_title("Improved vs Control 2 (CC+Equal)")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_paired_scatter.png")
print(f"Saved: fig3_paired_scatter.png", flush=True)
plt.close()

# ============ Figure 4: 逐方法 ARI 稳定性 ============
all_methods = sorted(set(m for r in all_results for m in r.get("per_method_ari", {})))

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(all_methods))
width = 0.25

for i, (config, label, key) in enumerate(GROUPS):
    means = []
    stds = []
    for m in all_methods:
        vals = [r["per_method_ari"].get(m, 0) for r in group_data[config] if m in r.get("per_method_ari", {})]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)
    ax.bar(x + i * width, means, width, yerr=stds, label=label.replace("\n", " "),
           color=COLORS_LIGHT[key], edgecolor=COLORS[key], capsize=3)

# 共识 ARI 虚线
for config, label, key in GROUPS:
    aris = [r["ari"] for r in group_data[config]]
    ax.axhline(y=np.mean(aris), color=COLORS[key], linestyle="--", alpha=0.4, linewidth=1)

ax.set_xticks(x + width)
ax.set_xticklabels(all_methods, rotation=30, ha="right")
ax.set_ylabel("ARI")
ax.set_title("Per-Method ARI Stability Across 5 Seeds")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_per_method_ari.png")
print(f"Saved: fig4_per_method_ari.png", flush=True)
plt.close()

# ============ Figure 5: Summary Table ============
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")

col_labels = ["Metric", "Control 1\n(Leiden+Equal)", "Control 2\n(CC+Equal)",
              "Improved\n(CC+Empirical)", "Δ vs Ctrl1", "Δ vs Ctrl2", "p (vs Ctrl1)"]
row_data = []

for key, label in [("ari", "ARI"), ("purity", "Purity"), ("nmi", "NMI"), ("retention", "Retention")]:
    vo = [r[key] for r in group_data["old_leiden_equal"]]
    vc = [r[key] for r in group_data["cc_equal_weights"]]
    vn = [r[key] for r in group_data["new_cc_empirical"]]
    mo, mc, mn = np.mean(vo), np.mean(vc), np.mean(vn)
    d1 = mn - mo
    d2 = mn - mc
    try:
        _, p1 = sp_stats.ttest_ind(vn, vo)
    except:
        p1 = 1.0
    row_data.append([
        label,
        f"{mo:.4f}±{np.std(vo):.3f}",
        f"{mc:.4f}±{np.std(vc):.3f}",
        f"{mn:.4f}±{np.std(vn):.3f}",
        f"+{d1:.4f} ({d1/mo*100:+.1f}%)",
        f"+{d2:.4f} ({d2/mc*100:+.1f}%)" if mc > 0 else "N/A",
        f"{p1:.4f}" if p1 >= 0.001 else "<0.001",
    ])

table = ax.table(cellText=row_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colColours=["#d9e2f3"] * len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)
for i in range(len(row_data)):
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        if i % 2 == 0:
            cell.set_facecolor("#f0f0f0")

ax.set_title("Three-Group Reproducibility Summary (5 seeds × 3 configs)", fontsize=13, pad=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_summary_table.png")
print(f"Saved: fig5_summary_table.png", flush=True)
plt.close()

# ============ Figure 6: 每种子折线图 ============
fig, ax = plt.subplots(figsize=(8, 5))

for config, label, key in GROUPS:
    aris_by_seed = {}
    for r in group_data[config]:
        aris_by_seed[r["seed"]] = r["ari"]
    sorted_seeds = sorted(aris_by_seed.keys())
    ax.plot(sorted_seeds, [aris_by_seed[s] for s in sorted_seeds],
            marker="o", color=COLORS[key], label=label.replace("\n", " "),
            linewidth=2, markersize=8)

ax.set_xlabel("Random Seed")
ax.set_ylabel("ARI")
ax.set_title("ARI Across Random Seeds (Reproducibility)")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(seeds)
ax.set_xticklabels([str(s) for s in seeds])

fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_seed_stability.png")
print(f"Saved: fig6_seed_stability.png", flush=True)
plt.close()

# ============ 打印最终汇总 ============
print(f"\n{'=' * 70}", flush=True)
print(f"Final Summary", flush=True)
print(f"{'=' * 70}", flush=True)

for config, label, key in GROUPS:
    data = group_data[config]
    aris = [r["ari"] for r in data]
    cv = np.std(aris) / np.mean(aris) * 100
    print(f"\n  {label.replace(chr(10), ' ')} (n={len(data)}):", flush=True)
    print(f"    ARI: {np.mean(aris):.4f} ± {np.std(aris):.4f}  CV={cv:.1f}%", flush=True)
    print(f"    [{min(aris):.4f} – {max(aris):.4f}]", flush=True)

# 统计检验
ari_n = [r["ari"] for r in group_data["new_cc_empirical"]]
ari_o = [r["ari"] for r in group_data["old_leiden_equal"]]
ari_c = [r["ari"] for r in group_data["cc_equal_weights"]]

t1, p1 = sp_stats.ttest_ind(ari_n, ari_o)
t2, p2 = sp_stats.ttest_ind(ari_n, ari_c)
print(f"\n  Improved vs Control 1: t={t1:.3f}, p={p1:.6f}", flush=True)
print(f"  Improved vs Control 2: t={t2:.3f}, p={p2:.6f}", flush=True)

try:
    w1, wp1 = sp_stats.wilcoxon(ari_n, ari_o, alternative="greater")
    print(f"  Wilcoxon (改进 vs Control 1): W={w1:.1f}, p={wp1:.6f}", flush=True)
except:
    pass
try:
    w2, wp2 = sp_stats.wilcoxon(ari_n, ari_c, alternative="greater")
    print(f"  Wilcoxon (改进 vs Control 2): W={w2:.1f}, p={wp2:.6f}", flush=True)
except:
    pass

print(f"\nFigures saved to: {FIG_DIR}/", flush=True)
for f in sorted(FIG_DIR.glob("fig*.png")):
    print(f"  {f.name}", flush=True)
print("\nVisualization complete!", flush=True)
