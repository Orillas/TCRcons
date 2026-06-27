#!/usr/bin/env python3
"""可复现性实验结果可视化。

从 reproducibility_results.json 读取数据，生成对比图表。
输出到 /home/jilin/DeepTCR/figures/
"""
import json
import sys
import os

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# 设置全局字体
plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DATA_PATH = "/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/reproducibility_results.json"
FIG_DIR = Path("/home/jilin/DeepTCR/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 配色
COLOR_NEW = "#2171b5"   # 蓝 - 改进后
COLOR_OLD = "#cb181d"   # 红 - 改进前
COLOR_NEW_LIGHT = "#6baed6"
COLOR_OLD_LIGHT = "#fb6a4a"

# ============ 加载数据 ============
with open(DATA_PATH) as f:
    data = json.load(f)

results = data["results"]
new_data = [r for r in results if r["config"] == "new_cc_empirical"]
old_data = [r for r in results if r["config"] == "old_leiden_equal"]

print(f"改进后: {len(new_data)} runs", flush=True)
print(f"改进前: {len(old_data)} runs", flush=True)

if not new_data or not old_data:
    print("ERROR: 数据不完整，无法绘图", flush=True)
    sys.exit(1)

seeds_new = [r["seed"] for r in new_data]
seeds_old = [r["seed"] for r in old_data]


# ============ Figure 1: ARI Box Plot + Scatter ============
fig, ax = plt.subplots(figsize=(6, 5))

ari_new = [r["ari"] for r in new_data]
ari_old = [r["ari"] for r in old_data]

bp = ax.boxplot(
    [ari_old, ari_new],
    labels=["Before\n(Leiden + Equal)", "After\n(CC + Empirical)"],
    patch_artist=True,
    widths=0.5,
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 6},
)
bp["boxes"][0].set_facecolor(COLOR_OLD_LIGHT)
bp["boxes"][0].set_edgecolor(COLOR_OLD)
bp["boxes"][1].set_facecolor(COLOR_NEW_LIGHT)
bp["boxes"][1].set_edgecolor(COLOR_NEW)

# 散点
for i, (ari_list, color) in enumerate([(ari_old, COLOR_OLD), (ari_new, COLOR_NEW)]):
    jitter = np.random.uniform(-0.08, 0.08, len(ari_list))
    ax.scatter([i + 1 + j for j in jitter], ari_list, color=color, s=40, zorder=5, alpha=0.8)

# 连线配对
common_seeds = sorted(set(seeds_new) & set(seeds_old))
for s in common_seeds:
    n = next(r for r in new_data if r["seed"] == s)
    o = next(r for r in old_data if r["seed"] == s)
    ax.plot([1, 2], [o["ari"], n["ari"]], color="gray", alpha=0.4, linewidth=0.8, linestyle="--")

ax.set_ylabel("Adjusted Rand Index (ARI)")
ax.set_title("Reproducibility: ARI Comparison (n=5 seeds)")
ax.grid(axis="y", alpha=0.3)

# 标注均值
ax.text(1, max(ari_old) + 0.02, f"μ={np.mean(ari_old):.3f}", ha="center", color=COLOR_OLD, fontweight="bold")
ax.text(2, max(ari_new) + 0.02, f"μ={np.mean(ari_new):.3f}", ha="center", color=COLOR_NEW, fontweight="bold")

# 标注提升百分比
delta = np.mean(ari_new) - np.mean(ari_old)
pct = delta / np.mean(ari_old) * 100
ax.annotate(
    f"+{pct:.0f}%",
    xy=(1.5, (np.mean(ari_old) + np.mean(ari_new)) / 2),
    fontsize=16, fontweight="bold", color="#2ca02c", ha="center",
)

fig.savefig(FIG_DIR / "fig1_ari_boxplot.png")
print(f"  Saved: fig1_ari_boxplot.png", flush=True)
plt.close()


# ============ Figure 2: Multi-metric Comparison ============
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

metrics = [
    ("ari", "ARI", "Adjusted Rand Index"),
    ("purity", "Purity", "Cluster Purity"),
    ("nmi", "NMI", "Normalized Mutual Info"),
    ("retention", "Retention", "Retention Rate"),
]

for ax, (key, label, title) in zip(axes, metrics):
    vals_new = [r[key] for r in new_data]
    vals_old = [r[key] for r in old_data]

    x = np.arange(2)
    means = [np.mean(vals_old), np.mean(vals_new)]
    stds = [np.std(vals_old), np.std(vals_new)]

    bars = ax.bar(
        x, means, yerr=stds, width=0.5,
        color=[COLOR_OLD_LIGHT, COLOR_NEW_LIGHT],
        edgecolor=[COLOR_OLD, COLOR_NEW],
        linewidth=1.5, capsize=5, error_kw={"linewidth": 1.5},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Before", "After"])
    ax.set_title(title)
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.3)

    # 标注数值
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Reproducibility: Multi-Metric Comparison (n=5 seeds)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_multi_metrics.png")
print(f"  Saved: fig2_multi_metrics.png", flush=True)
plt.close()


# ============ Figure 3: Per-method ARI Stability ============
all_methods = sorted(set(m for r in new_data + old_data for m in r["per_method_ari"]))

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(all_methods))
width = 0.35

# 改进后各方法 ARI（应该和改进前相同，因为是同样的 clusterers）
# 但可以展示方法间差异
means_new = []
stds_new = []
means_old = []
stds_old = []

for m in all_methods:
    vals_n = [r["per_method_ari"].get(m, 0) for r in new_data if m in r["per_method_ari"]]
    vals_o = [r["per_method_ari"].get(m, 0) for r in old_data if m in r["per_method_ari"]]
    means_new.append(np.mean(vals_n) if vals_n else 0)
    stds_new.append(np.std(vals_n) if vals_n else 0)
    means_old.append(np.mean(vals_o) if vals_o else 0)
    stds_old.append(np.std(vals_o) if vals_o else 0)

bars1 = ax.bar(x - width/2, means_old, width, yerr=stds_old,
               label="Before (Leiden)", color=COLOR_OLD_LIGHT, edgecolor=COLOR_OLD, capsize=3)
bars2 = ax.bar(x + width/2, means_new, width, yerr=stds_new,
               label="After (CC)", color=COLOR_NEW_LIGHT, edgecolor=COLOR_NEW, capsize=3)

# 共识结果线
ax.axhline(y=np.mean(ari_old), color=COLOR_OLD, linestyle="--", alpha=0.5, linewidth=1)
ax.axhline(y=np.mean(ari_new), color=COLOR_NEW, linestyle="--", alpha=0.5, linewidth=1)
ax.text(len(all_methods)-0.5, np.mean(ari_old)+0.01, f"Consensus Before: {np.mean(ari_old):.3f}",
        color=COLOR_OLD, fontsize=9, ha="right")
ax.text(len(all_methods)-0.5, np.mean(ari_new)+0.01, f"Consensus After: {np.mean(ari_new):.3f}",
        color=COLOR_NEW, fontsize=9, ha="right")

ax.set_xticks(x)
ax.set_xticklabels(all_methods, rotation=30, ha="right")
ax.set_ylabel("ARI")
ax.set_title("Per-Method ARI Stability Across 5 Seeds")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_per_method_ari.png")
print(f"  Saved: fig3_per_method_ari.png", flush=True)
plt.close()


# ============ Figure 4: Paired Comparison (Before → After) ============
fig, ax = plt.subplots(figsize=(6, 5))

for s in common_seeds:
    n = next(r for r in new_data if r["seed"] == s)
    o = next(r for r in old_data if r["seed"] == s)
    ax.annotate(
        "",
        xy=(o["ari"], n["ari"]),
        xytext=(o["ari"], o["ari"]),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
    )
    ax.scatter(o["ari"], n["ari"], color=COLOR_NEW, s=80, zorder=5, edgecolors="white", linewidths=1)
    ax.text(o["ari"] + 0.005, n["ari"] + 0.01, f"s={s}", fontsize=8, color="gray")

# 对角线
lim_min = min(min(ari_old), min(ari_new)) - 0.05
lim_max = max(max(ari_old), max(ari_new)) + 0.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, label="No change")

ax.set_xlabel("ARI (Before: Leiden + Equal)")
ax.set_ylabel("ARI (After: CC + Empirical)")
ax.set_title("Per-Seed ARI: Before → After Improvement")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_paired_ari.png")
print(f"  Saved: fig4_paired_ari.png", flush=True)
plt.close()


# ============ Figure 5: Summary Table as Figure ============
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

col_labels = ["Metric", "Before (Leiden+Equal)", "After (CC+Empirical)", "Δ", "p-value"]
row_data = []

from scipy import stats as sp_stats

for key, label in [("ari", "ARI"), ("purity", "Purity"), ("nmi", "NMI"), ("retention", "Retention")]:
    vn = [r[key] for r in new_data]
    vo = [r[key] for r in old_data]
    mn, mo = np.mean(vn), np.mean(vo)
    delta = mn - mo
    pct = delta / mo * 100 if mo != 0 else 0
    try:
        _, p = sp_stats.ttest_ind(vn, vo)
    except Exception:
        p = 1.0
    row_data.append([
        label,
        f"{mo:.4f} ± {np.std(vo):.4f}",
        f"{mn:.4f} ± {np.std(vn):.4f}",
        f"+{delta:.4f} ({pct:+.1f}%)",
        f"{p:.4f}" if p >= 0.001 else "<0.001",
    ])

table = ax.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colColours=["#d9e2f3"] * len(col_labels),
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# 交替行颜色
for i in range(len(row_data)):
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        if i % 2 == 0:
            cell.set_facecolor("#f0f0f0")

ax.set_title("Reproducibility Experiment Summary (5 seeds × 2 configs)", fontsize=13, pad=20)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_summary_table.png")
print(f"  Saved: fig5_summary_table.png", flush=True)
plt.close()


# ============ 完成 ============
print(f"\n所有图表已保存到 {FIG_DIR}/", flush=True)
for f in sorted(FIG_DIR.glob("fig*.png")):
    print(f"  {f.name}", flush=True)
print("可视化完成", flush=True)
