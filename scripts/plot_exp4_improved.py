#!/usr/bin/env python3
"""Exp4 Improved: Per-epitope visualization.

Compares: Improved (CC+Empirical), Control (CC+Equal), and individual methods.
Generates 6 figures to /home/jilin/DeepTCR/figures/
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DATA = "/home/jilin/DeepTCR/tcrconsensus/results/exp4_improved/exp4_improved_results.tsv"
FIG_DIR = Path("/home/jilin/DeepTCR/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Old Exp4 results for comparison
OLD_DATA = "/home/jilin/DeepTCR/tcrconsensus/results/exp4_mv_generalization/exp4_generalization_results.tsv"

df = pd.read_csv(DATA, sep="\t")
print(f"Loaded {len(df)} rows, {df['method'].nunique()} methods, {df['target_epitope'].nunique()} epitopes", flush=True)

# Load old results for old majority_vote comparison
old_df = pd.read_csv(OLD_DATA, sep="\t") if Path(OLD_DATA).exists() else None
if old_df is not None:
    print(f"Old Exp4: {len(old_df)} rows", flush=True)

# Colors
CONS_COLOR = {"improved": "#2171b5", "control": "#238b45", "old_mv": "#cb181d"}
METHOD_COLORS = {
    "giana": "#e6550d",
    "gliph2": "#756bb1",
    "clustcr": "#31a354",
    "hd_baseline": "#636363",
    "tcrmatch": "#e7298a",
    "tcrdist3": "#8c6d31",
    "deeptcr": "#7b4173",
}


# ============ Figure 1: Aggregate ARI Bar Chart ============
fig, ax = plt.subplots(figsize=(10, 5.5))

# Compute mean ARI per method, sort
agg = df.groupby("method")["ari"].agg(["mean", "std"]).sort_values("mean", ascending=True)

colors = []
for m in agg.index:
    if m == "improved_cc_empirical":
        colors.append(CONS_COLOR["improved"])
    elif m == "control_cc_equal":
        colors.append(CONS_COLOR["control"])
    else:
        colors.append(METHOD_COLORS.get(m, "#999"))

y = np.arange(len(agg))
bars = ax.barh(y, agg["mean"], xerr=agg["std"], height=0.6,
               color=colors, edgecolor="black", linewidth=0.5, capsize=4)

ax.set_yticks(y)
ax.set_yticklabels([m.replace("improved_cc_empirical", "Improved (CC+Empirical)")
                     .replace("control_cc_equal", "Control (CC+Equal)")
                     for m in agg.index])
ax.set_xlabel("Adjusted Rand Index (ARI)")
ax.set_title("Exp4: Per-Epitope ARI by Method (47 epitopes)")
ax.grid(axis="x", alpha=0.3)

# Value labels
for i, (v, s) in enumerate(zip(agg["mean"], agg["std"])):
    ax.text(v + s + 0.01, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

# Highlight consensus methods
for i, m in enumerate(agg.index):
    if m in ("improved_cc_empirical", "control_cc_equal"):
        ax.get_children()[i].set_edgecolor("black")
        ax.get_children()[i].set_linewidth(2)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_exp4_ari_bar.png")
print("Saved: fig_exp4_ari_bar.png", flush=True)
plt.close()


# ============ Figure 2: Per-epitope Improved vs Control Scatter ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 2a: Improved vs Control
ax = axes[0]
imp = df[df["method"] == "improved_cc_empirical"][["target_epitope", "ari"]].rename(columns={"ari": "ari_imp"})
ctrl = df[df["method"] == "control_cc_equal"][["target_epitope", "ari"]].rename(columns={"ari": "ari_ctrl"})
merged = imp.merge(ctrl, on="target_epitope")

ax.scatter(merged["ari_ctrl"], merged["ari_imp"], c=CONS_COLOR["improved"], s=40, alpha=0.7, edgecolors="white", linewidths=0.5)
lim = [0, max(merged["ari_ctrl"].max(), merged["ari_imp"].max()) + 0.05]
ax.plot(lim, lim, "k--", alpha=0.3, label="No change")
ax.set_xlabel("ARI - Control (CC+Equal)")
ax.set_ylabel("ARI - Improved (CC+Empirical)")
ax.set_title(f"Improved vs Control (n={len(merged)} epitopes)")
ax.legend()
ax.grid(alpha=0.3)

# Annotate wins
wins = (merged["ari_imp"] > merged["ari_ctrl"]).sum()
ties = (merged["ari_imp"] == merged["ari_ctrl"]).sum()
losses = len(merged) - wins - ties
ax.text(0.05, 0.95, f"Wins: {wins}  Ties: {ties}  Losses: {losses}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# 2b: Improved vs GIANA (best single)
ax = axes[1]
giana = df[df["method"] == "giana"][["target_epitope", "ari"]].rename(columns={"ari": "ari_giana"})
merged2 = imp.merge(giana, on="target_epitope")

ax.scatter(merged2["ari_giana"], merged2["ari_imp"], c=CONS_COLOR["improved"], s=40, alpha=0.7, edgecolors="white", linewidths=0.5)
lim = [0, max(merged2["ari_giana"].max(), merged2["ari_imp"].max()) + 0.05]
ax.plot(lim, lim, "k--", alpha=0.3, label="No change")
ax.set_xlabel("ARI - GIANA (best single)")
ax.set_ylabel("ARI - Improved (CC+Empirical)")
ax.set_title(f"Improved vs GIANA (n={len(merged2)} epitopes)")
ax.legend()
ax.grid(alpha=0.3)

wins2 = (merged2["ari_imp"] > merged2["ari_giana"]).sum()
ax.text(0.05, 0.95, f"Improved wins: {wins2}/{len(merged2)}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_exp4_paired_scatter.png")
print("Saved: fig_exp4_paired_scatter.png", flush=True)
plt.close()


# ============ Figure 3: Per-epitope heatmap (top 20 epitopes by size) ============
fig, ax = plt.subplots(figsize=(14, 8))

# Select top 20 epitopes by n_target
top_epis = df.drop_duplicates("target_epitope").nlargest(20, "n_target")["target_epitope"].tolist()

# Methods to show
methods_show = ["improved_cc_empirical", "control_cc_equal", "giana", "gliph2", "clustcr",
                "hd_baseline", "tcrmatch", "tcrdist3", "deeptcr"]
method_labels = ["Improved", "Control", "GIANA", "GLIPH2", "clusTCR",
                 "HD-Baseline", "TCRMatch", "TCRdist3", "DeepTCR"]

# Build matrix
matrix = np.zeros((len(top_epis), len(methods_show)))
for i, epi in enumerate(top_epis):
    for j, m in enumerate(methods_show):
        vals = df[(df["target_epitope"] == epi) & (df["method"] == m)]["ari"]
        matrix[i, j] = vals.values[0] if len(vals) > 0 else 0

im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.8)
ax.set_xticks(np.arange(len(methods_show)))
ax.set_xticklabels(method_labels, rotation=45, ha="right")
ax.set_yticks(np.arange(len(top_epis)))
ax.set_yticklabels([e[:15] for e in top_epis])

# Add text annotations
for i in range(len(top_epis)):
    for j in range(len(methods_show)):
        v = matrix[i, j]
        color = "white" if v > 0.4 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)

plt.colorbar(im, ax=ax, label="ARI", shrink=0.8)
ax.set_title("Exp4: Per-Epitope ARI Heatmap (Top 20 Epitopes by Size)")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_exp4_heatmap.png")
print("Saved: fig_exp4_heatmap.png", flush=True)
plt.close()


# ============ Figure 4: Multi-metric comparison ============
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

metrics_show = [
    ("ari", "ARI"),
    ("purity", "Purity"),
    ("sensitivity", "Sensitivity"),
    ("f1", "F1"),
]

key_methods = ["improved_cc_empirical", "control_cc_equal", "giana", "gliph2", "clustcr"]
key_labels = ["Improved", "Control", "GIANA", "GLIPH2", "clusTCR"]
key_colors = [CONS_COLOR["improved"], CONS_COLOR["control"],
              METHOD_COLORS["giana"], METHOD_COLORS["gliph2"], METHOD_COLORS["clustcr"]]

for ax, (metric, ylabel) in zip(axes, metrics_show):
    x = np.arange(len(key_methods))
    means = [df[df["method"] == m][metric].mean() for m in key_methods]
    stds = [df[df["method"] == m][metric].std() for m in key_methods]

    bars = ax.bar(x, means, yerr=stds, width=0.6, color=key_colors,
                  edgecolor="black", linewidth=0.5, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(key_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")

fig.suptitle("Exp4: Multi-Metric Comparison (47 epitopes)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_exp4_multi_metrics.png")
print("Saved: fig_exp4_multi_metrics.png", flush=True)
plt.close()


# ============ Figure 5: Summary Table ============
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis("off")

# Build table data
all_methods_sorted = df.groupby("method")["ari"].mean().sort_values(ascending=False).index.tolist()
col_labels = ["Method", "ARI", "Purity", "Sensitivity", "F1", "N Epitopes"]
row_data = []

for m in all_methods_sorted:
    sub = df[df["method"] == m]
    row_data.append([
        m.replace("improved_cc_empirical", "Improved (CC+Emp)")
          .replace("control_cc_equal", "Control (CC+Equal)"),
        f"{sub['ari'].mean():.4f} +/- {sub['ari'].std():.3f}",
        f"{sub['purity'].mean():.4f} +/- {sub['purity'].std():.3f}",
        f"{sub['sensitivity'].mean():.4f} +/- {sub['sensitivity'].std():.3f}",
        f"{sub['f1'].mean():.4f} +/- {sub['f1'].std():.3f}",
        str(len(sub)),
    ])

table = ax.table(cellText=row_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colColours=["#d9e2f3"] * len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.7)

for i in range(len(row_data)):
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        if i % 2 == 0:
            cell.set_facecolor("#f0f0f0")
    # Highlight consensus methods
    method_name = all_methods_sorted[i]
    if method_name in ("improved_cc_empirical", "control_cc_equal"):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor("#d4e6f1" if method_name == "improved_cc_empirical" else "#d5f5e3")

ax.set_title("Exp4: Per-Epitope Performance Summary (47 epitopes, leave-one-out with distractors)", fontsize=12, pad=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_exp4_summary_table.png")
print("Saved: fig_exp4_summary_table.png", flush=True)
plt.close()


# ============ Figure 6: Old MV vs Improved vs GIANA (if old data available) ============
if old_df is not None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Compare: old majority_vote, new improved, GIANA
    old_mv = old_df[old_df["method"] == "majority_vote"][["target_epitope", "ari"]].rename(columns={"ari": "ari_old"})
    new_imp = df[df["method"] == "improved_cc_empirical"][["target_epitope", "ari"]].rename(columns={"ari": "ari_new"})
    giana_df = df[df["method"] == "giana"][["target_epitope", "ari"]].rename(columns={"ari": "ari_giana"})

    compare = old_mv.merge(new_imp, on="target_epitope").merge(giana_df, on="target_epitope")
    compare = compare.sort_values("ari_giana", ascending=True)

    x = np.arange(len(compare))
    width = 0.25

    ax.bar(x - width, compare["ari_old"], width, label="Old MV (Leiden+Equal)",
           color=CONS_COLOR["old_mv"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x, compare["ari_new"], width, label="Improved (CC+Empirical)",
           color=CONS_COLOR["improved"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width, compare["ari_giana"], width, label="GIANA",
           color=METHOD_COLORS["giana"], alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([e[:12] for e in compare["target_epitope"]], rotation=90, fontsize=6)
    ax.set_ylabel("ARI")
    ax.set_title("Per-Epitope ARI: Old MV vs Improved vs GIANA")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_exp4_old_vs_new.png")
    print("Saved: fig_exp4_old_vs_new.png", flush=True)
    plt.close()

    # Paired stats
    print(f"\n--- Old MV vs Improved vs GIANA ---", flush=True)
    print(f"  Old MV:     {compare['ari_old'].mean():.4f}", flush=True)
    print(f"  Improved:   {compare['ari_new'].mean():.4f}", flush=True)
    print(f"  GIANA:      {compare['ari_giana'].mean():.4f}", flush=True)
    t_old_new, p_old_new = sp_stats.ttest_rel(compare["ari_new"], compare["ari_old"])
    print(f"  Old→New: Δ={compare['ari_new'].mean()-compare['ari_old'].mean():+.4f}, p={p_old_new:.4f}", flush=True)


# ============ Print final summary ============
print(f"\n{'=' * 70}", flush=True)
print(f"Exp4 Visualization Complete", flush=True)
print(f"{'=' * 70}", flush=True)

imp_ari = df[df["method"] == "improved_cc_empirical"]["ari"].mean()
ctrl_ari = df[df["method"] == "control_cc_equal"]["ari"].mean()
giana_ari = df[df["method"] == "giana"]["ari"].mean()

print(f"\n  Key Results:", flush=True)
print(f"    Improved (CC+Empirical): ARI = {imp_ari:.4f}", flush=True)
print(f"    Control (CC+Equal):      ARI = {ctrl_ari:.4f}", flush=True)
print(f"    GIANA (best single):     ARI = {giana_ari:.4f}", flush=True)
print(f"    Δ Improved-Control:      {imp_ari - ctrl_ari:+.4f}", flush=True)
print(f"    Gap to GIANA:            {giana_ari - imp_ari:.4f}", flush=True)

print(f"\n  Figures saved to {FIG_DIR}/:", flush=True)
for f in sorted(FIG_DIR.glob("fig_exp4_*.png")):
    print(f"    {f.name}", flush=True)
