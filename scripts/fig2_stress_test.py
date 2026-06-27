#!/usr/bin/env python3
"""Figure 2: Stress Test Performance — Consensus vs 7 Individual Methods.

6 subsets with increasing background noise.
Follows the visual style from run_full_comparison.py.

Reads from improved_stress_results.json (7 methods).
Generates publication-ready multi-panel figure.
"""

import json
import numpy as np
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

# ── Style constants (same as run_full_comparison.py) ──
DISPLAY_NAMES = {
    "tcrconsensus": "tcrconsensus",
    "giana": "GIANA",
    "clustcr": "clusTCR",
    "gliph2": "GLIPH2",
    "hd_baseline": "HD-Baseline",
    "tcrmatch": "TCRMatch",
    "tcrdist3": "TCRdist3",
    "deeptcr": "DeepTCR",
}

COLORS = {
    "tcrconsensus": "#2171b5",
    "giana": "#e6550d",
    "clustcr": "#31a354",
    "gliph2": "#756bb1",
    "hd_baseline": "#636363",
    "tcrmatch": "#e7298a",
    "tcrdist3": "#8c6d31",
    "deeptcr": "#7b4173",
}

MARKERS = {
    "tcrconsensus": "D",
    "giana": "o",
    "clustcr": "s",
    "gliph2": "^",
    "hd_baseline": "v",
    "tcrmatch": "P",
    "tcrdist3": "X",
    "deeptcr": "p",
}

METHOD_ORDER = [
    "tcrconsensus", "giana", "gliph2", "clustcr",
    "hd_baseline", "tcrdist3", "tcrmatch", "deeptcr",
]

METRICS_TO_PLOT = [
    ("ari_labeled", "ARI (labeled)"),
    ("nmi_labeled", "NMI (labeled)"),
    ("homogeneity", "Homogeneity"),
    ("v_measure", "V-measure"),
    ("retention", "Retention"),
    ("weighted_purity", "Weighted Purity"),
]

# ── Load data ──
DATA_PATH = "/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments/improved_stress/improved_stress_results.json"
FIG_DIR = "/home/jilin/DeepTCR/figures"

with open(DATA_PATH) as f:
    raw = json.load(f)

results = raw["results"]
n_subsets = len(results)

# ── Extract: build a unified matrix [method x subset x metric] ──
# Methods that appear per-subset vary (tcrdist3 missing from subsets 3-6)
# Build a padded matrix with NaN for missing entries

subsets = []
bg_ratios = []

# All methods ever seen
all_method_names = set()
for r in results:
    all_method_names.update(r["individual_methods"].keys())
all_method_names = sorted(all_method_names)

# method_matrix[method][metric][subset_idx] = value or NaN
method_matrix = {}
for m in all_method_names:
    method_matrix[m] = {}
    for key, _ in METRICS_TO_PLOT:
        method_matrix[m][key] = [np.nan] * n_subsets

consensus_matrix = {}
for key, _ in METRICS_TO_PLOT:
    consensus_matrix[key] = []

for i, r in enumerate(results):
    sid = r["subset"]
    c = r["consensus"]
    subsets.append(sid)
    bg_ratios.append(c["bg_ratio"])

    for key, _ in METRICS_TO_PLOT:
        consensus_matrix[key].append(c[key])

    for mname, mval in r["individual_methods"].items():
        for key, _ in METRICS_TO_PLOT:
            method_matrix[mname][key][i] = mval[key]

# Build ordered method list (exclude consensus, which is separate)
method_list = [m for m in METHOD_ORDER if m != "tcrconsensus" and m in all_method_names]
hm_methods = ["tcrconsensus"] + method_list  # consensus first for heatmap


# ── Compute fold improvement: consensus vs best single method ──
fold_improvement = {}
for key, label in METRICS_TO_PLOT:
    folds = []
    for i in range(n_subsets):
        singles = [method_matrix[m][key][i] for m in method_list
                   if not np.isnan(method_matrix[m][key][i])]
        best_single = max(singles) if singles else 0.0
        cons_val = consensus_matrix[key][i]
        if best_single > 0:
            folds.append(cons_val / best_single)
        else:
            folds.append(0.0)
    fold_improvement[key] = folds


# Helper: get values for a method, masking NaN
def get_method_vals(mname, key):
    """Return (x_valid, y_valid) arrays for a method, skipping NaN subsets."""
    vals = np.array(method_matrix[mname][key])
    valid = ~np.isnan(vals)
    return np.where(valid)[0], vals[valid]


# ================================================================
#  COMPOSITE FIGURE 2: 2x2 panel
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
x = np.arange(n_subsets)
x_labels = [f"S{s} ({bg:.1f}x)" for s, bg in zip(subsets, bg_ratios)]

# ── Panel A: ARI_labeled line plot ──
ax = axes[0, 0]

for mname in method_list:
    xv, yv = get_method_vals(mname, "ari_labeled")
    ax.plot(xv, yv, marker=MARKERS.get(mname, "o"), color=COLORS[mname],
            linewidth=1.5, markersize=6, alpha=0.75,
            label=DISPLAY_NAMES.get(mname, mname))

ax.plot(x, consensus_matrix["ari_labeled"],
        marker=MARKERS["tcrconsensus"], color=COLORS["tcrconsensus"],
        linewidth=3.0, markersize=9, label="tcrconsensus", zorder=10)

for i, fold in enumerate(fold_improvement["ari_labeled"]):
    ax.annotate(f"{fold:.1f}x", (x[i], consensus_matrix["ari_labeled"][i]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, fontweight="bold",
                color=COLORS["tcrconsensus"])

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_xlabel("Subset (Background Ratio)")
ax.set_ylabel("ARI (labeled)")
ax.set_title("(A) ARI: Consensus vs Individual Methods", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=9,
          edgecolor="#cccccc", fancybox=True)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=-0.02)

# ── Panel B: NMI line plot ──
ax = axes[0, 1]

for mname in method_list:
    xv, yv = get_method_vals(mname, "nmi_labeled")
    ax.plot(xv, yv, marker=MARKERS.get(mname, "o"), color=COLORS[mname],
            linewidth=1.5, markersize=6, alpha=0.75,
            label=DISPLAY_NAMES.get(mname, mname))

ax.plot(x, consensus_matrix["nmi_labeled"],
        marker=MARKERS["tcrconsensus"], color=COLORS["tcrconsensus"],
        linewidth=3.0, markersize=9, label="tcrconsensus", zorder=10)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_xlabel("Subset (Background Ratio)")
ax.set_ylabel("NMI (labeled)")
ax.set_title("(B) NMI: Consensus vs Individual Methods", fontsize=13, fontweight="bold")
ax.legend(loc="lower left", framealpha=0.9, ncol=2, fontsize=9,
          edgecolor="#cccccc", fancybox=True)
ax.grid(axis="y", alpha=0.3)
# Note: consensus NMI may be lower than giana/gliph2 at high noise because
# those methods over-cluster (high homogeneity), while consensus is more conservative.
# ARI is the stricter metric that penalizes over-clustering.
ax.text(0.02, 0.02,
        "Note: Single methods achieve higher NMI via\nover-clustering (high Homo, many tiny clusters).\nConsensus prioritizes ARI (pairwise accuracy).",
        transform=ax.transAxes, fontsize=7, verticalalignment="bottom",
        style="italic", color="#666666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9", edgecolor="#cccccc", alpha=0.9))

# ── Panel C: ARI heatmap (all methods x all subsets) ──
ax = axes[1, 0]

hm_rows = len(hm_methods)
hm_cols = n_subsets
matrix = np.full((hm_rows, hm_cols), np.nan)

for i, m in enumerate(hm_methods):
    for j in range(n_subsets):
        if m == "tcrconsensus":
            matrix[i, j] = consensus_matrix["ari_labeled"][j]
        else:
            val = method_matrix[m]["ari_labeled"][j]
            if not np.isnan(val):
                matrix[i, j] = val

# Use masked array for NaN display
masked = np.ma.masked_invalid(matrix)
cmap = plt.cm.YlOrRd.copy()
cmap.set_bad(color="#f0f0f0")  # gray for missing

im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0)
ax.set_xticks(np.arange(hm_cols))
ax.set_xticklabels([f"S{s}" for s in subsets])
ax.set_yticks(np.arange(hm_rows))
ax.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in hm_methods], fontsize=9)

for i in range(hm_rows):
    for j in range(hm_cols):
        v = matrix[i, j]
        if np.isnan(v):
            ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="#999999")
        else:
            color = "white" if v > np.nanmax(matrix) * 0.7 else "black"
            bold = "bold" if hm_methods[i] == "tcrconsensus" else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color=color, fontweight=bold)

# Highlight consensus row
ci = hm_methods.index("tcrconsensus")
for j in range(hm_cols):
    rect = plt.Rectangle((j - 0.5, ci - 0.5), 1, 1,
                         linewidth=3, edgecolor="#08306b", facecolor="none")
    ax.add_patch(rect)

# Mark best per column (star for non-consensus best)
for j in range(hm_cols):
    col_vals = [matrix[i, j] for i in range(hm_rows) if not np.isnan(matrix[i, j])]
    if not col_vals:
        continue
    best_val = max(col_vals)
    for i in range(hm_rows):
        if not np.isnan(matrix[i, j]) and matrix[i, j] == best_val:
            if i != ci:
                ax.plot(j, i, "*", color="#08306b", markersize=10)
            break

plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel("Subset")
ax.set_title("(C) ARI Heatmap: All Methods x All Subsets", fontsize=13, fontweight="bold")

# ── Panel D: Summary table (like run_full_comparison.py Fig 4) ──
ax = axes[1, 1]
ax.axis("off")

table_metrics = [
    ("ari_labeled", "ARI"),
    ("nmi_labeled", "NMI"),
    ("weighted_purity", "WPur"),
    ("retention", "Ret"),
]

# Build table data: one row per method, averaged across all 6 subsets
col_labels = ["Method", "ARI", "NMI", "WPur", "Ret", "ARI fold"]
row_data = []
for m in hm_methods:
    row = [DISPLAY_NAMES.get(m, m)]
    for key, _ in table_metrics:
        if m == "tcrconsensus":
            vals_list = consensus_matrix[key]
        else:
            vals_list = [v for v in method_matrix[m][key] if not np.isnan(v)]
        avg = np.mean(vals_list) if vals_list else 0.0
        row.append(f"{avg:.4f}")
    # Fold improvement (ARI, averaged)
    if m == "tcrconsensus":
        avg_fold = np.mean(fold_improvement["ari_labeled"])
        row.append(f"{avg_fold:.1f}x")
    else:
        row.append("1.0x (ref)")
    row_data.append(row)

# Create table
table = ax.table(
    cellText=row_data, colLabels=col_labels,
    cellLoc="center", loc="center",
    colColours=["#d4e6f1"] * len(col_labels),
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

# Highlight consensus row (row index 1, since header is row 0)
consensus_row = 1  # first data row = tcrconsensus
for j in range(len(col_labels)):
    cell = table[consensus_row, j]
    cell.set_facecolor("#d4e6f1")
    cell.set_text_props(fontweight="bold")

# Alternate row shading
for i in range(2, len(row_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[i, j].set_facecolor("#f7f7f7")

ax.set_title("(D) Average Performance Across 6 Subsets", fontsize=13, fontweight="bold", pad=20)

fig.suptitle("Figure 2: Stress Test — tcrconsensus vs 7 Individual Methods\n"
             "(10X Donor1, 6 Subsets with Increasing Background Noise)",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_stress_test_composite.png")
print("Saved: fig2_stress_test_composite.png")
plt.close()


# ================================================================
#  ALSO save individual panels at higher resolution
# ================================================================

# Fig 2A standalone
fig2a, ax = plt.subplots(figsize=(10, 6))
for mname in method_list:
    xv, yv = get_method_vals(mname, "ari_labeled")
    ax.plot(xv, yv, marker=MARKERS.get(mname, "o"), color=COLORS[mname],
            linewidth=1.5, markersize=7, alpha=0.75,
            label=DISPLAY_NAMES.get(mname, mname))

ax.plot(x, consensus_matrix["ari_labeled"],
        marker=MARKERS["tcrconsensus"], color=COLORS["tcrconsensus"],
        linewidth=3.0, markersize=10, label="tcrconsensus", zorder=10)

for i, fold in enumerate(fold_improvement["ari_labeled"]):
    ax.annotate(f"{fold:.1f}x", (x[i], consensus_matrix["ari_labeled"][i]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9, fontweight="bold",
                color=COLORS["tcrconsensus"])

ax.set_xticks(x)
ax.set_xticklabels([f"Subset {s}\n(bg {bg:.1f}x)" for s, bg in zip(subsets, bg_ratios)])
ax.set_xlabel("Subset (Background Ratio)")
ax.set_ylabel("ARI (labeled)")
ax.set_title("Stress Test: Consensus ARI vs Individual Methods")
ax.legend(loc="upper right", framealpha=0.9, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=-0.02)
fig2a.tight_layout()
fig2a.savefig(f"{FIG_DIR}/fig2a_ari_stress_line.png")
print("Saved: fig2a_ari_stress_line.png")
plt.close()

# Fig 2C standalone heatmap
fig2c, axes_hm = plt.subplots(1, 3, figsize=(22, 8))

heatmap_metrics = [
    ("ari_labeled", "ARI (labeled)"),
    ("nmi_labeled", "NMI (labeled)"),
    ("weighted_purity", "Weighted Purity"),
]

for ax, (metric_key, metric_label) in zip(axes_hm, heatmap_metrics):
    mat = np.full((hm_rows, hm_cols), np.nan)
    for i, m in enumerate(hm_methods):
        for j in range(n_subsets):
            if m == "tcrconsensus":
                mat[i, j] = consensus_matrix[metric_key][j]
            else:
                val = method_matrix[m][metric_key][j]
                if not np.isnan(val):
                    mat[i, j] = val

    masked_mat = np.ma.masked_invalid(mat)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#f0f0f0")

    im = ax.imshow(masked_mat, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(np.arange(hm_cols))
    ax.set_xticklabels([f"S{s}" for s in subsets])
    ax.set_yticks(np.arange(hm_rows))
    ax.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in hm_methods])

    for i in range(hm_rows):
        for j in range(hm_cols):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="#999999")
            else:
                color = "white" if v > np.nanmax(mat) * 0.7 else "black"
                bold = "bold" if hm_methods[i] == "tcrconsensus" else "normal"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color=color, fontweight=bold)

    ci = hm_methods.index("tcrconsensus")
    for j in range(hm_cols):
        rect = plt.Rectangle((j - 0.5, ci - 0.5), 1, 1,
                             linewidth=3, edgecolor="#08306b", facecolor="none")
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Subset")
    ax.set_title(metric_label)

fig2c.suptitle("Stress Test Heatmap: tcrconsensus vs Individual Methods",
               fontsize=14, y=1.02)
fig2c.tight_layout()
fig2c.savefig(f"{FIG_DIR}/fig2c_stress_heatmap.png")
print("Saved: fig2c_stress_heatmap.png")
plt.close()

print("\nAll Figure 2 files saved to", FIG_DIR)
