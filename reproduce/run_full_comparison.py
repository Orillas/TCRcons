#!/usr/bin/env python3
"""Full-dataset comparison: tcrconsensus (CC+Empirical) vs 7 individual methods.

Runs all methods once on the full high-confidence benchmark (seed=42),
computes ARI/Purity/AMI/Sensitivity/F1/Retention for each, then generates
comparison visualizations.
"""

import sys
import os
import time
import logging
import warnings
import json
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# Optional dev path hints (e.g. an editable tcrconsensus/src checkout or a local
# clusTCR clone). pip-installed users do not need these — both resolve via the
# installed packages. Set TCR_EXTRA_PATHS="/path/a:/path/b" to add directories.
for _p in os.environ.get("TCR_EXTRA_PATHS", "").split(os.pathsep):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "font.family": "sans-serif",
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

from tcrconsensus.io.parser import normalize
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper

# Paths are configurable via environment variables; defaults are repo-relative.
BENCHMARK = os.environ.get(
    "TCR_BENCHMARK", "results/paper_benchmark/paper_benchmark_v3_cd8.tsv")
OUT_DIR = Path(os.environ.get("TCR_OUT_DIR", "results/full_comparison"))
FIG_DIR = Path(os.environ.get("TCR_FIG_DIR", "results/figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============ Load data ============
print("=" * 70, flush=True)
print("Full-Dataset Comparison: tcrconsensus vs 7 Individual Methods", flush=True)
print("=" * 70, flush=True)

np.random.seed(42)

df = pd.read_csv(BENCHMARK, sep="\t", dtype=str)
rename = {c: c.lower() for c in df.columns
          if c.lower() != c and c.lower() in
          ["cdr3_alpha","cdr3_beta","v_alpha","v_beta","j_alpha","j_beta","tcr_id","epitope"]}
if rename:
    df = df.rename(columns=rename)
df_norm = normalize(df.copy())

epitope_map = {}
for _, row in df_norm.iterrows():
    tid = str(row.get("tcr_id", ""))
    epi = str(row.get("epitope", ""))
    if tid and epi:
        epitope_map[tid] = epi

n_total = len(df_norm)
n_epitopes = df_norm['epitope'].nunique()
print(f"Dataset: {n_total} TCRs, {n_epitopes} epitopes", flush=True)


# ============ Metrics computation ============
def compute_metrics_from_assignments(assignments, epitope_map, total_tcrs):
    """Compute full metrics from a list of ClusterAssignment objects."""
    tids = list(set(a.tcr_id for a in assignments))
    if len(tids) < 2:
        return None

    idx = {t: i for i, t in enumerate(tids)}
    pred = [-1] * len(tids)
    true_l = [-1] * len(tids)

    for a in assignments:
        i = idx[a.tcr_id]
        pred[i] = hash(a.cluster_id) % (10**8)
        if a.tcr_id in epitope_map:
            true_l[i] = hash(epitope_map[a.tcr_id]) % (10**8)

    valid = [j for j in range(len(tids)) if pred[j] != -1]
    if len(valid) < 2:
        return None

    lp = [pred[j] for j in valid]
    lt = [true_l[j] for j in valid]

    ari_val = adjusted_rand_score(lt, lp)
    ami_val = adjusted_mutual_info_score(lt, lp)

    # Purity
    cluster_epis = {}
    for i, p in enumerate(lp):
        cluster_epis.setdefault(p, []).append(lt[i])
    purity = sum(Counter(v).most_common(1)[0][1] for v in cluster_epis.values()) / len(valid)

    # Sensitivity (recall): fraction of true same-epitope pairs that are co-clustered
    from itertools import combinations
    same_epi_pairs = 0
    same_cluster_same_epi = 0
    true_clusters = {}
    for i, t in enumerate(lt):
        true_clusters.setdefault(t, []).append(i)
    for t, members in true_clusters.items():
        if len(members) < 2:
            continue
        for i, j in combinations(members, 2):
            same_epi_pairs += 1
            if lp[i] == lp[j]:
                same_cluster_same_epi += 1
    sensitivity = same_cluster_same_epi / same_epi_pairs if same_epi_pairs > 0 else 0

    # Precision: fraction of co-clustered pairs that share epitope
    pred_clusters = {}
    for i, p in enumerate(lp):
        pred_clusters.setdefault(p, []).append(i)
    co_clustered_pairs = 0
    correct_pairs = 0
    for p, members in pred_clusters.items():
        if len(members) < 2:
            continue
        for i, j in combinations(members, 2):
            co_clustered_pairs += 1
            if lt[i] == lt[j]:
                correct_pairs += 1
    precision = correct_pairs / co_clustered_pairs if co_clustered_pairs > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    retention = len(valid) / total_tcrs

    return {
        "ari": round(ari_val, 4),
        "purity": round(purity, 4),
        "ami": round(ami_val, 4),
        "sensitivity": round(sensitivity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "retention": round(retention, 4),
        "n_clustered": len(valid),
        "n_clusters": len(cluster_epis),
        "n_tcrs": len(tids),
    }


def compute_metrics_from_clusters(clusters, epitope_map, total_tcrs):
    """Compute full metrics from ConsensusCluster objects."""
    members = []
    for cc in clusters:
        members.extend(cc.member_ids)
    if not members:
        return None

    tids = list(set(members))
    idx = {t: i for i, t in enumerate(tids)}
    pred = [-1] * len(tids)
    true_l = [-1] * len(tids)

    for cc in clusters:
        cid = hash(cc.cluster_id) % (10**8)
        for tid in cc.member_ids:
            if tid in idx:
                pred[idx[tid]] = cid
            if tid in epitope_map:
                true_l[idx[tid]] = hash(epitope_map[tid]) % (10**8)

    valid = [i for i in range(len(tids)) if pred[i] != -1]
    if len(valid) < 2:
        return None

    lp = [pred[i] for i in valid]
    lt = [true_l[i] for i in valid]

    ari_val = adjusted_rand_score(lt, lp)
    ami_val = adjusted_mutual_info_score(lt, lp)

    cluster_epis = {}
    for i, p in enumerate(lp):
        cluster_epis.setdefault(p, []).append(lt[i])
    purity = sum(Counter(v).most_common(1)[0][1] for v in cluster_epis.values()) / len(valid)

    from itertools import combinations
    true_clusters = {}
    for i, t in enumerate(lt):
        true_clusters.setdefault(t, []).append(i)
    same_epi_pairs = 0
    same_cluster_same_epi = 0
    for t, members_list in true_clusters.items():
        if len(members_list) < 2:
            continue
        for i, j in combinations(members_list, 2):
            same_epi_pairs += 1
            if lp[i] == lp[j]:
                same_cluster_same_epi += 1
    sensitivity = same_cluster_same_epi / same_epi_pairs if same_epi_pairs > 0 else 0

    pred_clusters = {}
    for i, p in enumerate(lp):
        pred_clusters.setdefault(p, []).append(i)
    co_clustered_pairs = 0
    correct_pairs = 0
    for p, members_list in pred_clusters.items():
        if len(members_list) < 2:
            continue
        for i, j in combinations(members_list, 2):
            co_clustered_pairs += 1
            if lt[i] == lt[j]:
                correct_pairs += 1
    precision = correct_pairs / co_clustered_pairs if co_clustered_pairs > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    retention = len(valid) / total_tcrs

    return {
        "ari": round(ari_val, 4),
        "purity": round(purity, 4),
        "ami": round(ami_val, 4),
        "sensitivity": round(sensitivity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "retention": round(retention, 4),
        "n_clustered": len(valid),
        "n_clusters": len(clusters),
    }


# ============ Run all methods ============
clusterers = [
    ("hd_baseline", HDBaselineClusterer()),
    ("clustcr", ClusTCRWrapper()),
    ("tcrdist3", TCRDist3Wrapper()),
    ("gliph2", GLIPH2Wrapper()),
    ("giana", GIANAWrapper()),
    ("tcrmatch", TCRMatchWrapper()),
    ("deeptcr", DeepTCRWrapper()),
]

all_results = {}
all_assignments = {}

print(f"\n--- Running 7 individual methods ---", flush=True)
for name, wrapper in clusterers:
    mdir = OUT_DIR / name
    mdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        result = wrapper.safe_execute(df_norm, mdir, {})
        elapsed = time.time() - t0
        if result.assignments:
            all_assignments[name] = result.assignments
            metrics = compute_metrics_from_assignments(result.assignments, epitope_map, n_total)
            if metrics:
                all_results[name] = metrics
                print(f"  {name:12s}: ARI={metrics['ari']:.4f} Pur={metrics['purity']:.4f} "
                      f"AMI={metrics['ami']:.4f} Sen={metrics['sensitivity']:.4f} "
                      f"F1={metrics['f1']:.4f} Ret={metrics['retention']:.4f} "
                      f"({elapsed:.0f}s)", flush=True)
            else:
                print(f"  {name:12s}: metrics failed ({elapsed:.0f}s)", flush=True)
        else:
            print(f"  {name:12s}: FAILED ({elapsed:.0f}s)", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {name:12s}: ERROR - {e} ({elapsed:.0f}s)", flush=True)

# ============ Run tcrconsensus ============
print(f"\n--- Running tcrconsensus (CC+Empirical) ---", flush=True)
methods = sorted(set(a.method for a in sum(all_assignments.values(), [])))
print(f"  Active methods: {methods}", flush=True)

weights = empirical_weights(methods)
print(f"  Empirical weights:", flush=True)
for m in sorted(weights, key=weights.get, reverse=True):
    print(f"    {m}: {weights[m]:.4f}", flush=True)

all_a = []
for a_list in all_assignments.values():
    all_a.extend(a_list)

t0 = time.time()
edges = extract_pairwise_support(all_a, weights)
graph = build_consensus_graph(edges, threshold=0.3)
clusters = connected_components_clustering(graph)
if clusters:
    clusters = refine(clusters, edges, {})
elapsed = time.time() - t0
print(f"  Consensus: {len(clusters)} clusters ({elapsed:.1f}s)", flush=True)

consensus_metrics = compute_metrics_from_clusters(clusters, epitope_map, n_total)
if consensus_metrics:
    all_results["tcrconsensus"] = consensus_metrics
    print(f"  tcrconsensus: ARI={consensus_metrics['ari']:.4f} Pur={consensus_metrics['purity']:.4f} "
          f"AMI={consensus_metrics['ami']:.4f} Sen={consensus_metrics['sensitivity']:.4f} "
          f"F1={consensus_metrics['f1']:.4f} Ret={consensus_metrics['retention']:.4f}", flush=True)

# Save JSON
with open(OUT_DIR / "full_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'full_comparison.json'}", flush=True)


# ============================= VISUALIZATION =============================

DISPLAY_NAMES = {
    "tcrconsensus": "tcrconsensus\n(CC+Empirical)",
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

# Sort methods by ARI descending
method_order = ["tcrconsensus"] + sorted(
    [m for m in all_results if m != "tcrconsensus"],
    key=lambda m: all_results[m]["ari"], reverse=True
)


# ============ Fig 1: Multi-metric grouped bar chart ============
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

plot_metrics = [
    ("ari", "ARI"),
    ("purity", "Purity"),
    ("ami", "AMI"),
    ("sensitivity", "Sensitivity"),
    ("f1", "F1"),
    ("retention", "Retention"),
]

for ax, (metric, ylabel) in zip(axes.flatten(), plot_metrics):
    x = np.arange(len(method_order))
    vals = [all_results[m][metric] for m in method_order]
    colors = [COLORS[m] for m in method_order]

    bars = ax.bar(x, vals, width=0.7, color=colors,
                  edgecolor="black", linewidth=0.5)

    # Highlight tcrconsensus
    bars[0].set_edgecolor("#08306b")
    bars[0].set_linewidth(2.5)
    bars[0].set_hatch("///")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in method_order], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)

    # Mark best
    best_idx = vals.index(max(vals))
    for i, v in enumerate(vals):
        label = f"{v:.4f}"
        if i == best_idx:
            label = f"★ {v:.4f}"
        ax.text(i, v + 0.008, label, ha="center", fontsize=8,
                fontweight="bold" if i in (0, best_idx) else "normal",
                color="#08306b" if i == best_idx else "black")

fig.suptitle("Full-Dataset Comparison: tcrconsensus vs Individual Methods\n(High-Confidence Benchmark, 3118 TCRs, 47 Epitopes)", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_full_comparison_bar.png")
print(f"Saved: fig_full_comparison_bar.png", flush=True)
plt.close()


# ============ Fig 2: Radar chart ============
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

radar_methods = method_order[:5]  # consensus + top 4
radar_metrics = [("ari","ARI"), ("purity","Purity"), ("sensitivity","Sensitivity"),
                 ("f1","F1"), ("retention","Retention"), ("ami","AMI")]
n_m = len(radar_metrics)
angles = np.linspace(0, 2*np.pi, n_m, endpoint=False).tolist()
angles += angles[:1]

# Normalize
m_max = {}
for metric, _ in radar_metrics:
    m_max[metric] = max(all_results[m][metric] for m in radar_methods) * 1.15

for m in radar_methods:
    vals = [all_results[m][met] / m_max[met] for met, _ in radar_metrics]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2.5, label=DISPLAY_NAMES[m].replace("\n"," "),
            color=COLORS[m], markersize=6)
    ax.fill(angles, vals, alpha=0.06, color=COLORS[m])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([label for _, label in radar_metrics], fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_title("tcrconsensus vs Top Individual Methods\n(Normalized Metrics, Full Dataset)", fontsize=14, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=10)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_full_comparison_radar.png")
print(f"Saved: fig_full_comparison_radar.png", flush=True)
plt.close()


# ============ Fig 3: Heatmap ============
fig, ax = plt.subplots(figsize=(14, 7))

hm_metrics = [("ari","ARI"), ("purity","Purity"), ("sensitivity","Sensitivity"),
              ("f1","F1"), ("retention","Retention"), ("ami","AMI")]

matrix = np.zeros((len(method_order), len(hm_metrics)))
for i, m in enumerate(method_order):
    for j, (met, _) in enumerate(hm_metrics):
        matrix[i, j] = all_results[m][met]

im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

ax.set_xticks(np.arange(len(hm_metrics)))
ax.set_xticklabels([l for _, l in hm_metrics])
ax.set_yticks(np.arange(len(method_order)))
ax.set_yticklabels([DISPLAY_NAMES[m].replace("\n"," ") for m in method_order])

for i in range(len(method_order)):
    for j in range(len(hm_metrics)):
        v = matrix[i, j]
        color = "white" if v > matrix.max() * 0.7 else "black"
        bold = "bold" if method_order[i] == "tcrconsensus" else "normal"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color=color, fontweight=bold)

# Border for consensus row
for j in range(len(hm_metrics)):
    rect = plt.Rectangle((j-0.5, -0.5), 1, 1, linewidth=3, edgecolor="#08306b", facecolor="none")
    ax.add_patch(rect)

# Mark best per column
for j, (met, _) in enumerate(hm_metrics):
    best_i = np.argmax(matrix[:, j])
    if best_i != 0:
        ax.text(j, best_i, f"{matrix[best_i,j]:.3f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white" if matrix[best_i,j] > matrix.max()*0.7 else "#08306b")

plt.colorbar(im, ax=ax, label="Metric Value", shrink=0.8)
ax.set_title("Full-Dataset Multi-Metric Heatmap: tcrconsensus vs Individual Methods\n(3118 TCRs, 47 Epitopes)")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_full_comparison_heatmap.png")
print(f"Saved: fig_full_comparison_heatmap.png", flush=True)
plt.close()


# ============ Fig 4: Summary table ============
fig, ax = plt.subplots(figsize=(16, 5))
ax.axis("off")

col_labels = ["Method", "ARI", "Purity", "AMI", "Sensitivity", "Precision", "F1", "Retention", "Clusters"]
row_data = []

for m in method_order:
    r = all_results[m]
    row_data.append([
        DISPLAY_NAMES[m].replace("\n", " "),
        f"{r['ari']:.4f}",
        f"{r['purity']:.4f}",
        f"{r['ami']:.4f}",
        f"{r['sensitivity']:.4f}",
        f"{r['precision']:.4f}",
        f"{r['f1']:.4f}",
        f"{r['retention']:.4f}",
        str(r.get('n_clusters', 'N/A')),
    ])

table = ax.table(cellText=row_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colColours=["#d9e2f3"] * len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

for j in range(len(col_labels)):
    table[1, j].set_facecolor("#d4e6f1")
    table[1, j].set_text_props(fontweight="bold")

for i in range(1, len(row_data)+1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        if i > 1 and i % 2 == 0:
            cell.set_facecolor("#f7f7f7")

ax.set_title("Full-Dataset Performance: tcrconsensus vs Individual Methods\n(3118 TCRs, 47 Epitopes, seed=42)",
             fontsize=13, pad=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_full_comparison_table.png")
print(f"Saved: fig_full_comparison_table.png", flush=True)
plt.close()


# ============ Print summary ============
print(f"\n{'='*90}", flush=True)
print(f"Full-Dataset Comparison Summary", flush=True)
print(f"{'='*90}", flush=True)
print(f"{'Method':25s} {'ARI':>8s} {'Pur':>8s} {'AMI':>8s} {'Sen':>8s} {'Pre':>8s} {'F1':>8s} {'Ret':>8s} {'Cls':>6s}", flush=True)
print(f"{'-'*91}", flush=True)
for m in method_order:
    r = all_results[m]
    marker = " ◀" if m == "tcrconsensus" else ""
    print(f"{DISPLAY_NAMES[m].replace(chr(10),' '):25s} "
          f"{r['ari']:8.4f} {r['purity']:8.4f} {r['ami']:8.4f} "
          f"{r['sensitivity']:8.4f} {r['precision']:8.4f} {r['f1']:8.4f} "
          f"{r['retention']:8.4f} {r.get('n_clusters',0):6d}{marker}", flush=True)

print(f"\nFigures saved to {FIG_DIR}/:", flush=True)
for f in sorted(FIG_DIR.glob("fig_full_comparison_*.png")):
    print(f"  {f.name}", flush=True)
print("\nDone!", flush=True)
