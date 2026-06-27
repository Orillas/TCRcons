#!/usr/bin/env python3
"""Independent VDJdb validation: tcrconsensus vs 7 individual methods.

Splits VDJdb filtered data by species, runs all 8 methods on each split,
computes ARI/AMI/Purity/Sensitivity/F1/Retention, generates comparison figures.

Validation datasets:
  A) vdjdb_28423320 (Human, ~6035 TCRs, 2 epitopes)
  B) vdjdb_28636592 Human subset (~279 TCRs, 3 epitopes)
  C) vdjdb_28636592 Mouse subset (~1254 TCRs, 7 epitopes)
"""

import sys
import os
import time
import logging
import warnings
import json
from pathlib import Path
from collections import Counter
from itertools import combinations

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

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

DATA_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/data")
OUT_DIR  = Path("/home/jilin/DeepTCR/tcrconsensus/results/vdjdb_validation")
FIG_DIR  = Path("/home/jilin/DeepTCR/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============ Load & split VDJdb data ============
def load_vdjdb_filtered(path, species_filter=None):
    """Load a VDJdb filtered TSV and convert to benchmark format."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df[df["gene"] == "TRB"].copy()

    if species_filter:
        df = df[df["species"] == species_filter].copy()

    # Map columns to canonical lowercase names expected by normalize()
    out = pd.DataFrame()
    out["cdr3_beta"] = df["cdr3"]
    out["v_beta"] = df["v.segm"]
    out["j_beta"] = df["j.segm"]
    out["epitope"] = df["antigen.epitope"]
    out["tcr_id"] = [f"VDJDB_{i}" for i in range(len(out))]

    # Drop rows with missing CDR3 or Epitope
    out = out.dropna(subset=["cdr3_beta", "epitope"])
    out = out[out["cdr3_beta"].str.len() >= 6].copy()
    out = out.reset_index(drop=True)
    return out


# Prepare 3 validation datasets
datasets = {}

# A) vdjdb_28423320 — all human, 2 epitopes
ds_a = load_vdjdb_filtered(DATA_DIR / "vdjdb_28423320_filtered.tsv")
if len(ds_a) > 0:
    datasets["vdjdb_28423320_human"] = ds_a

# B) vdjdb_28636592 — human subset
ds_b = load_vdjdb_filtered(DATA_DIR / "vdjdb_28636592_filtered.tsv", species_filter="HomoSapiens")
if len(ds_b) > 0:
    datasets["vdjdb_28636592_human"] = ds_b

# C) vdjdb_28636592 — mouse subset
ds_c = load_vdjdb_filtered(DATA_DIR / "vdjdb_28636592_filtered.tsv", species_filter="MusMusculus")
if len(ds_c) > 0:
    datasets["vdjdb_28636592_mouse"] = ds_c

print("=" * 70, flush=True)
print("VDJdb Independent Validation: tcrconsensus vs 7 Methods", flush=True)
print("=" * 70, flush=True)
print(f"Prepared {len(datasets)} validation datasets:", flush=True)
for name, df in datasets.items():
    n_epi = df["epitope"].nunique()
    print(f"  {name}: {len(df)} TCRs, {n_epi} epitopes", flush=True)
    for epi, cnt in df["epitope"].value_counts().items():
        print(f"    {epi}: {cnt}", flush=True)


# ============ Metrics ============
def compute_metrics_from_assignments(assignments, epitope_map, total_tcrs):
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

    cluster_epis = {}
    for i, p in enumerate(lp):
        cluster_epis.setdefault(p, []).append(lt[i])
    purity = sum(Counter(v).most_common(1)[0][1] for v in cluster_epis.values()) / len(valid)

    true_clusters = {}
    for i, t in enumerate(lt):
        true_clusters.setdefault(t, []).append(i)
    same_epi_pairs = 0
    same_cluster_same_epi = 0
    for t, members in true_clusters.items():
        if len(members) < 2:
            continue
        for i, j in combinations(members, 2):
            same_epi_pairs += 1
            if lp[i] == lp[j]:
                same_cluster_same_epi += 1
    sensitivity = same_cluster_same_epi / same_epi_pairs if same_epi_pairs > 0 else 0

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


# ============ Clusterers ============
clusterers = [
    ("hd_baseline", HDBaselineClusterer()),
    ("clustcr",     ClusTCRWrapper()),
    ("tcrdist3",    TCRDist3Wrapper()),
    ("gliph2",      GLIPH2Wrapper()),
    ("giana",       GIANAWrapper()),
    ("tcrmatch",    TCRMatchWrapper()),
    ("deeptcr",     DeepTCRWrapper()),
]

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


# ============ Run validation on each dataset ============
all_validation = {}

for ds_name, df_raw in datasets.items():
    print(f"\n{'='*70}", flush=True)
    print(f"Dataset: {ds_name}", flush=True)
    print(f"{'='*70}", flush=True)

    n_total = len(df_raw)
    n_epitopes = df_raw["epitope"].nunique()
    print(f"  {n_total} TCRs, {n_epitopes} epitopes", flush=True)

    np.random.seed(42)

    # Normalize
    df_norm = normalize(df_raw.copy())

    # Build epitope map
    epitope_map = {}
    for _, row in df_norm.iterrows():
        tid = str(row.get("tcr_id", ""))
        epi = str(row.get("epitope", ""))
        if tid and epi:
            epitope_map[tid] = epi

    # Run individual methods
    ds_results = {}
    ds_assignments = {}

    print(f"  --- Running 7 individual methods ---", flush=True)
    for name, wrapper in clusterers:
        mdir = OUT_DIR / ds_name / name
        mdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = wrapper.safe_execute(df_norm, mdir, {})
            elapsed = time.time() - t0
            if result.assignments:
                ds_assignments[name] = result.assignments
                metrics = compute_metrics_from_assignments(result.assignments, epitope_map, n_total)
                if metrics:
                    ds_results[name] = metrics
                    print(f"    {name:12s}: ARI={metrics['ari']:.4f} Pur={metrics['purity']:.4f} "
                          f"AMI={metrics['ami']:.4f} Sen={metrics['sensitivity']:.4f} "
                          f"F1={metrics['f1']:.4f} Ret={metrics['retention']:.4f} "
                          f"({elapsed:.0f}s)", flush=True)
                else:
                    print(f"    {name:12s}: metrics failed ({elapsed:.0f}s)", flush=True)
            else:
                print(f"    {name:12s}: NO assignments ({elapsed:.0f}s)", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    {name:12s}: ERROR - {e} ({elapsed:.0f}s)", flush=True)

    # Run tcrconsensus
    print(f"  --- Running tcrconsensus ---", flush=True)
    active_methods = sorted(set(a.method for a in sum(ds_assignments.values(), [])))
    print(f"    Active methods: {active_methods}", flush=True)

    if len(active_methods) >= 2:
        weights = empirical_weights(active_methods)
        print(f"    Empirical weights:", flush=True)
        for m in sorted(weights, key=weights.get, reverse=True):
            print(f"      {m}: {weights[m]:.4f}", flush=True)

        all_a = []
        for a_list in ds_assignments.values():
            all_a.extend(a_list)

        t0 = time.time()
        edges = extract_pairwise_support(all_a, weights)
        graph = build_consensus_graph(edges, threshold=0.3)
        clusters = connected_components_clustering(graph)
        if clusters:
            clusters = refine(clusters, edges, {})
        elapsed = time.time() - t0
        print(f"    Consensus: {len(clusters) if clusters else 0} clusters ({elapsed:.1f}s)", flush=True)

        if clusters:
            consensus_metrics = compute_metrics_from_clusters(clusters, epitope_map, n_total)
            if consensus_metrics:
                ds_results["tcrconsensus"] = consensus_metrics
                print(f"    tcrconsensus: ARI={consensus_metrics['ari']:.4f} Pur={consensus_metrics['purity']:.4f} "
                      f"AMI={consensus_metrics['ami']:.4f} Sen={consensus_metrics['sensitivity']:.4f} "
                      f"F1={consensus_metrics['f1']:.4f} Ret={consensus_metrics['retention']:.4f}", flush=True)
    else:
        print(f"    SKIP: fewer than 2 individual methods succeeded", flush=True)

    all_validation[ds_name] = ds_results

    # Save per-dataset JSON
    with open(OUT_DIR / ds_name / "validation_results.json", "w") as f:
        json.dump(ds_results, f, indent=2)
    print(f"  Saved: {OUT_DIR / ds_name / 'validation_results.json'}", flush=True)


# ============ Save combined results ============
# Check if any dataset produced results
datasets_with_results = {k: v for k, v in all_validation.items() if v}
if not datasets_with_results:
    print("\nERROR: No methods produced assignments on any dataset!", flush=True)
    print("This likely means the column mapping is incorrect.", flush=True)
    with open(OUT_DIR / "all_vdjdb_validation.json", "w") as f:
        json.dump(all_validation, f, indent=2)
    sys.exit(1)

with open(OUT_DIR / "all_vdjdb_validation.json", "w") as f:
    json.dump(all_validation, f, indent=2)
print(f"\nSaved combined: {OUT_DIR / 'all_vdjdb_validation.json'}", flush=True)

# Only keep datasets with results for visualization
ds_names = list(datasets_with_results.keys())


# ============================= VISUALIZATION =============================

METRICS_LIST = [
    ("ari", "ARI"),
    ("purity", "Purity"),
    ("ami", "AMI"),
    ("sensitivity", "Sensitivity"),
    ("f1", "F1"),
    ("retention", "Retention"),
]


# ============ Fig 1: Per-dataset grouped bar (ARI + AMI + F1) ============
ds_names = list(datasets_with_results.keys())
ds_labels = {
    "vdjdb_28423320_human": "VDJdb 28423320\nHuman (6035 TCRs, 2 epi)",
    "vdjdb_28636592_human": "VDJdb 28636592\nHuman (279 TCRs, 3 epi)",
    "vdjdb_28636592_mouse": "VDJdb 28636592\nMouse (1254 TCRs, 7 epi)",
}

for ds_name in ds_names:
    ds_results = datasets_with_results[ds_name]
    if not ds_results:
        continue

    method_order = ["tcrconsensus"] + sorted(
        [m for m in ds_results if m != "tcrconsensus"],
        key=lambda m: ds_results[m]["ari"], reverse=True
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"VDJdb Validation: {ds_labels.get(ds_name, ds_name)}", fontsize=15, y=1.02)

    for ax, (metric, ylabel) in zip(axes.flatten(), METRICS_LIST):
        x = np.arange(len(method_order))
        vals = [ds_results[m][metric] for m in method_order]
        colors = [COLORS[m] for m in method_order]

        bars = ax.bar(x, vals, width=0.7, color=colors, edgecolor="black", linewidth=0.5)
        bars[0].set_edgecolor("#08306b")
        bars[0].set_linewidth(2.5)
        bars[0].set_hatch("///")

        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES[m] for m in method_order], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3)

        best_idx = vals.index(max(vals))
        for i, v in enumerate(vals):
            label = f"{v:.4f}"
            if i == best_idx:
                label = f"★ {v:.4f}"
            ax.text(i, v + 0.008, label, ha="center", fontsize=8,
                    fontweight="bold" if i in (0, best_idx) else "normal",
                    color="#08306b" if i == best_idx else "black")

    fig.tight_layout()
    safe_name = ds_name.replace("/", "_")
    fig.savefig(FIG_DIR / f"fig_vdjdb_val_{safe_name}_bar.png")
    print(f"Saved: fig_vdjdb_val_{safe_name}_bar.png", flush=True)
    plt.close()


# ============ Fig 2: Cross-dataset ARI comparison heatmap ============
# Methods present in all datasets
if len(ds_names) >= 2:
    common_methods = set.intersection(*[set(r.keys()) for r in datasets_with_results.values()])
    # Always include tcrconsensus if present
    if "tcrconsensus" in set().union(*[set(r.keys()) for r in datasets_with_results.values()]):
        all_methods_union = set().union(*[set(r.keys()) for r in datasets_with_results.values()])
    else:
        all_methods_union = common_methods

    heatmap_methods = sorted(all_methods_union, key=lambda m: (
        0 if m == "tcrconsensus" else 1,
        -np.mean([datasets_with_results[d].get(m, {}).get("ari", 0) for d in ds_names
                  if m in datasets_with_results[d]])
    ))

    fig, ax = plt.subplots(figsize=(14, 7))

    matrix = np.zeros((len(heatmap_methods), len(ds_names)))
    for i, m in enumerate(heatmap_methods):
        for j, ds in enumerate(ds_names):
            matrix[i, j] = datasets_with_results[ds].get(m, {}).get("ari", 0)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    col_labels = [ds_labels.get(d, d).replace("\n", " ") for d in ds_names]
    ax.set_xticks(np.arange(len(ds_names)))
    ax.set_xticklabels(col_labels, rotation=15, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(heatmap_methods)))
    ax.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in heatmap_methods])

    for i in range(len(heatmap_methods)):
        for j in range(len(ds_names)):
            v = matrix[i, j]
            color = "white" if v > matrix.max() * 0.7 else "black"
            bold = "bold" if heatmap_methods[i] == "tcrconsensus" else "normal"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=10, color=color, fontweight=bold)

    # Highlight consensus row
    if "tcrconsensus" in heatmap_methods:
        cons_idx = heatmap_methods.index("tcrconsensus")
        for j in range(len(ds_names)):
            rect = plt.Rectangle((j - 0.5, cons_idx - 0.5), 1, 1,
                                 linewidth=3, edgecolor="#08306b", facecolor="none")
            ax.add_patch(rect)

    # Mark best per column
    for j in range(len(ds_names)):
        best_i = np.argmax(matrix[:, j])
        if best_i != heatmap_methods.index("tcrconsensus") if "tcrconsensus" in heatmap_methods else True:
            ax.text(j, best_i, f"{matrix[best_i, j]:.4f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if matrix[best_i, j] > matrix.max() * 0.7 else "#08306b")

    plt.colorbar(im, ax=ax, label="ARI", shrink=0.8)
    ax.set_title("Cross-Dataset Validation: ARI on Independent VDJdb Data", fontsize=14)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_vdjdb_val_cross_dataset_heatmap.png")
    print(f"Saved: fig_vdjdb_val_cross_dataset_heatmap.png", flush=True)
    plt.close()


# ============ Fig 3: Summary table (all datasets) ============
n_ds = len(ds_names)
fig_height = 2 + 1.2 * (8 * n_ds + 1)
fig, ax = plt.subplots(figsize=(18, fig_height))
ax.axis("off")

col_labels = ["Dataset", "Method", "ARI", "Purity", "AMI", "Sensitivity", "F1", "Retention", "Clusters"]
row_data = []
row_colors = []

for ds_name in ds_names:
    ds_results = datasets_with_results[ds_name]
    if not ds_results:
        continue

    method_order = ["tcrconsensus"] + sorted(
        [m for m in ds_results if m != "tcrconsensus"],
        key=lambda m: ds_results[m]["ari"], reverse=True
    )

    for mi, m in enumerate(method_order):
        r = ds_results[m]
        row_data.append([
            ds_labels.get(ds_name, ds_name).replace("\n", " ") if mi == 0 else "",
            DISPLAY_NAMES.get(m, m),
            f"{r['ari']:.4f}",
            f"{r['purity']:.4f}",
            f"{r['ami']:.4f}",
            f"{r['sensitivity']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['retention']:.4f}",
            str(r.get("n_clusters", "N/A")),
        ])
        if m == "tcrconsensus":
            row_colors.append("#d4e6f1")
        elif len(row_data) % 2 == 0:
            row_colors.append("#f7f7f7")
        else:
            row_colors.append("white")

table = ax.table(cellText=row_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colColours=["#b3cde3"] * len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

for i, color in enumerate(row_colors):
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)
        if row_data[i][1] == "tcrconsensus":
            table[i + 1, j].set_text_props(fontweight="bold")

ax.set_title("VDJdb Independent Validation Summary\n(Human + Mouse, filtered data)",
             fontsize=14, pad=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_vdjdb_val_summary_table.png")
print(f"Saved: fig_vdjdb_val_summary_table.png", flush=True)
plt.close()


# ============ Fig 4: Cross-dataset multi-metric heatmap (ARI, AMI, F1) ============
if len(ds_names) >= 2:
    key_metrics = [("ari", "ARI"), ("ami", "AMI"), ("f1", "F1")]

    # Collect all methods that appear in any dataset
    all_methods = set()
    for ds_name in ds_names:
        all_methods.update(datasets_with_results[ds_name].keys())
    hm_methods = sorted(all_methods, key=lambda m: (
        0 if m == "tcrconsensus" else 1,
        -np.mean([datasets_with_results[d].get(m, {}).get("ari", 0) for d in ds_names])
    ))

    fig, axes = plt.subplots(1, len(key_metrics), figsize=(6 * len(key_metrics), 8))

    for ax_idx, (metric, mlabel) in enumerate(key_metrics):
        ax = axes[ax_idx]
        matrix = np.zeros((len(hm_methods), len(ds_names)))
        for i, m in enumerate(hm_methods):
            for j, ds in enumerate(ds_names):
                matrix[i, j] = datasets_with_results[ds].get(m, {}).get(metric, 0)

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(np.arange(len(ds_names)))
        short_labels = ["28423320\nHuman", "28636592\nHuman", "28636592\nMouse"]
        ax.set_xticklabels(short_labels[:len(ds_names)], fontsize=9)
        ax.set_yticks(np.arange(len(hm_methods)))
        ax.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in hm_methods])
        ax.set_title(mlabel, fontsize=13)

        for i in range(len(hm_methods)):
            for j in range(len(ds_names)):
                v = matrix[i, j]
                color = "white" if v > matrix.max() * 0.7 else "black"
                bold = "bold" if hm_methods[i] == "tcrconsensus" else "normal"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color=color, fontweight=bold)

        if "tcrconsensus" in hm_methods:
            cons_idx = hm_methods.index("tcrconsensus")
            for j in range(len(ds_names)):
                rect = plt.Rectangle((j - 0.5, cons_idx - 0.5), 1, 1,
                                     linewidth=2.5, edgecolor="#08306b", facecolor="none")
                ax.add_patch(rect)

        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("VDJdb Cross-Dataset Validation: Key Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_vdjdb_val_cross_metrics.png")
    print(f"Saved: fig_vdjdb_val_cross_metrics.png", flush=True)
    plt.close()


# ============ Print final summary ============
print(f"\n{'='*100}", flush=True)
print(f"VDJdb Independent Validation Summary", flush=True)
print(f"{'='*100}", flush=True)

for ds_name in ds_names:
    ds_results = datasets_with_results[ds_name]
    if not ds_results:
        continue

    method_order = ["tcrconsensus"] + sorted(
        [m for m in ds_results if m != "tcrconsensus"],
        key=lambda m: ds_results[m]["ari"], reverse=True
    )

    print(f"\n  [{ds_labels.get(ds_name, ds_name).replace(chr(10), ' ')}]", flush=True)
    print(f"  {'Method':20s} {'ARI':>8s} {'Pur':>8s} {'AMI':>8s} {'Sen':>8s} {'Pre':>8s} {'F1':>8s} {'Ret':>8s} {'Cls':>6s}", flush=True)
    print(f"  {'-'*82}", flush=True)

    for m in method_order:
        r = ds_results[m]
        marker = " ◀" if m == "tcrconsensus" else ""
        print(f"  {DISPLAY_NAMES.get(m,m):20s} "
              f"{r['ari']:8.4f} {r['purity']:8.4f} {r['ami']:8.4f} "
              f"{r['sensitivity']:8.4f} {r['precision']:8.4f} {r['f1']:8.4f} "
              f"{r['retention']:8.4f} {r.get('n_clusters',0):6d}{marker}", flush=True)

print(f"\nFigures saved to {FIG_DIR}/:", flush=True)
for f in sorted(FIG_DIR.glob("fig_vdjdb_val_*.png")):
    print(f"  {f.name}", flush=True)

print(f"\nResults saved to {OUT_DIR}/:", flush=True)
for f in sorted(OUT_DIR.rglob("*.json")):
    print(f"  {f.relative_to(OUT_DIR)}", flush=True)

print("\nVDJdb Validation Complete!", flush=True)
