#!/usr/bin/env python3
"""P0 Algorithm Ablation: Same co-association matrix, different clustering strategies.

Proves that connected components (CC) is optimal for TCR consensus clustering,
supporting the Occam's razor argument.

Strategies compared:
1. Connected Components (current) - threshold + CC
2. Leiden community detection
3. Louvain community detection
4. Hierarchical clustering (average linkage) + optimal dendrogram cut
5. Spectral clustering (normalized cut)

All use the SAME co-association matrix and weights.
Evaluation on v3_all (whole-dataset) + 10X stress test subset 1.
"""

import sys, time, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/scripts")
sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.consensus.coassociation import extract_pairwise_support, build_coassociation_matrix
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
from tcrconsensus.refinement.refiner import refine
from exp_shared import (
    get_all_clusterers, run_all_methods,
    assignments_to_labels, clusters_to_labels,
    evaluate_clustering,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from scipy.sparse.csgraph import connected_components as sp_connected_components
from scipy.sparse import csr_matrix
import networkx as nx

# Paths
DATA_BASE = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1")
GLIPH2_DIR = DATA_BASE / "input" / "Gliph2"
LABEL_JSON = "/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_h5.json"
V3_ALL = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_all.tsv"
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments/algorithm_ablation")
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(LABEL_JSON) as f:
    CDR3_EPI = json.load(f)["cdr3_to_epitopes"]

config_obj = load_config()
config = dict(config_obj.__dict__)
all_clusterers = get_all_clusterers()
clusterers = all_clusterers


def get_cdr3_epitope(cdr3):
    if cdr3 in CDR3_EPI:
        epis = CDR3_EPI[cdr3]
        return epis[0] if len(epis) == 1 else "MULTI:" + ";".join(sorted(epis))
    return None


def evaluate_with_metrics(pred, true_labels):
    """Evaluate with ARI + NMI, handling -1 unclustered."""
    valid = np.array([str(p) not in ("-1", "") for p in pred], dtype=bool)
    if valid.sum() < 2:
        return {"ari": 0.0, "nmi": 0.0, "n_clustered": int(valid.sum())}

    from sklearn.preprocessing import LabelEncoder
    le_t = LabelEncoder()
    le_p = LabelEncoder()
    true_str = true_labels[valid]
    pred_str = pred[valid].astype(str)
    t_enc = le_t.fit_transform(true_str)
    p_enc = le_p.fit_transform(pred_str)

    return {
        "ari": float(adjusted_rand_score(t_enc, p_enc)),
        "nmi": float(normalized_mutual_info_score(t_enc, p_enc)),
        "n_clustered": int(valid.sum()),
        "n_clusters": len(set(p_enc)),
    }


# ── Strategy implementations ──

def strategy_cc(edges, tcr_ids, threshold=0.3):
    """Connected Components (current approach)."""
    graph = build_consensus_graph(edges, threshold=threshold)
    clusters = connected_components_clustering(graph)
    return clusters_to_labels(clusters, tcr_ids)


def strategy_leiden(edges, tcr_ids, threshold=0.3, resolution=1.0):
    """Leiden community detection."""
    graph = build_consensus_graph(edges, threshold=threshold)
    if len(graph.edges) == 0:
        return np.full(len(tcr_ids), -1, dtype=object)

    try:
        import leidenalg
        import igraph as ig
        ig_graph = ig.Graph.from_networkx(graph)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights="weight",
        )
        label_map = {}
        for i, community in enumerate(partition):
            for v in ig_graph.vs[community]:
                name = v["_nx_name"] if "_nx_name" in v.attributes() else v["name"]
                label_map[str(name)] = f"leiden_{i:04d}"
        return np.array([label_map.get(tid, -1) for tid in tcr_ids], dtype=object)
    except ImportError:
        logger.warning("leidenalg not available, skipping")
        return None


def strategy_louvain(edges, tcr_ids, threshold=0.3, resolution=1.0):
    """Louvain community detection."""
    graph = build_consensus_graph(edges, threshold=threshold)
    if len(graph.edges) == 0:
        return np.full(len(tcr_ids), -1, dtype=object)

    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(graph, resolution=resolution, weight="weight")
    label_map = {}
    for i, community in enumerate(communities):
        for node in community:
            label_map[str(node)] = f"louvain_{i:04d}"
    return np.array([label_map.get(tid, -1) for tid in tcr_ids], dtype=object)


def strategy_hierarchical(edges, tcr_ids, all_assigns, weights, n_clusters=None):
    """Hierarchical clustering (average linkage) on co-association matrix.

    EAC (Evidence Accumulation Clustering) approach from Fred & Jain.
    """
    # Build dense co-association matrix
    matrix = build_coassociation_matrix(all_assigns, list(tcr_ids), weights)

    # Hierarchical clustering
    if n_clusters is None:
        n_clusters = max(5, int(np.sqrt(len(tcr_ids)) * 0.5))

    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    # Convert similarity to distance
    max_val = matrix.max() if matrix.max() > 0 else 1.0
    distance = max_val - matrix
    np.fill_diagonal(distance, 0)

    labels = hc.fit_predict(distance)
    return np.array([f"hc_{l:04d}" for l in labels], dtype=object)


def strategy_spectral(edges, tcr_ids, all_assigns, weights, n_clusters=None):
    """Spectral clustering on co-association matrix."""
    matrix = build_coassociation_matrix(all_assigns, list(tcr_ids), weights)

    if n_clusters is None:
        n_clusters = max(5, int(np.sqrt(len(tcr_ids)) * 0.5))

    # Add small constant for numerical stability
    matrix_shifted = matrix - matrix.min() + 1e-6

    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
            assign_labels="kmeans",
        )
        labels = sc.fit_predict(matrix_shifted)
        return np.array([f"spec_{l:04d}" for l in labels], dtype=object)
    except Exception as e:
        logger.warning(f"Spectral clustering failed: {e}")
        return None


# ── Run on 10X Subset 1 (fast, labeled data available) ──

print("=" * 78)
print("ALGORITHM ABLATION: CC vs Leiden vs Louvain vs Hierarchical vs Spectral")
print("=" * 78)

# Load 10X subset 1
df = pd.read_csv(GLIPH2_DIR / "subset_1.csv", sep="\t")
df = df.rename(columns={"CDR3b": "cdr3_beta", "TRBV": "v_beta", "TRBJ": "j_beta"})
df["tcr_id"] = df["cdr3_beta"]

def get_label(cdr3):
    if cdr3 in CDR3_EPI:
        epis = CDR3_EPI[cdr3]
        return epis[0] if len(epis) == 1 else "MULTI:" + ";".join(sorted(epis))
    return "BACKGROUND"

df["epitope"] = df["cdr3_beta"].apply(get_label)
df = df[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]].drop_duplicates(subset=["cdr3_beta"])

df_norm = normalize(df.copy())
true_labels = df_norm["epitope"].values
tcr_ids = df_norm["tcr_id"].values

print(f"\n10X Subset 1: {len(df_norm)} TCRs, {(df_norm['epitope'] != 'BACKGROUND').sum()} labeled")

# Run all methods
print("\nRunning individual methods...")
t0 = time.time()
method_results = run_all_methods(df_norm, clusterers, config, OUTDIR / "subset_1")
runtime = time.time() - t0
print(f"  Done in {runtime:.1f}s")

all_assigns = []
for mname, (assigns, rt) in method_results.items():
    all_assigns.extend(assigns)

methods_list = sorted(set(a.method for a in all_assigns))
weights = empirical_weights(methods_list)

print(f"\nMethods: {methods_list}")
print(f"Weights: { {m: f'{w:.4f}' for m, w in sorted(weights.items(), key=lambda x: -x[1])} }")

# Extract edges once
edges = extract_pairwise_support(all_assigns, weights)
print(f"Co-association edges: {len(edges)}")

# ── Evaluate all strategies ──
print(f"\n{'=' * 78}")
print(f"{'Strategy':<25s} {'Thresh':>6s} {'N_cls':>6s} {'ARI':>8s} {'NMI':>8s} {'N_clust':>8s} {'Time':>6s}")
print("-" * 78)

results = []

# 1. Connected Components at multiple thresholds
for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    t0 = time.time()
    pred = strategy_cc(edges, tcr_ids, threshold=thresh)
    elapsed = time.time() - t0
    ev = evaluate_with_metrics(pred, true_labels)
    label = f"CC (t={thresh})"
    print(f"  {label:<25s} {thresh:6.2f} {ev['n_clustered']:6d} {ev['ari']:8.4f} {ev['nmi']:8.4f} {ev['n_clusters']:8d} {elapsed:5.2f}s")
    results.append({"strategy": "CC", "threshold": thresh, **ev, "time_s": elapsed})

# 2. Leiden at multiple resolutions
for thresh in [0.3, 0.5]:
    for res in [0.5, 1.0, 2.0]:
        t0 = time.time()
        pred = strategy_leiden(edges, tcr_ids, threshold=thresh, resolution=res)
        elapsed = time.time() - t0
        if pred is not None:
            ev = evaluate_with_metrics(pred, true_labels)
            label = f"Leiden (t={thresh}, r={res})"
            print(f"  {label:<25s} {thresh:6.2f} {ev['n_clustered']:6d} {ev['ari']:8.4f} {ev['nmi']:8.4f} {ev['n_clusters']:8d} {elapsed:5.2f}s")
            results.append({"strategy": "Leiden", "threshold": thresh, "resolution": res, **ev, "time_s": elapsed})

# 3. Louvain at multiple resolutions
for thresh in [0.3, 0.5]:
    for res in [0.5, 1.0, 2.0]:
        t0 = time.time()
        pred = strategy_louvain(edges, tcr_ids, threshold=thresh, resolution=res)
        elapsed = time.time() - t0
        if pred is not None:
            ev = evaluate_with_metrics(pred, true_labels)
            label = f"Louvain (t={thresh}, r={res})"
            print(f"  {label:<25s} {thresh:6.2f} {ev['n_clustered']:6d} {ev['ari']:8.4f} {ev['nmi']:8.4f} {ev['n_clusters']:8d} {elapsed:5.2f}s")
            results.append({"strategy": "Louvain", "threshold": thresh, "resolution": res, **ev, "time_s": elapsed})

# 4. Hierarchical (EAC) at different cluster counts
for n_cl in [30, 50, 75, 100, 150]:
    t0 = time.time()
    pred = strategy_hierarchical(edges, tcr_ids, all_assigns, weights, n_clusters=n_cl)
    elapsed = time.time() - t0
    if pred is not None:
        ev = evaluate_with_metrics(pred, true_labels)
        label = f"Hierarch (k={n_cl})"
        print(f"  {label:<25s} {'--':>6s} {ev['n_clustered']:6d} {ev['ari']:8.4f} {ev['nmi']:8.4f} {ev['n_clusters']:8d} {elapsed:5.2f}s")
        results.append({"strategy": "Hierarchical", "n_clusters_param": n_cl, **ev, "time_s": elapsed})

# 5. Spectral at different cluster counts
for n_cl in [30, 50, 75, 100]:
    t0 = time.time()
    pred = strategy_spectral(edges, tcr_ids, all_assigns, weights, n_clusters=n_cl)
    elapsed = time.time() - t0
    if pred is not None:
        ev = evaluate_with_metrics(pred, true_labels)
        label = f"Spectral (k={n_cl})"
        print(f"  {label:<25s} {'--':>6s} {ev['n_clustered']:6d} {ev['ari']:8.4f} {ev['nmi']:8.4f} {ev['n_clusters']:8d} {elapsed:5.2f}s")
        results.append({"strategy": "Spectral", "n_clusters_param": n_cl, **ev, "time_s": elapsed})

# ── Summary ──
print(f"\n{'=' * 78}")
print("RANKING by ARI (top 10)")
print(f"{'=' * 78}")
ranked = sorted(results, key=lambda x: -x["ari"])
for i, r in enumerate(ranked[:10]):
    strat = r["strategy"]
    if "threshold" in r:
        detail = f"t={r['threshold']}"
    elif "n_clusters_param" in r:
        detail = f"k={r['n_clusters_param']}"
    else:
        detail = f"t={r.get('threshold', '?')}, r={r.get('resolution', '?')}"
    print(f"  {i+1:2d}. {strat} ({detail}): ARI={r['ari']:.4f}, NMI={r['nmi']:.4f}, N_cls={r['n_clusters']}")

print(f"\nRANKING by NMI (top 10)")
ranked_nmi = sorted(results, key=lambda x: -x["nmi"])
for i, r in enumerate(ranked_nmi[:10]):
    strat = r["strategy"]
    if "threshold" in r:
        detail = f"t={r['threshold']}"
    elif "n_clusters_param" in r:
        detail = f"k={r['n_clusters_param']}"
    else:
        detail = f"t={r.get('threshold', '?')}, r={r.get('resolution', '?')}"
    print(f"  {i+1:2d}. {strat} ({detail}): NMI={r['nmi']:.4f}, ARI={r['ari']:.4f}, N_cls={r['n_clusters']}")

# Save
with open(OUTDIR / "ablation_results.json", "w") as f:
    json.dump({"results": results, "n_tcrs": len(tcr_ids)}, f, indent=2, default=str)

print(f"\nSaved to {OUTDIR / 'ablation_results.json'}")

# ── Also run on v3_all if available ──
v3_path = Path(V3_ALL)
if v3_path.exists():
    print(f"\n{'=' * 78}")
    print("ALGORITHM ABLATION on v3_all (whole-dataset benchmark)")
    print(f"{'=' * 78}")

    df_v3 = pd.read_csv(v3_path, sep="\t", dtype=str)
    rename_lower = {}
    for col in df_v3.columns:
        low = col.lower()
        if low != col and low in ["cdr3_alpha", "cdr3_beta", "v_alpha", "v_beta",
                                   "j_alpha", "j_beta", "tcr_id", "epitope"]:
            rename_lower[col] = low
    if rename_lower:
        df_v3 = df_v3.rename(columns=rename_lower)

    df_v3_norm = normalize(df_v3.copy())
    true_v3 = df_v3_norm["epitope"].values
    tcr_v3 = df_v3_norm["tcr_id"].values

    print(f"v3_all: {len(df_v3_norm)} TCRs, {len(set(true_v3))} epitopes")

    method_results_v3 = run_all_methods(df_v3_norm, clusterers, config, OUTDIR / "v3_all")
    all_assigns_v3 = []
    for mname, (assigns, rt) in method_results_v3.items():
        all_assigns_v3.extend(assigns)

    methods_v3 = sorted(set(a.method for a in all_assigns_v3))
    weights_v3 = empirical_weights(methods_v3)
    edges_v3 = extract_pairwise_support(all_assigns_v3, weights_v3)

    print(f"\n  {'Strategy':<25s} {'ARI':>8s} {'NMI':>8s} {'N_cls':>8s}")
    print(f"  {'-' * 55}")

    v3_results = []
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        pred = strategy_cc(edges_v3, tcr_v3, threshold=thresh)
        ev = evaluate_with_metrics(pred, true_v3)
        print(f"  {'CC (t=' + str(thresh) + ')':<25s} {ev['ari']:8.4f} {ev['nmi']:8.4f} {ev['n_clusters']:8d}")
        v3_results.append({"strategy": "CC", "threshold": thresh, **ev})

    for res in [0.5, 1.0, 2.0]:
        pred = strategy_leiden(edges_v3, tcr_v3, threshold=0.3, resolution=res)
        if pred is not None:
            ev = evaluate_with_metrics(pred, true_v3)
            print(f"  {'Leiden (r=' + str(res) + ')':<25s} {ev['ari']:8.4f} {ev['nmi']:8.4f} {ev['n_clusters']:8d}")
            v3_results.append({"strategy": "Leiden", "resolution": res, **ev})

    for res in [0.5, 1.0, 2.0]:
        pred = strategy_louvain(edges_v3, tcr_v3, threshold=0.3, resolution=res)
        if pred is not None:
            ev = evaluate_with_metrics(pred, true_v3)
            print(f"  {'Louvain (r=' + str(res) + ')':<25s} {ev['ari']:8.4f} {ev['nmi']:8.4f} {ev['n_clusters']:8d}")
            v3_results.append({"strategy": "Louvain", "resolution": res, **ev})

    with open(OUTDIR / "ablation_v3_results.json", "w") as f:
        json.dump({"results": v3_results, "n_tcrs": len(tcr_v3)}, f, indent=2, default=str)

print("\nAlgorithm Ablation Done!")
