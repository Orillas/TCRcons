#!/usr/bin/env python3
"""Subset 1 Clustering Visualization.

Run tcrconsensus on 10X Donor1 subset 1, then generate:
(A) UMAP projection colored by consensus cluster + true epitope
(B) Cluster composition heatmap (epitope × cluster)
(C) Per-cluster purity bar chart
(D) Top clusters detail: sequence motif + V/J usage
"""

import sys, json, time, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True)
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/scripts")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 12, "font.family": "sans-serif",
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.refinement.refiner import refine
from exp_shared import get_all_clusterers, run_all_methods, clusters_to_labels

try:
    from sklearn.metrics import pairwise_distances
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not available, will use MDS fallback")

from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ── Paths ──
DATA_BASE = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1")
GLIPH2_DIR = DATA_BASE / "input" / "Gliph2"
LABEL_JSON = "/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_h5.json"
FIG_DIR = Path("/home/jilin/DeepTCR/figures")
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/subset1_vis")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(LABEL_JSON) as f:
    CDR3_EPI = json.load(f)["cdr3_to_epitopes"]

config_obj = load_config()
config = dict(config_obj.__dict__)

# ── Color palettes ──
# Epitope colors
EPI_COLORS = {
    "KLGGALQAK": "#e41a1c",
    "AVFDRKSDAK": "#377eb8",
    "GILGFVFTL": "#4daf4a",
    "RLRAEAQVK": "#984ea3",
    "RAKFKQLL": "#ff7f00",
    "IVTDFSVIK": "#a65628",
    "ELAGIGILTV": "#f781bf",
    "RPPIFIRRL": "#999999",
    "CYTWNQMNL": "#66c2a5",
    "FLRGRAYGL": "#fc8d62",
    "BACKGROUND": "#d9d9d9",
}
EPI_ORDER = ["KLGGALQAK", "AVFDRKSDAK", "GILGFVFTL", "RLRAEAQVK",
             "RAKFKQLL", "IVTDFSVIK", "ELAGIGILTV", "RPPIFIRRL"]

# Cluster colors
CLUSTER_CMAP = plt.cm.tab20

# ── Load subset 1 ──
print("=" * 70)
print("Subset 1 Clustering Visualization")
print("=" * 70)

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
n_total = len(tcr_ids)
n_labeled = (df_norm["epitope"] != "BACKGROUND").sum()

print(f"Subset 1: {n_total} TCRs, {n_labeled} labeled, {n_total - n_labeled} background")

# ── Run all methods ──
print("\nRunning 7 individual methods...")
all_clusterers = get_all_clusterers()
t0 = time.time()
method_results = run_all_methods(df_norm, all_clusterers, config, OUT_DIR / "methods")
elapsed = time.time() - t0
print(f"Methods done in {elapsed:.0f}s")

all_assigns = []
for mname, (assigns, rt) in method_results.items():
    all_assigns.extend(assigns)
    print(f"  {mname}: {len(assigns)} assignments")

methods_list = sorted(set(a.method for a in all_assigns))
print(f"\nActive methods: {methods_list}")

# ── Run consensus ──
print("\nRunning tcrconsensus...")
weights = empirical_weights(methods_list)
print("Empirical weights:")
for m, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {m}: {w:.4f}")

edges = extract_pairwise_support(all_assigns, weights)
print(f"Co-association edges: {len(edges)}")

graph = build_consensus_graph(edges, threshold=0.3)
clusters = connected_components_clustering(graph)
clusters = refine(clusters, edges, config)
print(f"Consensus clusters: {len(clusters)}")

# ── Build label arrays ──
pred = clusters_to_labels(clusters, tcr_ids)
pred_arr = np.asarray(pred, dtype=object)

# Map cluster IDs to sequential integers for coloring
unique_clusters = sorted(set(pred_arr) - {-1})
cluster_id_map = {cid: i for i, cid in enumerate(unique_clusters)}
cluster_labels = np.array([cluster_id_map.get(p, -1) for p in pred_arr])

# Epitope labels for coloring
labeled_idx = np.array([tid in CDR3_EPI for tid in tcr_ids])

# ── Compute per-cluster stats ──
cluster_stats = []
for c in clusters:
    members = c.member_ids
    epis = []
    for m in members:
        if m in CDR3_EPI:
            e = CDR3_EPI[m]
            epis.append(e[0] if len(e) == 1 else "MULTI")
    n_labeled_members = len(epis)
    n_bg = len(members) - n_labeled_members

    if epis:
        dominant = Counter(epis).most_common(1)[0]
        purity = dominant[1] / len(members)
        dominant_epi = dominant[0]
    else:
        purity = 0.0
        dominant_epi = "none"

    cluster_stats.append({
        "cluster_id": c.cluster_id,
        "n_members": len(members),
        "n_labeled": n_labeled_members,
        "n_bg": n_bg,
        "purity": purity,
        "dominant_epitope": dominant_epi,
        "confidence": c.cluster_confidence,
        "members": members,
    })

# Sort by purity * size (interesting clusters)
cluster_stats.sort(key=lambda x: -x["purity"] * x["n_labeled"])

print(f"\nTop 10 clusters:")
print(f"  {'ID':<15s} {'Size':>5s} {'Lab':>4s} {'BG':>4s} {'Pur':>6s} {'Dominant':<15s}")
for cs in cluster_stats[:10]:
    print(f"  {cs['cluster_id']:<15s} {cs['n_members']:5d} {cs['n_labeled']:4d} "
          f"{cs['n_bg']:4d} {cs['purity']:6.3f} {cs['dominant_epitope']:<15s}")

# ── Save cluster data JSON ──
save_data = {
    "n_total": n_total,
    "n_labeled": int(n_labeled),
    "n_clusters": len(clusters),
    "n_clustered": int((pred_arr != -1).sum()),
    "clusters": [
        {
            "cluster_id": cs["cluster_id"],
            "n_members": cs["n_members"],
            "n_labeled": cs["n_labeled"],
            "purity": cs["purity"],
            "dominant_epitope": cs["dominant_epitope"],
            "confidence": cs["confidence"],
        }
        for cs in cluster_stats
    ],
}
with open(OUT_DIR / "subset1_clusters.json", "w") as f:
    json.dump(save_data, f, indent=2)


# ================================================================
#  VISUALIZATION
# ================================================================

# ── Figure 1: UMAP projection ──
print("\nComputing dimensionality reduction...")

# Build distance matrix from CDR3 sequences using simple k-mer distance
def kmer_features(seqs, k=3):
    """Convert CDR3 sequences to k-mer frequency vectors."""
    all_kmers = set()
    for s in seqs:
        for i in range(len(s) - k + 1):
            all_kmers.add(s[i:i+k])
    kmer_list = sorted(all_kmers)
    kmer_idx = {km: i for i, km in enumerate(kmer_list)}

    mat = np.zeros((len(seqs), len(kmer_list)))
    for i, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            km = s[j:j+k]
            if km in kmer_idx:
                mat[i, kmer_idx[km]] += 1
    return mat

cdr3s = df_norm["cdr3_beta"].values
feat = kmer_features(cdr3s, k=3)
print(f"K-mer features: {feat.shape}")

if HAS_UMAP:
    print("Running UMAP...")
    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42, metric="cosine")
    embedding = reducer.fit_transform(feat)
else:
    print("Running MDS (UMAP not available)...")
    from sklearn.metrics.pairwise import cosine_distances
    dist = cosine_distances(feat)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=3)
    embedding = mds.fit_transform(dist)

print(f"Embedding shape: {embedding.shape}")


# ── Fig 1A: UMAP colored by TRUE epitope ──
fig1, axes = plt.subplots(1, 2, figsize=(20, 8))

ax = axes[0]
# Plot background first (gray)
bg_mask = true_labels == "BACKGROUND"
ax.scatter(embedding[bg_mask, 0], embedding[bg_mask, 1],
           c="#d9d9d9", s=8, alpha=0.3, rasterized=True, label="Background")

# Plot each epitope
for epi in EPI_ORDER:
    mask = true_labels == epi
    if mask.sum() > 0:
        color = EPI_COLORS.get(epi, "#333333")
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=15, alpha=0.7, rasterized=True, label=f"{epi} ({mask.sum()})")

# Plot other labeled
other_mask = np.array([
    tl not in EPI_ORDER and tl != "BACKGROUND"
    for tl in true_labels
])
if other_mask.sum() > 0:
    ax.scatter(embedding[other_mask, 0], embedding[other_mask, 1],
               c="#333333", s=10, alpha=0.4, rasterized=True, label=f"Other ({other_mask.sum()})")

ax.set_title("(A) True Epitope Labels", fontsize=14, fontweight="bold")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, markerscale=1.5,
          framealpha=0.9, edgecolor="#cccccc")


# ── Fig 1B: UMAP colored by consensus cluster ──
ax = axes[1]

# Unclustered = gray
uncl_mask = pred_arr == -1
ax.scatter(embedding[uncl_mask, 0], embedding[uncl_mask, 1],
           c="#d9d9d9", s=8, alpha=0.2, rasterized=True, label=f"Unclustered ({uncl_mask.sum()})")

# Color each cluster (only those with labeled members)
n_clusters_vis = min(len(unique_clusters), 20)
top_clusters = [cs["cluster_id"] for cs in cluster_stats[:n_clusters_vis] if cs["n_labeled"] > 0]
top_clusters_set = set(top_clusters)

for i, cid in enumerate(unique_clusters):
    if cid in top_clusters_set:
        mask = pred_arr == cid
        cs_entry = next((c for c in cluster_stats if c["cluster_id"] == cid), None)
        n_lab = cs_entry["n_labeled"] if cs_entry else 0
        if n_lab > 0:
            color = CLUSTER_CMAP(i / max(len(unique_clusters), 1))
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color], s=20, alpha=0.8, rasterized=True)
    else:
        mask = pred_arr == cid
        if mask.sum() > 0:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c="#bbbbbb", s=8, alpha=0.3, rasterized=True)

# Build legend for top clusters
handles = [plt.Rectangle((0, 0), 1, 1, color="#d9d9d9", label=f"Unclustered ({uncl_mask.sum()})")]
for i, cid in enumerate(unique_clusters):
    if cid in top_clusters_set:
        cs_entry = next((c for c in cluster_stats if c["cluster_id"] == cid), None)
        n_lab = cs_entry["n_labeled"] if cs_entry else 0
        if n_lab > 0:
            epi = cs_entry["dominant_epitope"] if cs_entry else "?"
            color = CLUSTER_CMAP(i / max(len(unique_clusters), 1))
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color,
                           label=f"C{i}: {epi} (n={cs_entry['n_members']}, pur={cs_entry['purity']:.2f})"))

ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6,
          framealpha=0.9, edgecolor="#cccccc")
ax.set_title("(B) Consensus Clusters", fontsize=14, fontweight="bold")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

fig1.suptitle("Subset 1: tcrconsensus Clustering Results (10X Donor1)\n"
              f"{n_total} TCRs, {len(clusters)} clusters, {int((pred_arr != -1).sum())} clustered",
              fontsize=15, fontweight="bold", y=1.04)
fig1.tight_layout()
fig1.savefig(FIG_DIR / "fig_subset1_umap.png")
print(f"Saved: fig_subset1_umap.png")
plt.close()


# ── Figure 2: Cluster composition heatmap ──
fig2, axes = plt.subplots(1, 2, figsize=(20, 10))

# Panel A: Epitope composition per cluster (top 20 clusters by labeled members)
ax = axes[0]
top20 = [cs for cs in cluster_stats if cs["n_labeled"] > 0][:20]
if top20:
    n_show = len(top20)
    epi_set = set()
    for cs in top20:
        for m in cs["members"]:
            if m in CDR3_EPI:
                e = CDR3_EPI[m]
                epi_set.add(e[0] if len(e) == 1 else "MULTI")
    epi_list = sorted(epi_set)

    matrix = np.zeros((n_show, len(epi_list)))
    for i, cs in enumerate(top20):
        for m in cs["members"]:
            if m in CDR3_EPI:
                e = CDR3_EPI[m]
                ename = e[0] if len(e) == 1 else "MULTI"
                if ename in epi_list:
                    j = epi_list.index(ename)
                    matrix[i, j] += 1

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(epi_list)))
    ax.set_xticklabels(epi_list, rotation=45, ha="right", fontsize=8)
    row_labels = [f"C{i} ({cs['dominant_epitope'][:8]}, n={cs['n_members']})"
                  for i, cs in enumerate(top20)]
    ax.set_yticks(np.arange(n_show))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Epitope")
    ax.set_ylabel("Cluster")
    ax.set_title("(A) Epitope Composition per Cluster", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Count")

    # Annotate cells
    for i in range(n_show):
        for j in range(len(epi_list)):
            v = matrix[i, j]
            if v > 0:
                color = "white" if v > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7, color=color)

# Panel B: Per-cluster purity bar chart
ax = axes[1]
if top20:
    labels = [f"C{i}" for i in range(len(top20))]
    purities = [cs["purity"] for cs in top20]
    n_labs = [cs["n_labeled"] for cs in top20]
    n_bgs = [cs["n_bg"] for cs in top20]

    x = np.arange(len(top20))
    bars_lab = ax.bar(x, n_labs, color="#2171b5", label="Labeled (epitope)", edgecolor="black", linewidth=0.5)
    bars_bg = ax.bar(x, n_bgs, bottom=n_labs, color="#d9d9d9", label="Background", edgecolor="black", linewidth=0.5)

    # Annotate purity on top
    for i, (n_l, pur) in enumerate(zip(n_labs, purities)):
        ax.text(i, n_labs[i] + n_bgs[i] + 0.5, f"{pur:.2f}",
                ha="center", fontsize=7, fontweight="bold", color="#08306b")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Members")
    ax.set_title("(B) Cluster Size & Purity (top 20)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")

fig2.suptitle("Subset 1: Cluster Composition Analysis", fontsize=15, fontweight="bold", y=1.02)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "fig_subset1_composition.png")
print(f"Saved: fig_subset1_composition.png")
plt.close()


# ── Figure 3: Confusion-style matrix + method comparison ──
fig3, axes = plt.subplots(1, 2, figsize=(20, 8))

# Panel A: Confusion matrix (true epitope vs predicted cluster for top epitopes)
ax = axes[0]
# Focus on top 8 epitopes + background
conf_epis = EPI_ORDER[:8]
# Build mapping: epitope -> cluster_id (majority vote)
epi_cluster_dist = defaultdict(lambda: Counter())
for i in range(n_total):
    tl = true_labels[i]
    pl = pred_arr[i]
    if pl != -1 and tl in conf_epis:
        epi_cluster_dist[tl][pl] += 1

# Get top clusters that contain these epitopes
relevant_clusters = set()
for epi in conf_epis:
    relevant_clusters.update(epi_cluster_dist[epi].keys())
# Sort by total count
cluster_total = Counter()
for epi in conf_epis:
    for cid, cnt in epi_cluster_dist[epi].items():
        cluster_total[cid] += cnt
top_cl_ids = [cid for cid, _ in cluster_total.most_common(15)]

# Build confusion matrix
cmatrix = np.zeros((len(conf_epis), len(top_cl_ids)))
for i, epi in enumerate(conf_epis):
    for j, cid in enumerate(top_cl_ids):
        cmatrix[i, j] = epi_cluster_dist[epi].get(cid, 0)

im = ax.imshow(cmatrix, cmap="Blues", aspect="auto")
ax.set_xticks(np.arange(len(top_cl_ids)))
# Label clusters by their dominant epitope
cl_labels = []
for cid in top_cl_ids:
    cs_entry = next((c for c in cluster_stats if c["cluster_id"] == cid), None)
    if cs_entry and cs_entry["n_labeled"] > 0:
        cl_labels.append(f"{cs_entry['dominant_epitope'][:6]}\n(n={cs_entry['n_members']})")
    else:
        cl_labels.append(f"C{cid[:6]}")
ax.set_xticklabels(cl_labels, fontsize=7, rotation=45, ha="right")
ax.set_yticks(np.arange(len(conf_epis)))
ax.set_yticklabels(conf_epis, fontsize=9)
ax.set_xlabel("Consensus Cluster")
ax.set_ylabel("True Epitope")
ax.set_title("(A) Epitope × Cluster Assignment", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.7)

# Annotate
for i in range(len(conf_epis)):
    for j in range(len(top_cl_ids)):
        v = cmatrix[i, j]
        if v > 0:
            color = "white" if v > cmatrix.max() * 0.6 else "black"
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8, color=color)

# Panel B: Per-method comparison on subset 1
ax = axes[1]

# ── Helper functions ──
def clusters_to_labels_from_assigns(assigns, tcr_ids):
    label_map = defaultdict(lambda: -1)
    for a in assigns:
        label_map[a.tcr_id] = a.cluster_id
    return np.array([label_map.get(tid, -1) for tid in tcr_ids], dtype=object)

def compute_quick_metrics(pred, true_labels, labeled_idx):
    lp = pred[labeled_idx]
    lt = true_labels[labeled_idx]
    clustered = np.array([str(p) not in ("-1", "") for p in lp])
    if clustered.sum() < 2:
        return 0.0, 0.0
    ari = adjusted_rand_score(lt[clustered], lp[clustered].astype(str))
    nmi = normalized_mutual_info_score(lt[clustered], lp[clustered].astype(str))
    return ari, nmi
from sklearn.metrics import homogeneity_completeness_v_measure

method_names = sorted(method_results.keys()) + ["CONSENSUS"]
method_ari = []
method_nmi = []

for mname in sorted(method_results.keys()):
    assigns, _ = method_results[mname]
    mpred = clusters_to_labels_from_assigns(assigns, tcr_ids)
    ari, nmi = compute_quick_metrics(mpred, true_labels, labeled_idx)
    method_ari.append(ari)
    method_nmi.append(nmi)

# Consensus metrics
cons_ari = adjusted_rand_score(
    true_labels[labeled_idx][pred_arr[labeled_idx] != -1],
    pred_arr[labeled_idx][pred_arr[labeled_idx] != -1].astype(str)
) if (pred_arr[labeled_idx] != -1).sum() > 1 else 0
cons_nmi = normalized_mutual_info_score(
    true_labels[labeled_idx][pred_arr[labeled_idx] != -1],
    pred_arr[labeled_idx][pred_arr[labeled_idx] != -1].astype(str)
) if (pred_arr[labeled_idx] != -1).sum() > 1 else 0
method_ari.append(cons_ari)
method_nmi.append(cons_nmi)

DISPLAY = {
    "clustcr": "clusTCR", "deeptcr": "DeepTCR", "giana": "GIANA",
    "gliph2": "GLIPH2", "hd_baseline": "HD-Base", "tcrdist3": "TCRdist3",
    "tcrmatch": "TCRMatch", "CONSENSUS": "tcrconsensus",
}
display_names = [DISPLAY.get(m, m) for m in method_names]
x = np.arange(len(method_names))
width = 0.35

bars1 = ax.bar(x - width/2, method_ari, width, label="ARI", color="#2171b5", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, method_nmi, width, label="NMI", color="#e6550d", edgecolor="black", linewidth=0.5)

# Highlight consensus
idx_cons = len(method_names) - 1
bars1[idx_cons].set_edgecolor("#08306b")
bars1[idx_cons].set_linewidth(2.5)
bars1[idx_cons].set_hatch("///")
bars2[idx_cons].set_edgecolor("#08306b")
bars2[idx_cons].set_linewidth(2.5)
bars2[idx_cons].set_hatch("///")

for i in range(len(method_names)):
    ax.text(i - width/2, method_ari[i] + 0.01, f"{method_ari[i]:.3f}", ha="center", fontsize=7)
    ax.text(i + width/2, method_nmi[i] + 0.01, f"{method_nmi[i]:.3f}", ha="center", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(display_names, rotation=30, ha="right")
ax.set_ylabel("Score")
ax.set_title("(B) Method Comparison (Subset 1)", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig3.suptitle("Subset 1: tcrconsensus Clustering Detail", fontsize=15, fontweight="bold", y=1.02)
fig3.tight_layout()
fig3.savefig(FIG_DIR / "fig_subset1_detail.png")
print(f"Saved: fig_subset1_detail.png")
plt.close()


print(f"\nAll figures saved to {FIG_DIR}/")
print("Done!")
