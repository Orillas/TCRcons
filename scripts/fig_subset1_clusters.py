#!/usr/bin/env python3
"""Subset 1: run tcrconsensus, visualize TCR clusters.

Output:
  1. Network graph of top clusters (nodes=TCRs, edges=co-association)
  2. UMAP colored by cluster + true epitope
  3. Cluster summary table
"""
import sys, json, time, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
for n in ['numba','tensorflow','absl','matplotlib']:
    logging.getLogger(n).setLevel(logging.ERROR)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/scripts")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.refinement.refiner import refine
from exp_shared import get_all_clusterers, run_all_methods, clusters_to_labels

# ── Paths ──
DATA = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1/input/Gliph2")
LABELS = "/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_h5.json"
FIGS = Path("/home/jilin/DeepTCR/figures")
OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/subset1_vis")
FIGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

with open(LABELS) as f:
    CDR3_EPI = json.load(f)["cdr3_to_epitopes"]

config = dict(load_config().__dict__)

# ── Load subset 1 ──
df = pd.read_csv(DATA / "subset_1.csv", sep="\t")
df = df.rename(columns={"CDR3b": "cdr3_beta", "TRBV": "v_beta", "TRBJ": "j_beta"})
df["tcr_id"] = df["cdr3_beta"]

def get_label(c):
    if c in CDR3_EPI:
        e = CDR3_EPI[c]
        return e[0] if len(e) == 1 else "MULTI:" + ";".join(sorted(e))
    return "BACKGROUND"

df["epitope"] = df["cdr3_beta"].apply(get_label)
df = df[["tcr_id","cdr3_beta","v_beta","j_beta","epitope"]].drop_duplicates(subset=["cdr3_beta"])
df_norm = normalize(df.copy())
true_labels = df_norm["epitope"].values
tcr_ids = df_norm["tcr_id"].values
n_total = len(tcr_ids)
labeled_idx = np.array([tid in CDR3_EPI for tid in tcr_ids])

print(f"Subset 1: {n_total} TCRs, {labeled_idx.sum()} labeled")

# ── Run methods + consensus ──
print("Running methods...")
t0 = time.time()
clusterers = get_all_clusterers()
method_results = run_all_methods(df_norm, clusterers, config, OUT / "methods")
print(f"  Methods done: {time.time()-t0:.0f}s")

all_assigns = []
for mn, (assigns, _) in method_results.items():
    all_assigns.extend(assigns)
    print(f"    {mn}: {len(assigns)} assignments")

methods_list = sorted(set(a.method for a in all_assigns))
weights = empirical_weights(methods_list)

print("Running consensus...")
edges = extract_pairwise_support(all_assigns, weights)
graph = build_consensus_graph(edges, threshold=0.3)
clusters = connected_components_clustering(graph)
clusters = refine(clusters, edges, config)
print(f"  {len(clusters)} clusters, {len(edges)} edges")

pred = clusters_to_labels(clusters, tcr_ids)
pred_arr = np.asarray(pred, dtype=object)

# ── Cluster statistics ──
cluster_stats = []
for c in clusters:
    members = c.member_ids
    epis = []
    for m in members:
        if m in CDR3_EPI:
            e = CDR3_EPI[m]
            epis.append(e[0] if len(e) == 1 else "MULTI")
    n_lab = len(epis)
    n_bg = len(members) - n_lab
    if epis:
        dom = Counter(epis).most_common(1)[0]
        purity = dom[1] / len(members)
        dom_epi = dom[0]
    else:
        purity = 0.0
        dom_epi = "none"
    cluster_stats.append({
        "cluster_id": c.cluster_id,
        "n": len(members), "n_lab": n_lab, "n_bg": n_bg,
        "purity": purity, "dom_epi": dom_epi,
        "confidence": c.cluster_confidence,
        "members": members,
    })

# Sort: labeled members first, then by purity*size
cluster_stats.sort(key=lambda x: (-x["n_lab"], -x["purity"]))
n_with_lab = sum(1 for c in cluster_stats if c["n_lab"] > 0)

print(f"\nClusters with labeled members: {n_with_lab}/{len(cluster_stats)}")
print(f"{'#':<4s} {'ID':<20s} {'Size':>5s} {'Lab':>4s} {'BG':>4s} {'Pur':>6s} {'Dominant':<15s}")
for i, cs in enumerate(cluster_stats[:15]):
    print(f"{i:<4d} {cs['cluster_id']:<20s} {cs['n']:5d} {cs['n_lab']:4d} "
          f"{cs['n_bg']:4d} {cs['purity']:6.3f} {cs['dom_epi']:<15s}")

# ── Save cluster assignments ──
rows = []
for cs in cluster_stats:
    for m in cs["members"]:
        epi = get_label(m) if m in CDR3_EPI else "BACKGROUND"
        rows.append({"cdr3": m, "cluster_id": cs["cluster_id"], "epitope": epi,
                      "cluster_size": cs["n"], "cluster_purity": cs["purity"],
                      "dominant_epitope": cs["dom_epi"]})
# Add unclustered
for tid in tcr_ids:
    if pred_arr[list(tcr_ids).index(tid)] == -1:
        epi = get_label(tid) if tid in CDR3_EPI else "BACKGROUND"
        rows.append({"cdr3": tid, "cluster_id": "unclustered", "epitope": epi,
                      "cluster_size": 0, "cluster_purity": 0, "dominant_epitope": "none"})

pd.DataFrame(rows).to_csv(OUT / "subset1_cluster_assignments.csv", index=False)
print(f"\nSaved cluster assignments: {OUT / 'subset1_cluster_assignments.csv'}")


# ================================================================
#  FIGURE 1: Network graph of top clusters
# ================================================================
print("\nGenerating network graph...")

# Pick top 8 clusters with labeled members
top_clusters = [cs for cs in cluster_stats if cs["n_lab"] > 0][:8]
top_ids = set(cs["cluster_id"] for cs in top_clusters)

# Build subgraph from consensus graph for top cluster members only
top_members = set()
for cs in top_clusters:
    top_members.update(cs["members"])

# Build networkx graph from edges
subG = nx.Graph()
for m in top_members:
    subG.add_node(m)

# Add edges from consensus edges
edge_map = {}
for e in edges:
    pair = (e.tcr_id_a, e.tcr_id_b) if hasattr(e, 'tcr_id_a') else (e[0], e[1])
    score = e.final_score if hasattr(e, 'final_score') else e[2] if len(e) > 2 else 1.0
    if pair[0] in top_members and pair[1] in top_members:
        a, b = pair
        key = tuple(sorted([a, b]))
        if key not in edge_map or score > edge_map[key]:
            edge_map[key] = score
            subG.add_edge(a, b, weight=score)

# Color nodes by cluster
CLUSTER_COLORS = plt.cm.Set1(np.linspace(0, 1, max(len(top_clusters), 2)))
node_colors = []
node_sizes = []
for n in subG.nodes():
    for i, cs in enumerate(top_clusters):
        if n in cs["members"]:
            node_colors.append(CLUSTER_COLORS[i])
            node_sizes.append(30 if n in CDR3_EPI else 10)
            break
    else:
        node_colors.append("#cccccc")
        node_sizes.append(8)

# Layout
print(f"  Network: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
if subG.number_of_nodes() > 0:
    pos = nx.spring_layout(subG, k=0.8, iterations=50, seed=42)

    fig_net, ax = plt.subplots(figsize=(16, 14))
    nx.draw_networkx_edges(subG, pos, alpha=0.1, width=0.5, ax=ax)
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.7, ax=ax, edgecolors="black", linewidths=0.3)

    # Legend
    handles = []
    for i, cs in enumerate(top_clusters):
        label = f"C{i+1}: {cs['dom_epi']} (n={cs['n']}, pur={cs['purity']:.2f})"
        handles.append(mpatches.Patch(color=CLUSTER_COLORS[i], label=label))
    handles.append(mpatches.Patch(color="#cccccc", label="Background"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=9, framealpha=0.9)

    ax.set_title(f"tcrconsensus Cluster Network — Subset 1 (Top {len(top_clusters)} Clusters)\n"
                 f"{n_total} TCRs, {len(clusters)} clusters",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig_net.tight_layout()
    fig_net.savefig(FIGS / "fig_subset1_network.png")
    print(f"  Saved: fig_subset1_network.png")
    plt.close()


# ================================================================
#  FIGURE 2: UMAP of ALL TCRs colored by cluster
# ================================================================
print("Computing UMAP...")

# K-mer features
def kmer_features(seqs, k=3):
    kmers = set()
    for s in seqs:
        for i in range(len(s) - k + 1):
            kmers.add(s[i:i+k])
    kml = sorted(kmers)
    kmi = {km: j for j, km in enumerate(kml)}
    mat = np.zeros((len(seqs), len(kml)))
    for i, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            km = s[j:j+k]
            if km in kmi:
                mat[i, kmi[km]] += 1
    return mat

cdr3s = df_norm["cdr3_beta"].values
feat = kmer_features(cdr3s)

try:
    from umap import UMAP
    emb = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42, metric="cosine").fit_transform(feat)
except ImportError:
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_distances
    emb = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(cosine_distances(feat))

# Assign cluster index for coloring
unique_cl = sorted(set(pred_arr) - {-1})
cl_map = {cid: i for i, cid in enumerate(unique_cl)}

fig_umap, axes = plt.subplots(1, 2, figsize=(22, 9))

# Panel A: colored by true epitope
ax = axes[0]
EPI_COLORS = {
    "KLGGALQAK": "#e41a1c", "AVFDRKSDAK": "#377eb8", "GILGFVFTL": "#4daf4a",
    "RLRAEAQVK": "#984ea3", "RAKFKQLL": "#ff7f00", "IVTDFSVIK": "#a65628",
    "ELAGIGILTV": "#f781bf", "RPPIFIRRL": "#999999",
}
EPI_ORDER = list(EPI_COLORS.keys())

bg_m = true_labels == "BACKGROUND"
ax.scatter(emb[bg_m, 0], emb[bg_m, 1], c="#e0e0e0", s=4, alpha=0.2, rasterized=True)
for epi in EPI_ORDER:
    m = true_labels == epi
    if m.sum() > 0:
        ax.scatter(emb[m, 0], emb[m, 1], c=EPI_COLORS[epi], s=12, alpha=0.6,
                   rasterized=True, label=f"{epi} ({m.sum()})")
other_m = np.array([tl not in EPI_ORDER and tl != "BACKGROUND" for tl in true_labels])
if other_m.sum() > 0:
    ax.scatter(emb[other_m, 0], emb[other_m, 1], c="#555555", s=6, alpha=0.3, rasterized=True,
               label=f"Other ({other_m.sum()})")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, markerscale=1.5)
ax.set_title("(A) True Epitope Labels", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

# Panel B: colored by consensus cluster
ax = axes[1]
uncl_m = pred_arr == -1
ax.scatter(emb[uncl_m, 0], emb[uncl_m, 1], c="#e0e0e0", s=4, alpha=0.15, rasterized=True)

# Color top clusters distinctly, rest gray
top_set = set(cs["cluster_id"] for cs in top_clusters)
cmap_clusters = plt.cm.Set1(np.linspace(0, 1, max(len(top_clusters), 2)))
cluster_color_map = {}
for i, cs in enumerate(top_clusters):
    cluster_color_map[cs["cluster_id"]] = cmap_clusters[i]

for cid in unique_cl:
    m = pred_arr == cid
    if cid in top_set:
        ax.scatter(emb[m, 0], emb[m, 1], c=[cluster_color_map[cid]], s=18, alpha=0.8,
                   rasterized=True, edgecolors="black", linewidths=0.2)
    elif m.sum() > 0:
        ax.scatter(emb[m, 0], emb[m, 1], c="#bbbbbb", s=6, alpha=0.3, rasterized=True)

# Legend
handles = [mpatches.Patch(color="#e0e0e0", label=f"Unclustered ({uncl_m.sum()})")]
for i, cs in enumerate(top_clusters):
    handles.append(mpatches.Patch(color=cmap_clusters[i],
                   label=f"C{i+1}: {cs['dom_epi']} (n={cs['n']}, pur={cs['purity']:.2f})"))
handles.append(mpatches.Patch(color="#bbbbbb", label="Other clusters"))
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
ax.set_title("(B) Consensus Clusters", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

fig_umap.suptitle(f"Subset 1: tcrconsensus Clustering ({n_total} TCRs, {len(clusters)} clusters)",
                  fontsize=15, fontweight="bold", y=1.02)
fig_umap.tight_layout()
fig_umap.savefig(FIGS / "fig_subset1_umap.png")
print(f"  Saved: fig_subset1_umap.png")
plt.close()


# ================================================================
#  FIGURE 3: Cluster composition heatmap
# ================================================================
fig_heat, ax = plt.subplots(figsize=(16, 10))

top20 = [cs for cs in cluster_stats if cs["n_lab"] > 0][:20]
epi_set = set()
for cs in top20:
    for m in cs["members"]:
        if m in CDR3_EPI:
            e = CDR3_EPI[m]
            epi_set.add(e[0] if len(e) == 1 else "MULTI")
epi_list = sorted(epi_set)

mat = np.zeros((len(top20), len(epi_list)))
for i, cs in enumerate(top20):
    for m in cs["members"]:
        if m in CDR3_EPI:
            e = CDR3_EPI[m]
            en = e[0] if len(e) == 1 else "MULTI"
            if en in epi_list:
                mat[i, epi_list.index(en)] += 1

im = ax.imshow(mat, cmap="Blues", aspect="auto")
ax.set_xticks(np.arange(len(epi_list)))
ax.set_xticklabels(epi_list, rotation=45, ha="right", fontsize=9)
row_labels = [f"C{i+1}: {cs['dom_epi'][:10]} (n={cs['n']}, pur={cs['purity']:.2f})"
              for i, cs in enumerate(top20)]
ax.set_yticks(np.arange(len(top20)))
ax.set_yticklabels(row_labels, fontsize=8)
ax.set_xlabel("True Epitope")
ax.set_ylabel("Consensus Cluster")
ax.set_title("Cluster × Epitope Composition (Top 20 Clusters)", fontsize=14, fontweight="bold")

for i in range(len(top20)):
    for j in range(len(epi_list)):
        v = mat[i, j]
        if v > 0:
            color = "white" if v > mat.max() * 0.6 else "black"
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8, color=color)

plt.colorbar(im, ax=ax, shrink=0.6, label="TCR Count")
fig_heat.tight_layout()
fig_heat.savefig(FIGS / "fig_subset1_heatmap.png")
print(f"  Saved: fig_subset1_heatmap.png")
plt.close()


# ================================================================
#  FIGURE 4: Cluster summary table
# ================================================================
fig_tbl, ax = plt.subplots(figsize=(18, 6))
ax.axis("off")

cols = ["#", "Cluster", "Size", "Labeled", "BG", "Purity", "Confidence", "Dominant Epitope",
        "Top V gene", "Median CDR3 len"]
rows_data = []
cdr3_to_v = dict(zip(df_norm["cdr3_beta"], df_norm["v_beta"]))

for i, cs in enumerate(cluster_stats[:20]):
    # V gene enrichment
    v_genes = [cdr3_to_v.get(m, "?") for m in cs["members"] if cdr3_to_v.get(m)]
    top_v = Counter(v_genes).most_common(1)[0][0] if v_genes else "?"
    # CDR3 lengths
    lens = [len(m) for m in cs["members"]]
    med_len = np.median(lens) if lens else 0
    rows_data.append([
        str(i+1),
        cs["cluster_id"][:12],
        str(cs["n"]),
        str(cs["n_lab"]),
        str(cs["n_bg"]),
        f"{cs['purity']:.3f}",
        f"{cs['confidence']:.3f}",
        cs["dom_epi"],
        top_v,
        f"{med_len:.0f}",
    ])

tbl = ax.table(cellText=rows_data, colLabels=cols, cellLoc="center", loc="center",
               colColours=["#d4e6f1"] * len(cols))
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.6)

ax.set_title(f"Cluster Summary — Subset 1 (Top 20 of {len(clusters)} clusters)",
             fontsize=14, fontweight="bold", pad=20)
fig_tbl.tight_layout()
fig_tbl.savefig(FIGS / "fig_subset1_table.png")
print(f"  Saved: fig_subset1_table.png")
plt.close()

print(f"\nAll figures saved to {FIGS}/")
print("Done!")
