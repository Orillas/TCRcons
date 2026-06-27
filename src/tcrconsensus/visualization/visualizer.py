"""Post-clustering visualization: 6 core plots.

1. Cluster size distribution
2. Co-association heatmap
3. UMAP embedding colored by cluster
4. Cluster network graph
5. Cluster confidence distribution
6. Method agreement matrix
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Matplotlib setup ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    raise ImportError("matplotlib required for visualization")

try:
    import seaborn as sns
except ImportError:
    sns = None  # type: ignore

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore

try:
    from umap import UMAP
except ImportError:
    UMAP = None  # type: ignore


def _save_fig(fig: plt.Figure, output_dir: Path, name: str, formats: list[str], dpi: int = 150) -> list[Path]:
    """Save figure in requested formats."""
    paths = []
    for fmt in formats:
        p = output_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _cdr3_distance_matrix(sequences: list[str]) -> np.ndarray:
    """Compute pairwise normalized Levenshtein distance for CDR3 sequences."""
    n = len(sequences)
    dist = np.zeros((n, n), dtype=np.float64)

    # Fast Levenshtein via scipy or manual
    try:
        from scipy.spatial.distance import pdist, squareform

        def _lev(a: str, b: str) -> float:
            """Normalized Levenshtein distance."""
            if a == b:
                return 0.0
            la, lb = len(a), len(b)
            if la == 0 or lb == 0:
                return 1.0
            # DP Levenshtein
            prev = list(range(lb + 1))
            for i in range(1, la + 1):
                curr = [i] + [0] * lb
                for j in range(1, lb + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
                prev = curr
            return prev[lb] / max(la, lb)

        condensed = pdist(sequences, metric=_lev)
        dist = squareform(condensed)
    except Exception:
        # Fallback: simple length-based distance
        lengths = np.array([len(s) for s in sequences], dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(lengths[i] - lengths[j]) / max(lengths[i], lengths[j], 1)
                dist[i, j] = d
                dist[j, i] = d

    return dist


# ============================================================
# Plot 1: Cluster size distribution
# ============================================================
def plot_cluster_size_distribution(
    clusters: list[dict],
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Histogram + KDE of cluster sizes."""
    formats = formats or ["png"]
    sizes = [len(c.get("member_ids", [])) for c in clusters]
    if not sizes:
        logger.warning("No clusters to plot size distribution")
        return []

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(30, max(5, len(set(sizes))))
    ax.hist(sizes, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)

    if sns is not None:
        try:
            sns.kdeplot(sizes, ax=ax, color="darkred", linewidth=2, warn_singular=False)
        except Exception:
            pass

    ax.set_xlabel("Cluster Size (n members)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster Size Distribution (n={len(sizes)})")
    ax.axvline(np.median(sizes), color="orange", linestyle="--", label=f"Median={np.median(sizes):.0f}")
    ax.legend()

    # Stats text box
    stats_text = f"Min: {min(sizes)}  Max: {max(sizes)}  Mean: {np.mean(sizes):.1f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    return _save_fig(fig, output_dir, "cluster_size_distribution", formats)


# ============================================================
# Plot 2: Co-association heatmap
# ============================================================
def plot_coassociation_heatmap(
    edges: list[dict],
    tcr_ids: list[str],
    output_dir: Path,
    formats: list[str] | None = None,
    max_display: int = 200,
) -> list[Path]:
    """Heatmap of pairwise co-association scores."""
    formats = formats or ["png"]
    if not edges or len(tcr_ids) < 2:
        logger.warning("Insufficient data for co-association heatmap")
        return []

    # Subsample if too many TCRs
    display_ids = tcr_ids[:max_display]
    id_set = set(display_ids)
    n = len(display_ids)
    idx = {tid: i for i, tid in enumerate(display_ids)}

    matrix = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        a, b = e.get("tcr_id_a", ""), e.get("tcr_id_b", "")
        if a in id_set and b in id_set:
            score = e.get("weighted_support", e.get("final_score", 0.0))
            i, j = idx[a], idx[b]
            matrix[i, j] = score
            matrix[j, i] = score

    fig, ax = plt.subplots(figsize=(10, 8))
    if n > 50:
        # No tick labels for large matrices
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    else:
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short = [t[:8] for t in display_ids]
        ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticklabels(short, fontsize=6)

    plt.colorbar(im, ax=ax, label="Weighted Co-association Score")
    ax.set_title(f"Co-association Heatmap (n={n} TCRs)")

    return _save_fig(fig, output_dir, "coassociation_heatmap", formats)


# ============================================================
# Plot 3: UMAP embedding
# ============================================================
def plot_umap_embedding(
    df: pd.DataFrame,
    clusters: list[dict],
    output_dir: Path,
    formats: list[str] | None = None,
    seq_col: str = "cdr3_beta",
    id_col: str = "tcr_id",
) -> list[Path]:
    """UMAP 2D projection of TCR sequences, colored by cluster."""
    formats = formats or ["png"]

    if UMAP is None:
        logger.warning("umap-learn not installed, skipping UMAP plot")
        return []

    if id_col not in df.columns or seq_col not in df.columns:
        # Try alternate column names
        for alt in ["cdr3", "junction_aa", "CDR3"]:
            if alt in df.columns:
                seq_col = alt
                break
        if id_col not in df.columns:
            logger.warning(f"No {id_col} column found, skipping UMAP")
            return []

    sequences = df[seq_col].fillna("").astype(str).tolist()
    tcr_ids = df[id_col].astype(str).tolist()
    n = len(sequences)

    if n < 3:
        logger.warning("Too few TCRs for UMAP")
        return []

    # Build cluster label map
    label_map: dict[str, str] = {}
    for c in clusters:
        cid = c.get("cluster_id", "")
        for mid in c.get("member_ids", []):
            label_map[mid] = cid

    # Compute distance matrix
    logger.info(f"Computing CDR3 distance matrix for {n} sequences...")
    dist = _cdr3_distance_matrix(sequences)

    # UMAP embedding
    n_neighbors = min(15, n - 1)
    min_dist = 0.1
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=42,
    )
    embedding = reducer.fit_transform(dist)

    # Assign labels and colors
    labels = [label_map.get(tid, "unclustered") for tid in tcr_ids]
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)

    # Color palette
    if n_labels <= 20:
        cmap = plt.cm.tab20 if n_labels <= 20 else plt.cm.hsv
        color_map = {lbl: cmap(i / max(n_labels - 1, 1)) for i, lbl in enumerate(unique_labels)}
    else:
        cmap = plt.cm.hsv
        color_map = {lbl: cmap(i / n_labels) for i, lbl in enumerate(unique_labels)}

    colors = [color_map.get(l, (0.7, 0.7, 0.7, 1.0)) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        xs = [embedding[i, 0] for i in range(n) if mask[i]]
        ys = [embedding[i, 1] for i in range(n) if mask[i]]
        ax.scatter(xs, ys, c=[color_map[lbl]], label=lbl if n_labels <= 25 else None,
                   s=20, alpha=0.7, edgecolors="none")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"TCR Cluster UMAP (n={n}, {n_labels} clusters)")

    if n_labels <= 25:
        ax.legend(markerscale=2, fontsize=7, ncol=2, loc="best",
                  bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    return _save_fig(fig, output_dir, "umap_clusters", formats)


# ============================================================
# Plot 4: Cluster network graph
# ============================================================
def plot_cluster_network(
    clusters: list[dict],
    edges: list[dict],
    output_dir: Path,
    formats: list[str] | None = None,
    max_nodes: int = 300,
    max_edges: int = 2000,
) -> list[Path]:
    """Network graph of consensus clusters with edges."""
    formats = formats or ["png"]

    if nx is None:
        logger.warning("networkx not installed, skipping network plot")
        return []

    # Build cluster assignment map
    node_cluster: dict[str, str] = {}
    for c in clusters:
        cid = c.get("cluster_id", "")
        for mid in c.get("member_ids", []):
            node_cluster[mid] = cid

    # Collect nodes
    all_nodes = list(node_cluster.keys())
    if len(all_nodes) > max_nodes:
        # Subsample: keep all from small clusters, sample from large
        sampled = []
        cluster_members: dict[str, list[str]] = defaultdict(list)
        for n_id in all_nodes:
            cluster_members[node_cluster[n_id]].append(n_id)
        for cid, members in cluster_members.items():
            if len(members) <= max_nodes // len(cluster_members or 1):
                sampled.extend(members)
            else:
                rng = np.random.RandomState(42)
                k = max(3, max_nodes // len(cluster_members))
                sampled.extend(rng.choice(members, min(k, len(members)), replace=False).tolist())
        all_nodes = sampled
        node_cluster = {n: node_cluster[n] for n in all_nodes}

    node_set = set(all_nodes)

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    # Add edges (filtered)
    edge_count = 0
    for e in edges:
        a, b = e.get("tcr_id_a", ""), e.get("tcr_id_b", "")
        if a in node_set and b in node_set:
            score = e.get("weighted_support", e.get("final_score", 1.0))
            G.add_edge(a, b, weight=score)
            edge_count += 1
            if edge_count >= max_edges:
                break

    # Color by cluster
    unique_clusters = sorted(set(node_cluster.values()))
    n_clusters = len(unique_clusters)
    if n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.hsv
    cluster_color = {cid: cmap(i / max(n_clusters - 1, 1)) for i, cid in enumerate(unique_clusters)}
    node_colors = [cluster_color.get(node_cluster.get(n, ""), (0.7, 0.7, 0.7)) for n in G.nodes()]

    # Layout
    if len(G.nodes()) < 500:
        pos = nx.spring_layout(G, k=0.5, seed=42, weight="weight")
    else:
        pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(14, 12))
    edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_alphas = [0.1 + 0.6 * (w / max_w) for w in edge_weights]

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax,
                           edge_color=edge_alphas)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30,
                           alpha=0.8, ax=ax, edgecolors="none")

    ax.set_title(f"Consensus Cluster Network ({len(G.nodes())} nodes, {len(G.edges())} edges)")
    ax.axis("off")

    return _save_fig(fig, output_dir, "cluster_network", formats, dpi=200)


# ============================================================
# Plot 5: Cluster confidence distribution
# ============================================================
def plot_confidence_distribution(
    clusters: list[dict],
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Histogram of cluster confidence scores."""
    formats = formats or ["png"]
    confidences = [c.get("cluster_confidence", 0.0) for c in clusters]
    if not confidences:
        logger.warning("No confidence scores to plot")
        return []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    ax = axes[0]
    ax.hist(confidences, bins=min(30, max(5, len(set(confidences)))),
            color="teal", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(confidences), color="red", linestyle="--",
               label=f"Median={np.median(confidences):.3f}")
    ax.axvline(np.mean(confidences), color="orange", linestyle=":",
               label=f"Mean={np.mean(confidences):.3f}")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Confidence Distribution")
    ax.legend()

    # Right: confidence vs cluster size scatter
    ax2 = axes[1]
    sizes = [len(c.get("member_ids", [])) for c in clusters]
    ax2.scatter(sizes, confidences, alpha=0.6, s=20, color="teal", edgecolors="none")
    ax2.set_xlabel("Cluster Size")
    ax2.set_ylabel("Confidence Score")
    ax2.set_title("Confidence vs Cluster Size")

    if len(confidences) > 3:
        try:
            r = np.corrcoef(sizes, confidences)[0, 1]
            if np.isfinite(r):
                z = np.polyfit(sizes, confidences, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(sizes), max(sizes), 100)
                ax2.plot(x_line, p(x_line), "r--", alpha=0.7, label=f"r={r:.3f}")
                ax2.legend()
        except (np.linalg.LinAlgError, ValueError):
            pass

    plt.tight_layout()
    return _save_fig(fig, output_dir, "confidence_distribution", formats)


# ============================================================
# Plot 6: Method agreement matrix
# ============================================================
def plot_method_agreement(
    assignments: list[dict],
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Heatmap showing pairwise agreement between clustering methods."""
    formats = formats or ["png"]

    if not assignments:
        logger.warning("No assignments for method agreement plot")
        return []

    # Group by method -> {tcr_id: cluster_id}
    method_labels: dict[str, dict[str, str]] = defaultdict(dict)
    for a in assignments:
        method_labels[a.get("method", "")][a.get("tcr_id", "")] = a.get("cluster_id", "")

    methods = sorted(method_labels.keys())
    if len(methods) < 2:
        logger.warning("Need >= 2 methods for agreement matrix")
        return []

    # Compute pairwise ARI-like agreement
    n_methods = len(methods)
    agreement = np.zeros((n_methods, n_methods), dtype=np.float64)

    for i in range(n_methods):
        agreement[i, i] = 1.0
        for j in range(i + 1, n_methods):
            m1, m2 = methods[i], methods[j]
            common_tcrs = set(method_labels[m1].keys()) & set(method_labels[m2].keys())
            if len(common_tcrs) < 2:
                agreement[i, j] = 0.0
                agreement[j, i] = 0.0
                continue

            # Simple agreement fraction: same-cluster pairs overlap
            pairs1 = set()
            pairs2 = set()
            common_list = sorted(common_tcrs)
            for a, b in combinations(common_list, 2):
                if method_labels[m1][a] == method_labels[m1][b]:
                    pairs1.add((a, b))
                if method_labels[m2][a] == method_labels[m2][b]:
                    pairs2.add((a, b))

            if not pairs1 and not pairs2:
                score = 1.0  # both methods say nothing is clustered
            else:
                intersection = len(pairs1 & pairs2)
                union = len(pairs1 | pairs2)
                score = intersection / union if union > 0 else 0.0

            agreement[i, j] = score
            agreement[j, i] = score

    fig, ax = plt.subplots(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(agreement, annot=True, fmt=".3f", cmap="YlGnBu",
                    xticklabels=methods, yticklabels=methods, ax=ax,
                    vmin=0, vmax=1, square=True)
    else:
        im = ax.imshow(agreement, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_yticklabels(methods)
        for i in range(n_methods):
            for j in range(n_methods):
                ax.text(j, i, f"{agreement[i, j]:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if agreement[i, j] > 0.5 else "black")
        plt.colorbar(im, ax=ax)

    ax.set_title("Method Agreement (Jaccard on co-clustered pairs)")

    return _save_fig(fig, output_dir, "method_agreement", formats)


# ============================================================
# Main entry point
# ============================================================
def generate_cluster_visualizations(
    df: pd.DataFrame,
    clusters: list[dict],
    edges: list[dict] | None = None,
    assignments: list[dict] | None = None,
    output_dir: str | Path = ".",
    formats: list[str] | None = None,
    seq_col: str = "cdr3_beta",
    id_col: str = "tcr_id",
) -> list[Path]:
    """Generate all post-clustering visualizations.

    Parameters
    ----------
    df : normalized TCR DataFrame
    clusters : list of cluster dicts (cluster_id, member_ids, core_member_ids, confidence)
    edges : list of ConsensusEdge dicts
    assignments : list of ClusterAssignment dicts (for method agreement)
    output_dir : directory to write figures
    formats : output formats (default: ["png"])
    seq_col : CDR3 sequence column name
    id_col : TCR ID column name

    Returns
    -------
    list of Path to generated figure files
    """
    formats = formats or ["png"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    # 1. Cluster size distribution
    try:
        paths = plot_cluster_size_distribution(clusters, output_dir, formats)
        generated.extend(paths)
        logger.info(f"[viz] cluster_size_distribution -> {paths}")
    except Exception as e:
        logger.error(f"[viz] cluster_size_distribution failed: {e}")

    # 2. Co-association heatmap
    if edges:
        try:
            tcr_ids = sorted({tid for c in clusters for tid in c.get("member_ids", [])})
            paths = plot_coassociation_heatmap(edges, tcr_ids, output_dir, formats)
            generated.extend(paths)
            logger.info(f"[viz] coassociation_heatmap -> {paths}")
        except Exception as e:
            logger.error(f"[viz] coassociation_heatmap failed: {e}")

    # 3. UMAP embedding
    try:
        paths = plot_umap_embedding(df, clusters, output_dir, formats, seq_col, id_col)
        generated.extend(paths)
        logger.info(f"[viz] umap_clusters -> {paths}")
    except Exception as e:
        logger.error(f"[viz] umap_clusters failed: {e}")

    # 4. Cluster network graph
    if edges:
        try:
            paths = plot_cluster_network(clusters, edges, output_dir, formats)
            generated.extend(paths)
            logger.info(f"[viz] cluster_network -> {paths}")
        except Exception as e:
            logger.error(f"[viz] cluster_network failed: {e}")

    # 5. Confidence distribution
    try:
        paths = plot_confidence_distribution(clusters, output_dir, formats)
        generated.extend(paths)
        logger.info(f"[viz] confidence_distribution -> {paths}")
    except Exception as e:
        logger.error(f"[viz] confidence_distribution failed: {e}")

    # 6. Method agreement matrix
    if assignments and len(assignments) > 0:
        try:
            paths = plot_method_agreement(assignments, output_dir, formats)
            generated.extend(paths)
            logger.info(f"[viz] method_agreement -> {paths}")
        except Exception as e:
            logger.error(f"[viz] method_agreement failed: {e}")

    logger.info(f"[viz] Generated {len(generated)} figures in {output_dir}")
    return generated
