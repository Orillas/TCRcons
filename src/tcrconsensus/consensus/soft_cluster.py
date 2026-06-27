"""Tier-1/F1(b): soft-overlap consensus via the Community-Affiliation Model
(BigCLAM, Yang & Leskovec WWW 2013).

Unlike connected components (hard partition) and Leiden (disjoint communities),
BigCLAM fits a non-negative affiliation matrix F ∈ R^{n×k}_+ so that each TCR
may belong to SEVERAL communities with graded strength. The generative model:

    P(edge (i,j)) = 1 - exp(-F_i · F_j)

Maximise the log-likelihood via projected gradient ascent (F ≥ 0). A TCR whose
F_i has several large entries spans multiple specificity groups — the
**cross-reactivity candidate** signal that hard clustering cannot express.

This module consumes the SAME consensus co-association graph that
connected_components_clustering would (edges surviving the F2/F3 threshold),
so soft-vs-hard is an apples-to-apples comparison on the thresholded graph.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


def _build_index(graph: nx.Graph) -> tuple[list[str], dict[str, int]]:
    nodes = sorted(graph.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    return nodes, idx


def bigclam(
    graph: nx.Graph,
    k: int,
    *,
    learning_rate: float = 0.05,
    iterations: int = 500,
    lambda_reg: float = 0.01,
    init: Optional[np.ndarray] = None,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Fit BigCLAM affiliation matrix F.

    Args:
        graph: weighted undirected graph (edge weight used as multiplier on the
            edge-gradient; if absent, treated as 1).
        k: number of latent communities.
        learning_rate: gradient-ascent step size.
        iterations: gradient steps.
        lambda_reg: L2 regularisation on F (prevents degenerate large values).
        init: optional (n,k) warm start; else small uniform+noise.
        seed: RNG seed (deterministic).

    Returns:
        F: (n,k) non-negative affiliation matrix.
        nodes: node id per row (sorted).
    """
    nodes, idx = _build_index(graph)
    n = len(nodes)
    if n == 0 or k <= 0:
        return np.zeros((n, k)), nodes

    rng = np.random.RandomState(seed)
    # Adjacency as dense weight matrix (n is modest for TCR consensus graphs).
    W = np.zeros((n, n), dtype=float)
    for u, v, d in graph.edges(data=True):
        i, j = idx[u], idx[v]
        w = d.get("weight", 1.0)
        W[i, j] = w
        W[j, i] = w
    np.fill_diagonal(W, 0.0)
    has_edge = W > 0

    # Init: small positive values + noise (BigCLAM convention).
    if init is not None:
        F = np.maximum(init, 1e-6).astype(float)
    else:
        F = rng.rand(n, k) * 0.1 + 1e-3

    total_sum = F.sum(axis=0)  # k-vector, recomputed each iter

    for it in range(iterations):
        # Predicted dot products on edges
        # Gradient of LL w.r.t. F_i:
        #   edge term:   sum_{j in N(i)} w_ij * F_j * exp(-d_ij)/(1-exp(-d_ij)) * (sign)
        #   non-edge:   -(total_sum - F_i - sum_{j in N(i)} F_j)   [BigCLAM trick]
        # We add the edge-weight as a multiplier (consensus edges carry confidence).
        # vectorise over nodes
        FFi = F @ F.T                     # (n,n) dot products
        np.fill_diagonal(FFi, 0.0)
        # avoid log(0): clamp d on edges to >= 1e-12
        d_edge = np.where(has_edge, np.maximum(FFi, 1e-12), 0.0)
        # exp(-d)/(1-exp(-d)) = 1/(exp(d)-1)
        with np.errstate(over="ignore", divide="ignore"):
            coef = np.where(has_edge, 1.0 / (np.exp(np.minimum(d_edge, 50.0)) - 1.0), 0.0)
        coef = np.clip(coef, 0.0, 1e4)   # numerical guard: 1/(exp(d)-1) -> inf as d->0
        # edge gradient contribution to F_i: sum_j w_ij * coef_ij * F_j
        grad_edge = (coef * W) @ F         # (n,k)
        # non-edge contribution: -(total_sum - F_i - sum_{j in N(i)} F_j)
        neighbor_sum = (has_edge.astype(float)) @ F   # sum of F_j over neighbors
        grad_nonedge = -(total_sum[None, :] - F - neighbor_sum)
        grad = grad_edge + grad_nonedge - lambda_reg * F
        F = F + learning_rate * grad
        F = np.maximum(F, 0.0)             # project to non-negative
        total_sum = F.sum(axis=0)

        if verbose and (it % 50 == 0 or it == iterations - 1):
            ll = _log_likelihood(F, W, has_edge, lambda_reg)
            logger.info(f"BigCLAM iter {it}: LL={ll:.2f}")

    return F, nodes


def _log_likelihood(F, W, has_edge, lambda_reg):
    FFi = F @ F.T
    np.fill_diagonal(FFi, 0.0)
    d = np.maximum(FFi, 1e-12)
    # edge term: sum log(1 - exp(-d_ij)); non-edge: -d_ij
    edge_ll = np.where(has_edge, np.log(np.maximum(1 - np.exp(-d), 1e-12)), 0.0)
    nonedge_ll = np.where((~has_edge) & (FFi > 0), -FFi, 0.0)
    return float(edge_ll.sum() + 0.5 * nonedge_ll.sum() - 0.5 * lambda_reg * (F ** 2).sum())


def extract_memberships(
    F: np.ndarray,
    nodes: list[str],
    *,
    abs_threshold: float = 0.0,
    rel_threshold: float = 0.1,
    min_strength: float = 1e-3,
) -> dict[str, set[int]]:
    """Map each TCR to the SET of communities it belongs to.

    A community c is assigned to node i if F[i,c] >= max(abs_threshold,
    rel_threshold * max_c F[i,c]) and >= min_strength. Nodes with no community
    above threshold get the empty set (effectively unclustered).
    """
    out: dict[str, set[int]] = {}
    for i, node in enumerate(nodes):
        row = F[i]
        peak = row.max() if row.size else 0.0
        thr = max(abs_threshold, rel_threshold * peak)
        comms = {int(c) for c in np.where(row >= max(thr, min_strength))[0]}
        out[node] = comms
    return out


def memberships_to_clusters(
    memberships: dict[str, set[int]], k: int
) -> list[set[str]]:
    """Invert: for each community, the set of member TCRs (overlapping)."""
    clusters = [set() for _ in range(k)]
    for node, comms in memberships.items():
        for c in comms:
            if 0 <= c < k:
                clusters[c].add(node)
    return [c for c in clusters if len(c) > 0]


def spectral_soft(
    graph: nx.Graph,
    k: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Soft-overlap embedding via the normalised Laplacian's top eigenvectors.

    More numerically robust than BigCLAM on sparse consensus graphs (BigCLAM's
    global non-edge penalty collapses F→0 when avg degree ≪ n). The smallest-k
    non-trivial eigenvectors of L = I - D^{-1/2} W D^{-1/2} embed nodes so that
    community membership is graded: a TCR bridging two specificity groups has
    comparable |loading| on two eigenvectors. Sign of eigenvectors is arbitrary,
    so memberships use |loading|.

    Returns U (n,k) loadings and node order. Pass to ``extract_memberships``.
    """
    nodes, idx = _build_index(graph)
    n = len(nodes)
    if n == 0 or k <= 0:
        return np.zeros((n, k)), nodes
    W = np.zeros((n, n), dtype=float)
    for u, v, d in graph.edges(data=True):
        i, j = idx[u], idx[v]
        w = float(d.get("weight", 1.0))
        W[i, j] = w; W[j, i] = w
    deg = W.sum(axis=1)
    deg[deg == 0] = 1.0
    Dinv = 1.0 / np.sqrt(deg)
    L = np.eye(n) - (Dinv[:, None] * W * Dinv[None, :])
    # n is modest (TCR consensus graphs); dense eigendecomposition is robust & fast.
    vals, vecs = np.linalg.eigh(L)
    # drop the smallest (trivial constant) eigenvector, keep next k
    order = np.argsort(vals)
    take = order[1:min(k + 1, len(order))] if len(order) > 1 else order[:1]
    U = np.abs(vecs[:, take])           # (n, k') loadings, sign-ambiguous
    if U.shape[1] < k:                   # pad if not enough eigenvectors
        U = np.hstack([U, np.zeros((n, k - U.shape[1]))])
    return U, nodes
