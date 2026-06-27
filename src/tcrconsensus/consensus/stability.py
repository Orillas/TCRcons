"""Tier-2 innovation 4a: stability- and agreement-weighted ensemble.

Replaces the static EMPIRICAL_PRIORS weighting with a data-aware weight:

    w_m  ∝  acc(m | D)  ×  stab(m | D)  ×  prior_m^alpha   (Bayesian shrinkage)

where
  acc(m)   = leave-one-out consensus consistency: NMI between method m's clustering
             and the consensus built from the OTHER M-1 methods. High NMI => m agrees
             with the collective => trustworthy. Needs NO labels and NO method re-runs.
  stab(m)  = (optional) bootstrap subsampling stability: 1 - normalized mean
             variation-of-information across B subsampled runs. Default off (==1.0)
             because deterministic methods have no run-to-run variance; enable only
             for methods with internal auto-tuning (DeepTCR, TCRdist3).
  prior_m  = the static empirical prior score (cold start / regularizer).
  alpha    = shrinkage toward the prior (0 = pure data-driven, 1 = pure prior).

This structurally eliminates the "stale prior" problem (e.g. tcrdist3 after the
infer_cdrs fix) because acc() is recomputed on the current dataset every run.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from ..schema.records import ClusterAssignment
from .coassociation import extract_pairwise_support
from .graph import build_consensus_graph, connected_components_clustering
from .weights import EMPIRICAL_PRIORS, DEFAULT_COEFFICIENTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: clustering -> per-tcr label array, aligned over a tcr_id universe
# ---------------------------------------------------------------------------

def _assignments_by_method(
    assignments: list[ClusterAssignment],
) -> dict[str, list[ClusterAssignment]]:
    groups: dict[str, list[ClusterAssignment]] = defaultdict(list)
    for a in assignments:
        groups[a.method].append(a)
    return groups


def _label_map(assignments: list[ClusterAssignment]) -> dict[str, str]:
    """tcr_id -> cluster_id (first assignment wins; one label per tcr)."""
    m: dict[str, str] = {}
    for a in assignments:
        if a.tcr_id not in m:
            m[a.tcr_id] = a.cluster_id
    return m


def _consensus_label_map(
    assignments: list[ClusterAssignment],
    weights: dict[str, float],
    threshold: float = 0.3,
) -> dict[str, str]:
    """Build a balanced-style consensus from the given assignments and return
    a tcr_id -> consensus_cluster_id map."""
    edges = extract_pairwise_support(assignments, weights)
    graph = build_consensus_graph(edges, threshold=threshold)
    out: dict[str, str] = {}
    for cluster in connected_components_clustering(graph):
        for tid in cluster.member_ids:
            out[tid] = cluster.cluster_id
    return out


def _aligned_labels(
    map_a: dict[str, str],
    map_b: dict[str, str],
    universe: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return two aligned int label arrays over `universe`. TCRs absent from a
    map get a unique singleton label so they never spuriously agree."""
    la, lb = [], []
    next_singleton = 10_000_000
    for tid in universe:
        ca = map_a.get(tid)
        cb = map_b.get(tid)
        la.append(ca if ca is not None else f"__solo_{next_singleton}")
        lb.append(cb if cb is not None else f"__solo_{next_singleton}")
        if ca is None or cb is None:
            next_singleton += 1
    from sklearn.preprocessing import LabelEncoder
    ea = LabelEncoder().fit_transform(la)
    eb = LabelEncoder().fit_transform(lb)
    return ea, eb


# ---------------------------------------------------------------------------
# Leave-one-out consensus consistency  (= accuracy proxy, label-free)
# ---------------------------------------------------------------------------

def loo_consensus_accuracy(
    assignments: list[ClusterAssignment],
    weights: dict[str, float] | None = None,
    threshold: float = 0.3,
) -> dict[str, float]:
    """For each method m, NMI(m, consensus of the other M-1 methods).

    Returns dict method_name -> nmi in [0, 1]. Higher = method agrees with the
    collective evidence of the rest = more trustworthy on THIS dataset.
    """
    weights = weights or {}
    by_method = _assignments_by_method(assignments)
    methods = list(by_method.keys())
    universe_set: set[str] = set()
    for ml in by_method.values():
        for a in ml:
            universe_set.add(a.tcr_id)
    universe = sorted(universe_set)

    acc: dict[str, float] = {}
    for m in methods:
        others = [a for mm in methods if mm != m for a in by_method[mm]]
        if not others:
            acc[m] = 0.5
            continue
        cons_map = _consensus_label_map(others, weights, threshold=threshold)
        method_map = _label_map(by_method[m])
        ea, eb = _aligned_labels(method_map, cons_map, universe)
        acc[m] = float(normalized_mutual_info_score(ea, eb))
    return acc


# ---------------------------------------------------------------------------
# (Optional) bootstrap subsampling stability
# ---------------------------------------------------------------------------

def bootstrap_stability(
    run_method: Callable[[pd.DataFrame], list[ClusterAssignment]],
    tcr_table: pd.DataFrame,
    seed: int = 42,
    n_bootstrap: int = 20,
    subsample_frac: float = 0.8,
) -> float:
    """1 - normalized mean variation-of-information across B subsampled runs.

    Variation of Information (VI) is an unbiased clustering-distance metric.
    stab = 1 - mean(VI) / log(n). 1.0 = perfectly stable.

    Only call this for methods with internal randomness/auto-tuning; for
    deterministic methods the result is trivially 1.0 and the B re-runs are
    wasted compute.
    """
    rng = np.random.RandomState(seed)
    n = len(tcr_table)
    clusterings: list[dict[str, str]] = []
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=max(2, int(n * subsample_frac)), replace=True)
        sub = tcr_table.iloc[idx].reset_index(drop=True)
        try:
            assigns = run_method(sub)
        except Exception as e:  # pragma: no cover - method-dependent
            logger.warning(f"stability bootstrap run {b} failed: {e}")
            continue
        clusterings.append(_label_map(assigns))
    if len(clusterings) < 2:
        return 1.0
    # mean pairwise VI over shared tcr_id universe
    vis: list[float] = []
    keys = sorted(set().union(*[set(c) for c in clusterings]))
    for i in range(len(clusterings)):
        for j in range(i + 1, len(clusterings)):
            ea, eb = _aligned_labels(clusterings[i], clusterings[j], keys)
            vis.append(_variation_of_information(ea, eb))
    mean_vi = float(np.mean(vis)) if vis else 0.0
    return float(max(0.0, 1.0 - mean_vi / max(1.0, np.log(max(2, len(keys))))))


def _variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """VI = H(a|b) + H(b|a). Implemented via entropy / mutual information."""
    from sklearn.metrics import mutual_info_score
    from sklearn.metrics.cluster import entropy
    ha, hb = entropy(labels_a), entropy(labels_b)
    mi = mutual_info_score(labels_a, labels_b)
    return float(max(0.0, ha + hb - 2.0 * mi))


# ---------------------------------------------------------------------------
# Combined stability-weighted weights
# ---------------------------------------------------------------------------

def _prior_score(method: str, priors: dict | None) -> float:
    priors = priors or EMPIRICAL_PRIORS
    p = priors.get(method, {
        "purity": 0.5, "sensitivity": 0.5, "ari": 0.05,
        "ami": 0.1, "noise_robust": 0.5,
    })
    c = DEFAULT_COEFFICIENTS
    return (
        c["ari"] * p.get("ari", 0.05)
        + c["ami"] * p.get("ami", 0.1)
        + c["purity"] * p.get("purity", 0.5)
        + c["sensitivity"] * p.get("sensitivity", 0.5)
        + c["noise_robust"] * p.get("noise_robust", 0.5)
    )


def stability_weighted_weights(
    assignments: list[ClusterAssignment],
    priors: dict | None = None,
    alpha: float = 0.5,
    min_weight: float = 0.03,
    threshold: float = 0.3,
    stability: dict[str, float] | None = None,
    acc: dict[str, float] | None = None,
) -> dict[str, float]:
    """Data-aware weights:

        w_m ∝ acc(m) * stab(m) * prior_m^alpha

    acc and stab default to label-free estimates computed from `assignments`
    (acc) / 1.0 (stab). prior_m^alpha is Bayesian shrinkage toward the static
    empirical prior (alpha=0 fully data-driven, alpha=1 fully prior).
    """
    by_method = _assignments_by_method(assignments)
    methods = list(by_method.keys())
    if acc is None:
        acc = loo_consensus_accuracy(assignments, threshold=threshold)
    if stability is None:
        stability = {m: 1.0 for m in methods}

    raw: dict[str, float] = {}
    for m in methods:
        a = max(acc.get(m, 0.5), 1e-3)
        s = max(stability.get(m, 1.0), 1e-3)
        prior = max(_prior_score(m, priors), 1e-3)
        raw[m] = a * s * (prior ** alpha)

    total = sum(raw.values())
    if total <= 0:
        n = len(methods)
        return {m: 1.0 / n for m in methods}
    # floor + renormalize
    floored = {m: max(w / total, min_weight) for m, w in raw.items()}
    total2 = sum(floored.values())
    return {m: w / total2 for m, w in floored.items()}
