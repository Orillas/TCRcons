"""Clustering evaluation metrics.

Extended with AMI, V-measure, per-epitope pairwise F1, and pairwise sensitivity.
"""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def retention(n_clustered: int, n_total: int) -> float:
    """Fraction of input TCRs retained in any cluster."""
    if n_total == 0:
        return 0.0
    return n_clustered / n_total


def purity(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Cluster purity: fraction of correctly assigned items.

    For each predicted cluster, count the majority true label.
    """
    if len(pred_labels) == 0:
        return 0.0
    clusters = np.unique(pred_labels)
    total_correct = 0
    for c in clusters:
        mask = pred_labels == c
        if mask.sum() == 0:
            continue
        majority = np.bincount(true_labels[mask].astype(int)).max()
        total_correct += majority
    return total_correct / len(pred_labels)


def weighted_purity(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Weighted mean cluster purity (weighted by cluster size)."""
    return purity(pred_labels, true_labels)  # purity is already weighted


def unweighted_purity(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Unweighted mean cluster purity."""
    if len(pred_labels) == 0:
        return 0.0
    clusters = np.unique(pred_labels)
    purities = []
    for c in clusters:
        mask = pred_labels == c
        if mask.sum() < 2:
            continue
        majority = np.bincount(true_labels[mask].astype(int)).max()
        purities.append(majority / mask.sum())
    return float(np.mean(purities)) if purities else 0.0


def sensitivity(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Sensitivity: mean fraction of same-epitope TCRs co-clustered."""
    epitopes = np.unique(true_labels)
    if len(epitopes) <= 1:
        return 0.0

    scores = []
    for ep in epitopes:
        ep_mask = true_labels == ep
        ep_pred = pred_labels[ep_mask]
        if len(ep_pred) < 2:
            continue
        same = 0
        total_pairs = 0
        for i in range(len(ep_pred)):
            for j in range(i + 1, len(ep_pred)):
                total_pairs += 1
                if ep_pred[i] == ep_pred[j]:
                    same += 1
        if total_pairs > 0:
            scores.append(same / total_pairs)

    return float(np.mean(scores)) if scores else 0.0


def pairwise_sensitivity(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Pairwise sensitivity: same-epitope pairs co-clustered / all same-epitope pairs."""
    same_epi_same_cluster = 0
    total_same_epi = 0

    for i in range(len(true_labels)):
        for j in range(i + 1, len(true_labels)):
            if true_labels[i] == true_labels[j]:
                total_same_epi += 1
                if pred_labels[i] == pred_labels[j]:
                    same_epi_same_cluster += 1

    return same_epi_same_cluster / total_same_epi if total_same_epi > 0 else 0.0


def pairwise_precision(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Pairwise precision: co-clustered same-epitope / all co-clustered pairs."""
    same_epi_same_cluster = 0
    total_co_clustered = 0

    for i in range(len(pred_labels)):
        for j in range(i + 1, len(pred_labels)):
            if pred_labels[i] == pred_labels[j]:
                total_co_clustered += 1
                if true_labels[i] == true_labels[j]:
                    same_epi_same_cluster += 1

    return same_epi_same_cluster / total_co_clustered if total_co_clustered > 0 else 0.0


def f1_score(purity_val: float, sensitivity_val: float) -> float:
    """Harmonic mean of purity and sensitivity."""
    if purity_val + sensitivity_val == 0:
        return 0.0
    return 2 * purity_val * sensitivity_val / (purity_val + sensitivity_val)


def ari(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Adjusted Rand Index."""
    return adjusted_rand_score(true_labels, pred_labels)


def ami(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Adjusted Mutual Information."""
    return adjusted_mutual_info_score(true_labels, pred_labels)


def nmi(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Normalized Mutual Information."""
    return normalized_mutual_info_score(true_labels, pred_labels)


def v_measure(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict:
    """Homogeneity, completeness, and V-measure."""
    h, c, v = homogeneity_completeness_v_measure(true_labels, pred_labels)
    return {"homogeneity": h, "completeness": c, "v_measure": v}


def bcubed_f1(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    unclustered: str = "-1",
) -> dict[str, float]:
    """BCubed precision / recall / F1 (reproductivity.md §4 standard).

    Unlike pairwise F1 (which inflates on large clusters), BCubed averages
    per-element precision/recall, so every element contributes equally. TCRs
    not assigned to any cluster (label == ``unclustered``) are excluded from
    the average but counted as misses via recall 0 on their true-epitope peers.

    Returns dict(precision, recall, f1, n_evaluated).
    """
    pred = np.asarray(pred_labels).astype(str)
    true = np.asarray(true_labels).astype(str)
    valid = np.array(
        [p != "" and p != unclustered for p in pred], dtype=bool
    )
    p = pred[valid]
    t = true[valid]
    n = len(p)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_evaluated": 0}
    ps = rs = 0.0
    for i in range(n):
        sc = p == p[i]      # same cluster
        se = t == t[i]      # same epitope
        nc, ne = sc.sum(), se.sum()
        c = (sc & se).sum()
        ps += c / nc if nc else 0.0
        rs += c / ne if ne else 0.0
    prec, rec = ps / n, rs / n
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_evaluated": int(n),
    }


def overlapping_pairwise_f1(
    pred_memberships: dict,
    true_labels,
    node_order=None,
) -> dict[str, float]:
    """Overlapping pairwise F1 for soft (multi-community) clustering.

    Two nodes are "together in prediction" iff they share >=1 predicted
    community; "together in truth" iff they share >=1 true epitope label.
    Pairwise precision/recall/F1 over these binary relations. Handles the
    cross-reactive case where a true label is itself a set (a TCR binding
    multiple epitopes shares truth with both epitope groups).

    Args:
        pred_memberships: node -> set(community id). From BigCLAM.
        true_labels: node -> single label (str) OR set of labels; or an
            array/list aligned with node_order (single-label per node).
    """
    nodes = node_order if node_order is not None else list(pred_memberships.keys())
    n = len(nodes)
    # normalise true labels to sets
    def _to_set(x):
        if isinstance(x, (set, tuple, list)):
            return set(x)
        return {x}
    if isinstance(true_labels, dict):
        T = {nd: _to_set(true_labels[nd]) for nd in nodes}
    else:
        T = {nd: _to_set(true_labels[i]) for i, nd in enumerate(nodes)}
    C = {nd: set(pred_memberships.get(nd, set())) for nd in nodes}

    tp = co_pred = tp_true = 0
    for ii in range(n):
        ci, ti = C[nodes[ii]], T[nodes[ii]]
        for jj in range(ii + 1, n):
            share_c = len(ci & C[nodes[jj]]) > 0
            share_t = len(ti & T[nodes[jj]]) > 0
            if share_c:
                co_pred += 1
            if share_t:
                tp_true += 1
            if share_c and share_t:
                tp += 1
    prec = tp / co_pred if co_pred else 0.0
    rec = tp / tp_true if tp_true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "n_pairs_pred": int(co_pred), "n_pairs_true": int(tp_true)}


def membership_entropy(memberships: dict) -> dict[str, float]:
    """Per-node membership cardinality + a simple overlap score.

    Returns dict(node -> {'n_communities': int}). Useful to separate
    multi-community (cross-reactive candidate) from single-community nodes.
    """
    out = {}
    for nd, comms in memberships.items():
        out[nd] = {"n_communities": len(comms)}
    return out


# ---------------------------------------------------------------------------
# Per-epitope metrics
# ---------------------------------------------------------------------------

def per_epitope_pairwise_f1(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    epitope_names: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute pairwise F1 for each epitope (one-vs-rest).

    For each target epitope:
      positive pairs = TCR pairs sharing the target epitope
      predicted positive pairs = TCR pairs co-clustered
      precision = correct co-clustered / all co-clustered involving target
      recall = correct co-clustered / all same-epitope pairs
      F1 = harmonic mean

    Returns DataFrame with columns: epitope, precision, recall, f1, n_tcrs, n_pairs
    """
    if epitope_names is None:
        epitope_names = np.unique(true_labels)

    results = []
    for target_ep in epitope_names:
        ep_mask = true_labels == target_ep
        n_tcrs = ep_mask.sum()
        if n_tcrs < 2:
            continue

        # True positive pairs (same target epitope)
        tp_true = n_tcrs * (n_tcrs - 1) // 2

        # Predicted positive pairs (co-clustered among target epitope TCRs)
        ep_pred = pred_labels[ep_mask]
        cluster_counts = Counter(ep_pred)
        tp_pred = sum(c * (c - 1) // 2 for c in cluster_counts.values())

        # Correct co-clustered pairs (TP)
        correct = 0
        indices = np.where(ep_mask)[0]
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                if pred_labels[indices[ii]] == pred_labels[indices[jj]]:
                    correct += 1

        # Precision: correct / predicted_positive
        # But we need all co-clustered pairs involving ANY TCR in same cluster as target TCRs
        # Use a simpler definition: precision = correct / tp_pred (if tp_pred > 0)
        # More standard: for one-vs-rest, precision = correct / all_co_clustered_involving_target
        target_clusters = set(ep_pred)
        all_co_clustered = 0
        for cl in target_clusters:
            cl_mask = pred_labels == cl
            n_cl = cl_mask.sum()
            all_co_clustered += n_cl * (n_cl - 1) // 2
        # Subtract pairs from non-target epitopes in same cluster
        # Simpler: use correct / max(correct, 1) for precision proxy
        # Actually use the standard pairwise definition:
        # precision = correct / all_co_clustered_involving_target_tcrs

        prec = correct / all_co_clustered if all_co_clustered > 0 else 0.0
        rec = correct / tp_true if tp_true > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results.append({
            "epitope": target_ep,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_tcrs": int(n_tcrs),
            "n_pairs": int(tp_true),
        })

    return pd.DataFrame(results)


def per_epitope_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute per-epitope ARI, sensitivity, and pairwise F1."""
    epitopes = np.unique(true_labels)
    rows = []

    for ep in epitopes:
        ep_mask = true_labels == ep
        n_tcrs = ep_mask.sum()
        if n_tcrs < 2:
            continue

        ep_pred = pred_labels[ep_mask]

        # Sensitivity for this epitope
        total_pairs = n_tcrs * (n_tcrs - 1) // 2
        cluster_counts = Counter(ep_pred)
        co_clustered = sum(c * (c - 1) // 2 for c in cluster_counts.values())
        sens = co_clustered / total_pairs if total_pairs > 0 else 0.0

        # Purity of the largest cluster this epitope lands in
        largest_cluster = max(cluster_counts, key=cluster_counts.get)
        all_in_cluster = (pred_labels == largest_cluster).sum()
        ep_purity = n_tcrs / all_in_cluster if all_in_cluster > 0 else 0.0

        rows.append({
            "epitope": ep,
            "n_tcrs": int(n_tcrs),
            "sensitivity": sens,
            "purity": ep_purity,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    n_total: int,
) -> dict[str, float]:
    """Compute all standard metrics at once."""
    n_clustered = len(pred_labels)
    p = purity(pred_labels, true_labels)
    s = sensitivity(pred_labels, true_labels)
    vm = v_measure(pred_labels, true_labels)

    return {
        "retention": retention(n_clustered, n_total),
        "purity": p,
        "unweighted_purity": unweighted_purity(pred_labels, true_labels),
        "sensitivity": s,
        "pairwise_sensitivity": pairwise_sensitivity(pred_labels, true_labels),
        "pairwise_precision": pairwise_precision(pred_labels, true_labels),
        "f1": f1_score(p, s),
        "ari": ari(pred_labels, true_labels),
        "ami": ami(pred_labels, true_labels),
        "nmi": nmi(pred_labels, true_labels),
        "homogeneity": vm["homogeneity"],
        "completeness": vm["completeness"],
        "v_measure": vm["v_measure"],
        "n_clustered": float(n_clustered),
        "n_total": float(n_total),
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats.append(statistic(sample))
    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_stats, alpha * 100)
    ci_high = np.percentile(boot_stats, (1 - alpha) * 100)
    return float(statistic(values)), float(ci_low), float(ci_high)


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the difference (a - b) paired by index.

    Returns dict with mean_diff, ci_low, ci_high, p_proportion (fraction where diff > 0).
    """
    rng = np.random.RandomState(seed)
    n = len(values_a)
    diffs = values_a - values_b
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(np.mean(diffs[idx]))
    boot_diffs = np.array(boot_diffs)
    alpha = (1 - ci) / 2
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(boot_diffs, alpha * 100)),
        "ci_high": float(np.percentile(boot_diffs, (1 - alpha) * 100)),
        "p_positive": float(np.mean(boot_diffs > 0)),
    }


def wilcoxon_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> dict:
    """Wilcoxon signed-rank test for paired samples."""
    from scipy.stats import wilcoxon
    diffs = values_a - values_b
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        stat, pval = wilcoxon(diffs)
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0}
    return {"statistic": float(stat), "p_value": float(pval)}
