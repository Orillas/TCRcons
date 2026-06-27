"""
Innovation #2: Calibration-in-the-loop reweighting.

Closes the loop between consensus and calibration: instead of calibration
being a passive "label" on the final clusters, it actively drives method
weight updates in an iterative refinement.

Algorithm:
  1. Run consensus with initial weights → clusters + edges
  2. For each method m, compute calibrated reliability:
     - Find clusters where method m contributed (its members)
     - Compute each such cluster's raw_confidence → calibrated via isotonic
     - r_m = size-weighted mean calibrated confidence of m's clusters
  3. Update weights: w_m' = (1-α)·w_m + α·r_m  (EMA, α=0.5 default)
  4. Normalize w' to Σ=1
  5. Repeat from step 1 until convergence (weight delta < ε) or max_iter.

This is the first TCR consensus method to use calibration feedback for
active weight optimization; existing methods (EAC, Strehl-Ghosh) use static
priors or fixed quality heuristics.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..schema.records import ClusterAssignment, ConsensusCluster, ConsensusEdge

logger = logging.getLogger(__name__)


def method_cluster_map(
    assignments: list[ClusterAssignment],
) -> dict[str, dict[str, set[str]]]:
    """Build method → cluster_id → {tcr_id, ...} map."""
    from collections import defaultdict
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for a in assignments:
        out[a.method][a.cluster_id].add(a.tcr_id)
    return {m: dict(cs) for m, cs in out.items()}


def method_reliability_from_calibration(
    clusters: list[ConsensusCluster],
    calibrator,      # Calibrator instance with .predict(raw_conf) method
    raw_confidences: list[float] | None = None,
    method_map: dict[str, dict[str, set[str]]] | None = None,
    assignments: list[ClusterAssignment] | None = None,
) -> dict[str, float]:
    """Compute per-method reliability from calibrated cluster confidences.

    For each consensus cluster, we compute which methods contributed members.
    A method's reliability is the size-weighted mean calibrated confidence
    of clusters it contributed to.

    Args:
        clusters: consensus clusters with member_ids.
        calibrator: fitted Calibrator with predict() method.
        raw_confidences: pre-computed raw_confidence per cluster (optional).
        method_map: pre-built method→cluster→members map.
        assignments: used to build method_map if not provided.

    Returns:
        dict[method_name → reliability_score in [0,1]].
    """
    if method_map is None and assignments is not None:
        method_map = method_cluster_map(assignments)

    if raw_confidences is None:
        from ..calibration.calibrator import raw_confidence
        raw_confidences = []  # Need edges too; handled differently
        # We'll compute per-cluster inline

    # Build cluster_id → {tcr_id, ...} for consensus clusters
    cluster_members = {c.cluster_id: set(c.member_ids) for c in clusters}

    # For each method, find which consensus clusters overlap with its clusters
    method_scores: dict[str, list[float]] = {}
    method_total_members: dict[str, int] = {}

    if method_map is None:
        logger.warning("Cannot compute method reliability: no method_map or assignments")
        return {}

    for mname, m_clusters in method_map.items():
        scores: list[float] = []
        total_members = 0

        for m_cluster_id, m_members in m_clusters.items():
            # Find overlapping consensus clusters
            for c in clusters:
                overlap = m_members & set(c.member_ids)
                if len(overlap) >= 2:  # meaningful overlap
                    # Weight by overlap size
                    cal_conf = c.cluster_confidence if c.cluster_confidence > 0 else 0.5
                    for _ in range(len(overlap)):
                        scores.append(cal_conf)
                    total_members += len(overlap)

        if scores:
            method_scores[mname] = scores
            method_total_members[mname] = total_members

    # Compute per-method reliability
    reliability: dict[str, float] = {}
    for mname, scores in method_scores.items():
        if scores:
            reliability[mname] = float(np.mean(scores))
        else:
            reliability[mname] = 0.5  # neutral default

    logger.info(
        f"Method reliability from calibration: "
        f"{ {m: f'{r:.3f}' for m, r in reliability.items()} }"
    )
    return reliability


def iterative_reweight(
    assignments: list[ClusterAssignment],
    initial_weights: dict[str, float],
    consensus_fn,       # callable: (assignments, weights) -> (clusters, edges)
    calibrator_factory, # callable: (clusters, edges) -> Calibrator
    *,
    max_iter: int = 5,
    alpha: float = 0.6,     # EMA rate: 0=static, 1=fully replace each iteration
    convergence_eps: float = 1e-3,
    min_weight: float = 0.05,
) -> tuple[dict[str, float], list[dict]]:
    """Iteratively refine method weights using calibration feedback.

    Args:
        assignments: initial ClusterAssignments from all methods.
        initial_weights: starting weights (e.g., from empirical_weights).
        consensus_fn: fn(assignments, weights) → (clusters, edges).
        calibrator_factory: fn(clusters, edges) → Calibrator.
        max_iter: maximum refinement iterations.
        alpha: EMA smoothing (0=no update, 1=full replacement).
        convergence_eps: stop when max(|w_new - w_old|) < eps.
        min_weight: floor to prevent zeroing out methods.

    Returns:
        (final_weights, history) where history is a list of per-iteration
        dicts with keys: iter, weights, reliability, clusters, delta.
    """
    weights = dict(initial_weights)
    # Normalize initial weights
    total = sum(weights.values())
    if total > 0:
        weights = {m: max(w / total, min_weight) for m, w in weights.items()}

    method_map = method_cluster_map(assignments)
    active_methods = [m for m in weights if m in method_map]
    history: list[dict] = []

    logger.info(
        f"Calibration-in-the-loop: {len(active_methods)} methods, "
        f"max_iter={max_iter}, α={alpha}"
    )

    for iteration in range(max_iter):
        # Step 1: Run consensus
        clusters, edges = consensus_fn(assignments, weights)

        if not clusters:
            logger.warning(f"Iter {iteration}: no clusters produced; stopping")
            break

        # Step 2: Calibrate
        calibrator = calibrator_factory(clusters, edges)

        # Compute raw confidences and calibrate
        try:
            from ..calibration.calibrator import raw_confidence
        except ImportError:
            raw_confidence = None

        # Set cluster_confidence from calibrator
        for c in clusters:
            try:
                rc = raw_confidence(c, edges) if raw_confidence else 0.5
                c.cluster_confidence = float(calibrator.predict(rc))
            except Exception:
                c.cluster_confidence = 0.5

        # Step 3: Compute per-method reliability
        reliability = method_reliability_from_calibration(
            clusters, calibrator, method_map=method_map,
        )

        if not reliability:
            logger.warning(f"Iter {iteration}: could not compute reliability; stopping")
            break

        # Step 4: Update weights (EMA)
        new_weights: dict[str, float] = {}
        for m in active_methods:
            old_w = weights.get(m, min_weight)
            r = reliability.get(m, 0.5)
            new_weights[m] = (1 - alpha) * old_w + alpha * r

        # Apply floor and normalize
        new_weights = {m: max(w, min_weight) for m, w in new_weights.items()}
        total = sum(new_weights.values())
        new_weights = {m: w / total for m, w in new_weights.items()}

        # Step 5: Check convergence
        deltas = {m: abs(new_weights.get(m, 0) - weights.get(m, 0)) for m in active_methods}
        max_delta = max(deltas.values()) if deltas else 0.0

        history.append({
            "iter": iteration,
            "weights": dict(new_weights),
            "reliability": dict(reliability),
            "n_clusters": len(clusters),
            "max_delta": max_delta,
        })

        logger.info(
            f"Iter {iteration}: max_delta={max_delta:.4f}, "
            f"n_clusters={len(clusters)}, "
            f"weights={ {m: f'{w:.3f}' for m, w in new_weights.items()} }"
        )

        weights = new_weights

        if max_delta < convergence_eps:
            logger.info(f"Converged at iteration {iteration}")
            break

    if not history:
        # At least record initial state
        history.append({"iter": 0, "weights": dict(weights), "reliability": {},
                       "n_clusters": 0, "max_delta": 0.0})

    return weights, history
