"""Co-association matrix and pairwise support extraction.

Tier-2/4b extension: SIGNED consensus edges. When a `purity_lookup` is supplied,
each attractive edge also accumulates REPULSION from high-purity methods that
SEPARATED the pair (assigned both TCRs but to different clusters). The edge's
final_score becomes the NET evidence (weighted_support - repulsion_support), so
a pair co-clustered only by a noisy method but consistently split by
high-purity methods can be dropped even under a low (coverage) threshold.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from ..schema.records import ClusterAssignment, ConsensusEdge

logger = logging.getLogger(__name__)


def extract_pairwise_support(
    assignments: list[ClusterAssignment],
    weights: dict[str, float] | None = None,
    purity_lookup: dict[str, float] | None = None,
    high_purity_threshold: float = 0.9,
    use_signed: bool = False,
    repulsion_discount: float = 1.0,
    min_repulsion_methods: int = 1,
) -> list[ConsensusEdge]:
    """Extract pairwise co-association edges from cluster assignments.

    For each method, TCR pairs in the same cluster get a positive support link.
    When `use_signed` is True and `purity_lookup` is provided, high-purity
    methods (purity >= high_purity_threshold) that placed a pair in DIFFERENT
    clusters add negative (repulsion) evidence to that pair's edge.

    repulsion_discount: fraction of the separating methods' weight actually
        subtracted (default 1.0 = full). <1.0 makes repulsion a modulator
        rather than a veto, preventing a single high-purity method from
        overriding multiple agreeing methods.
    min_repulsion_methods: require at least this many high-purity methods to
        separate the pair before any repulsion is applied (default 1).
    """
    weights = weights or {}

    # Group assignments by method
    method_clusters: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for a in assignments:
        method_clusters[a.method][a.cluster_id].add(a.tcr_id)

    # Optional: per-method tcr_id -> cluster_id map, for repulsion computation
    signed = use_signed and bool(purity_lookup)
    method_tcr_cluster: dict[str, dict[str, str]] = {}
    repulsion_methods: set[str] = set()
    if signed:
        for method, clusters in method_clusters.items():
            pur = purity_lookup.get(method, 0.0)  # type: ignore[union-attr]
            if pur >= high_purity_threshold:
                repulsion_methods.add(method)
                tcm: dict[str, str] = {}
                for cid, members in clusters.items():
                    for tid in members:
                        tcm[tid] = cid
                method_tcr_cluster[method] = tcm
        if repulsion_methods:
            logger.info(
                f"Signed consensus: repulsion from high-purity methods "
                f"({sorted(repulsion_methods)}, purity>={high_purity_threshold})"
            )

    # Accumulate pairwise positive support
    pair_scores: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"method_count": 0, "weighted_sum": 0.0, "methods": []}
    )

    for method, clusters in method_clusters.items():
        w = weights.get(method, 1.0)
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue
            for a, b in combinations(sorted(members), 2):
                key = (a, b)
                pair_scores[key]["method_count"] += 1
                pair_scores[key]["weighted_sum"] += w
                pair_scores[key]["methods"].append(method)

    # Build ConsensusEdge list
    edges = []
    for (a, b), scores in pair_scores.items():
        repulsion = 0.0
        if signed and repulsion_methods:
            sep_methods = []
            sep_weight = 0.0
            for m, tcm in method_tcr_cluster.items():
                ca = tcm.get(a)
                cb = tcm.get(b)
                # both assigned by m but to different clusters => separation
                if ca is not None and cb is not None and ca != cb:
                    sep_methods.append(m)
                    sep_weight += weights.get(m, 1.0)
            if len(sep_methods) >= min_repulsion_methods:
                repulsion = sep_weight * repulsion_discount
        net = scores["weighted_sum"] - repulsion
        edges.append(
            ConsensusEdge(
                tcr_id_a=a,
                tcr_id_b=b,
                method_support_count=scores["method_count"],
                weighted_support=scores["weighted_sum"],
                repulsion_support=repulsion,
                final_score=net,  # base score; refined later
            )
        )

    return edges


def build_coassociation_matrix(
    assignments: list[ClusterAssignment],
    tcr_ids: list[str],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Build dense co-association matrix.

    C[i,j] = sum of weights for methods that cluster i and j together.
    """
    weights = weights or {}
    n = len(tcr_ids)
    idx_map = {tid: i for i, tid in enumerate(tcr_ids)}
    matrix = np.zeros((n, n), dtype=np.float64)

    method_clusters: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for a in assignments:
        method_clusters[a.method][a.cluster_id].add(a.tcr_id)

    for method, clusters in method_clusters.items():
        w = weights.get(method, 1.0)
        for cluster_id, members in clusters.items():
            for a, b in combinations(members, 2):
                if a in idx_map and b in idx_map:
                    i, j = idx_map[a], idx_map[b]
                    matrix[i, j] += w
                    matrix[j, i] += w

    return matrix
