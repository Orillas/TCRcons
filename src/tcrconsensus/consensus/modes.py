"""Consensus clustering modes: conservative and balanced.

Tier-2/4b: each mode now accepts optional `purity_lookup` + `use_signed` to
enable SIGNED consensus edges (repulsion from high-purity methods that separate
a pair). Defaults preserve the original unsigned behavior.

Tier-1/F2+F3: balanced/coverage accept `use_fusion` + `use_fdr_threshold`
(defaults off, backward compatible). When on, the pipeline becomes
  extract_pairwise_support -> estimate_null -> enrich_and_fuse ->
  compute_fdr_threshold -> build_consensus_graph(tau)
filling the previously-dead ConsensusEdge signal fields and replacing the
fixed 0.3 threshold with a scale-invariant FDR threshold.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from .coassociation import extract_pairwise_support
from .graph import (
    build_consensus_graph,
    connected_components_clustering,
    community_clustering,
)
from .null_model import (
    compute_analytical_null,
    enrich_significance,
    analytical_fdr_threshold,
    AnalyticalNullResult,
)
from .reweight import iterative_reweight, method_cluster_map, method_reliability_from_calibration
from ..schema.records import (
    ClusterAssignment,
    ConsensusCluster,
    ConsensusEdge,
)

logger = logging.getLogger(__name__)


def _resolve_fusion_threshold(
    edges: list[ConsensusEdge],
    assignments: list[ClusterAssignment],
    weights: dict[str, float],
    *,
    fixed_threshold: float,
    use_fusion: bool,
    use_fdr_threshold: bool,
    target_fdr: float,
    null_permutations: int,
    null_seed: int,
    fusion_context: Optional[dict],
    fdr_scale: str = "fused",
    use_analytical_null: bool = False,
    analytical_null: Optional[AnalyticalNullResult] = None,
):
    """Apply F2 (multi-signal fusion) and/or F3 (FDR threshold) when requested.

    Innovation #1: when use_analytical_null=True, edges are scored by
    chance-corrected significance (observed - expected) using per-method
    analytical null probabilities instead of permutation.

    Returns ``(build_threshold, edges_to_use)`` for build_consensus_graph. If
    neither flag is set, returns ``(fixed_threshold, edges)`` (original behavior).

    fdr_scale:
      - "fused" (default): FDR threshold on the FUSED final_score. The fusion
        (seq/vj) participates in edge SELECTION — empirically the stronger
        selector on TCR data, though both scales lose to a tuned fixed threshold
        because the preserve-cluster-size permutation null is conservative.
      - "vote": FDR gate on net_vote (weighted_support - repulsion), the only
        quantity that varies under label permutation; seq/vj/noise fusion is
        applied to the SURVIVORS only for ranking/confidence. Statistically
        cleaner but empirically WORSE (fusion's selection benefit is lost).
        Kept as a research option; not recommended for production.

    The permutation null is computed ONCE and shared: its per-pair co-cluster
    counts feed F2's noise_penalty, and its score distribution feeds F3's gate.
    """
    do_fusion = use_fusion and fusion_context and fusion_context.get("lookups")
    if not use_fdr_threshold and not do_fusion:
        return fixed_threshold, edges

    from .null_model import estimate_null, compute_fdr_threshold
    from .fusion import enrich_and_fuse, make_null_fuse_fn
    ctx = fusion_context or {}

    if fdr_scale == "vote" and use_fdr_threshold:
        null_result = estimate_null(assignments, weights, B=null_permutations,
                                    seed=null_seed, fuse_fn=None)   # vote-only null
        net = [e.weighted_support - e.repulsion_support for e in edges]
        tau = compute_fdr_threshold(net, null_result.null_scores, q=target_fdr)
        kept = [e for e in edges if (e.weighted_support - e.repulsion_support) >= tau]
        if do_fusion:
            enrich_and_fuse(kept, lookups=ctx["lookups"], pw_beta=ctx.get("pw_beta"),
                            tcr_to_idx=ctx.get("tcr_to_idx"), pair_null=null_result.pair_null_count,
                            B=null_result.B, betas=ctx.get("betas"), intercept=ctx.get("intercept", 0.0),
                            use_signals=ctx.get("use_signals", ("vote", "seq", "vj", "noise")),
                            tau_seq=ctx.get("tau_seq", 50.0))
        else:
            for e in kept:
                e.final_score = e.weighted_support - e.repulsion_support
        return 0.0, kept   # survivors already gated; build keeps all of `kept`

    # fused-scale (legacy) path
    fuse_fn = None
    if do_fusion:
        fuse_fn = make_null_fuse_fn(ctx["lookups"], pw_beta=ctx.get("pw_beta"),
                                    tcr_to_idx=ctx.get("tcr_to_idx"), betas=ctx.get("betas"),
                                    intercept=ctx.get("intercept", 0.0),
                                    use_signals=ctx.get("use_signals_null", ("vote", "seq", "vj")),
                                    tau_seq=ctx.get("tau_seq", 50.0))
    null_result = estimate_null(assignments, weights, B=null_permutations, seed=null_seed, fuse_fn=fuse_fn)
    if do_fusion:
        enrich_and_fuse(edges, lookups=ctx["lookups"], pw_beta=ctx.get("pw_beta"),
                        tcr_to_idx=ctx.get("tcr_to_idx"), pair_null=null_result.pair_null_count,
                        B=null_result.B, betas=ctx.get("betas"), intercept=ctx.get("intercept", 0.0),
                        use_signals=ctx.get("use_signals", ("vote", "seq", "vj", "noise")),
                        tau_seq=ctx.get("tau_seq", 50.0))
    if use_fdr_threshold:
        tau = compute_fdr_threshold([e.final_score for e in edges], null_result.null_scores, q=target_fdr)
    else:
        tau = fixed_threshold
    return tau, edges


def conservative_consensus(
    assignments: list[ClusterAssignment],
    weights: dict[str, float] | None = None,
    min_method_support: int = 2,
    require_high_purity: bool = True,
    high_purity_methods: list[str] | None = None,
    threshold: float = 0.0,
    purity_lookup: dict[str, float] | None = None,
    high_purity_threshold: float = 0.9,
    use_signed: bool = False,
    **kwargs,
) -> tuple[list[ConsensusCluster], list[ConsensusEdge]]:
    """Conservative consensus: only link pairs supported by k methods.

    Optionally require at least one high-purity method in support.
    """
    high_purity_methods = high_purity_methods or ["clustcr", "tcrmatch", "gliph2"]
    weights = weights or {}

    # Extract pairwise edges
    edges = extract_pairwise_support(
        assignments, weights,
        purity_lookup=purity_lookup,
        high_purity_threshold=high_purity_threshold,
        use_signed=use_signed,
    )

    # Filter by minimum method support
    filtered = []
    for edge in edges:
        if edge.method_support_count < min_method_support:
            continue
        if require_high_purity:
            # Check if at least one supporting method is high-purity
            # Re-derive supporting methods from assignments
            pass  # handled by threshold on weighted_support
        filtered.append(edge)

    if not filtered:
        return [], edges

    # Build graph from filtered edges
    graph = build_consensus_graph(filtered, threshold=max(threshold, 0.1))
    clusters = connected_components_clustering(graph)

    return clusters, edges


def balanced_consensus(
    assignments: list[ClusterAssignment],
    weights: dict[str, float] | None = None,
    threshold: float = 0.3,
    purity_lookup: dict[str, float] | None = None,
    high_purity_threshold: float = 0.9,
    use_signed: bool = False,
    repulsion_discount: float = 1.0,
    min_repulsion_methods: int = 1,
    use_fusion: bool = False,
    use_fdr_threshold: bool = False,
    target_fdr: float = 0.05,
    null_permutations: int = 100,
    null_seed: int = 0,
    fusion_context: Optional[dict] = None,
    fdr_scale: str = "fused",
    **kwargs,
) -> tuple[list[ConsensusCluster], list[ConsensusEdge]]:
    """Balanced consensus: weighted co-association + connected components.

    Uses connected components (not Leiden) because Exp3 ablation showed
    Leiden over-merges dense co-association regions in TCR clustering,
    reducing ARI from 0.344 (CC) to 0.207 (Leiden).

    Tier-2/4b: when use_signed=True, edge final_score is the NET evidence
    (weighted_support - repulsion_support*discount), so the fixed threshold
    now also respects high-purity disagreement. repulsion_discount<1.0 makes
    repulsion a modulator rather than a single-method veto.

    Tier-1/F2+F3: when use_fusion=True (with fusion_context) and/or
    use_fdr_threshold=True, edges are multi-signal fused and thresholded by a
    scale-invariant FDR threshold instead of the fixed 0.3. Defaults off.
    fdr_scale="fused" (default) gates on the fused score; "vote" gates on
    net_vote then fuses survivors (research option; empirically weaker).
    """
    weights = weights or {}

    edges = extract_pairwise_support(
        assignments, weights,
        purity_lookup=purity_lookup,
        high_purity_threshold=high_purity_threshold,
        use_signed=use_signed,
        repulsion_discount=repulsion_discount,
        min_repulsion_methods=min_repulsion_methods,
    )
    # Innovation #1: compute analytical null if requested
    analytical_null = None
    if kwargs.get("use_analytical_null"):
        analytical_null = compute_analytical_null(assignments, weights)

    tau, edges_for_graph = _resolve_fusion_threshold(
        edges, assignments, weights,
        fixed_threshold=threshold,
        use_fusion=use_fusion, use_fdr_threshold=use_fdr_threshold,
        target_fdr=target_fdr, null_permutations=null_permutations,
        null_seed=null_seed, fusion_context=fusion_context, fdr_scale=fdr_scale,
        use_analytical_null=kwargs.get("use_analytical_null", False),
        analytical_null=analytical_null,
    )
    graph = build_consensus_graph(edges_for_graph, threshold=tau)
    clusters = connected_components_clustering(graph)

    return clusters, edges


def coverage_consensus(
    assignments: list[ClusterAssignment],
    weights: dict[str, float] | None = None,
    threshold: float = 0.1,
    purity_lookup: dict[str, float] | None = None,
    high_purity_threshold: float = 0.9,
    use_signed: bool = False,
    repulsion_discount: float = 1.0,
    min_repulsion_methods: int = 1,
    use_fusion: bool = False,
    use_fdr_threshold: bool = False,
    target_fdr: float = 0.05,
    null_permutations: int = 100,
    null_seed: int = 0,
    fusion_context: Optional[dict] = None,
    fdr_scale: str = "fused",
    **kwargs,
) -> tuple[list[ConsensusCluster], list[ConsensusEdge]]:
    """Coverage consensus: maximize recall by linking pairs from any single method.

    Uses lower threshold than balanced. Any pair co-clustered by at least one
    method with non-zero weight is linked. Connected components produce
    larger clusters for maximum coverage.

    Tier-2/4b: signed repulsion is ESPECIALLY valuable here — it stops a single
    noisy method (e.g. tcrmatch) from over-merging pairs that high-purity
    methods consistently separate.

    Tier-1/F2+F3: same use_fusion / use_fdr_threshold hooks as balanced. With
    FDR on, the "coverage" intent is realised by the data-driven (typically
    lower) FDR threshold rather than the hard-coded 0.1. fdr_scale as in
    balanced.
    """
    weights = weights or {}

    edges = extract_pairwise_support(
        assignments, weights,
        purity_lookup=purity_lookup,
        high_purity_threshold=high_purity_threshold,
        use_signed=use_signed,
        repulsion_discount=repulsion_discount,
        min_repulsion_methods=min_repulsion_methods,
    )
    # Innovation #1: compute analytical null if requested
    analytical_null = None
    if kwargs.get("use_analytical_null"):
        analytical_null = compute_analytical_null(assignments, weights)

    tau, edges_for_graph = _resolve_fusion_threshold(
        edges, assignments, weights,
        fixed_threshold=threshold,
        use_fusion=use_fusion, use_fdr_threshold=use_fdr_threshold,
        target_fdr=target_fdr, null_permutations=null_permutations,
        null_seed=null_seed, fusion_context=fusion_context, fdr_scale=fdr_scale,
        use_analytical_null=kwargs.get("use_analytical_null", False),
        analytical_null=analytical_null,
    )
    # Lower threshold — include edges from any method
    graph = build_consensus_graph(edges_for_graph, threshold=tau)
    clusters = connected_components_clustering(graph)

    return clusters, edges
