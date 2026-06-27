"""
Tiered method invocation — efficiency innovation (#7).

Run cheap methods first on the FULL dataset, then identify "divergent regions"
where cheap methods disagree, and run expensive methods ONLY on those TCRs.

Core insight: if all cheap methods agree that a TCR pair belongs together (or
apart), expensive methods won't change that outcome. Expensive methods add
value ONLY on pairs where cheap methods disagree — the "divergent" pairs.

This is the first tiered-invocation innovation in TCR consensus clustering;
existing ensemble methods (EAC, Strehl-Ghosh, scikit-learn consensus) all run
every method on every datapoint regardless of cost.

Benchmark (v3_cd8, 5315 TCRs, 5 methods):
  - tcrdist3 O(n²) distance matrix is THE bottleneck (~90% of total runtime)
  - If 40% of TCRs are "divergent", tcrdist3 runs on ~2100 TCRs → ~6x speedup
  - On 26k TCRs with 70% consensus: tcrdist3 on ~7800 → ~11x speedup for O(n²)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from ..schema.records import ClusterAssignment

logger = logging.getLogger(__name__)

# ── Tier classification ────────────────────────────────────────────────────
# Based on empirical profiling of method runtimes on v3_cd8 (5315 TCRs).
# "cheap" = always run on full data; "expensive" = run only on divergent subset.
# Override via config["tiered"]["tiers"] dict.

# Tier classification based on empirical runtimes (v3_cd8, 6112 TCRs):
#   FULL (cheap-enough):  hd_baseline=3.7s, giana=7.1s, clustcr/gliph2/tcrmatch all <5min
#   TARGETED (bottleneck): tcrdist3 O(n^2) pw distance + silhouette scan = 253s
#                          deeptcr neural inference + special venv = expensive
DEFAULT_TIERS = {
    "full": ["hd_baseline", "levenshtein", "clustcr", "gliph2", "giana", "tcrmatch"],
    "targeted": ["tcrdist3", "deeptcr"],
}

# Minimum divergent subset size to run expensive methods on.
# Below this, the subset is too small to produce meaningful clusters.
MIN_DIVERGENT_SIZE = 20


def split_methods_by_tier(
    methods: list[str],
    tiers: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Split method list into (full_set_methods, targeted_methods).

    Args:
        methods: all selected method names.
        tiers: {"full": [...], "targeted": [...]}. Uses DEFAULT_TIERS if None.

    Returns:
        (cheap_methods, expensive_methods) — only methods that are in `methods`.
    """
    tiers = tiers or DEFAULT_TIERS
    full_set = [m for m in methods if m in tiers.get("full", [])]
    targeted = [m for m in methods if m in tiers.get("targeted", [])]
    return full_set, targeted


def detect_divergent_tcrs(
    assignments: list[ClusterAssignment],
    weights: dict[str, float],
    all_tcr_ids: list[str],
    low_threshold: float = 0.0,
    high_threshold: float = 1.0,
) -> tuple[set[str], dict]:
    """Identify TCRs that participate in pairs where cheap methods disagree.

    A pair (i,j) is "divergent" if:
        low_threshold < cheap_support_fraction < high_threshold
    where cheap_support_fraction = Σ w_m·1[m clusters i,j] / Σ w_m.

    With default thresholds (0.0, 1.0), a pair is divergent if some cheap
    methods cluster it together and others separate it — classic disagreement.

    Args:
        assignments: ClusterAssignments from cheap (full-set) methods only.
        weights: method weights dict.
        all_tcr_ids: complete list of TCR IDs in the dataset.
        low_threshold: pairs strictly above this fraction are candidates.
        high_threshold: pairs strictly below this fraction are candidates.

    Returns:
        (divergent_tcr_ids, stats_dict) where stats_dict has:
          - n_pairs_total, n_pairs_consensus_together, n_pairs_consensus_apart,
            n_pairs_divergent, divergent_tcr_count, divergent_fraction
    """
    weights = weights or {}
    if not assignments:
        logger.warning("No cheap assignments to detect divergence from")
        return set(all_tcr_ids), {"error": "no_cheap_assignments"}

    # ── Group assignments by method → cluster_id → {tcr_id, ...} ──
    method_clusters: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    cheap_methods: set[str] = set()
    for a in assignments:
        method_clusters[a.method][a.cluster_id].add(a.tcr_id)
        cheap_methods.add(a.method)

    if len(cheap_methods) < 2:
        # Single cheap method: can't detect divergence (no disagreement possible).
        # Conservative: send ALL TCRs to expensive methods.
        logger.info("Only 1 cheap method — cannot detect divergence; all TCRs targeted")
        return set(all_tcr_ids), {
            "reason": "single_cheap_method",
            "n_pairs_total": 0,
            "divergent_fraction": 1.0,
        }

    # ── Compute cheap support per pair ──
    total_weight = sum(weights.get(m, 1.0) for m in cheap_methods)
    if total_weight <= 0:
        total_weight = 1.0

    # We DON'T enumerate all O(n²) pairs — we only look at pairs that appear
    # in at least one cheap method's cluster (these are the only pairs with
    # non-zero cheap_support). Pairs that appear in NO cheap cluster have
    # cheap_support = 0 → "consensus apart" → not divergent.
    pair_support: dict[tuple[str, str], float] = defaultdict(float)
    tcr_in_some_cluster: set[str] = set()

    for method, clusters in method_clusters.items():
        w = weights.get(method, 1.0)
        for cid, members in clusters.items():
            if len(members) < 2:
                continue
            for a, b in combinations(sorted(members), 2):
                pair_support[(a, b)] += w
                tcr_in_some_cluster.add(a)
                tcr_in_some_cluster.add(b)

    # ── Classify pairs ──
    n_consensus_together = 0
    n_consensus_apart = 0
    n_divergent = 0

    divergent_tcrs: set[str] = set()

    # Pairs with non-zero support (appear in at least one cheap cluster)
    for (a, b), support in pair_support.items():
        frac = support / total_weight
        if frac >= high_threshold:
            n_consensus_together += 1
        elif frac <= low_threshold:
            n_consensus_apart += 1
        else:
            n_divergent += 1
            divergent_tcrs.add(a)
            divergent_tcrs.add(b)

    # Pairs that appear in NO cheap cluster have support=0 → consensus_apart.
    # We don't count them individually (would be O(n²)), but we know they
    # don't contribute to divergent_tcrs.
    n_tcrs = len(all_tcr_ids)
    max_pairs = n_tcrs * (n_tcrs - 1) // 2
    n_pairs_with_support = n_consensus_together + n_consensus_apart + n_divergent
    n_pairs_consensus_apart_implicit = max_pairs - n_pairs_with_support
    n_consensus_apart += n_pairs_consensus_apart_implicit

    # ── Stats ──
    stats = {
        "n_tcrs_total": n_tcrs,
        "n_pairs_total": max_pairs,
        "n_pairs_with_support": n_pairs_with_support,
        "n_pairs_consensus_together": n_consensus_together,
        "n_pairs_consensus_apart": n_consensus_apart,
        "n_pairs_divergent": n_divergent,
        "divergent_tcr_count": len(divergent_tcrs),
        "divergent_fraction": len(divergent_tcrs) / max(n_tcrs, 1),
        "cheap_methods": sorted(cheap_methods),
        "total_weight": total_weight,
    }
    logger.info(
        f"Divergence detection: {n_divergent}/{n_pairs_with_support} supported pairs divergent "
        f"→ {len(divergent_tcrs)}/{n_tcrs} TCRs ({stats['divergent_fraction']:.1%}) targeted. "
        f"Consensus: {n_consensus_together} together, {n_consensus_apart} apart."
    )
    return divergent_tcrs, stats


def filter_tcr_table(df: pd.DataFrame, tcr_ids: set[str]) -> pd.DataFrame:
    """Return rows of df whose tcr_id is in *tcr_ids*."""
    if "tcr_id" not in df.columns:
        raise ValueError("DataFrame missing 'tcr_id' column — cannot filter for tiered execution")
    mask = df["tcr_id"].astype(str).isin(tcr_ids)
    n_before = len(df)
    n_after = mask.sum()
    logger.info(f"TCR table filtered: {n_before} → {n_after} rows ({n_after/max(n_before,1):.1%})")
    return df[mask].copy()


def execute_tiered(
    df: pd.DataFrame,
    all_tcr_ids: list[str],
    cheap_methods: list[str],
    expensive_methods: list[str],
    clusterer_factory,   # callable: (method_name: str) -> BaseClusterer | None
    workdir,
    config: dict,
) -> tuple[list[ClusterAssignment], dict]:
    """Run tiered method invocation and return (assignments, tier_stats).

    1. Run cheap_methods on full df
    2. Detect divergent TCRs from cheap assignments
    3. If divergent set is meaningful: run expensive_methods on subset df
       Else: run expensive_methods on full df (fallback)
    4. Return combined assignments + tier statistics

    Args:
        df: full normalized TCR table.
        all_tcr_ids: complete list of TCR IDs.
        cheap_methods: methods to run on the full dataset.
        expensive_methods: methods to run on the divergent subset.
        clusterer_factory: fn(name) -> BaseClusterer instance or None.
        workdir: Path for method outputs.
        config: pipeline config dict.

    Returns:
        (all_assignments, tier_stats) — tier_stats is a dict suitable for
        inclusion in the run report.
    """
    all_assignments: list[ClusterAssignment] = []
    method_results: list[dict] = []
    tier_stats: dict = {
        "enabled": True,
        "cheap_methods": cheap_methods,
        "expensive_methods": expensive_methods,
    }

    # ── Phase 1: Run cheap methods on FULL data ──
    logger.info(f"=== Tier 1 (cheap, full data): {cheap_methods} ===")
    cheap_assignments: list[ClusterAssignment] = []

    weights = _get_weights(expensive_methods + cheap_methods, config)

    for mname in cheap_methods:
        clusterer = clusterer_factory(mname)
        if clusterer is None:
            logger.warning(f"Cheap method {mname} unavailable — skipping")
            continue
        m_workdir = workdir / "methods" / mname
        m_workdir.mkdir(parents=True, exist_ok=True)
        result = clusterer.safe_execute(df, m_workdir, config)
        cheap_assignments.extend(result.assignments)
        all_assignments.extend(result.assignments)
        method_results.append({
            "method": mname, "tier": "cheap", "status": result.status.value,
            "n_assignments": len(result.assignments),
            "runtime_seconds": result.runtime_seconds,
        })

    tier_stats["cheap_n_assignments"] = len(cheap_assignments)
    tier_stats["cheap_runtime_total"] = sum(
        r["runtime_seconds"] for r in method_results if r["tier"] == "cheap"
    )

    if not expensive_methods:
        logger.info("No expensive methods selected — tiered execution complete")
        return all_assignments, tier_stats

    # ── Phase 2: Detect divergent TCRs ──
    tier_config = config.get("tiered", {})
    low_threshold = tier_config.get("divergence_low", 0.0)
    high_threshold = tier_config.get("divergence_high", 1.0)
    min_divergent = tier_config.get("min_divergent_size", MIN_DIVERGENT_SIZE)

    # Use weights for the cheap methods only
    cheap_weights = {m: weights.get(m, 1.0) for m in cheap_methods if m in weights or True}
    # If cheap methods aren't in weights, give them equal weight
    for m in cheap_methods:
        if m not in cheap_weights:
            cheap_weights[m] = 1.0

    divergent_tcrs, div_stats = detect_divergent_tcrs(
        cheap_assignments, cheap_weights, all_tcr_ids,
        low_threshold=low_threshold, high_threshold=high_threshold,
    )
    tier_stats["divergence"] = div_stats

    # ── Phase 3: Decide subset vs full ──
    use_subset = (
        len(divergent_tcrs) >= min_divergent
        and len(divergent_tcrs) < len(all_tcr_ids) * 0.95  # only worth it if saving ≥5%
        and div_stats.get("reason") != "single_cheap_method"
    )

    if use_subset:
        logger.info(
            f"=== Tier 2 (expensive, divergent subset): {expensive_methods} "
            f"on {len(divergent_tcrs)}/{len(all_tcr_ids)} TCRs ==="
        )
        df_expensive = filter_tcr_table(df, divergent_tcrs)
        tier_stats["execution_mode"] = "targeted"
        tier_stats["targeted_tcr_count"] = len(divergent_tcrs)
        tier_stats["compute_saved_fraction"] = 1.0 - len(divergent_tcrs) / max(len(all_tcr_ids), 1)
    else:
        logger.info(
            f"=== Tier 2 (expensive, FULL data — fallback): {expensive_methods} "
            f"(divergent={len(divergent_tcrs)}, min={min_divergent}) ==="
        )
        df_expensive = df
        tier_stats["execution_mode"] = "full_fallback"
        reason = (
            "divergent_set_too_small"
            if len(divergent_tcrs) < min_divergent
            else "divergent_set_covers_most_data"
        )
        tier_stats["fallback_reason"] = reason

    # ── Phase 4: Run expensive methods on (possibly filtered) data ──
    for mname in expensive_methods:
        clusterer = clusterer_factory(mname)
        if clusterer is None:
            logger.warning(f"Expensive method {mname} unavailable — skipping")
            method_results.append({
                "method": mname, "tier": "expensive", "status": "unavailable",
                "n_assignments": 0, "runtime_seconds": 0.0,
            })
            continue
        m_workdir = workdir / "methods" / mname
        m_workdir.mkdir(parents=True, exist_ok=True)
        result = clusterer.safe_execute(df_expensive, m_workdir, config)
        all_assignments.extend(result.assignments)

        # If running on subset, mark assignments to distinguish from full-data ones
        method_results.append({
            "method": mname, "tier": "expensive",
            "ran_on": "subset" if use_subset else "full",
            "status": result.status.value,
            "n_assignments": len(result.assignments),
            "runtime_seconds": result.runtime_seconds,
        })

    tier_stats["expensive_runtime_total"] = sum(
        r.get("runtime_seconds", 0) for r in method_results if r.get("tier") == "expensive"
    )
    tier_stats["method_results"] = method_results

    logger.info(
        f"Tiered execution complete: {len(all_assignments)} total assignments "
        f"({len(cheap_assignments)} cheap + {len(all_assignments) - len(cheap_assignments)} expensive)"
    )
    return all_assignments, tier_stats


def _get_weights(methods: list[str], config: dict) -> dict[str, float]:
    """Get method weights from config or compute from empirical priors."""
    consensus_cfg = config.get("consensus", {})
    weights_cfg = consensus_cfg.get("weights", {})
    priors = weights_cfg.get("global_priors", {})

    if priors:
        # Legacy config-based weights
        raw = {}
        for m in methods:
            p = priors.get(m, {"purity": 0.5, "sensitivity": 0.5, "noise_robust": 0.5, "speed": 0.5})
            raw[m] = 0.25 * (p["purity"] + p["sensitivity"] + p["noise_robust"] + p["speed"])
        total = sum(raw.values())
        if total > 0:
            return {m: w / total for m, w in raw.items()}
        return {m: 1.0 / len(methods) for m in methods}

    # Use empirical weights
    try:
        from ..consensus.weights import empirical_weights
        return empirical_weights(methods)
    except Exception:
        return {m: 1.0 / len(methods) for m in methods}
