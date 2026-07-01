"""Rule-based method selection from dataset profile and user objective."""

from __future__ import annotations

import logging
from typing import Any

from ..schema.records import (
    DatasetProfile,
    RunPlan,
    Objective,
    ConsensusMode,
    RepertoireType,
)

logger = logging.getLogger(__name__)

# Default available methods (check availability at runtime)
ALL_METHODS = ['hd_baseline', 'clustcr', 'gliph2', 'tcrdist3', 'giana', 'tcrmatch', 'deeptcr']


def select_methods(
    profile: DatasetProfile,
    objective: Objective | str = Objective.BALANCED,
    config: dict | None = None,
    available_methods: list[str] | None = None,
) -> RunPlan:
    """Select methods, consensus mode, and parameters from profile + objective."""
    config = config or {}
    available = set(available_methods or ALL_METHODS)

    if isinstance(objective, str):
        objective = Objective(objective)

    # Determine scenario from profile
    scenario = _classify_scenario(profile)

    # Get rule from config or use defaults
    selection_rules = config.get("selection", {}).get("rules", {})
    rule = selection_rules.get(scenario, {})

    if rule:
        methods = [m for m in rule.get("methods", []) if m in available]
        consensus_mode_str = rule.get("consensus_mode", "balanced")
    else:
        methods, consensus_mode_str = _default_selection(profile, objective, available)

    # Ensure at least HD baseline is included
    if "hd_baseline" in available and "hd_baseline" not in methods:
        methods.append("hd_baseline")

    consensus_mode = ConsensusMode(consensus_mode_str)

    use_tiered = (
        objective == Objective.FAST_SCREENING
        or config.get("tiered", {}).get("enabled", False)
    )

    return RunPlan(
        objective=objective,
        selected_methods=methods,
        consensus_mode=consensus_mode,
        use_tiered=use_tiered,
        method_params={},
        weighting_profile=_get_weight_profile(objective),
        refinement_params=config.get("refinement", {}),
        reporting_flags=config.get("reporting", {}),
    )


def _classify_scenario(profile: DatasetProfile) -> str:
    """Classify dataset into a named scenario."""
    if profile.background_noise_score > 0.5 and profile.chain_mode.value == "beta_only":
        return "bulk_noisy_beta"
    if profile.repertoire_type in (RepertoireType.ANTIGEN_ENRICHED, RepertoireType.CURATED_DB):
        return "antigen_enriched"
    return "balanced"


def _default_selection(
    profile: DatasetProfile,
    objective: Objective,
    available: set[str],
) -> tuple[list[str], str]:
    """Fallback selection when no config rule matches."""
    methods = []
    consensus_mode = "balanced"

    if objective == Objective.HIGH_PURITY:
        candidates = ["clustcr", "gliph2", "tcrmatch", "hd_baseline"]
        consensus_mode = "conservative"
    elif objective == Objective.HIGH_RECALL:
        candidates = ["deeptcr", "giana", "tcrdist3", "hd_baseline"]
        consensus_mode = "coverage"
    elif objective == Objective.FAST_SCREENING:
        # Tiered: cheap methods on full data, expensive only on divergent subset
        candidates = ["hd_baseline", "levenshtein", "giana", "tcrdist3", "gliph2", "tcrmatch"]
        consensus_mode = "balanced"
        # FAST_SCREENING always uses tiered execution
    elif objective == Objective.NOISE_ROBUST:
        candidates = ["tcrdist3", "gliph2", "hd_baseline"]
        consensus_mode = "conservative"
    else:
        candidates = list(available)

    methods = [m for m in candidates if m in available]
    if not methods:
        methods = list(available)

    return methods, consensus_mode


def _get_weight_profile(objective: Objective) -> str:
    """Map objective to weighting profile name."""
    mapping = {
        Objective.HIGH_PURITY: "high_purity",
        Objective.HIGH_RECALL: "high_recall",
        Objective.NOISE_ROBUST: "noise_robust",
        Objective.FAST_SCREENING: "fast_screening",
        Objective.BALANCED: "balanced",
    }
    return mapping.get(objective, "balanced")
