"""Method selection from dataset profile and user objective.

Tiered execution (cheap methods on full data, expensive on divergent subset)
is always the default. Consensus mode is no longer driven by objective —
defaults to ``balanced``; users override via ``mode`` at run time.
"""

from __future__ import annotations

import logging

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
    objective: Objective | str = Objective.FAST_SCREENING,
    config: dict | None = None,
    available_methods: list[str] | None = None,
) -> RunPlan:
    """Select methods from profile + objective.

    Tiered execution runs cheap methods on the full dataset and expensive
    methods only on divergent TCR subsets where cheap methods disagree.

    Consensus mode is always ``balanced`` by default; override via the
    ``mode`` parameter on ``TCRConsensus()`` or CLI ``--mode`` flag.
    """
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
    else:
        methods = _default_selection(profile, objective, available)

    # Ensure at least HD baseline is included
    if "hd_baseline" in available and "hd_baseline" not in methods:
        methods.append("hd_baseline")

    # Tiered execution is always enabled by default
    # Opt out via config: tiered.enabled: false
    use_tiered = config.get("tiered", {}).get("enabled", True)

    return RunPlan(
        objective=objective,
        selected_methods=methods,
        consensus_mode=ConsensusMode.BALANCED,
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
) -> list[str]:
    """Fallback method selection when no config rule matches.

    Returns only the method list — consensus mode is always ``balanced``
    and tiered execution is always enabled by default.
    """
    if objective == Objective.HIGH_PURITY:
        candidates = ["clustcr", "gliph2", "tcrmatch", "hd_baseline"]
    elif objective == Objective.HIGH_RECALL:
        candidates = ["deeptcr", "giana", "tcrdist3", "hd_baseline"]
    elif objective == Objective.FAST_SCREENING:
        # Tiered: cheap methods on full data, expensive only on divergent subset
        candidates = ["hd_baseline", "levenshtein", "giana", "tcrdist3", "gliph2", "tcrmatch"]
    elif objective == Objective.NOISE_ROBUST:
        candidates = ["tcrdist3", "gliph2", "hd_baseline"]
    else:
        candidates = list(available)

    methods = [m for m in candidates if m in available]
    if not methods:
        methods = list(available)

    return methods


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
