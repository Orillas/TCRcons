"""Method weight computation from priors, scenario, and empirical data."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Empirical performance priors from Exp1 v3 (6,112 TCRs, 74 epitopes).
# AMI from Exp1 v3 benchmark; sensitivity/purity/ari from per_method_detail.tsv.
# ---------------------------------------------------------------------------

EMPIRICAL_PRIORS = {
    # Methods ranked by ARI contribution in Exp3 ablation.
    # AMI (Adjusted Mutual Information) corrects for chance, unlike NMI.
    # - GLIPH2: best single-method ARI (0.196), high AMI (0.458)
    # - GIANA: second best ARI (0.183), moderate AMI (0.362)
    # - HD Baseline: moderate ARI (0.112), AMI (0.400)
    # - TCRMatch: high sensitivity (0.86) but low purity (0.45), ARI=0.093
    # - clusTCR: very high purity (0.97) but only 12.9% retention
    # - DeepTCR: 100% retention but ARI=0.016, AMI=0.185, noise source
    # - TCRdist3: near-random ARI=0.004, AMI=0.092, noise source
    "gliph2": {
        "purity": 0.89, "sensitivity": 0.22, "ari": 0.196,
        "ami": 0.458, "noise_robust": 0.8,
    },
    "giana": {
        "purity": 0.69, "sensitivity": 0.08, "ari": 0.183,
        "ami": 0.362, "noise_robust": 0.7,
    },
    "hd_baseline": {
        "purity": 0.90, "sensitivity": 0.25, "ari": 0.112,
        "ami": 0.400, "noise_robust": 0.7,
    },
    "tcrmatch": {
        "purity": 0.45, "sensitivity": 0.86, "ari": 0.093,
        "ami": 0.333, "noise_robust": 0.6,
    },
    "clustcr": {
        "purity": 0.97, "sensitivity": 0.42, "ari": 0.036,
        "ami": 0.336, "noise_robust": 0.3,
    },
    "deeptcr": {
        "purity": 0.70, "sensitivity": 0.03, "ari": 0.016,
        "ami": 0.185, "noise_robust": 0.2,
    },
    "tcrdist3": {
        "purity": 0.98, "sensitivity": 0.01, "ari": 0.004,
        "ami": 0.092, "noise_robust": 0.1,
    },
    "levenshtein": {
        "purity": 0.84, "sensitivity": 0.30, "ari": 0.237,
        "ami": 0.436, "noise_robust": 0.7,
    },
}

# Default weighting coefficients: ARI + AMI + purity + sensitivity + noise_robust
DEFAULT_COEFFICIENTS = {
    "ari": 0.40,
    "ami": 0.10,
    "purity": 0.30,
    "sensitivity": 0.10,
    "noise_robust": 0.10,
}


def empirical_weights(
    methods: list[str],
    priors: dict[str, dict] | None = None,
    coefficients: dict[str, float] | None = None,
    min_weight: float = 0.05,
) -> dict[str, float]:
    """Compute method weights from empirical benchmark data.

    w_m = a * ari + b * ami + c * purity + d * sensitivity + e * noise_robust

    Methods below min_weight are set to min_weight (not zero, to preserve
    their pairwise support in co-association graph).

    Args:
        methods: list of method names that are available
        priors: override EMPIRICAL_PRIORS (for testing)
        coefficients: override DEFAULT_COEFFICIENTS (for testing)
        min_weight: minimum weight floor (prevents zeroing out)

    Returns:
        dict mapping method_name -> weight (normalized to sum=1)
    """
    priors = priors or EMPIRICAL_PRIORS
    coeffs = coefficients or DEFAULT_COEFFICIENTS

    raw_weights = {}
    for method in methods:
        prior = priors.get(method, {
            "purity": 0.5, "sensitivity": 0.5, "ari": 0.05,
            "ami": 0.1, "noise_robust": 0.5,
        })
        w = (
            coeffs["ari"] * prior.get("ari", 0.05)
            + coeffs["ami"] * prior.get("ami", 0.1)
            + coeffs["purity"] * prior.get("purity", 0.5)
            + coeffs["sensitivity"] * prior.get("sensitivity", 0.5)
            + coeffs["noise_robust"] * prior.get("noise_robust", 0.5)
        )
        raw_weights[method] = max(w, min_weight)

    # Normalize to sum to 1
    total = sum(raw_weights.values())
    if total > 0:
        raw_weights = {m: w / total for m, w in raw_weights.items()}

    return raw_weights


def get_active_methods(
    methods: list[str],
    priors: dict[str, dict] | None = None,
    min_score: float = 0.10,
) -> list[str]:
    """Filter methods by minimum empirical quality score.

    Methods with very low ARI and noise_robustness are excluded.
    Uses the same scoring as empirical_weights() but as a binary filter.

    Args:
        methods: available method names
        priors: override EMPIRICAL_PRIORS
        min_score: minimum combined score to include method

    Returns:
        filtered list of method names
    """
    priors = priors or EMPIRICAL_PRIORS
    coeffs = DEFAULT_COEFFICIENTS

    active = []
    for method in methods:
        prior = priors.get(method, None)
        if prior is None:
            # Unknown method — include by default (no prior = no reason to exclude)
            active.append(method)
            continue

        score = (
            coeffs["ari"] * prior.get("ari", 0.05)
            + coeffs["ami"] * prior.get("ami", 0.1)
            + coeffs["purity"] * prior.get("purity", 0.5)
            + coeffs["sensitivity"] * prior.get("sensitivity", 0.5)
            + coeffs["noise_robust"] * prior.get("noise_robust", 0.5)
        )
        if score >= min_score:
            active.append(method)
        else:
            logger.info(f"  Excluding {method}: score={score:.3f} < {min_score}")

    return active


# ---------------------------------------------------------------------------
# Legacy: config-based weight computation (kept for backward compat)
# ---------------------------------------------------------------------------

def compute_method_weights(
    methods: list[str],
    scenario: str = "balanced",
    config: dict | None = None,
    method_status: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compute method weights from config priors and scenario.

    Falls back to empirical_weights() if no config priors are set.
    """
    config = config or {}
    method_status = method_status or {}

    consensus_cfg = config.get("consensus", {})
    weights_cfg = consensus_cfg.get("weights", {})
    priors = weights_cfg.get("global_priors", {})

    # If no config priors, use empirical data
    if not priors:
        return empirical_weights(methods)

    # Legacy path: use config priors
    coefficients = weights_cfg.get("coefficients", {})
    coeffs = coefficients.get(scenario, {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    a, b, c, d = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"]

    weights = {}
    for method in methods:
        if method_status.get(method) == "failed":
            weights[method] = 0.0
            continue

        prior = priors.get(method, {
            "purity": 0.5, "sensitivity": 0.5,
            "noise_robust": 0.5, "speed": 0.5,
        })

        w = (
            a * prior.get("purity", 0.5)
            + b * prior.get("sensitivity", 0.5)
            + c * prior.get("noise_robust", 0.5)
            + d * prior.get("speed", 0.5)
        )
        weights[method] = w

    total = sum(weights.values())
    if total > 0:
        scale = len(methods) / total
        weights = {m: w * scale for m, w in weights.items()}

    return weights
