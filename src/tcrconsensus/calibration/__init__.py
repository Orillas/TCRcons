"""Tier-2 innovation 5: calibrated probabilistic cluster confidence.

Turns the raw `mean(edge.final_score)` confidence (which has no probability
semantics) into a calibrated P(cluster is pure) via isotonic regression, with
a coverage guarantee: clusters with calibrated confidence >= t have empirical
purity >= t (on held-out data).
"""

from .calibrator import (
    Calibrator,
    raw_confidence,
    extract_cluster_features,
    fit_calibration,
    coverage_curve,
    expected_calibration_error,
)

__all__ = [
    "Calibrator",
    "raw_confidence",
    "extract_cluster_features",
    "fit_calibration",
    "coverage_curve",
    "expected_calibration_error",
]
