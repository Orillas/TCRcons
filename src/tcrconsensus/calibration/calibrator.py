"""Isotonic calibration of cluster confidence -> P(pure)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..schema.records import ConsensusCluster, ConsensusEdge

logger = logging.getLogger(__name__)

# Background amino-acid frequencies (approximate, for IC). Used by motif module
# primarily; kept here for shared reference.
BG_AA_FREQS = {
    "A": 0.074, "C": 0.025, "D": 0.054, "E": 0.054, "F": 0.047,
    "G": 0.074, "H": 0.026, "I": 0.068, "K": 0.058, "L": 0.099,
    "M": 0.025, "N": 0.045, "P": 0.039, "Q": 0.034, "R": 0.052,
    "S": 0.057, "T": 0.051, "V": 0.073, "W": 0.013, "Y": 0.034,
}


def _edge_map(edges: list[ConsensusEdge]) -> dict[tuple[str, str], ConsensusEdge]:
    m: dict[tuple[str, str], ConsensusEdge] = {}
    for e in edges:
        m[tuple(sorted([e.tcr_id_a, e.tcr_id_b]))] = e
    return m


def _member_confidence(
    member: str, members: list[str], emap: dict[tuple[str, str], ConsensusEdge],
) -> float:
    scores = []
    for other in members:
        if other == member:
            continue
        e = emap.get(tuple(sorted([member, other])))
        if e is not None:
            scores.append(e.final_score)
    return float(np.mean(scores)) if scores else 0.0


def raw_confidence(
    cluster: ConsensusCluster,
    edges: list[ConsensusEdge],
) -> float:
    """Mean intra-cluster edge final_score, with a small multi-method-agreement
    bonus. This is the UNCALIBRATED confidence the Calibrator maps to P(pure)."""
    emap = _edge_map(edges)
    members = cluster.member_ids
    if len(members) < 2:
        return 0.0
    edge_scores: list[float] = []
    method_counts: list[int] = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            e = emap.get(tuple(sorted([members[i], members[j]])))
            if e is not None:
                edge_scores.append(e.final_score)
                method_counts.append(e.method_support_count)
    if not edge_scores:
        return 0.0
    mean_score = float(np.mean(edge_scores))
    # entropy bonus: reward broad method agreement (high mean method_support_count)
    mean_support = float(np.mean(method_counts))
    bonus = 0.05 * (mean_support - 1.0)  # ~0 for single-method, positive for multi
    return float(np.clip(mean_score + bonus, 0.0, 1.0))


def extract_cluster_features(
    clusters: list[ConsensusCluster],
    edges: list[ConsensusEdge],
) -> list[dict]:
    """Per-cluster feature dict (for analysis / richer calibrators)."""
    emap = _edge_map(edges)
    rows = []
    for c in clusters:
        members = c.member_ids
        es = []
        mc = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                e = emap.get(tuple(sorted([members[i], members[j]])))
                if e is not None:
                    es.append(e.final_score)
                    mc.append(e.method_support_count)
        rows.append({
            "cluster_id": c.cluster_id,
            "size": len(members),
            "mean_edge": float(np.mean(es)) if es else 0.0,
            "min_edge": float(np.min(es)) if es else 0.0,
            "median_edge": float(np.median(es)) if es else 0.0,
            "mean_method_support": float(np.mean(mc)) if mc else 0.0,
            "n_edges": len(es),
        })
    return rows


class Calibrator:
    """Isotonic-regression map raw_confidence -> calibrated P(pure).

    Isotonic regression guarantees a monotone mapping, which gives the coverage
    property: on the fit set, {clusters: calib >= t} have empirical purity >= t.
    """

    def __init__(self):
        self._iso = None
        self._xmin = 0.0
        self._xmax = 1.0
        self._default = 0.5

    def fit(self, raw_confidences: np.ndarray, purities: np.ndarray) -> "Calibrator":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:  # pragma: no cover
            logger.warning("sklearn not available; Calibrator is identity")
            self._iso = None
            return self
        raw = np.asarray(raw_confidences, dtype=float)
        pur = np.asarray(purities, dtype=float)
        if len(raw) < 5:
            self._iso = None
            return self
        order = np.argsort(raw)
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(raw[order], pur[order])
        self._xmin = float(raw.min())
        self._xmax = float(raw.max())
        self._default = float(np.mean(pur))
        return self

    def predict(self, raw_confidence: float) -> float:
        if self._iso is None:
            return float(np.clip(raw_confidence, 0.0, 1.0))
        return float(self._iso.predict([float(raw_confidence)])[0])

    def predict_array(self, raw: np.ndarray) -> np.ndarray:
        if self._iso is None:
            return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        return np.asarray(self._iso.predict(np.asarray(raw, dtype=float)), dtype=float)


def fit_calibration(
    clusters: list[ConsensusCluster],
    edges: list[ConsensusEdge],
    purities: list[float],
) -> Calibrator:
    """Convenience: build Calibrator from clusters + their true purities."""
    raw = np.array([raw_confidence(c, edges) for c in clusters], dtype=float)
    pur = np.asarray(purities, dtype=float)
    cal = Calibrator().fit(raw, pur)
    return cal


def coverage_curve(
    calibrated: np.ndarray,
    purities: np.ndarray,
    n_thresholds: int = 50,
) -> dict:
    """For each confidence threshold t (sweep), compute:
       - retained: fraction of clusters with calibrated >= t
       - mean_purity: mean purity among retained
       - n_retained: count
    A well-calibrated system has mean_purity(t) ~= t."""
    calibrated = np.asarray(calibrated, dtype=float)
    purities = np.asarray(purities, dtype=float)
    n = len(calibrated)
    ts = np.linspace(0.0, 1.0, n_thresholds)
    retained, purity, counts = [], [], []
    for t in ts:
        mask = calibrated >= t
        counts.append(int(mask.sum()))
        retained.append(float(mask.sum()) / n if n else 0.0)
        purity.append(float(purities[mask].mean()) if mask.any() else 0.0)
    return {
        "thresholds": ts.tolist(),
        "retained_fraction": retained,
        "mean_purity": purity,
        "n_retained": counts,
    }


def expected_calibration_error(
    calibrated: np.ndarray,
    purities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE = sum_b (|B_b|/N) * |acc(b) - conf(b)| where conf is the mean
    calibrated confidence in bin b and acc is the mean purity in bin b."""
    calibrated = np.asarray(calibrated, dtype=float)
    purities = np.asarray(purities, dtype=float)
    n = len(calibrated)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (calibrated >= lo) & (calibrated < (hi if b < n_bins - 1 else hi + 1e-9))
        if mask.any():
            conf = float(calibrated[mask].mean())
            acc = float(purities[mask].mean())
            ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
