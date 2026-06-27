"""Tier-2 innovation 6: CDR3 motif analysis for refinement.

Position-weight matrices (PWM), information content (IC), and pairwise PWM
KL divergence — used by the refiner to (a) split clusters with bimodal motifs
and (b) revive the dead merge step with a biologically grounded criterion.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
N_AA = len(AA)

# Rough uniform background; refine per-dataset if desired.
BG = np.full(N_AA, 1.0 / N_AA)


def _pad_to_max(seqs: list[str]) -> list[str]:
    if not seqs:
        return seqs
    m = max(len(s) for s in seqs)
    return [s + "-" * (m - len(s)) for s in seqs]


def build_pwm(seqs: list[str], pseudocount: float = 0.5) -> np.ndarray:
    """PWM of shape (L, N_AA). Gap/pad positions distribute uniformly."""
    seqs = _pad_to_max([s for s in seqs if s])
    if not seqs:
        return np.zeros((1, N_AA))
    L = len(seqs[0])
    pwm = np.full((L, N_AA), pseudocount, dtype=np.float64)
    for s in seqs:
        for pos, aa in enumerate(s):
            if aa in AA_INDEX:
                pwm[pos, AA_INDEX[aa]] += 1.0
            else:
                # gap or non-standard: spread mass uniformly to avoid bias
                pwm[pos, :] += 1.0 / N_AA
    pwm /= pwm.sum(axis=1, keepdims=True)
    return pwm


def information_content(pwm: np.ndarray, bg: np.ndarray | None = None) -> float:
    """Sum over positions of KL(pwm_pos || bg). High = strong, specific motif."""
    bg = bg if bg is not None else BG
    pwm = np.clip(pwm, 1e-9, 1.0)
    kl = pwm * np.log2(pwm / bg[None, :])
    return float(kl.sum())


def pwm_kl(pwm1: np.ndarray, pwm2: np.ndarray) -> float:
    """Symmetric KL between two PWMs (aligned by min length). High = different motifs."""
    L = min(pwm1.shape[0], pwm2.shape[0])
    if L == 0:
        return 0.0
    p = np.clip(pwm1[:L], 1e-9, 1.0)
    q = np.clip(pwm2[:L], 1e-9, 1.0)
    kl_pq = np.sum(p * np.log2(p / q))
    kl_qp = np.sum(q * np.log2(q / p))
    return float(0.5 * (kl_pq + kl_qp))


def cluster_motif_ic(seqs: list[str]) -> float:
    """IC of the cluster's PWM — measures motif strength/coherence."""
    if len(seqs) < 2:
        return 0.0
    return information_content(build_pwm(seqs))


def onehot_embed(seqs: list[str]) -> np.ndarray:
    """One-hot embedding of CDR3s (padded) for k-means split. Shape (n, L*N_AA)."""
    seqs = _pad_to_max([s for s in seqs if s])
    if not seqs:
        return np.zeros((0, N_AA))
    L = len(seqs[0])
    mat = np.zeros((len(seqs), L * N_AA), dtype=np.float64)
    for r, s in enumerate(seqs):
        for pos, aa in enumerate(s):
            if aa in AA_INDEX:
                mat[r, pos * N_AA + AA_INDEX[aa]] = 1.0
    return mat


def motif_subgroups(
    seqs: list[str], k: int = 2, random_state: int = 42,
) -> list[list[int]] | None:
    """Split a list of sequences into k motif subgroups via k-means on the
    one-hot embedding. Returns list of index-lists, or None if not splittable."""
    if len(seqs) < 4:
        return None
    X = onehot_embed(seqs)
    if X.shape[0] < k:
        return None
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        labels = km.fit_predict(X)
    except Exception as e:
        logger.debug(f"motif_subgroups kmeans failed: {e}")
        return None
    groups: list[list[int]] = [[] for _ in range(k)]
    for i, lab in enumerate(labels):
        if 0 <= lab < k:
            groups[lab].append(i)
    # drop empty groups
    return [g for g in groups if g]
