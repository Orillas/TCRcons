"""Tier-1/F2: multi-signal evidence fusion for consensus edges.

The :class:`ConsensusEdge` schema declared ``sequence_support`` /
``vj_support`` / ``noise_penalty`` fields that were **always 0.0** — they had
never been wired in. This module fills them and fuses them with the method-vote
(net of Tier-2/4b repulsion) into a single calibrated ``final_score``:

    final_score = σ(β0 + β_vote·net_vote + β_seq·seq + β_vj·vj + β_noise·noise_sig)

where ``net_vote = weighted_support - repulsion_support`` (so F2 composes with
the Tier-2 signed graph: repulsion is folded into the vote term, not
double-counted as a separate signal).

Signals
-------
- ``sequence_support`` ∈ [0,1]: exp(-TCRdist/τ) from the tcrdist3 ``pw_beta``
  matrix (re-used, not recomputed); falls back to a logistic map of the mean
  BLOSUM62 substitution score when tcrdist3 did not run.
- ``vj_support`` ∈ [0,1]: 0.5·1[shared V] + 0.5·1[shared J] (co-restriction).
- ``noise_penalty`` = -log10(p_null): significance of the co-clustering against
  the permutation null (shared with :mod:`null_model`). Normalised to [0,1]
  before fusion.

Default β are principled; ``learn_fusion_beta`` is a leave-one-dataset-out
logistic-regression interface (reviewer.md §5 leak control).
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

import numpy as np

from ..schema.records import ConsensusEdge

logger = logging.getLogger(__name__)

DEFAULT_BETAS = {"vote": 1.0, "seq": 0.8, "vj": 0.4, "noise": 0.6}
DEFAULT_INTERCEPT = 0.0
TAU_SEQ = 50.0          # tcrdist distance scale (typical intra-specificity ~10-60)
NOISE_CAP = 3.0         # -log10(p_null) at which noise_sig saturates to 1.0 (p_null≈1e-3)


# ---------------------------------------------------------------------------
# BLOSUM62 (NCBI canonical block) — fallback sequence similarity, no deps
# ---------------------------------------------------------------------------

_BLOSUM62_TEXT = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
R -1  5 -2 -3 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -2  0 -1 -4
N -2 -2  6  1 -3  0 -1  0  0 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
D -2 -3  1  6 -3 -1  3 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
Q -1  1  0 -1 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
E -1  0 -1  3 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
H -2  0  0 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  9 -1 -1 -4 -3 -3 -2 -1 -2 -4
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -1  0  0 -1 -4
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0 -1 -1 -4
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -3 -1  0 -3 -1  4 -3 -2 -1 -4
B -2 -2  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -3 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2 -1 -1 -2 -1 -1 -1 -1 -1 -4
* -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""


def _build_blosum62() -> dict[tuple[str, str], int]:
    mat: dict[tuple[str, str], int] = {}
    header: list[str] | None = None
    for line in _BLOSUM62_TEXT.strip().splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if header is None:
            header = parts
            continue
        row_aa = parts[0]
        vals = parts[1:]
        for col_aa, v in zip(header, vals):
            mat[(row_aa, col_aa)] = int(v)
    return mat


_BLOSUM62 = _build_blosum62()


def blosum_score(s1: str, s2: str) -> float:
    """Mean BLOSUM62 substitution score over the aligned prefix (min length).

    Position-wise; indels ignored (CDR3 lengths usually match within a method's
    cluster). Range roughly [-4, 11] per position.
    """
    L = min(len(s1), len(s2))
    if L == 0:
        return 0.0
    total = 0
    for i in range(L):
        a, b = s1[i], s2[i]
        total += _BLOSUM62.get((a, b), _BLOSUM62.get((b, a), -4))
    return total / L


# ---------------------------------------------------------------------------
# Signal primitives
# ---------------------------------------------------------------------------

def sequence_similarity(
    cdr3_a: str, cdr3_b: str, pw_dist: Optional[float], tau_seq: float = TAU_SEQ
) -> float:
    """sequence_support ∈ [0,1].

    If a TCRdist distance is available (lower = structurally closer):
    ``exp(-pw_dist / tau_seq)``. Otherwise fall back to a logistic map of the
    mean BLOSUM62 score (centred at ~2, scale 3).
    """
    if pw_dist is not None and pw_dist == pw_dist and pw_dist >= 0:  # not NaN
        return float(math.exp(-pw_dist / tau_seq))
    bs = blosum_score(cdr3_a or "", cdr3_b or "")
    return 1.0 / (1.0 + math.exp(-(bs - 2.0) / 3.0))


def vj_shared(v_a, v_b, j_a, j_b) -> float:
    """vj_support ∈ [0,1]: 0.5 per shared V / shared J gene."""
    s = 0.0
    if v_a and v_b and v_a == v_b:
        s += 0.5
    if j_a and j_b and j_a == j_b:
        s += 0.5
    return s


def noise_significance(p_null: float) -> float:
    """noise_sig ∈ [0,1]: normalised -log10(p_null), capped at NOISE_CAP.

    High when the pair rarely co-clusters under the permutation null
    (i.e. the co-clustering is unlikely to be chance).
    """
    if p_null <= 0:
        return 1.0
    return min(-math.log10(p_null) / NOISE_CAP, 1.0)


def _edge_key(tcr_id_a: str, tcr_id_b: str) -> tuple[str, str]:
    return (tcr_id_a, tcr_id_b) if tcr_id_a <= tcr_id_b else (tcr_id_b, tcr_id_a)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fused_score(
    net_vote: float,
    seq: float,
    vj: float,
    noise_sig: float,
    *,
    betas: dict | None = None,
    intercept: float = DEFAULT_INTERCEPT,
    use_signals: Iterable[str] = ("vote", "seq", "vj", "noise"),
) -> float:
    """Single-edge sigmoid fusion. Pure function (used by both observed and
    null edges, so the null lives on the same scale)."""
    betas = betas or DEFAULT_BETAS
    use = set(use_signals)
    z = intercept
    if "vote" in use:
        z += betas["vote"] * net_vote
    if "seq" in use:
        z += betas["seq"] * seq
    if "vj" in use:
        z += betas["vj"] * vj
    if "noise" in use:
        z += betas["noise"] * noise_sig
    if z >= 50:
        return 1.0
    if z <= -50:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-z)))


def enrich_and_fuse(
    edges: list[ConsensusEdge],
    *,
    lookups: dict,
    pw_beta: Optional[np.ndarray] = None,
    tcr_to_idx: Optional[dict] = None,
    pair_null: Optional[dict] = None,
    B: int = 0,
    betas: Optional[dict] = None,
    intercept: float = DEFAULT_INTERCEPT,
    use_signals: Iterable[str] = ("vote", "seq", "vj", "noise"),
    tau_seq: float = TAU_SEQ,
) -> list[ConsensusEdge]:
    """Fill sequence/vj/noise fields and recompute ``final_score`` in place.

    Args:
        edges: from :func:`extract_pairwise_support` (vote + repulsion already
            accumulated). ``net_vote = weighted_support - repulsion_support``.
        lookups: ``tcr_id -> {cdr3_beta, v_beta, j_beta, ...}``.
        pw_beta: optional n_clonotype × n_clonotype TCRdist distance matrix.
        tcr_to_idx: ``tcr_id -> pw_beta row index`` (clonotype index).
        pair_null: optional ``(a,b) -> null co-cluster count`` (from estimate_null).
        B: permutation count (for p_null normalisation).
        betas/intercept/use_signals: fusion control (ablation sweeps these).
    """
    betas = betas or DEFAULT_BETAS
    for e in edges:
        la = lookups.get(e.tcr_id_a, {})
        lb = lookups.get(e.tcr_id_b, {})

        pw = None
        if pw_beta is not None and tcr_to_idx is not None:
            ia = tcr_to_idx.get(e.tcr_id_a)
            ib = tcr_to_idx.get(e.tcr_id_b)
            if ia is not None and ib is not None:
                pw = float(pw_beta[ia, ib])

        e.sequence_support = sequence_similarity(
            la.get("cdr3_beta", ""), lb.get("cdr3_beta", ""), pw, tau_seq
        )
        e.vj_support = vj_shared(
            la.get("v_beta", ""), lb.get("v_beta", ""),
            la.get("j_beta", ""), lb.get("j_beta", ""),
        )

        if pair_null is not None and B > 0:
            cnt = pair_null.get(_edge_key(e.tcr_id_a, e.tcr_id_b), 0)
            p_null = (cnt + 1) / (B + 1)
            e.noise_penalty = float(-math.log10(p_null)) if p_null > 0 else float(NOISE_CAP)
            noise_sig = noise_significance(p_null)
        else:
            e.noise_penalty = 0.0
            noise_sig = 0.0

        e.final_score = fused_score(
            e.weighted_support - e.repulsion_support,
            e.sequence_support, e.vj_support, noise_sig,
            betas=betas, intercept=intercept, use_signals=use_signals,
        )
    return edges


def make_null_fuse_fn(
    lookups: dict,
    pw_beta: Optional[np.ndarray] = None,
    tcr_to_idx: Optional[dict] = None,
    betas: Optional[dict] = None,
    intercept: float = DEFAULT_INTERCEPT,
    use_signals: Iterable[str] = ("vote", "seq", "vj"),
    tau_seq: float = TAU_SEQ,
):
    """Return a ``fuse_fn(edges) -> list[float]`` for :func:`null_model.estimate_null`.

    Maps a permutation's unsigned edges to fused scores using the SAME β. Note:
    the null deliberately EXCLUDES the noise term (noise is itself derived from
    the permutation null — including it would be circular). So null edges are
    scored on vote + seq + vj only.
    """
    betas = betas or DEFAULT_BETAS
    # seq/vj are TCR-pair-fixed (permutation-invariant), so memoise per pair —
    # across B permutations the same pairs recur, and the BLOSUM62 fallback for
    # tcrdist3-uncovered TCRs is O(L) per call. Without this cache the null loop
    # is O(B · null_edges · L); with it, the heavy work happens once per pair.
    seqvj_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def _seqvj(a: str, b: str) -> tuple[float, float]:
        key = (a, b) if a <= b else (b, a)
        cached = seqvj_cache.get(key)
        if cached is not None:
            return cached
        la = lookups.get(a, {})
        lb = lookups.get(b, {})
        pw = None
        if pw_beta is not None and tcr_to_idx is not None:
            ia = tcr_to_idx.get(a)
            ib = tcr_to_idx.get(b)
            if ia is not None and ib is not None:
                pw = float(pw_beta[ia, ib])
        seq = sequence_similarity(
            la.get("cdr3_beta", ""), lb.get("cdr3_beta", ""), pw, tau_seq
        )
        vj = vj_shared(
            la.get("v_beta", ""), lb.get("v_beta", ""),
            la.get("j_beta", ""), lb.get("j_beta", ""),
        )
        seqvj_cache[key] = (seq, vj)
        return seq, vj

    def fuse_fn(edges):
        out = []
        for e in edges:
            seq, vj = _seqvj(e.tcr_id_a, e.tcr_id_b)
            out.append(
                fused_score(
                    e.weighted_support, seq, vj, 0.0,
                    betas=betas, intercept=intercept, use_signals=use_signals,
                )
            )
        return out

    return fuse_fn


def learn_fusion_beta(labeled_pairs: list[dict]) -> dict:
    """Fit β on a labelled pair set via logistic regression (interface only).

    ``labeled_pairs`` items: ``{"vote","seq","vj","noise","label(0/1)}``.
    Strict leave-one-dataset-out leak control (reviewer.md §5). Returns a β
    dict on the DEFAULT_BETAS schema. Callers must hold out the dataset used
    for final evaluation. Not invoked by default paths.
    """
    from sklearn.linear_model import LogisticRegression

    if not labeled_pairs:
        return dict(DEFAULT_BETAS)
    X = np.array(
        [[p["vote"], p["seq"], p["vj"], p.get("noise", 0.0)] for p in labeled_pairs],
        dtype=float,
    )
    y = np.array([p["label"] for p in labeled_pairs], dtype=int)
    if len(np.unique(y)) < 2:
        return dict(DEFAULT_BETAS)
    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    lr.fit(X, y)
    return {
        "vote": float(lr.coef_[0][0]),
        "seq": float(lr.coef_[0][1]),
        "vj": float(lr.coef_[0][2]),
        "noise": float(lr.coef_[0][3]),
    }
"""
Innovation #3: Learnable fusion β + noise auto-gating.

Two additions to consensus/fusion.py:

1. learn_fusion_beta(): logistic regression on labeled pairs to fit β
   coefficients (fills the existing interface stub with a full implementation).

2. noise_auto_gate(): automatically determines whether to include the noise
   signal in fusion, based on the dataset's background_noise_score from the
   profiler. When noise_score < threshold (clean data), noise is excluded;
   when noise_score >= threshold (noisy data), noise is included.

   This is data-driven: the Dash experiment showed noise HELPS on noisy data
   (+52%) but HURTS on clean data (-8.6× false_merge). The auto-gate turns
   this empirical finding into an automated control.

3. build_labeled_pairs(): construct training pairs from labeled TCR data
   with strict leave-one-dataset-out leak control (reviewer.md §5).
"""


import logging
from typing import Optional

import numpy as np

from ..schema.records import ConsensusEdge

logger = logging.getLogger(__name__)

# Default noise gate threshold from empirical findings:
# v3_cd8 noise_score ≈ 0.2 (clean) → gate OFF
# Dash noise_score ≈ 0.6 (noisy) → gate ON
DEFAULT_NOISE_GATE_THRESHOLD = 0.4


# ── Noise auto-gating ──────────────────────────────────────────────────────

def noise_auto_gate(
    background_noise_score: float,
    threshold: float = DEFAULT_NOISE_GATE_THRESHOLD,
) -> bool:
    """Determine whether to include noise signal in fusion.

    Based on empirical finding: noise_penalty harms clean data (v3_cd8:
    false_merge 0.011→0.095, 8.6x) but helps noisy data (Dash: +52%).

    Args:
        background_noise_score: from profiler (0=clean, 1=very noisy).
        threshold: scores >= threshold enable noise.

    Returns:
        True if noise signal should be included in fusion.
    """
    enabled = background_noise_score >= threshold
    if enabled:
        logger.info(
            f"Noise signal ENABLED (background_noise={background_noise_score:.3f} "
            f">= {threshold})"
        )
    else:
        logger.info(
            f"Noise signal DISABLED (background_noise={background_noise_score:.3f} "
            f"< {threshold}) — clean data, noise would inflate false_merge"
        )
    return enabled


def resolve_use_signals(
    use_signals_base: tuple[str, ...] = ("vote", "seq", "vj", "noise"),
    background_noise_score: Optional[float] = None,
    noise_gate_threshold: float = DEFAULT_NOISE_GATE_THRESHOLD,
) -> tuple[str, ...]:
    """Resolve which signals to use, optionally gating noise.

    If background_noise_score is None, noise is included (backward compatible).
    """
    if background_noise_score is None:
        return use_signals_base
    if not noise_auto_gate(background_noise_score, noise_gate_threshold):
        return tuple(s for s in use_signals_base if s != "noise")
    return use_signals_base


# ── Labeled pair construction ──────────────────────────────────────────────

def build_labeled_pairs(
    edges: list[ConsensusEdge],
    assignments: list[ClusterAssignment],
    true_labels: dict[str, str],      # tcr_id -> epitope
    lookups: dict | None = None,
    pw_beta: np.ndarray | None = None,
    tcr_to_idx: dict | None = None,
    *,
    max_pairs: int = 50000,
    balance: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Build labeled training pairs for logistic regression.

    Each pair has features [vote, seq, vj, noise] and label 1 if same epitope.

    Strict leak control: labeled pairs should come from a DIFFERENT dataset
    than the one being evaluated (reviewer.md §5). Caller is responsible for
    holding out the evaluation dataset.

    Args:
        edges: consensus edges with filled signal fields.
        assignments: method cluster assignments.
        true_labels: tcr_id → epitope mapping.
        lookups/pw_beta/tcr_to_idx: for computing seq/vj/noise if not in edges.
        max_pairs: cap to avoid O(n²) blowup.
        balance: if True, sample equal number of positive and negative pairs.

    Returns:
        list of {"vote", "seq", "vj", "noise", "label"} dicts.
    """
    rng = np.random.RandomState(seed)

    positives: list[dict] = []
    negatives: list[dict] = []

    for e in edges:
        net_vote = e.weighted_support - e.repulsion_support
        label_a = true_labels.get(e.tcr_id_a)
        label_b = true_labels.get(e.tcr_id_b)
        if label_a is None or label_b is None:
            continue

        is_positive = (label_a == label_b)
        pair_dict = {
            "vote": net_vote,
            "seq": e.sequence_support,
            "vj": e.vj_support,
            "noise": e.noise_penalty,
            "label": 1 if is_positive else 0,
        }
        if is_positive:
            positives.append(pair_dict)
        else:
            negatives.append(pair_dict)

    n_pos = len(positives)
    n_neg = len(negatives)

    if balance and n_pos > 0 and n_neg > 0:
        # Sample to balance
        target = min(n_pos, n_neg, max_pairs // 2)
        if n_pos > target:
            pos_idx = rng.choice(n_pos, target, replace=False)
            positives = [positives[i] for i in pos_idx]
        if n_neg > target:
            neg_idx = rng.choice(n_neg, target, replace=False)
            negatives = [negatives[i] for i in neg_idx]
        pairs = positives + negatives
        rng.shuffle(pairs)
    else:
        pairs = positives + negatives
        if len(pairs) > max_pairs:
            idx = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in idx]

    logger.info(
        f"Labeled pairs: {sum(1 for p in pairs if p['label'])} positive, "
        f"{sum(1 for p in pairs if p['label']==0)} negative "
        f"(from {n_pos}+{n_neg} raw)"
    )
    return pairs


# ── Learnable fusion β ─────────────────────────────────────────────────────

def learn_fusion_beta(
    labeled_pairs: list[dict],
    *,
    signal_names: tuple[str, ...] = ("vote", "seq", "vj", "noise"),
    C: float = 1.0,
    max_iter: int = 500,
) -> dict[str, float]:
    """Learn fusion β coefficients via L2-regularized logistic regression.

    Maps the hand-tuned DEFAULT_BETAS to data-driven coefficients using a
    held-out labeled dataset (strict leave-one-dataset-out for leak control).

    Args:
        labeled_pairs: from build_labeled_pairs(), each dict has keys matching
            signal_names + "label" (0 or 1).
        signal_names: which signals to include as features.
        C: inverse regularization strength (smaller = stronger regularization).
        max_iter: solver iterations.

    Returns:
        β dict mapping signal name → learned coefficient.
        Unused signals get 0.0. If training fails (e.g., only one class),
        returns DEFAULT_BETAS.
    """
    from ..consensus.fusion import DEFAULT_BETAS

    if not labeled_pairs:
        logger.warning("No labeled pairs for β learning; using defaults")
        return dict(DEFAULT_BETAS)

    # Build feature matrix
    active_signals = [s for s in signal_names if s in labeled_pairs[0]]
    X = np.array(
        [[p.get(s, 0.0) for s in active_signals] for p in labeled_pairs],
        dtype=float,
    )
    y = np.array([p["label"] for p in labeled_pairs], dtype=int)

    if len(np.unique(y)) < 2:
        logger.warning("Only one class in labeled pairs; using default β")
        return dict(DEFAULT_BETAS)

    if len(X) < 10:
        logger.warning(f"Only {len(X)} pairs; using default β (need >10)")
        return dict(DEFAULT_BETAS)

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        logger.warning("sklearn not available; using default β")
        return dict(DEFAULT_BETAS)

    # Standardize features for stable learning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std

    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(X_scaled, y)

    # Extract coefficients (on standardized scale, convert back)
    coefs = lr.coef_[0] / X_std

    learned: dict[str, float] = {}
    for i, s in enumerate(active_signals):
        learned[s] = float(coefs[i])

    # Fill in missing signals with 0.0
    for s in signal_names:
        if s not in learned:
            learned[s] = 0.0

    logger.info(
        f"Learned fusion β: { {k: f'{v:.3f}' for k, v in learned.items()} } "
        f"(accuracy={lr.score(X_scaled, y):.3f}, "
        f"intercept={float(lr.intercept_[0]):.3f})"
    )
    return learned
