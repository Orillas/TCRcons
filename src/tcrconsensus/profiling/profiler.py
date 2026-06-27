"""Dataset profiling — compute statistics and infer repertoire characteristics."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..schema.records import DatasetProfile, ChainMode, RepertoireType

logger = logging.getLogger(__name__)


def profile(
    df: pd.DataFrame,
    config: dict | None = None,
) -> DatasetProfile:
    """Generate a DatasetProfile from a normalized TCR DataFrame."""
    config = config or {}
    n = len(df)
    if n == 0:
        return DatasetProfile()

    # Chain mode
    chain_mode_str = df["chain_mode"].mode().iloc[0] if "chain_mode" in df.columns else "beta_only"
    chain_mode = ChainMode(chain_mode_str)

    # V/J completeness
    vj_completeness = _compute_vj_completeness(df, chain_mode)

    # CDR3 length summary
    cdr3_col = "cdr3_beta" if chain_mode != ChainMode.ALPHA_ONLY else "cdr3_alpha"
    cdr3_lengths = df[cdr3_col].dropna().str.len()
    cdr3_summary = {
        "mean": float(cdr3_lengths.mean()) if len(cdr3_lengths) > 0 else 0.0,
        "std": float(cdr3_lengths.std()) if len(cdr3_lengths) > 0 else 0.0,
        "min": int(cdr3_lengths.min()) if len(cdr3_lengths) > 0 else 0,
        "max": int(cdr3_lengths.max()) if len(cdr3_lengths) > 0 else 0,
    }

    # Unique ratio
    unique_ratio = _compute_unique_ratio(df, cdr3_col)

    # Clone expansion score
    clone_expansion_score = _compute_clone_expansion(df)

    # Publicity score (shared sequences across subjects)
    publicity_score = _compute_publicity(df)

    # Background noise score
    noise_cfg = config.get("profiling", {}).get("noise_estimation", {})
    background_noise_score = _compute_noise_score(df, cdr3_col, noise_cfg)

    # Label availability
    label_availability = "epitope" in df.columns and df["epitope"].notna().any()

    # Repertoire type inference
    repertoire_type = _infer_repertoire_type(df, n, label_availability)

    notes = []
    if background_noise_score > 0.5:
        notes.append("High background noise detected")
    if vj_completeness < 0.3:
        notes.append("Low V/J gene completeness")

    return DatasetProfile(
        n_tcrs=n,
        chain_mode=chain_mode,
        vj_completeness=vj_completeness,
        cdr3_length_summary=cdr3_summary,
        unique_ratio=unique_ratio,
        clone_expansion_score=clone_expansion_score,
        publicity_score=publicity_score,
        background_noise_score=background_noise_score,
        label_availability=label_availability,
        repertoire_type=repertoire_type,
        notes=notes,
    )


def _compute_vj_completeness(df: pd.DataFrame, chain_mode: ChainMode) -> float:
    """Fraction of records with V and J genes filled."""
    n = len(df)
    if n == 0:
        return 0.0
    v_cols, j_cols = [], []
    if chain_mode in (ChainMode.BETA_ONLY, ChainMode.PAIRED_AB):
        v_cols.append("v_beta")
        j_cols.append("j_beta")
    if chain_mode in (ChainMode.ALPHA_ONLY, ChainMode.PAIRED_AB):
        v_cols.append("v_alpha")
        j_cols.append("j_alpha")
    v_present = df[v_cols].notna().any(axis=1).sum() if v_cols else 0
    j_present = df[j_cols].notna().any(axis=1).sum() if j_cols else 0
    return (v_present + j_present) / (2 * n)


def _compute_unique_ratio(df: pd.DataFrame, cdr3_col: str) -> float:
    """Ratio of unique CDR3 sequences to total."""
    seqs = df[cdr3_col].dropna()
    if len(seqs) == 0:
        return 0.0
    return float(seqs.nunique() / len(seqs))


def _compute_clone_expansion(df: pd.DataFrame) -> float:
    """Clone expansion score based on count distribution skew."""
    if "count" not in df.columns:
        return 0.0
    counts = df["count"].dropna()
    if len(counts) == 0:
        return 0.0
    total = counts.sum()
    if total == 0:
        return 0.0
    max_count = counts.max()
    return float(max_count / total)


def _compute_publicity(df: pd.DataFrame) -> float:
    """Fraction of CDR3 sequences appearing in more than one subject."""
    if "subject_id" not in df.columns or "cdr3_beta" not in df.columns:
        return 0.0
    valid = df.dropna(subset=["subject_id", "cdr3_beta"])
    if len(valid) == 0:
        return 0.0
    per_seq = valid.groupby("cdr3_beta")["subject_id"].nunique()
    public = (per_seq > 1).sum()
    return float(public / len(per_seq))


def _compute_noise_score(
    df: pd.DataFrame,
    cdr3_col: str,
    config: dict,
) -> float:
    """Estimate background noise from proxy features."""
    seqs = df[cdr3_col].dropna()
    if len(seqs) == 0:
        return 0.5

    # Singleton fraction
    counts = seqs.value_counts()
    singleton_frac = (counts == 1).sum() / len(counts)

    # Low frequency fraction
    lowfreq_thresh = config.get("lowfreq_threshold", 0.01)
    total = len(seqs)
    lowfreq_frac = (counts / total < lowfreq_thresh).sum() / len(counts)

    # V/J skewness proxy
    vj_skew = 0.0
    if "v_beta" in df.columns:
        v_counts = df["v_beta"].dropna().value_counts()
        if len(v_counts) > 1:
            vj_skew = float(v_counts.iloc[0] / v_counts.sum())

    # Weighted combination
    w_singleton = config.get("singleton_weight", 0.3)
    w_lowfreq = config.get("lowfreq_weight", 0.2)
    w_vj = config.get("vj_skew_weight", 0.2)
    w_density = config.get("density_weight", 0.15)
    w_motif = config.get("motif_weight", 0.15)

    # Density: avg sequences per unique CDR3
    density = min(1.0, total / max(seqs.nunique(), 1))

    noise = (
        w_singleton * singleton_frac
        + w_lowfreq * lowfreq_frac
        + w_vj * vj_skew
        + w_density * density
        + w_motif * 0.3  # placeholder
    )
    return min(1.0, max(0.0, noise))


def _infer_repertoire_type(
    df: pd.DataFrame, n: int, has_labels: bool
) -> RepertoireType:
    """Infer repertoire type from dataset characteristics."""
    if has_labels and n < 5000:
        return RepertoireType.CURATED_DB
    if has_labels:
        return RepertoireType.ANTIGEN_ENRICHED
    if n > 100000:
        return RepertoireType.BULK
    if "sample_id" in df.columns and df["sample_id"].notna().nunique() > 1:
        return RepertoireType.BULK
    return RepertoireType.BULK
