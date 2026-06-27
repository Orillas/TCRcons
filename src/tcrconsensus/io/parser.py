"""Input parsing and normalization for TCR data."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..schema.records import TCRRecord, ChainMode

logger = logging.getLogger(__name__)

# Column mapping presets
AIRR_COLUMNS = {
    "sequence_id": "tcr_id",
    "cdr3": "cdr3_beta",
    "v_call": "v_beta",
    "j_call": "j_beta",
    "clone_count": "count",
    "clone_frequency": "frequency",
}

VDJDB_COLUMNS = {
    "CDR3": "cdr3_beta",
    "V": "v_beta",
    "J": "j_beta",
    "Epitope": "epitope",
    "MHC": "hla",
    "Species": "subject_id",
}

CUSTOM_COLUMNS = {
    "cdr3": "cdr3_beta",
    "cdr3_beta": "cdr3_beta",
    "cdr3_alpha": "cdr3_alpha",
    "v_gene": "v_beta",
    "v_beta": "v_beta",
    "v_alpha": "v_alpha",
    "j_gene": "j_beta",
    "j_beta": "j_beta",
    "j_alpha": "j_alpha",
    "epitope": "epitope",
    "hla": "hla",
    "subject": "subject_id",
    "count": "count",
    "frequency": "frequency",
}


def detect_format(path: str) -> str:
    """Auto-detect input file format from header columns."""
    df = pd.read_csv(path, sep=None, engine="python", nrows=0)
    cols = set(df.columns.str.lower())

    if "sequence_id" in cols and "cdr3" in cols:
        return "airr"
    if "cdr3" in cols and "epitope" in cols:
        return "vdjdb"
    return "custom"


def load_airr_tsv(path: str) -> pd.DataFrame:
    """Load AIRR TSV format."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return _apply_column_map(df, AIRR_COLUMNS)


def load_vdjdb(path: str) -> pd.DataFrame:
    """Load VDJdb export format."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return _apply_column_map(df, VDJDB_COLUMNS)


def load_custom(
    path: str,
    column_map: Optional[dict[str, str]] = None,
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """Load custom CSV/TSV with optional column mapping."""
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, engine="python")
    cmap = column_map or CUSTOM_COLUMNS
    return _apply_column_map(df, cmap)


def load_file(
    path: str,
    fmt: str = "auto",
    column_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Load any supported format and return normalized DataFrame."""
    if fmt == "auto":
        fmt = detect_format(path)

    loaders = {
        "airr": load_airr_tsv,
        "vdjdb": load_vdjdb,
    }

    if fmt in loaders:
        return loaders[fmt](path)
    return load_custom(path, column_map)


def _apply_column_map(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Rename columns according to mapping and keep only known fields."""
    df = df.rename(columns=col_map)
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize loaded DataFrame into canonical schema.

    - Generate tcr_id if missing
    - Fill defaults for count/frequency
    - Infer chain_mode
    - Remove rows without any CDR3
    - Strip whitespace from string fields
    """
    # Strip whitespace
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Generate tcr_id if missing
    if "tcr_id" not in df.columns or df["tcr_id"].isna().all():
        df["tcr_id"] = [f"tcr_{i:06d}" for i in range(len(df))]
    else:
        df["tcr_id"] = df["tcr_id"].fillna(
            pd.Series([f"tcr_{i:06d}" for i in range(len(df))])
        )

    # Fill defaults
    if "count" not in df.columns:
        df["count"] = 1
    else:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)

    if "frequency" not in df.columns:
        df["frequency"] = None
    else:
        df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")

    # Infer chain mode
    has_alpha = "cdr3_alpha" in df.columns and df["cdr3_alpha"].notna().any()
    has_beta = "cdr3_beta" in df.columns and df["cdr3_beta"].notna().any()
    if has_alpha and has_beta:
        df["chain_mode"] = "paired_ab"
    elif has_alpha:
        df["chain_mode"] = "alpha_only"
    else:
        df["chain_mode"] = "beta_only"

    # Drop rows without any CDR3
    cdr3_cols = [c for c in ["cdr3_alpha", "cdr3_beta"] if c in df.columns]
    if cdr3_cols:
        mask = df[cdr3_cols].notna().any(axis=1)
        dropped = (~mask).sum()
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows without CDR3 sequence")
        df = df[mask].copy()

    # Ensure canonical columns exist
    canonical = [
        "tcr_id", "chain_mode", "cdr3_alpha", "cdr3_beta",
        "v_alpha", "j_alpha", "v_beta", "j_beta",
        "subject_id", "sample_id", "epitope", "hla",
        "count", "frequency", "source_dataset",
    ]
    for col in canonical:
        if col not in df.columns:
            df[col] = None

    df = df.reset_index(drop=True)
    return df[canonical]


def to_records(df: pd.DataFrame) -> list[TCRRecord]:
    """Convert normalized DataFrame to list of TCRRecord."""
    records = []
    for _, row in df.iterrows():
        records.append(
            TCRRecord(
                tcr_id=str(row.get("tcr_id", "")),
                chain_mode=ChainMode(row.get("chain_mode", "beta_only")),
                cdr3_alpha=_str(row.get("cdr3_alpha")),
                cdr3_beta=_str(row.get("cdr3_beta")),
                v_alpha=_str(row.get("v_alpha")),
                j_alpha=_str(row.get("j_alpha")),
                v_beta=_str(row.get("v_beta")),
                j_beta=_str(row.get("j_beta")),
                subject_id=_str(row.get("subject_id")),
                sample_id=_str(row.get("sample_id")),
                epitope=_str(row.get("epitope")),
                hla=_str(row.get("hla")),
                count=int(row.get("count", 1) or 1),
                frequency=_float(row.get("frequency")),
                source_dataset=_str(row.get("source_dataset")),
            )
        )
    return records


def _str(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return str(val) if str(val) != "None" else None


def _float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
