"""Data preprocessing pipeline for TCR data.

Provides a chainable Preprocessor that handles format detection,
column mapping, CDR3 cleaning, chain filtering, score filtering,
deduplication, and metadata extraction.

Usage:
    from tcrconsensus.io.preprocess import Preprocessor

    df = (Preprocessor()
        .load("input.tsv")
        .map_columns()
        .filter_chain(gene="TRB")
        .filter_score(min_score=1)
        .clean_cdr3()
        .dedup()
        .extract_metadata()
        .normalize()
        .result())
    print(Preprocessor.report())
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical column names (internal standard)
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "tcr_id", "chain_mode",
    "cdr3_alpha", "cdr3_beta",
    "v_alpha", "j_alpha", "v_beta", "j_beta",
    "subject_id", "sample_id", "epitope", "hla",
    "count", "frequency", "source_dataset",
]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# Format-specific column mappings
# ---------------------------------------------------------------------------
FORMAT_MAPPINGS = {
    "vdjdb": {
        "CDR3": "cdr3_beta",
        "V": "v_beta",
        "J": "j_beta",
        "Gene": "_gene",
        "Score": "_score",
        "Epitope": "epitope",
        "Epitope gene": "epitope_gene",
        "Epitope species": "epitope_species",
        "MHC A": "hla",
        "MHC B": "mhc_b",
        "MHC class": "mhc_class",
        "Species": "species",
        "Reference": "reference",
        "Method": "ident_method",
        "Meta": "_meta",
        "CDR3fix": "_cdr3fix",
    },
    "mcpas": {
        "CDR3.beta.aa": "cdr3_beta",
        "CDR3.alpha.aa": "cdr3_alpha",
        "TRBV": "v_beta",
        "TRBJ": "j_beta",
        "TRAV": "v_alpha",
        "TRAJ": "j_alpha",
        "TRBD": "d_beta",
        "Species": "species",
        "Category": "category",
        "Pathology": "pathology",
        "Epitope.peptide": "epitope",
        "Epitope.ID": "epitope_id",
        "MHC": "hla",
        "Tissue": "tissue",
        "T.Cell.Type": "cell_type",
        "Antigen.protein": "antigen_protein",
        "Protein.ID": "protein_id",
        "PubMed.ID": "subject_id",
        "Antigen.identification.method": "ident_method",
    },
    "airr": {
        "sequence_id": "tcr_id",
        "cdr3": "cdr3_beta",
        "cdr3_aa": "cdr3_beta",
        "v_call": "v_beta",
        "j_call": "j_beta",
        "d_call": "d_beta",
        "clone_count": "count",
        "clone_frequency": "frequency",
        "locus": "_locus",
        "productive": "_productive",
        "sequence_alignment": "_seq_alignment",
    },
    "10x": {
        "barcode": "tcr_id",
        "cdr3": "cdr3_beta",
        "cdr3s_aa": "cdr3_beta",
        "v_gene": "v_beta",
        "j_gene": "j_beta",
        "d_gene": "d_beta",
        "chain": "_chain",
        "clono_id": "_clono_id",
        "raw_clono_id": "_raw_clono_id",
        "freq": "frequency",
    },
}

# Fuzzy column name mapping for custom / unknown formats
FUZZY_MAPPINGS = {
    # CDR3 beta
    "cdr3_beta": "cdr3_beta", "cdr3": "cdr3_beta", "cdr3b": "cdr3_beta",
    "junction_aa": "cdr3_beta", "junctionaa": "cdr3_beta",
    "amino_acid": "cdr3_beta",
    # CDR3 alpha
    "cdr3_alpha": "cdr3_alpha", "cdr3a": "cdr3_alpha",
    # V/J genes
    "v_beta": "v_beta", "v_gene": "v_beta", "v_call": "v_beta",
    "vseg": "v_beta", "trbv": "v_beta",
    "j_beta": "j_beta", "j_gene": "j_beta", "j_call": "j_beta",
    "jseg": "j_beta", "trbj": "j_beta",
    "v_alpha": "v_alpha", "tra_v_call": "v_alpha", "trav": "v_alpha",
    "j_alpha": "j_alpha", "tra_j_call": "j_alpha", "traj": "j_alpha",
    # Epitope
    "epitope": "epitope", "epitope_peptide": "epitope",
    "epitope_peptid": "epitope", "antigen": "epitope",
    "epitope.aa": "epitope",
    # HLA
    "hla": "hla", "mhc": "hla", "mhc_a": "hla",
    # Subject
    "subject_id": "subject_id", "subject": "subject_id",
    "donor": "subject_id", "patient": "subject_id",
    "pubmed_id": "subject_id",
    # Metadata
    "species": "species", "count": "count", "frequency": "frequency",
}


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
def detect_format(df: pd.DataFrame) -> str:
    """Auto-detect TCR data format from DataFrame columns.

    Checks for unique column signatures of known formats.
    Returns one of: 'vdjdb', 'mcpas', 'airr', '10x', 'custom'.
    """
    cols = set(df.columns)
    cols_lower = {c.lower(): c for c in cols}
    lower_set = set(cols_lower.keys())

    # VDJdb: has Gene + CDR3 + V + J + Epitope + Score
    if {"gene", "cdr3", "v", "j"} <= lower_set and "score" in lower_set:
        return "vdjdb"

    # McPAS-TCR: has CDR3.beta.aa + TRBV + Pathology
    if "cdr3.beta.aa" in lower_set or ("trbv" in lower_set and "pathology" in lower_set):
        return "mcpas"

    # AIRR: has sequence_id + cdr3 + v_call
    if {"sequence_id", "v_call"} <= lower_set:
        return "airr"

    # 10x: has barcode + chain + v_gene
    if "barcode" in lower_set and "chain" in lower_set:
        return "10x"

    return "custom"


def detect_format_from_file(path: str) -> str:
    """Detect format by reading just the header."""
    df = pd.read_csv(path, sep=None, engine="python", nrows=1)
    return detect_format(df)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class Preprocessor:
    """Chainable TCR data preprocessing pipeline.

    Each method returns self for chaining. Call .result() to get the DataFrame.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._raw_df: Optional[pd.DataFrame] = None
        self._fmt: Optional[str] = None
        self._report: dict = {}
        self._source: Optional[str] = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, path: str, fmt: str = "auto", **kwargs) -> "Preprocessor":
        """Load TCR data file.

        Args:
            path: File path (TSV, CSV, etc.)
            fmt: 'auto', 'vdjdb', 'mcpas', 'airr', '10x', 'custom'
            **kwargs: Passed to pd.read_csv
        """
        self._source = str(path)

        # Auto-detect separator
        sep = kwargs.pop("sep", None)
        if sep is None:
            with open(path) as f:
                first_line = f.readline()
            sep = "\t" if "\t" in first_line else ","

        self._raw_df = pd.read_csv(path, sep=sep, dtype=str,
                                   keep_default_na=True, na_values=["NA", "na", "N/A", ""],
                                   engine="python", **kwargs)

        # Fill NaN with empty string for internal processing
        self._df = self._raw_df.fillna("")
        self._report["source"] = str(path)
        self._report["input_rows"] = len(self._df)

        if fmt == "auto":
            self._fmt = detect_format(self._df)
        else:
            self._fmt = fmt

        logger.info(f"Loaded {len(self._df)} rows from {path} (format: {self._fmt})")

        # Auto-map columns so subsequent steps always see canonical names
        if self._fmt in FORMAT_MAPPINGS:
            self.map_columns()
        return self

    def load_dataframe(self, df: pd.DataFrame, fmt: str = "auto") -> "Preprocessor":
        """Load from an existing DataFrame."""
        self._raw_df = df.copy()
        self._df = df.fillna("")
        self._report["input_rows"] = len(self._df)
        self._fmt = detect_format(self._df) if fmt == "auto" else fmt
        logger.info(f"Loaded {len(self._df)} rows (format: {self._fmt})")

        if self._fmt in FORMAT_MAPPINGS:
            self.map_columns()
        return self

    # ------------------------------------------------------------------
    # Column mapping
    # ------------------------------------------------------------------
    def map_columns(self) -> "Preprocessor":
        """Rename columns to canonical names based on detected format."""
        if self._df is None:
            raise RuntimeError("Call .load() first")

        n_before = len(self._df.columns)

        if self._fmt in FORMAT_MAPPINGS:
            mapping = FORMAT_MAPPINGS[self._fmt]
        else:
            # Fuzzy match for custom formats
            mapping = {}
            cols_lower = {c.lower(): c for c in self._df.columns}
            for col_lower, canonical in FUZZY_MAPPINGS.items():
                if col_lower in cols_lower:
                    mapping[cols_lower[col_lower]] = canonical

        self._df = self._df.rename(columns=mapping)
        n_mapped = sum(1 for v in mapping.values() if v in CANONICAL_COLUMNS or v.startswith("_"))
        self._report["columns_mapped"] = n_mapped
        self._report["format_detected"] = self._fmt

        logger.info(f"Mapped {n_mapped} columns for format '{self._fmt}'")
        return self

    # ------------------------------------------------------------------
    # Chain filtering
    # ------------------------------------------------------------------
    def filter_chain(self, gene: Optional[str] = None) -> "Preprocessor":
        """Filter to specific chain (TRB or TRA).

        For VDJdb: filters by Gene column.
        For 10x: filters by chain column.
        For others: filters by presence of cdr3_beta / cdr3_alpha.
        """
        if self._df is None:
            raise RuntimeError("Call .load() first")

        n_before = len(self._df)

        # Check both mapped and original column names
        gene_col = None
        for col in ["_gene", "Gene", "gene", "chain", "_chain", "locus", "_locus"]:
            if col in self._df.columns:
                gene_col = col
                break

        if gene_col is not None:
            if gene_col in ("_gene", "Gene", "gene", "locus", "_locus"):
                self._df = self._df[self._df[gene_col].astype(str).str.upper() == gene.upper()].copy()
            elif gene_col in ("chain", "_chain"):
                self._df = self._df[self._df[gene_col].astype(str).str.upper() == gene.upper()].copy()
        elif gene == "TRB":
            cdr3_col = None
            for col in ["cdr3_beta", "CDR3", "CDR3.beta.aa", "cdr3", "cdr3b"]:
                if col in self._df.columns:
                    cdr3_col = col
                    break
            if cdr3_col:
                self._df = self._df[self._df[cdr3_col].astype(str).str.len() > 0].copy()
        elif gene == "TRA":
            cdr3_col = None
            for col in ["cdr3_alpha", "CDR3.alpha.aa", "CDR3.alpha"]:
                if col in self._df.columns:
                    cdr3_col = col
                    break
            if cdr3_col:
                self._df = self._df[self._df[cdr3_col].astype(str).str.len() > 0].copy()

        n_after = len(self._df)
        self._report["chain_filter"] = {"gene": gene, "removed": n_before - n_after}
        logger.info(f"Chain filter ({gene}): {n_before} -> {n_after} rows")
        return self

    # ------------------------------------------------------------------
    # Score / quality filtering
    # ------------------------------------------------------------------
    def filter_score(self, min_score: int = 0, score_col: Optional[str] = None) -> "Preprocessor":
        """Filter by quality/score column.

        For VDJdb: uses Score column by default.
        For others: uses specified column or auto-detects.
        """
        if self._df is None:
            raise RuntimeError("Call .load() first")

        if min_score <= 0:
            return self

        n_before = len(self._df)

        # Determine score column
        if score_col is None:
            candidates = ["_score", "Score", "score", "quality", "confidence"]
            col = None
            for c in candidates:
                if c in self._df.columns:
                    col = c
                    break
        else:
            col = score_col

        if col is None:
            logger.warning(f"No score column found, skipping filter_score")
            return self

        scores = pd.to_numeric(self._df[col], errors="coerce").fillna(0)
        self._df = self._df[scores >= min_score].copy()

        n_after = len(self._df)
        self._report["score_filter"] = {"min_score": min_score, "removed": n_before - n_after}
        logger.info(f"Score filter (>= {min_score}): {n_before} -> {n_after} rows")
        return self

    # ------------------------------------------------------------------
    # CDR3 cleaning
    # ------------------------------------------------------------------
    def clean_cdr3(self,
                   min_length: int = 5,
                   max_length: int = 30,
                   require_c_start: bool = False,
                   uppercase: bool = True,
                   remove_invalid: bool = True) -> "Preprocessor":
        """Clean CDR3 sequences.

        - Uppercase all sequences
        - Remove sequences with invalid characters
        - Filter by length range
        - Optionally require C-start
        """
        if self._df is None:
            raise RuntimeError("Call .load() first")

        n_before = len(self._df)
        reasons = {}

        for cdr3_col in ["cdr3_beta", "cdr3_alpha"]:
            if cdr3_col not in self._df.columns:
                # Try original column names
                orig_map = {"cdr3_beta": ["CDR3", "CDR3.beta.aa", "cdr3", "cdr3b", "junction_aa"],
                            "cdr3_alpha": ["CDR3.alpha.aa", "CDR3.alpha"]}
                for orig in orig_map.get(cdr3_col, []):
                    if orig in self._df.columns:
                        self._df = self._df.rename(columns={orig: cdr3_col})
                        break
                if cdr3_col not in self._df.columns:
                    continue

            seqs = self._df[cdr3_col].astype(str)
            mask = seqs.str.len() > 0  # non-empty

            if uppercase:
                self._df.loc[mask, cdr3_col] = self._df.loc[mask, cdr3_col].str.upper()

            seqs = self._df[cdr3_col].astype(str)
            mask = seqs.str.len() > 0

            # Check for invalid characters
            if remove_invalid:
                def has_invalid(s):
                    if not s or len(s) == 0:
                        return False
                    return bool(set(s) - VALID_AA)

                invalid_mask = seqs.apply(has_invalid)
                n_invalid = invalid_mask.sum()
                if n_invalid > 0:
                    reasons[f"{cdr3_col}_invalid_aa"] = int(n_invalid)
                    self._df.loc[invalid_mask, cdr3_col] = ""

            seqs = self._df[cdr3_col].astype(str)
            mask = seqs.str.len() > 0

            # Length filter
            too_short = mask & (seqs.str.len() < min_length)
            too_long = mask & (seqs.str.len() > max_length)
            n_short, n_long = too_short.sum(), too_long.sum()
            if n_short > 0:
                reasons[f"{cdr3_col}_too_short"] = int(n_short)
                self._df.loc[too_short, cdr3_col] = ""
            if n_long > 0:
                reasons[f"{cdr3_col}_too_long"] = int(n_long)
                self._df.loc[too_long, cdr3_col] = ""

            # C-start filter
            if require_c_start:
                seqs = self._df[cdr3_col].astype(str)
                mask = seqs.str.len() > 0
                no_c = mask & ~seqs.str.startswith("C")
                n_no_c = no_c.sum()
                if n_no_c > 0:
                    reasons[f"{cdr3_col}_no_c_start"] = int(n_no_c)
                    self._df.loc[no_c, cdr3_col] = ""

        # Drop rows where all CDR3 columns are empty
        cdr3_cols = [c for c in ["cdr3_beta", "cdr3_alpha"] if c in self._df.columns]
        if cdr3_cols:
            has_cdr3 = self._df[cdr3_cols].apply(
                lambda col: col.astype(str).str.len() > 0
            ).any(axis=1)
            self._df = self._df[has_cdr3].copy()

        n_after = len(self._df)
        reasons["total_removed"] = n_before - n_after
        self._report["cdr3_clean"] = reasons
        logger.info(f"CDR3 clean: {n_before} -> {n_after} rows ({reasons})")
        return self

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def dedup(self, strategy: str = "keep_first",
              by: Optional[list[str]] = None) -> "Preprocessor":
        """Remove duplicate TCR entries.

        Args:
            strategy: 'keep_first' or 'keep_most_complete'
            by: Columns to deduplicate on. Default: cdr3_beta + v_beta
        """
        if self._df is None:
            raise RuntimeError("Call .load() first")

        n_before = len(self._df)

        if by is None:
            by = [c for c in ["cdr3_beta", "v_beta"] if c in self._df.columns]
            if not by:
                by = ["cdr3_beta"]

        if strategy == "keep_most_complete":
            # Sort so that rows with more non-empty fields come first
            completeness = self._df.apply(
                lambda row: sum(1 for v in row if str(v).strip() not in ("", "nan", "None")),
                axis=1
            )
            self._df = self._df.assign(_completeness=completeness)
            self._df = self._df.sort_values("_completeness", ascending=False)
            self._df = self._df.drop_duplicates(subset=by, keep="first")
            self._df = self._df.drop(columns=["_completeness"])
        else:
            self._df = self._df.drop_duplicates(subset=by, keep="first")

        n_after = len(self._df)
        self._report["dedup"] = {"by": by, "strategy": strategy, "removed": n_before - n_after}
        logger.info(f"Dedup: {n_before} -> {n_after} rows")
        return self

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------
    def extract_metadata(self, json_cols: Optional[list[str]] = None) -> "Preprocessor":
        """Extract metadata from JSON columns.

        For VDJdb: parses Meta → subject_id, cell_subset, tissue, etc.
        For VDJdb: parses CDR3fix → quality flags.
        For others: parses specified JSON columns.
        """
        if self._df is None:
            raise RuntimeError("Call .load() first")

        # VDJdb Meta extraction
        if self._fmt == "vdjdb" and "_meta" in self._df.columns:
            meta_fields = ["subject.id", "cell.subset", "tissue",
                          "subject.cohort", "donor.MHC"]

            for field in meta_fields:
                col_name = field.replace(".", "_")
                self._df[col_name] = self._df["_meta"].apply(
                    lambda s: _extract_json_field(s, field)
                )

            # subject_id from Meta
            if "subject_id" not in self._df.columns or self._df["subject_id"].astype(str).str.len().sum() == 0:
                self._df["subject_id"] = self._df["_meta"].apply(
                    lambda s: _extract_json_field(s, "subject.id")
                )

        # VDJdb CDR3fix extraction
        if self._fmt == "vdjdb" and "_cdr3fix" in self._df.columns:
            for field in ["good", "fixNeeded"]:
                col_name = f"cdr3fix_{field}"
                self._df[col_name] = self._df["_cdr3fix"].apply(
                    lambda s: _extract_json_field(s, field)
                )

        # Custom JSON columns
        if json_cols:
            for col in json_cols:
                if col in self._df.columns:
                    self._df[f"_parsed_{col}"] = self._df[col].apply(
                        lambda s: _safe_json_parse(s)
                    )

        self._report["metadata_extracted"] = True
        return self

    # ------------------------------------------------------------------
    # Normalize (final step)
    # ------------------------------------------------------------------
    def normalize(self) -> "Preprocessor":
        """Normalize into canonical schema.

        - Generate tcr_id if missing
        - Fill defaults for count/frequency
        - Infer chain_mode
        - Ensure all canonical columns exist
        """
        if self._df is None:
            raise RuntimeError("Call .load() first")

        df = self._df

        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": "", "None": "", "nan": ""})

        # Generate tcr_id if missing
        if "tcr_id" not in df.columns or df["tcr_id"].astype(str).str.strip().str.len().sum() == 0:
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
        has_alpha = "cdr3_alpha" in df.columns and df["cdr3_alpha"].astype(str).str.len().sum() > 0
        has_beta = "cdr3_beta" in df.columns and df["cdr3_beta"].astype(str).str.len().sum() > 0
        if has_alpha and has_beta:
            df["chain_mode"] = "paired_ab"
        elif has_alpha:
            df["chain_mode"] = "alpha_only"
        else:
            df["chain_mode"] = "beta_only"

        # Ensure canonical columns exist
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df.reset_index(drop=True)

        # Keep canonical + any extra columns (prefixed with _)
        extra_cols = [c for c in df.columns if c.startswith("_") or c not in CANONICAL_COLUMNS]
        keep_cols = CANONICAL_COLUMNS + [c for c in extra_cols if c in df.columns]
        # Remove duplicates in keep_cols
        seen = set()
        unique_keep = []
        for c in keep_cols:
            if c not in seen:
                seen.add(c)
                unique_keep.append(c)

        self._df = df[unique_keep].copy()

        self._report["output_rows"] = len(self._df)
        logger.info(f"Normalized: {len(self._df)} rows")
        return self

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def result(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        if self._df is None:
            raise RuntimeError("Call .load() first")
        return self._df.copy()

    def report(self) -> dict:
        """Return the preprocessing report."""
        return self._report.copy()

    def report_summary(self) -> str:
        """Return a human-readable report summary."""
        lines = ["Preprocessing Report", "=" * 40]
        for key, val in self._report.items():
            if isinstance(val, dict):
                lines.append(f"{key}:")
                for k, v in val.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {val}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _extract_json_field(s: str, field: str) -> str:
    """Extract a field from a JSON string."""
    try:
        d = json.loads(s) if isinstance(s, str) and s else {}
        val = d.get(field, "")
        return str(val) if val else ""
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


def _safe_json_parse(s: str) -> dict:
    """Safely parse a JSON string."""
    try:
        return json.loads(s) if isinstance(s, str) and s else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def preprocess_file(path: str, **kwargs) -> pd.DataFrame:
    """One-shot preprocessing: load, clean, normalize.

    Args:
        path: Input file path
        **kwargs: Options for pipeline steps:
            format/fmt: data format hint (vdjdb, mcpas, airr, 10x, auto)
            gene/chain: filter to chain (TRB, TRA)
            min_score: minimum quality score
            dedup: whether to deduplicate (default True)
            clean_cdr3: whether to clean CDR3 (default True)
            extract_meta: whether to extract metadata (default True)
            min_length, max_length: CDR3 length bounds
            require_c_start: require CDR3 starts with C

    Returns:
        Normalized DataFrame
    """
    # Pop all non-load kwargs before passing to load()
    fmt = kwargs.pop("format", kwargs.pop("fmt", "auto"))
    gene = kwargs.pop("gene", kwargs.pop("chain", None))
    min_score = kwargs.pop("min_score", 0)
    do_dedup = kwargs.pop("dedup", True)
    do_clean = kwargs.pop("clean_cdr3", True)
    do_meta = kwargs.pop("extract_meta", True)
    clean_kwargs = {}
    for k in ["min_length", "max_length", "require_c_start"]:
        if k in kwargs:
            clean_kwargs[k] = kwargs.pop(k)

    pp = (Preprocessor()
          .load(path, fmt=fmt, **kwargs)
          .map_columns())

    if gene:
        pp = pp.filter_chain(gene=gene)
    if min_score > 0:
        pp = pp.filter_score(min_score=min_score)

    if do_clean:
        pp = pp.clean_cdr3(**clean_kwargs)

    pp = pp.dedup()

    if do_meta:
        pp = pp.extract_metadata()

    return pp.normalize().result()
