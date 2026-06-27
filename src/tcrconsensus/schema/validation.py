"""Field-level validation for TCR data records."""

from __future__ import annotations

import re
from typing import Optional

from .records import TCRRecord, ChainMode

# CDR3 pattern: starts with C, ends with F/W, amino acids in between
CDR3_PATTERN = re.compile(r"^C[A-Z]{3,}[FW]$", re.IGNORECASE)

# V/J gene pattern: e.g. TRBV1-1, TRBJ1-1
VJ_GENE_PATTERN = re.compile(r"^TR[AB][VDJ]\d+(-\d+)?$", re.IGNORECASE)


def validate_tcr_record(record: TCRRecord) -> list[str]:
    """Validate a TCR record. Returns list of error messages (empty = valid)."""
    errors: list[str] = []

    if not record.tcr_id:
        errors.append("tcr_id is required")

    # Validate chain-specific CDR3
    if record.chain_mode in (ChainMode.BETA_ONLY, ChainMode.PAIRED_AB):
        if not record.cdr3_beta:
            errors.append("cdr3_beta required for beta/paired mode")
        elif not _valid_cdr3(record.cdr3_beta):
            errors.append(f"cdr3_beta format invalid: {record.cdr3_beta}")

    if record.chain_mode in (ChainMode.ALPHA_ONLY, ChainMode.PAIRED_AB):
        if not record.cdr3_alpha:
            errors.append("cdr3_alpha required for alpha/paired mode")
        elif not _valid_cdr3(record.cdr3_alpha):
            errors.append(f"cdr3_alpha format invalid: {record.cdr3_alpha}")

    # Validate V/J genes if present
    for gene_name, gene_val in [
        ("v_alpha", record.v_alpha),
        ("j_alpha", record.j_alpha),
        ("v_beta", record.v_beta),
        ("j_beta", record.j_beta),
    ]:
        if gene_val and not _valid_vj_gene(gene_val):
            errors.append(f"{gene_name} format invalid: {gene_val}")

    if record.count < 1:
        errors.append("count must be >= 1")

    if record.frequency is not None and not (0 <= record.frequency <= 1):
        errors.append("frequency must be in [0, 1]")

    return errors


def _valid_cdr3(seq: str) -> bool:
    """Check if CDR3 sequence matches expected pattern."""
    return bool(CDR3_PATTERN.match(seq.upper()))


def _valid_vj_gene(gene: str) -> bool:
    """Check if V/J gene name matches expected pattern."""
    return bool(VJ_GENE_PATTERN.match(gene))


def validate_cdr3_basic(seq: Optional[str]) -> bool:
    """Basic CDR3 check — non-empty amino acid string, len >= 5."""
    if not seq:
        return False
    seq = seq.upper()
    return len(seq) >= 5 and seq.isalpha()
