"""TCRMatch wrapper adapter (subprocess-based).

Runs TCRMatch C++ binary against the IEDB database (paper approach).
Groups input CDR3s by shared IEDB receptor_group matches into clusters.

Paper approach:
  - Single-chain independent analysis (TRB only)
  - Match against IEDB database with similarity threshold
  - Cluster query CDR3s by shared receptor_group (IEDB TCR grouping),
    not by shared epitope — much more specific grouping
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .base import BaseClusterer
from ..schema.records import ClusterAssignment

logger = logging.getLogger(__name__)


def _trim_cdr3(seq: str) -> str:
    """Trim CDR3 the way TCRMatch does: remove leading C and trailing F/W/Y."""
    s = seq
    if s.startswith("C"):
        s = s[1:]
    if s and s[-1] in ("F", "W", "Y"):
        s = s[:-1]
    return s


class TCRMatchWrapper(BaseClusterer):
    """Wrapper around TCRMatch binary for TCR clustering via IEDB matching."""

    name = "tcrmatch"

    def __init__(
        self,
        tcrmatch_bin: str = "",
        iedb_db: str = "",
        threshold: float = 0.97,
        n_threads: int = 4,
        max_memory_gb: int = 8,
    ):
        self.tcrmatch_bin = tcrmatch_bin
        self.iedb_db = iedb_db
        self.threshold = threshold
        self.n_threads = n_threads
        self.max_memory_gb = max_memory_gb

    def prepare_input(self, tcr_table: pd.DataFrame, config: dict) -> Path:
        """Write newline-separated CDR3 file (TRB only — paper: single-chain analysis)."""
        tmpdir = tempfile.mkdtemp()
        input_path = Path(tmpdir) / "tcrmatch_input.txt"

        lines = []
        # Map trimmed CDR3 -> list of original tcr_ids
        self._trimmed_to_tcr_ids: dict[str, list[str]] = defaultdict(list)
        seen = set()
        for _, r in tcr_table.iterrows():
            tcr_id = str(r.get("tcr_id", ""))
            # Paper: single-chain analysis — TRB only
            cdr3 = r.get("cdr3_beta")
            if pd.isna(cdr3) or not cdr3:
                continue
            cdr3 = str(cdr3)
            trimmed = _trim_cdr3(cdr3)
            self._trimmed_to_tcr_ids[trimmed].append(tcr_id)
            if trimmed not in seen:
                seen.add(trimmed)
                lines.append(cdr3)  # TCRMatch accepts full CDR3, trims internally

        input_path.write_text("\n".join(lines) + "\n")
        logger.info(f"TCRMatch input: {len(lines)} unique CDR3 sequences")
        return input_path

    def run(self, prepared_input: Path, workdir: Path) -> Path:
        """Execute TCRMatch against IEDB database."""
        output_dir = workdir / "tcrmatch_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        binary = (
            self.tcrmatch_bin
            or os.environ.get("TCR_TCRMATCH_BIN")
            or shutil.which("tcrmatch")
        )
        if not binary:
            raise FileNotFoundError(
                "TCRMatch binary not found. Pass tcrmatch_bin=... to "
                "TCRMatchWrapper, set the TCR_TCRMATCH_BIN environment "
                "variable, or install the `tcrmatch` binary on PATH."
            )

        iedb_db = self.iedb_db or os.environ.get("TCR_TCRMATCH_IEDB")
        if not iedb_db:
            iedb_db = prepared_input
            logger.warning(
                "IEDB database not configured (set TCR_TCRMATCH_IEDB); "
                "using self-comparison"
            )
        elif not Path(iedb_db).exists():
            iedb_db = prepared_input
            logger.warning(
                "IEDB database not found at %s; using self-comparison", iedb_db
            )

        cmd = [binary, "-i", str(prepared_input), "-t", str(self.n_threads),
               "-s", str(self.threshold), "-d", str(iedb_db),
               "-m", str(self.max_memory_gb)]

        logger.info(f"TCRMatch cmd: {' '.join(cmd[:6])}...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        (output_dir / "matches.tsv").write_text(result.stdout)
        (workdir / "tcrmatch_stderr.log").write_text(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"TCRMatch failed: {result.stderr[:500]}")

        return self.parse_output(workdir)

    def parse_output(self, workdir: Path) -> pd.DataFrame:
        """Parse TCRMatch TSV output.

        Output columns: trimmed_input_sequence, match_sequence, score,
                         receptor_group, epitope, antigen, organism
        """
        fpath = workdir / "tcrmatch_output" / "matches.tsv"
        if not fpath.exists():
            return pd.DataFrame()

        rows = []
        with open(fpath) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    try:
                        score = float(parts[2])
                    except (ValueError, TypeError):
                        score = 0.0
                    receptor_group = parts[3] if len(parts) > 3 else ""
                    epitope = parts[4] if len(parts) > 4 else ""
                    rows.append({
                        "query_seq": parts[0],
                        "match_seq": parts[1],
                        "score": score,
                        "receptor_group": receptor_group,
                        "epitope": epitope,
                    })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def normalize(self, raw_output: pd.DataFrame) -> list[ClusterAssignment]:
        """Group query CDR3s by shared IEDB receptor_group via Union-Find.

        Paper approach: queries that match the same IEDB receptor_group
        (a specific TCR clonotype grouping) are clustered together.
        This is much more specific than epitope-based grouping.
        """
        if raw_output is None or raw_output.empty:
            return []

        # Build receptor_group -> set of query CDR3s
        rg_to_cdr3s: dict[str, set[str]] = defaultdict(set)
        for _, row in raw_output.iterrows():
            q = str(row["query_seq"])
            rg = str(row.get("receptor_group", ""))
            if not rg:
                continue
            rg_to_cdr3s[rg].add(q)

        # Union-Find on query CDR3s that share the same receptor_group
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], x)
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for cdr3s in rg_to_cdr3s.values():
            cdr3_list = list(cdr3s)
            for i in range(1, len(cdr3_list)):
                union(cdr3_list[0], cdr3_list[i])

        # Ensure all in parent
        for cdr3s in rg_to_cdr3s.values():
            for c in cdr3s:
                if c not in parent:
                    parent[c] = c

        # Group by root -> clusters
        clusters: dict[str, list[str]] = defaultdict(list)
        for cdr3 in parent:
            clusters[find(cdr3)].append(cdr3)

        # Convert to assignments via trimmed->tcr_id map
        assignments = []
        idx = 0
        n_singletons = 0
        for root, members in clusters.items():
            if len(members) < 2:
                n_singletons += 1
                continue
            cid = f"tcrmatch_{idx:04d}"
            idx += 1
            for trimmed_cdr3 in members:
                for tid in self._trimmed_to_tcr_ids.get(trimmed_cdr3, []):
                    assignments.append(
                        ClusterAssignment(
                            method=self.name,
                            tcr_id=tid,
                            cluster_id=cid,
                            membership_score=1.0,
                        )
                    )

        logger.info(
            f"TCRMatch: {len(clusters)} raw groups "
            f"({len(clusters) - n_singletons} valid clusters, "
            f"{n_singletons} singletons), "
            f"{len(assignments)} assignments"
        )
        return assignments
