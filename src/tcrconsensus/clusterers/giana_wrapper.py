"""GIANA wrapper adapter (subprocess-based) — matching paper approach.

Paper approach:
  - GIANA runs on ALL unique CDR3 sequences from both alpha and beta chains
    (rbind(dataframe_alpha, dataframe_beta), then unique CDR3s)
  - Input: one CDR3 per line, Vgene=False
  - GIANA outputs (CDR3, cluster_id, info) — info columns are echoed from input
  - Paper merges GIANA output back to alpha/beta chain data by CDR3

Our adaptation:
  - Prepare input: extract all unique CDR3 sequences (alpha + beta) from tcr_table
  - Build a CDR3 → tcr_ids reverse mapping (one CDR3 can map to multiple tcr_ids)
  - Run GIANA with Vgene=False, exact=True, thr=7.0
  - Parse output: for each (CDR3, cluster_id), expand to all associated tcr_ids
  - Normalize to ClusterAssignment list

Reference: https://github.com/s175573/GIANA
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseClusterer
from ..schema.records import ClusterAssignment

logger = logging.getLogger(__name__)


class GIANAWrapper(BaseClusterer):
    """Wrapper around GIANA clustering tool (paper approach).

    Collects unique CDR3 sequences from both alpha and beta chains,
    runs GIANA with Vgene=False, then maps clusters back to tcr_ids
    via CDR3→tcr_ids reverse lookup.
    """

    name = "giana"

    def __init__(
        self,
        giana_script: str = "",
        threshold: float = 7.0,
        threshold_score: float = 3.5,
        threshold_vgene: float = 3.7,
        exact: bool = True,
        n_threads: int = 1,
    ):
        self.giana_script = giana_script
        self.threshold = threshold
        self.threshold_score = threshold_score
        self.threshold_vgene = threshold_vgene
        self.exact = exact
        self.n_threads = n_threads

    # ------------------------------------------------------------------
    # Pipeline: prepare_input → run → normalize
    # ------------------------------------------------------------------

    def prepare_input(self, tcr_table: pd.DataFrame, config: dict) -> dict:
        """Prepare GIANA input with CDR3→tcr_ids mapping.

        Paper approach: merge all unique CDR3 sequences from alpha and beta
        chains into a single deduplicated list. Build reverse mapping so
        we can expand GIANA output back to individual tcr_ids.
        """
        cdr3_to_tcr_ids: dict[str, list[str]] = {}
        n_alpha = 0
        n_beta = 0

        # Resolve actual column names (case-insensitive)
        beta_col = None
        alpha_col = None
        for col in tcr_table.columns:
            if col.lower() == "cdr3_beta":
                beta_col = col
            elif col.lower() == "cdr3_alpha":
                alpha_col = col
        if not beta_col and not alpha_col:
            logger.warning("No CDR3 columns found in tcr_table")
            return {"cdr3_to_tcr_ids": {}, "unique_cdr3s": [], "total_tcrs": 0}

        for _, r in tcr_table.iterrows():
            tcr_id = str(r.get("tcr_id", ""))

            # Beta chain CDR3 → add tcr_id (avoid duplicates via set)
            if beta_col:
                val = r.get(beta_col)
                if pd.notna(val) and str(val).strip():
                    cdr3b = str(val).strip()
                    if cdr3b not in cdr3_to_tcr_ids:
                        cdr3_to_tcr_ids[cdr3b] = []
                    if tcr_id not in cdr3_to_tcr_ids[cdr3b]:
                        cdr3_to_tcr_ids[cdr3b].append(tcr_id)
                        n_beta += 1

            # Alpha chain CDR3 → add tcr_id (avoid duplicates via set)
            if alpha_col:
                val = r.get(alpha_col)
                if pd.notna(val) and str(val).strip():
                    cdr3a = str(val).strip()
                    if cdr3a not in cdr3_to_tcr_ids:
                        cdr3_to_tcr_ids[cdr3a] = []
                    if tcr_id not in cdr3_to_tcr_ids[cdr3a]:
                        cdr3_to_tcr_ids[cdr3a].append(tcr_id)
                        n_alpha += 1

        total_tcrs = sum(len(v) for v in cdr3_to_tcr_ids.values())
        logger.info(
            f"GIANA input: {len(cdr3_to_tcr_ids)} unique CDR3s "
            f"(from {n_beta} beta + {n_alpha} alpha entries, "
            f"{total_tcrs} total tcr_id associations)"
        )

        return {
            "cdr3_to_tcr_ids": cdr3_to_tcr_ids,
            "unique_cdr3s": list(cdr3_to_tcr_ids.keys()),
            "total_tcrs": total_tcrs,
        }

    def run(self, prepared_input: dict, workdir: Path) -> dict:
        """Execute GIANA subprocess and return cluster assignments.

        Input file format: one CDR3 per line (Vgene=False mode).
        GIANA echoes input columns after CDR3 in the output,
        so we don't need extra info columns — we match by CDR3.
        """
        cdr3_to_tcr_ids = prepared_input["cdr3_to_tcr_ids"]
        unique_cdr3s = prepared_input["unique_cdr3s"]

        if not unique_cdr3s:
            return {"assignments": []}

        tmpdir = tempfile.mkdtemp()
        try:
            # ---- Write input file: one CDR3 per line ----
            input_path = os.path.join(tmpdir, "giana_input.txt")
            with open(input_path, "w") as f:
                for cdr3 in unique_cdr3s:
                    f.write(cdr3 + "\n")
            logger.info(f"GIANA input file: {input_path} ({len(unique_cdr3s)} CDR3s)")

            # ---- Build output directory ----
            output_dir = workdir / "giana_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # ---- Locate GIANA script (param > env > PATH > error) ----
            script = self.giana_script or os.environ.get("TCR_GIANA_SCRIPT")
            if not script:
                script = shutil.which("GIANA4.1.py") or shutil.which("giana")
            if not script:
                raise FileNotFoundError(
                    "GIANA script not found. Pass giana_script=... to "
                    "GIANAWrapper, set the TCR_GIANA_SCRIPT environment "
                    "variable, or place GIANA4.1.py on PATH."
                )

            # ---- Build command ----
            # Paper: Vgene=False (-v), exact=True (default), thr=7.0
            cmd = [
                sys.executable, script,
                "-f", input_path,
                "-t", str(self.threshold),
                "-S", str(self.threshold_score),
                "-o", str(output_dir),
                "-N", str(self.n_threads),
                "-v",  # Vgene=False — CDR3 only mode
            ]
            if not self.exact:
                cmd.append("-e")

            logger.info(f"GIANA command: {' '.join(cmd)}")

            # ---- Execute ----
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            (workdir / "giana_stdout.log").write_text(result.stdout)
            (workdir / "giana_stderr.log").write_text(result.stderr)

            if result.returncode != 0:
                raise RuntimeError(f"GIANA failed (rc={result.returncode}): {result.stderr[:500]}")

            # ---- Parse output ----
            candidates_out = list(output_dir.glob("*RotationEncodingBL62*"))
            if not candidates_out:
                # Fallback: any non-hidden file
                candidates_out = [
                    p for p in output_dir.iterdir()
                    if p.is_file() and not p.name.startswith(".")
                ]
            if not candidates_out:
                logger.warning("GIANA produced no output files")
                return {"assignments": []}

            data_file = candidates_out[0]
            logger.info(f"GIANA output file: {data_file}")

            # Parse: CDR3\tcluster_id\tinfo
            cdr3_clusters: dict[str, str] = {}
            n_data_lines = 0
            with open(data_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("##"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        cdr3_seq = parts[0]
                        cluster_id = parts[1]
                        cdr3_clusters[cdr3_seq] = cluster_id
                        n_data_lines += 1

            logger.info(
                f"GIANA parsed: {n_data_lines} CDR3s in "
                f"{len(set(cdr3_clusters.values()))} clusters"
            )

            # ---- Map back to tcr_ids (deduplicate cross-chain) ----
            assignments = []
            seen_pairs: set[tuple[str, str]] = set()
            unmatched = 0
            for cdr3, cluster_id in cdr3_clusters.items():
                tcr_ids = cdr3_to_tcr_ids.get(cdr3, [])
                if not tcr_ids:
                    unmatched += 1
                    continue
                for tid in tcr_ids:
                    key = (tid, cluster_id)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        assignments.append({
                            "tcr_id": tid,
                            "cluster": cluster_id,
                        })

            logger.info(
                f"GIANA mapped: {len(assignments)} tcr_id assignments "
                f"({unmatched} CDR3s unmatched)"
            )
            return {"assignments": assignments}

        except FileNotFoundError:
            logger.error(f"GIANA script not found: {script}")
            raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def parse_output(self, workdir: Path) -> dict:
        """No-op: run() returns dict directly."""
        return {}

    def normalize(self, raw_output: dict) -> list[ClusterAssignment]:
        """Convert GIANA cluster assignments to ClusterAssignment list."""
        assignments = []
        for item in raw_output.get("assignments", []):
            cid = item["cluster"]
            # Handle both string and int cluster IDs
            cluster_str = f"{int(cid):04d}" if isinstance(cid, str) else f"{cid:04d}"
            assignments.append(
                ClusterAssignment(
                    method=self.name,
                    tcr_id=str(item["tcr_id"]),
                    cluster_id=f"giana_{cluster_str}",
                    membership_score=1.0,
                )
            )
        return assignments
