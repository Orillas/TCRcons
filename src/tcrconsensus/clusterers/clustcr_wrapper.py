"""clusTCR wrapper adapter (Python API) — matching paper approach.

Paper approach:
  - clusTCR v1.0.2, default parameters
  - method="two-step" (FAISS preclustering + MCL refinement)
  - Integrates TRA/TRB paired chains via native alpha parameter
  - fit(data, alpha=alpha_series) — clusTCR internally concatenates beta+alpha

Reference: https://github.com/svalkiers/clusTCR
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import BaseClusterer
from ..schema.records import ClusterAssignment

logger = logging.getLogger(__name__)


class ClusTCRWrapper(BaseClusterer):
    """Wrapper around clusTCR — paper-matched default parameters with dual-chain."""

    name = "clustcr"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import clustcr  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self, method: str = "two-step", n_cpus: int = 1):
        # Paper: default parameters → method="two-step" (FAISS + MCL)
        self.method = method
        self.n_cpus = n_cpus

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def prepare_input(self, tcr_table: pd.DataFrame, config: dict) -> dict:
        """Extract CDR3b + CDR3a pairs for clusTCR native dual-chain.

        Deduplicates by (CDR3b, CDR3a) pair to preserve pairing diversity.
        Maps concatenated junction (CDR3b+CDR3a) back to tcr_ids.
        """
        beta_col = alpha_col = None
        for col in tcr_table.columns:
            if col.lower() == "cdr3_beta":
                beta_col = col
            elif col.lower() == "cdr3_alpha":
                alpha_col = col

        if not beta_col:
            raise ValueError("cdr3_beta column required for clusTCR")

        # Dedup by (CDR3b, CDR3a) pair → list of tcr_ids
        # Also track junction (CDR3b+CDR3a) → CDR3b for reverse mapping
        junction_to_ids: dict[str, list[str]] = {}
        beta_list: list[str] = []
        alpha_list: list[str] = []
        skipped = 0
        has_alpha = False

        for _, r in tcr_table.iterrows():
            cdr3b = str(r.get(beta_col, "")).strip()
            if not cdr3b:
                skipped += 1
                continue
            tcr_id = str(r.get("tcr_id", ""))

            if alpha_col:
                cdr3a = str(r.get(alpha_col, "")).strip()
            else:
                cdr3a = ""

            if cdr3a:
                has_alpha = True

            # Junction = CDR3b + CDR3a (same as clusTCR's data.add(alpha))
            junction = cdr3b + cdr3a if cdr3a else cdr3b

            if junction not in junction_to_ids:
                junction_to_ids[junction] = []
                beta_list.append(cdr3b)
                alpha_list.append(cdr3a)

            junction_to_ids[junction].append(tcr_id)

        total = sum(len(v) for v in junction_to_ids.values())
        logger.info(
            f"clusTCR input: {len(junction_to_ids)} unique junctions "
            f"({total} tcr_id associations, alpha={'yes' if has_alpha else 'no'}, "
            f"{skipped} skipped)"
        )
        return {
            "junction_to_ids": junction_to_ids,
            "beta_list": beta_list,
            "alpha_list": alpha_list,
            "has_alpha": has_alpha,
        }

    def run(self, prepared_input: dict, workdir: Path) -> dict:
        """Run clusTCR with default two-step method and native alpha parameter."""
        junction_to_ids = prepared_input["junction_to_ids"]
        beta_list = prepared_input["beta_list"]
        alpha_list = prepared_input["alpha_list"]
        has_alpha = prepared_input["has_alpha"]

        if not junction_to_ids:
            return {"assignments": []}

        try:
            from clustcr import Clustering
        except ImportError:
            logger.error("clusTCR not installed. pip install clustcr")
            raise

        # Paper: default parameters → method="two-step", chain="B"
        clustering = Clustering(
            method=self.method,  # "two-step" (FAISS + MCL)
            chain="B",
            n_cpus=self.n_cpus,
        )

        beta_series = pd.Series(beta_list, name="cdr3")

        # Paper: native dual-chain via alpha parameter
        if has_alpha:
            alpha_series = pd.Series(alpha_list, name="alpha")
            logger.info("clusTCR: using native alpha parameter for dual-chain")
            result = clustering.fit(beta_series, alpha=alpha_series)
        else:
            logger.info("clusTCR: CDR3b only (no alpha)")
            result = clustering.fit(beta_series)

        # Build assignments from result
        clusters_df = result.clusters_df
        cdr3_col = (
            "junction_aa"
            if "junction_aa" in clusters_df.columns
            else clusters_df.columns[0]
        )
        cluster_col = (
            "cluster"
            if "cluster" in clusters_df.columns
            else clusters_df.columns[1]
        )

        assignments = []
        n_no_match = 0
        for _, row in clusters_df.iterrows():
            junction = str(row[cdr3_col])
            cluster_id = str(row[cluster_col])

            tcr_ids = junction_to_ids.get(junction, [])
            if not tcr_ids:
                n_no_match += 1
                continue

            for tid in tcr_ids:
                assignments.append({"tcr_id": tid, "cluster": cluster_id})

        n_clusters = len(set(a["cluster"] for a in assignments))
        if n_no_match:
            logger.warning(f"clusTCR: {n_no_match} junctions unmatched")
        logger.info(
            f"clusTCR: {len(assignments)} tcr_id assignments ({n_clusters} clusters)"
        )
        return {"assignments": assignments}

    def parse_output(self, workdir: Path) -> dict:
        return {}

    def normalize(self, raw_output: dict) -> list[ClusterAssignment]:
        assignments = []
        for item in raw_output.get("assignments", []):
            assignments.append(
                ClusterAssignment(
                    method=self.name,
                    tcr_id=str(item["tcr_id"]),
                    cluster_id=f"clustcr_{int(item['cluster']):04d}",
                    membership_score=1.0,
                )
            )
        return assignments
