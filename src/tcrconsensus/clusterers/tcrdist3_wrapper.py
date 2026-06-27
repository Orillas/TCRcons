"""TCRdist3 wrapper adapter — paper approach (silhouette-optimized hierarchical).

Paper approach (as described by user):
  1. Dual-chain integration: Compute separate distance matrices for TRA and TRB
  2. Distance matrix synthesis: ADD (not average) the TRA and TRB matrices
  3. Hierarchical clustering: Ward linkage on combined distance matrix
  4. Silhouette score optimization: Find optimal threshold via silhouette score

Uses the TCRrep class from the tcrdist package (tcrdist3 v0.3).
Non-default: researchers optimized hierarchical clustering with silhouette score
on the distance matrix output (no standard clustering pipeline exists for tcrdist3).

Reference: Dash et al., 2017; Mayer-Blackwell et al., 2021
"""

from __future__ import annotations

import logging
from pathlib import Path

from collections import Counter

import numpy as np
import pandas as pd

from .base import BaseClusterer
from ..schema.records import ClusterAssignment

logger = logging.getLogger(__name__)


class TCRDist3Wrapper(BaseClusterer):
    """Wrapper around tcrdist TCRrep with silhouette-optimized hierarchical clustering."""

    name = "tcrdist3"

    def __init__(
        self,
        organism: str = "human",
        chains: list[str] | None = None,
        n_cpus: int = 1,
    ):
        self.organism = organism
        self.chains = chains or ["alpha", "beta"]
        self.n_cpus = n_cpus
        # Tier-1/F2: expose the TRB distance matrix (+ clonotype index) for the
        # consensus multi-signal edge fusion. Populated by run(); None until then.
        self.last_pw_beta: "np.ndarray | None" = None
        self.last_tcr_to_idx: dict[str, int] = {}

    @staticmethod
    def _add_allele(v: str) -> str:
        if not v or pd.isna(v):
            return ""
        v = str(v).strip()
        if "*" not in v:
            return f"{v}*01"
        return v

    @staticmethod
    def _safe_str(val) -> str:
        if pd.isna(val) or val is None:
            return ""
        return str(val).strip()

    def prepare_input(self, tcr_table: pd.DataFrame, config: dict) -> dict:
        """Map columns to tcrdist format and build clonotype→tcr_ids mapping."""
        col_map = {}
        for col in tcr_table.columns:
            low = col.lower()
            if low == "cdr3_beta":
                col_map["cdr3_b_aa"] = col
            elif low == "cdr3_alpha":
                col_map["cdr3_a_aa"] = col
            elif low in ("v_beta", "trbv"):
                col_map["v_b_gene"] = col
            elif low in ("v_alpha", "trav"):
                col_map["v_a_gene"] = col
            elif low in ("j_beta", "trbj"):
                col_map["j_b_gene"] = col
            elif low in ("j_alpha", "traj"):
                col_map["j_a_gene"] = col

        if "cdr3_b_aa" not in col_map:
            raise ValueError("cdr3_beta column required for tcrdist3")

        records: list[dict] = []
        clone_to_tcr_ids: dict[str, list[str]] = {}
        seen_clones: set[str] = set()

        for _, r in tcr_table.iterrows():
            cdr3b = self._safe_str(r.get(col_map.get("cdr3_b_aa", ""), ""))
            if not cdr3b:
                continue

            tcr_id = str(r.get("tcr_id", ""))
            cdr3a = self._safe_str(r.get(col_map.get("cdr3_a_aa", ""), "")) if "cdr3_a_aa" in col_map else ""
            vb = self._add_allele(self._safe_str(r.get(col_map.get("v_b_gene", ""), ""))) if "v_b_gene" in col_map else ""
            jb = self._safe_str(r.get(col_map.get("j_b_gene", ""), "")) if "j_b_gene" in col_map else ""
            va = self._add_allele(self._safe_str(r.get(col_map.get("v_a_gene", ""), ""))) if "v_a_gene" in col_map else ""
            ja = self._safe_str(r.get(col_map.get("j_a_gene", ""), "")) if "j_a_gene" in col_map else ""

            clone_key = f"{va}_{cdr3a}_{ja}_{vb}_{cdr3b}_{jb}"

            if clone_key not in seen_clones:
                seen_clones.add(clone_key)
                rec = {
                    "cdr3_b_aa": cdr3b,
                    "v_b_gene": vb,
                    "j_b_gene": jb,
                    "cdr3_a_aa": cdr3a,
                    "v_a_gene": va,
                    "j_a_gene": ja,
                    "count": 1,
                }
                records.append(rec)
                clone_to_tcr_ids[clone_key] = []

            clone_to_tcr_ids[clone_key].append(tcr_id)

        total = sum(len(v) for v in clone_to_tcr_ids.values())
        uses_alpha = "cdr3_a_aa" in col_map
        logger.info(
            f"TCRdist3 input: {len(records)} unique clonotypes "
            f"({total} tcr_id associations, alpha={'yes' if uses_alpha else 'no'})"
        )
        return {
            "records": records,
            "clone_to_tcr_ids": clone_to_tcr_ids,
            "uses_alpha": uses_alpha,
        }

    def run(self, prepared_input: dict, workdir: Path) -> dict:
        """Compute tcrdist distances and perform silhouette-optimized hierarchical clustering.

        Paper method:
          - Compute TRB distance matrix (pw_beta) and TRA distance matrix (pw_alpha) separately
          - Synthesize: dist_matrix = pw_beta + pw_alpha (ADD, not average)
          - Ward linkage hierarchical clustering
          - Optimize threshold via silhouette score
        """
        records = prepared_input["records"]
        clone_to_tcr_ids = prepared_input["clone_to_tcr_ids"]
        uses_alpha = prepared_input["uses_alpha"]

        n = len(records)
        if n < 2:
            return {"assignments": []}

        # ---- Step 1: Compute distances via TCRrep ----
        from tcrdist.repertoire import TCRrep

        cell_df = pd.DataFrame(records)
        chains = self.chains if uses_alpha else ["beta"]

        tr = TCRrep(
            cell_df=cell_df,
            organism=self.organism,
            chains=chains,
            db_file="alphabeta_gammadelta_db.tsv",
            compute_distances=False,
            infer_cdrs=True,       # infer CDR1/CDR2/pmhc from V gene — full tcrdist metric
            deduplicate=False,
            cpus=self.n_cpus,
            store_all_cdr=True,
        )

        # Fix None values in inferred CDR columns
        for _col in ["cdr1_b_aa", "cdr2_b_aa", "pmhc_b_aa",
                      "cdr1_a_aa", "cdr2_a_aa", "pmhc_a_aa"]:
            if _col in tr.clone_df.columns:
                tr.clone_df[_col] = tr.clone_df[_col].fillna("")
            else:
                tr.clone_df[_col] = ""

        # CDR3 weight = 5 (emphasize CDR3 while retaining structural info)
        tr.weights_b["cdr3_b_aa"] = 5
        if uses_alpha:
            tr.weights_a["cdr3_a_aa"] = 5
        tr.compute_distances()

        logger.info(f"TCRrep computed distances for {n} clonotypes")

        # ---- Step 2: Build combined distance matrix by ADDING TRA + TRB ----
        def _to_np(x):
            if x is None:
                return None
            if hasattr(x, "values"):
                return x.values
            return np.asarray(x)

        pw_beta = _to_np(getattr(tr, "pw_beta", None))
        pw_alpha = _to_np(getattr(tr, "pw_alpha", None))

        if uses_alpha and pw_alpha is not None:
            if pw_beta is not None:
                # Paper method: ADD (not average) the two chain distance matrices
                dist_matrix = pw_beta + pw_alpha
                logger.info("Distance matrix: pw_beta + pw_alpha (summed)")
            else:
                dist_matrix = pw_alpha
                logger.info("Distance matrix: pw_alpha only (pw_beta unavailable)")
        elif pw_beta is not None:
            dist_matrix = pw_beta
            logger.info("Distance matrix: pw_beta only (no alpha)")
        else:
            logger.warning("No distance matrix available from TCRrep")
            return {"assignments": []}

        dist_matrix = np.nan_to_num(dist_matrix, nan=100.0, posinf=100.0, neginf=0.0)

        # ---- Step 3: Hierarchical clustering ----
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        dist_sym = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_sym, 0)
        condensed = squareform(dist_sym)
        Z = linkage(condensed, method="ward")

        # ---- Step 4: Silhouette score optimization ----
        # Scan thresholds and pick the one with the best silhouette score.
        # Upper bound: 50th percentile of condensed distances.
        p50 = np.percentile(condensed, 50)
        t_max = max(10, min(int(p50), 200))

        best_sil, best_t = -1.0, -1
        for t_val in np.arange(2, t_max + 1, 1):
            labels = fcluster(Z, t_val, criterion="distance")
            n_uniq = len(np.unique(labels))
            if n_uniq <= 1 or n_uniq >= n:
                continue
            unique, counts = np.unique(labels, return_counts=True)
            non_single = (counts >= 2).sum()
            if non_single < 1:
                continue
            try:
                from sklearn.metrics import silhouette_score
                sil = silhouette_score(dist_sym, labels)
                if sil > best_sil:
                    best_sil, best_t = sil, t_val
            except Exception:
                continue

        if best_t < 0:
            best_t = t_max // 2  # fallback

        cluster_labels = fcluster(Z, best_t, criterion="distance")
        n_clusters_raw = len(set(cluster_labels))
        logger.info(
            f"TCRdist3 hierarchical: best_t={best_t:.1f}, "
            f"silhouette={best_sil:.4f}, {n_clusters_raw} raw clusters"
        )

        # ---- Filter: keep only clusters with size ≥ 2 ----
        # Paper approach: singletons are not considered "clustered" —
        # they are excluded from assignments (retention < 1.0).
        label_counts = Counter(cluster_labels)
        valid_labels = {lab for lab, cnt in label_counts.items() if cnt >= 2}

        # ---- Map clusters back to tcr_ids ----
        clone_keys = list(clone_to_tcr_ids.keys())
        assignments = []
        seen_pairs: set[tuple[str, str]] = set()
        n_valid_clusters = 0
        assigned_clones = 0

        # Renumber valid clusters sequentially
        label_remap = {}
        for i, label in enumerate(cluster_labels):
            if label not in valid_labels:
                continue
            if label not in label_remap:
                n_valid_clusters += 1
                label_remap[label] = n_valid_clusters

            new_label = label_remap[label]
            if i >= len(clone_keys):
                break
            ck = clone_keys[i]
            tcr_ids = clone_to_tcr_ids.get(ck, [])
            cluster_str = str(new_label)
            assigned_clones += 1
            for tid in tcr_ids:
                key = (tid, cluster_str)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    assignments.append({"tcr_id": tid, "cluster": cluster_str})

        logger.info(
            f"TCRdist3 mapped: {len(assignments)} tcr_id assignments "
            f"({n_valid_clusters} valid clusters, "
            f"{assigned_clones}/{n} clonotypes retained)"
        )

        # Tier-1/F2: cache pw_beta (TRB-only TCRdist distance) + a
        # tcr_id -> clonotype-index map, so the consensus fusion layer can
        # look up the structural distance for any pair without recomputing.
        # Matrix rows/cols correspond to deduplicated clonotypes in `records`
        # order == clone_keys order (clone_to_tcr_ids.keys()).
        try:
            self.last_pw_beta = pw_beta
            self.last_tcr_to_idx = {}
            for ci, ck in enumerate(clone_keys):
                for tid in clone_to_tcr_ids.get(ck, []):
                    self.last_tcr_to_idx[tid] = ci
            logger.info(
                f"TCRdist3 exposed pw_beta ({pw_beta.shape if pw_beta is not None else None}) "
                f"for {len(self.last_tcr_to_idx)} tcr_ids"
            )
        except Exception as e:  # never let fusion-exposure break clustering
            logger.warning(f"TCRdist3 pw_beta exposure failed: {e}")
            self.last_pw_beta = None
            self.last_tcr_to_idx = {}

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
                    cluster_id=f"tcrdist3_{int(item['cluster']):04d}",
                    membership_score=1.0,
                )
            )
        return assignments
