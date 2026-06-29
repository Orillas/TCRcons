"""DeepTCR wrapper adapter (Python API) — two default clustering options.

Option 1 (default): Hierarchical + silhouette threshold optimization
  - Ward linkage, distance criterion, silhouette-optimized threshold (t=0..99)
  - Produces many small, high-purity clusters (mostly singletons)
  - Best for fine-grained purity analysis

Option 2: Phenograph (Leiden modularity optimization, k=30)
  - Graph-based community detection via nearest-neighbor graph
  - Produces few large clusters (typically 20-30)
  - Best for broad structural grouping

Reference: https://github.com/sidhomj/DeepTCR
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseClusterer
from ..schema.records import ClusterAssignment

logger = logging.getLogger(__name__)

# Valid clustering option names
_CLUSTER_OPTIONS = frozenset({"option1", "option2"})

# Map named options to DeepTCR native method names
_OPTION_METHOD = {
    "option1": "hierarchical",  # hierarchical + silhouette
    "option2": "phenograph",    # native phenograph
}


class DeepTCRWrapper(BaseClusterer):
    """Wrapper around DeepTCR unsupervised clustering.

    Two default clustering options:
      Option 1 (default): Hierarchical + silhouette threshold optimization
      Option 2: Phenograph (Leiden modularity optimization)

    VAE parameters use DeepTCR v2.0 defaults (latent_dim=256, epochs_min=0,
    batch_size=10000). Dual-chain (alpha+beta) with V/J gene features,
    deduplicated by full clonotype, clusters mapped back via clonotype key.
    """

    name = "deeptcr"

    def __init__(
        self,
        clustering_method: str = "option1",
        latent_dim: int = 256,
        use_alpha: bool = True,
        use_v_beta: bool = True,
        use_j_beta: bool = True,
        use_v_alpha: bool = True,
        use_j_alpha: bool = True,
        n_threads: int = 4,
        epochs_min: int = 0,
        batch_size: int = 10000,
    ):
        self.clustering_method = str(clustering_method).lower()
        if self.clustering_method not in _CLUSTER_OPTIONS:
            raise ValueError(
                f"Unknown clustering method '{self.clustering_method}'. "
                f"Use: {sorted(_CLUSTER_OPTIONS)}"
            )
        self._resolved_method = _OPTION_METHOD[self.clustering_method]
        self.latent_dim = latent_dim
        self.use_alpha = use_alpha
        self.use_v_beta = use_v_beta
        self.use_j_beta = use_j_beta
        self.use_v_alpha = use_v_alpha
        self.use_j_alpha = use_j_alpha
        self.n_threads = n_threads
        self.epochs_min = epochs_min
        self.batch_size = batch_size

    @staticmethod
    def _add_allele(v: str) -> str:
        """Append *01 IMGT allele suffix if not already present."""
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

    def _clonotype_key(self, cdr3a, cdr3b, va, ja, vb, jb) -> str:
        return f"{va}_{cdr3a}_{ja}_{vb}_{cdr3b}_{jb}"

    def prepare_input(self, tcr_table: pd.DataFrame, config: dict) -> dict:
        """Prepare data for DeepTCR with full clonotype deduplication."""
        records: list[dict] = []
        clonotype_to_tcr_ids: dict[str, list[str]] = {}
        seen: set[str] = set()
        skipped = 0

        for _, r in tcr_table.iterrows():
            cdr3b = self._safe_str(r.get("cdr3_beta"))
            if not cdr3b:
                skipped += 1
                continue

            cdr3a = self._safe_str(r.get("cdr3_alpha"))
            vb = self._add_allele(self._safe_str(r.get("v_beta")))
            jb = self._add_allele(self._safe_str(r.get("j_beta")))
            va = self._add_allele(self._safe_str(r.get("v_alpha")))
            ja = self._add_allele(self._safe_str(r.get("j_alpha")))
            tcr_id = str(r.get("tcr_id", ""))

            ck = self._clonotype_key(cdr3a, cdr3b, va, ja, vb, jb)

            if ck not in seen:
                seen.add(ck)
                rec: dict = {"CDR3b": cdr3b}
                if self.use_alpha and cdr3a:
                    rec["CDR3a"] = cdr3a
                if self.use_v_beta and vb:
                    rec["v_beta"] = vb
                if self.use_j_beta and jb:
                    rec["j_beta"] = jb
                if self.use_v_alpha and va:
                    rec["v_alpha"] = va
                if self.use_j_alpha and ja:
                    rec["j_alpha"] = ja
                records.append(rec)
                clonotype_to_tcr_ids[ck] = []

            clonotype_to_tcr_ids[ck].append(tcr_id)

        total_tcrs = sum(len(v) for v in clonotype_to_tcr_ids.values())
        logger.info(
            f"DeepTCR input: {len(records)} unique clonotypes "
            f"from {total_tcrs} tcr_ids ({skipped} skipped, no CDR3b)"
        )

        return {
            "records": records,
            "clonotype_to_tcr_ids": clonotype_to_tcr_ids,
            "total_tcrs": total_tcrs,
        }

    def run(self, prepared_input: dict, workdir: Path) -> dict:
        """Execute DeepTCR_U: VAE training + native clustering on GPU."""
        records = prepared_input["records"]
        clonotype_map = prepared_input["clonotype_to_tcr_ids"]
        n = len(records)

        if n == 0:
            return {"assignments": []}

        # --- Set up CUDA library path for GPU, if nvidia pip packages exist locally ---
        import site as _site_mod
        import sys as _sys
        _sp_dirs = list(_site_mod.getsitepackages()) + [_site_mod.getusersitepackages()]
        _sp_dirs.append(
            os.path.join(
                _sys.prefix, "lib",
                f"python{_sys.version_info.major}.{_sys.version_info.minor}",
                "site-packages",
            )
        )
        _nvidia_pkgs = [
            "cublas", "cuda_cupti", "cuda_nvrtc", "cuda_runtime",
            "cudnn", "cufft", "curand", "cusolver", "cusparse", "nccl",
        ]
        nvidia_lib = ""
        for _sp in _sp_dirs:
            if all(os.path.isdir(os.path.join(_sp, "nvidia", _p, "lib")) for _p in _nvidia_pkgs):
                nvidia_lib = ":".join(
                    os.path.join(_sp, "nvidia", _p, "lib") for _p in _nvidia_pkgs
                )
                break
        if nvidia_lib:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{nvidia_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            )
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import subprocess as _sp
        try:
            _r = _sp.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5)
            _free = [int(x.strip()) for x in _r.stdout.strip().split(chr(10)) if x.strip()]
            if _free:
                _best = max(range(len(_free)), key=lambda i: _free[i])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(_best)
                logger.info(f"DeepTCR using GPU {_best} ({_free[_best]}MiB free)")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                logger.info("DeepTCR: no GPU detected, using CPU")
        except Exception:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        from DeepTCR.DeepTCR import DeepTCR_U

        data_dir = tempfile.mkdtemp()
        try:
            sample_dir = os.path.join(data_dir, "sample_0")
            os.makedirs(sample_dir, exist_ok=True)

            for rec in records:
                rec["cloneCount"] = 1

            input_df = pd.DataFrame(records)

            tsv_col_order = []
            if "CDR3a" in input_df.columns:
                tsv_col_order.append("CDR3a")
            tsv_col_order.append("CDR3b")
            tsv_col_order.append("cloneCount")
            if "v_alpha" in input_df.columns:
                tsv_col_order.append("v_alpha")
            if "j_alpha" in input_df.columns:
                tsv_col_order.append("j_alpha")
            if "v_beta" in input_df.columns:
                tsv_col_order.append("v_beta")
            if "j_beta" in input_df.columns:
                tsv_col_order.append("j_beta")
            input_df = input_df[tsv_col_order]

            tsv_path = os.path.join(sample_dir, "tcr.tsv")
            input_df.to_csv(tsv_path, sep="\t", index=False)

            logger.info(f"DeepTCR input TSV: {tsv_path} ({n} rows)")
            logger.info(f"  Columns: {list(input_df.columns)}")

            model_dir = os.path.join(data_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            dtn = DeepTCR_U(os.path.join(model_dir, "deeptcr"))

            load_kw: dict = {
                "directory": sample_dir,
                "Load_Prev_Data": False,
                "n_jobs": self.n_threads,
                "aa_column_beta": "CDR3b",
                "count_column": "cloneCount",
                "aggregate_by_aa": False,
            }
            if self.use_alpha and "CDR3a" in input_df.columns:
                load_kw["aa_column_alpha"] = "CDR3a"
            if self.use_v_beta and "v_beta" in input_df.columns:
                load_kw["v_beta_column"] = "v_beta"
            if self.use_j_beta and "j_beta" in input_df.columns:
                load_kw["j_beta_column"] = "j_beta"
            if self.use_v_alpha and "v_alpha" in input_df.columns:
                load_kw["v_alpha_column"] = "v_alpha"
            if self.use_j_alpha and "j_alpha" in input_df.columns:
                load_kw["j_alpha_column"] = "j_alpha"

            dtn.Get_Data(**load_kw)
            loaded = len(dtn.beta_sequences)
            logger.info(f"DeepTCR loaded {loaded} sequences (input was {n})")

            if loaded == 0:
                return {"assignments": []}

            dtn.Train_VAE(
                latent_dim=self.latent_dim,
                epochs_min=self.epochs_min,
                suppress_output=True,
                batch_size=min(self.batch_size, loaded),
            )
            logger.info("VAE training complete")

            # ---- Monkey-patch scipy lil_array for phenograph compat ----
            # scipy >=1.13 changed lil_array.rows from list to ndarray,
            # breaking phenograph's internal jaccard kernel.
            import scipy.sparse as _sp
            if hasattr(_sp, "lil_array"):
                _orig_tocsr = _sp.lil_array.tocsr
                def _patched_tocsr(self, copy=False):
                    if hasattr(self.rows, "tolist"):
                        self.rows = self.rows.tolist()
                    if hasattr(self.data, "tolist"):
                        self.data = self.data.tolist()
                    return _orig_tocsr(self, copy=copy)
                _sp.lil_array.tocsr = _patched_tocsr
                logger.info("Applied scipy lil_array.tocsr monkey-patch for phenograph compat")

            # ---- Use DeepTCR clustering (option1 or option2) ----
            _actual_method = self._resolved_method

            # For option1 (hierarchical + silhouette), apply the
            # hierarchical_optimization patch to handle t=0 edge case
            # where silhouette_score crashes when n_labels == n_samples.
            if self.clustering_method == "option1":
                import inspect as _inspect
                import DeepTCR.functions.utils_u as _utils_u
                import sklearn.metrics as _skm
                _orig_src = _inspect.getsource(_utils_u.hierarchical_optimization)
                if "if n_labels <= 1" not in _orig_src:
                    _orig_hier = _utils_u.hierarchical_optimization
                    def _patched_hierarchical(d, f, method="ward", criterion="distance"):
                        from scipy.cluster.hierarchy import linkage, fcluster
                        dd = copy.deepcopy(d)
                        Z = linkage(dd, method=method)
                        tl = np.arange(0, 100, 1)
                        sil = []
                        for t in tl:
                            IDX = fcluster(Z, t, criterion=criterion)
                            sel = IDX > 0
                            nl = len(np.unique(IDX[sel]))
                            ns = np.sum(sel)
                            if nl <= 1 or nl >= ns:
                                sil.append(-1.0)
                            else:
                                sil.append(_skm.silhouette_score(f[sel, :], IDX[sel]))
                        sil = np.array(sil)
                        topt = tl[np.argmax(sil)]
                        return fcluster(Z, topt, criterion=criterion)
                    _utils_u.hierarchical_optimization = _patched_hierarchical
                    import DeepTCR.DeepTCR as _dtcr_mod
                    _dtcr_mod.hierarchical_optimization = _patched_hierarchical
                    logger.info(
                        "Applied hierarchical_optimization patch "
                        "(silhouette edge case) for option1"
                    )

            try:
                dtn.Cluster(
                    set="all",
                    clustering_method=_actual_method,
                    n_jobs=self.n_threads,
                )
            except Exception as _clust_err:
                if self.clustering_method == "option2" and "Expected list, got numpy" in str(_clust_err):
                    logger.warning(
                        "Phenograph (option2) failed (scipy compat), "
                        "falling back to dbscan: %s",
                        _clust_err,
                    )
                    _actual_method = "dbscan"
                    dtn.Cluster(
                        set="all",
                        clustering_method=_actual_method,
                        n_jobs=self.n_threads,
                    )
                else:
                    raise

            n_clusters = len(set(dtn.Cluster_Assignments))
            logger.info(
                f"DeepTCR native clustering ({_actual_method}): "
                f"{n_clusters} clusters from {loaded} sequences"
            )

            # ---- Map clusters back to tcr_ids ----
            assignments = []
            ck_list = list(clonotype_map.keys())

            for i, cluster_num in enumerate(dtn.Cluster_Assignments):
                if i >= len(ck_list):
                    seq_a = ""
                    seq_b = dtn.beta_sequences[i] if i < len(dtn.beta_sequences) else ""
                    v = self._add_allele(
                        dtn.v_beta[i] if hasattr(dtn, "v_beta") and i < len(dtn.v_beta) else ""
                    )
                    j = self._add_allele(
                        dtn.j_beta[i] if hasattr(dtn, "j_beta") and i < len(dtn.j_beta) else ""
                    )
                    va = self._add_allele(
                        dtn.v_alpha[i] if hasattr(dtn, "v_alpha") and i < len(dtn.v_alpha) else ""
                    )
                    ja = self._add_allele(
                        dtn.j_alpha[i] if hasattr(dtn, "j_alpha") and i < len(dtn.j_alpha) else ""
                    )
                    if hasattr(dtn, "alpha_sequences") and dtn.alpha_sequences is not None and len(dtn.alpha_sequences) > 0:
                        seq_a = dtn.alpha_sequences[i] if i < len(dtn.alpha_sequences) else ""
                    ck = self._clonotype_key(seq_a, seq_b, va, ja, v, j)
                else:
                    ck = ck_list[i]

                tcr_ids = clonotype_map.get(ck, [])
                for tid in tcr_ids:
                    assignments.append({
                        "tcr_id": tid,
                        "cluster": int(cluster_num),
                    })

            logger.info(
                f"DeepTCR mapped {len(assignments)} tcr_id assignments "
                f"({n_clusters} clusters)"
            )
            return {"assignments": assignments}

        except Exception as e:
            logger.error(f"DeepTCR failed: {e}")
            raise RuntimeError(f"DeepTCR execution error: {e}") from e

        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def parse_output(self, workdir: Path) -> dict:
        return {}

    def normalize(self, raw_output: dict) -> list[ClusterAssignment]:
        assignments = []
        for item in raw_output.get("assignments", []):
            assignments.append(
                ClusterAssignment(
                    method=self.name,
                    tcr_id=str(item["tcr_id"]),
                    cluster_id=f"deeptcr_{item['cluster']:04d}",
                    membership_score=1.0,
                )
            )
        return assignments
