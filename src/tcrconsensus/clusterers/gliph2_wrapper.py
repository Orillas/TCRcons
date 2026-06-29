"""GLIPH2 wrapper adapter (irtools.centos binary) — matching paper approach.

Paper approach:
  - Input: CDR3b, TRBV, TRBJ, CDR3a, Subject_Condition, count (tab-separated)
  - Algorithm: motif-based clustering with Fisher's exact test
  - Output: one line per cluster with motif + space-separated CDR3b members
  - Singletons are filtered out (pattern != "single")

Uses the compiled GLIPH2 binary (irtools.centos) from the clusTCR package
with the GLIPH2 v2.0 reference database (ref_CD8_v2.0.fa).

Reference: Huang et al., Nat. Biotechnol. 2020
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .base import BaseClusterer
from ..schema.records import ClusterAssignment

logger = logging.getLogger(__name__)

def _gliph2_lib_from_clustcr() -> str | None:
    """Locate the irtools binary + reference bundle shipped inside the clusTCR
    package, if clusTCR is installed.

    clusTCR bundles ``irtools.centos`` and the GLIPH2 v2.0 reference files under
    ``clustcr/modules/gliph2/lib/``. Using that copy avoids redistributing GLIPH2
    separately (it carries a restrictive academic license).
    """
    try:
        import clustcr  # noqa: F401
    except Exception:
        return None
    cand = Path(clustcr.__file__).resolve().parent / "modules" / "gliph2" / "lib"
    if cand.is_dir() and any(p.name.startswith("irtools") for p in cand.iterdir()):
        return str(cand)
    return None


def _resolve_gliph2_lib(gliph2_lib: str | None) -> str:
    """Resolve the GLIPH2 lib directory (param > env > clusTCR bundle > error).

    The lib directory must contain the compiled ``irtools`` binary and the
    GLIPH2 v2.0 reference files (``ref_CD8_v2.0.fa`` etc.).
    """
    lib = gliph2_lib or os.environ.get("TCR_GLIPH2_LIB")
    if not lib:
        from ..backends import gliph2_lib_path
        cand = gliph2_lib_path()
        if cand.is_dir() and any(p.name.startswith("irtools") for p in cand.iterdir()):
            lib = str(cand)
    if not lib:
        lib = _gliph2_lib_from_clustcr()
    if not lib:
        raise FileNotFoundError(
            "GLIPH2 lib directory not configured. Run `tcrconsensus install-backends "
            "--gliph2`, set TCR_GLIPH2_LIB to a directory holding the irtools binary "
            "and GLIPH2 v2.0 reference files, or pass gliph2_lib=... to GLIPH2Wrapper."
        )
    return lib


class GLIPH2Wrapper(BaseClusterer):
    """Wrapper around GLIPH2 via compiled irtools.centos binary."""

    name = "gliph2"

    def __init__(
        self,
        gliph2_lib: str | None = None,
        lcminp: float = 0.001,
        p_depth: int = 1000,
        kmer_min_depth: int = 3,
        local_min_OVE: int = 10,
        refer_file: str = "ref_CD8_v2.0.fa",
    ):
        self.gliph2_lib = gliph2_lib
        self.lcminp = lcminp
        self.p_depth = p_depth
        self.kmer_min_depth = kmer_min_depth
        self.local_min_OVE = local_min_OVE
        self.refer_file = refer_file

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def prepare_input(self, tcr_table: pd.DataFrame, config: dict) -> dict:
        """Build CDR3b→tcr_ids mapping for GLIPH2 clustering.

        GLIPH2 clusters beta-chain CDR3s by motif. Alpha chain info
        is passed through input but not used for clustering.
        """
        # Resolve column names (case-insensitive)
        beta_col = alpha_col = v_beta_col = j_beta_col = None
        for col in tcr_table.columns:
            low = col.lower()
            if low == "cdr3_beta":
                beta_col = col
            elif low == "cdr3_alpha":
                alpha_col = col
            elif low in ("v_beta", "trbv"):
                v_beta_col = col
            elif low in ("j_beta", "trbj"):
                j_beta_col = col

        if not beta_col:
            raise ValueError("cdr3_beta column required for GLIPH2")

        # Build CDR3b → tcr_ids mapping (one CDR3b can map to multiple tcr_ids)
        cdr3b_to_tcr_ids: dict[str, list[str]] = {}
        # Also track full records for input file
        records: dict[str, dict] = {}
        seen = set()

        for _, r in tcr_table.iterrows():
            cdr3b = str(r.get(beta_col, "")).strip()
            if not cdr3b:
                continue
            tcr_id = str(r.get("tcr_id", ""))

            if cdr3b not in cdr3b_to_tcr_ids:
                cdr3b_to_tcr_ids[cdr3b] = []
            if tcr_id not in cdr3b_to_tcr_ids[cdr3b]:
                cdr3b_to_tcr_ids[cdr3b].append(tcr_id)

            # Build record (one per unique CDR3b)
            if cdr3b not in seen:
                seen.add(cdr3b)
                cdr3a = str(r.get(alpha_col, "")).strip() if alpha_col else ""
                vb = str(r.get(v_beta_col, "")).strip() if v_beta_col else ""
                jb = str(r.get(j_beta_col, "")).strip() if j_beta_col else ""
                records[cdr3b] = {
                    "CDR3b": cdr3b,
                    "TRBV": vb,
                    "TRBJ": jb,
                    "CDR3a": cdr3a if cdr3a else "NA",
                    "Subject": "database:CD8+",
                    "count": "1",
                }

        total = sum(len(v) for v in cdr3b_to_tcr_ids.values())
        logger.info(
            f"GLIPH2 input: {len(records)} unique CDR3b sequences "
            f"({total} tcr_id associations)"
        )
        return {
            "records": records,
            "cdr3b_to_tcr_ids": cdr3b_to_tcr_ids,
        }

    def run(self, prepared_input: dict, workdir: Path) -> dict:
        """Execute GLIPH2 via irtools.centos binary."""
        records = prepared_input["records"]
        cdr3b_to_tcr_ids = prepared_input["cdr3b_to_tcr_ids"]

        if not records:
            return {"assignments": []}
        workdir.mkdir(parents=True, exist_ok=True)
        self.gliph2_lib = _resolve_gliph2_lib(self.gliph2_lib)

        # ---- Write parameters file ----
        param_path = os.path.join(self.gliph2_lib, "parameters_tcrconsensus")
        params = f"""# tcrconsensus GLIPH2 parameters
out_prefix=tcrconsensus_output
cdr3_file=tcrconsensus_input.txt
refer_file={self.refer_file}
v_usage_freq_file=ref_V_CD48_v2.0.txt
cdr3_length_freq_file=ref_L_CD48_v2.0.txt
local_min_pvalue={self.lcminp}
p_depth={self.p_depth}
global_convergence_cutoff=1
simulation_depth=1000
kmer_min_depth={self.kmer_min_depth}
local_min_OVE={self.local_min_OVE}
algorithm=GLIPH2
all_aa_interchangeable=1
"""
        with open(param_path, "w") as f:
            f.write(params)

        # ---- Write input file ----
        # Format: CDR3b TAB TRBV TAB TRBJ TAB CDR3a TAB Subject_Condition TAB count
        input_path = os.path.join(self.gliph2_lib, "tcrconsensus_input.txt")
        lines = []
        for cdr3b, rec in records.items():
            line = "\t".join([
                rec["CDR3b"], rec["TRBV"], rec["TRBJ"],
                rec["CDR3a"], rec["Subject"], rec["count"],
            ])
            lines.append(line)
        with open(input_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"GLIPH2 input: {input_path} ({len(lines)} CDR3b)")

        # ---- Run irtools.centos ----
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.gliph2_lib)
            result = subprocess.run(
                ["./irtools.centos", "-c", "parameters_tcrconsensus"],
                capture_output=True, text=True, timeout=3600,
            )
            (workdir / "gliph2_stdout.log").write_text(result.stdout)
            (workdir / "gliph2_stderr.log").write_text(result.stderr)

            if result.returncode != 0:
                raise RuntimeError(f"GLIPH2 failed (rc={result.returncode}): {result.stderr[:500]}")
        finally:
            os.chdir(orig_cwd)

        # ---- Parse output ----
        output_file = os.path.join(self.gliph2_lib, "tcrconsensus_output_cluster.txt")
        if not os.path.exists(output_file):
            logger.warning("GLIPH2 produced no output file")
            return {"assignments": []}

        assignments = []
        n_clusters = 0
        n_singletons = 0
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                # Format: pvalue ? size motif/type member1 member2 ...
                cluster_type = parts[3]
                if cluster_type == "single":
                    n_singletons += 1
                    continue

                # cluster members are CDR3b sequences
                members = parts[4:]
                cluster_id = str(n_clusters)
                n_clusters += 1

                for cdr3b in members:
                    tcr_ids = cdr3b_to_tcr_ids.get(cdr3b, [cdr3b])
                    for tid in tcr_ids:
                        assignments.append({
                            "tcr_id": tid,
                            "cluster": cluster_id,
                        })

        logger.info(
            f"GLIPH2: {n_clusters} clusters, {len(assignments)} tcr_id "
            f"assignments, {n_singletons} singletons filtered"
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
                    cluster_id=f"gliph2_{int(item['cluster']):04d}",
                    membership_score=1.0,
                )
            )
        return assignments
