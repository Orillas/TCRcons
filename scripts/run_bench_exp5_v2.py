#!/usr/bin/env python3
"""Exp5: Biological Case Study on core benchmark (4,779 pairs).

For 6 well-studied epitopes: per-cluster membership, method support,
V/J enrichment, 4-mer motifs, CDR3 length distribution.
"""

import sys, time, logging, collections
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.refinement.refiner import refine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/benchmark_data")
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/bench_exp5")
CASE_EPITOPES = ["GILGFVFTL", "NLVPMVATV", "GLCTLVAML", "YLQPRTFLL", "LLWNGPMAV", "TTDPSFLGRY"]


def get_clusterers():
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    for name, mod in [("clustcr", "clustcr_wrapper"), ("gliph2", "gliph2_wrapper")]:
        try:
            m = __import__(f"tcrconsensus.clusterers.{mod}", fromlist=[name.title().replace("_","")+"Wrapper"])
            cls = getattr(m, [c for c in dir(m) if "Wrapper" in c][0])
            clusterers[name] = cls()
        except: pass
    return clusterers


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()._raw
    clusterers = get_clusterers()
    log.info(f"Clusterers: {list(clusterers.keys())}")

    df = pd.read_csv(BENCHMARK_DIR / "benchmark_core_4779.tsv", sep="\t", dtype=str)
    rename_map = {"CDR3_beta": "cdr3_beta", "V_beta": "v_beta", "J_beta": "j_beta",
                  "CDR3_alpha": "cdr3_alpha", "V_alpha": "v_alpha", "J_alpha": "j_alpha",
                  "Epitope": "epitope"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for epitope in CASE_EPITOPES:
        epi_df = df[df["epitope"] == epitope].copy()
        if len(epi_df) < 10:
            log.warning(f"Skipping {epitope}: only {len(epi_df)} TCRs")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Case Study: {epitope} ({len(epi_df)} TCRs)")
        log.info(f"{'='*60}")

        epi_dir = OUT_DIR / epitope
        epi_dir.mkdir(parents=True, exist_ok=True)

        df_norm = normalize(epi_df.copy())
        tcr_ids = df_norm["tcr_id"].values

        # Run each method
        method_clusters = {}
        for mname, clusterer in clusterers.items():
            r = clusterer.safe_execute(df_norm, epi_dir / mname, config)
            if r.status.value == "success" and r.assignments:
                label_map = {}
                for a in r.assignments:
                    if a.tcr_id not in label_map:
                        label_map[a.tcr_id] = a.cluster_id
                method_clusters[mname] = label_map
                log.info(f"  {mname}: {len(set(label_map.values()))} clusters")

        # Run consensus
        all_a = []
        for mname, clusterer in clusterers.items():
            r = clusterer.safe_execute(df_norm, epi_dir / "consensus" / mname, config)
            if r.status.value == "success" and r.assignments:
                all_a.extend(r.assignments)
        if len(all_a) >= 2:
            methods = list(set(a.method for a in all_a))
            weights = compute_method_weights(methods, "balanced", config)
            clusters, edges = balanced_consensus(all_a, weights)
            clusters = refine(clusters, edges, config)

            label_map = {}
            for c in clusters:
                for mid in c.member_ids:
                    label_map[mid] = c.cluster_id
            method_clusters["consensus"] = label_map
            log.info(f"  consensus: {len(clusters)} clusters")

        # Analyze each consensus cluster
        records = []
        for method, label_map in method_clusters.items():
            for cid in set(label_map.values()):
                members = [tid for tid, c in label_map.items() if c == cid]
                member_df = epi_df[epi_df["tcr_id"].isin(members)]

                # V/J gene enrichment
                v_counts = member_df["v_beta"].value_counts().head(3) if "v_beta" in member_df.columns else pd.Series()
                j_counts = member_df["j_beta"].value_counts().head(3) if "j_beta" in member_df.columns else pd.Series()

                # CDR3 length distribution
                cdr3_lens = member_df["cdr3_beta"].str.len().value_counts().sort_index()

                # 4-mer motifs
                motifs = Counter()
                for seq in member_df["cdr3_beta"].dropna():
                    for i in range(len(seq) - 3):
                        motifs[seq[i:i+4]] += 1
                top_motifs = motifs.most_common(5)

                records.append({
                    "method": method, "cluster_id": cid, "n_members": len(members),
                    "top_V": v_counts.index[0] if len(v_counts) > 0 else "NA",
                    "top_V_freq": v_counts.iloc[0] if len(v_counts) > 0 else 0,
                    "top_J": j_counts.index[0] if len(j_counts) > 0 else "NA",
                    "mean_cdr3_len": member_df["cdr3_beta"].str.len().mean() if "cdr3_beta" in member_df.columns else 0,
                    "top_4mer": top_motifs[0][0] if top_motifs else "NA",
                    "top_4mer_count": top_motifs[0][1] if top_motifs else 0,
                })

        cluster_df = pd.DataFrame(records)
        cluster_df.to_csv(epi_dir / f"{epitope}_clusters.tsv", sep="\t", index=False)

        # CDR3 length distribution
        len_dist = epi_df.copy()
        len_dist["cdr3_beta_len"] = len_dist["cdr3_beta"].str.len()
        len_dist["cdr3_alpha_len"] = len_dist["cdr3_alpha"].str.len()
        len_dist[["tcr_id", "cdr3_beta_len", "cdr3_alpha_len"]].to_csv(
            epi_dir / f"{epitope}_cdr3_lengths.tsv", sep="\t", index=False)

        log.info(f"  Saved cluster analysis to {epi_dir}")

    print("\n" + "=" * 80)
    print("EXP5: CASE STUDY COMPLETE (core benchmark)")
    print("=" * 80)
    print(f"Results in: {OUT_DIR}")


if __name__ == "__main__":
    run()
