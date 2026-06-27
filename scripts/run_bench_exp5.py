#!/usr/bin/env python3
"""Exp5: Biological Case Study on benchmark dataset.

Top epitopes: KLGGALQAK (CMV), GILGFVFTL (Flu), NLVPMVATV (CMV), AVFDRKSDAK (EBV).
Per-epitope: cluster membership, method support, motif, V/J enrichment.
"""

import sys, time, logging, collections
from pathlib import Path
import numpy as np, pandas as pd

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


def get_clusterers():
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    for name, mod in [("clustcr", "clustcr_wrapper"), ("tcrdist3", "tcrdist3_wrapper"),
                      ("gliph2", "gliph2_wrapper")]:
        try:
            m = __import__(f"tcrconsensus.clusterers.{mod}", fromlist=["X"])
            cls = getattr(m, [c for c in dir(m) if "Wrapper" in c][0])
            clusterers[name] = cls()
        except: pass
    return clusterers


def run_case_study(epitope, df_epi, clusterers, config, out_dir):
    """Run detailed case study for one epitope."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tcrs = len(df_epi)

    # Normalize
    df_norm = normalize(df_epi.copy())
    tcr_ids = df_norm["tcr_id"].values

    # Run each method + consensus
    all_assignments = {}
    method_results = {}
    for mname, cl in clusterers.items():
        r = cl.safe_execute(df_norm, out_dir / mname, config)
        if r.status.value == "success" and r.assignments:
            all_assignments[mname] = r.assignments
            # Build cluster map
            clusters = {}
            for a in r.assignments:
                clusters.setdefault(a.cluster_id, []).append(a.tcr_id)
            method_results[mname] = clusters

    # Consensus
    consensus_clusters = {}
    if len(all_assignments) >= 2:
        methods = list(all_assignments.keys())
        all_a = []
        for a in all_assignments.values():
            all_a.extend(a)
        weights = compute_method_weights(methods, "balanced", config)
        clusters, edges = balanced_consensus(all_a, weights)
        clusters = refine(clusters, edges, config)

        for c in clusters:
            consensus_clusters[c.cluster_id] = list(c.member_ids)

    # Analysis
    results = {"epitope": epitope, "n_tcrs": n_tcrs}

    # Cluster stats per method
    for mname, cl_dict in method_results.items():
        n_cl = len(cl_dict)
        sizes = [len(v) for v in cl_dict.values()]
        results[f"{mname}_n_clusters"] = n_cl
        results[f"{mname}_max_size"] = max(sizes) if sizes else 0
        results[f"{mname}_mean_size"] = np.mean(sizes) if sizes else 0

    results["consensus_n_clusters"] = len(consensus_clusters)
    if consensus_clusters:
        sizes = [len(v) for v in consensus_clusters.values()]
        results["consensus_max_size"] = max(sizes)
        results["consensus_mean_size"] = np.mean(sizes)

    # V/J gene enrichment in consensus clusters
    if consensus_clusters:
        v_counts = collections.Counter()
        j_counts = collections.Counter()
        tcr_to_row = dict(zip(df_epi["tcr_id"], df_epi.itertuples()))
        for cid, members in consensus_clusters.items():
            for tid in members:
                row = tcr_to_row.get(tid)
                if row:
                    v = getattr(row, "v_beta", "")
                    j = getattr(row, "j_beta", "")
                    if v and str(v) != "nan":
                        v_counts[str(v).split("*")[0]] += 1
                    if j and str(j) != "nan":
                        j_counts[str(j).split("*")[0]] += 1

        results["top_v_genes"] = dict(v_counts.most_common(5))
        results["top_j_genes"] = dict(j_counts.most_common(5))

    # CDR3 motif analysis (simple k-mer)
    cdr3s = df_epi["cdr3_beta"].dropna().values
    kmer_counts = collections.Counter()
    for seq in cdr3s:
        seq = str(seq)
        for i in range(len(seq) - 3):
            kmer_counts[seq[i:i+4]] += 1
    results["top_4mers"] = dict(kmer_counts.most_common(10))

    # CDR3 length distribution
    lengths = [len(str(s)) for s in cdr3s if str(s) != "nan"]
    results["cdr3_len_mean"] = np.mean(lengths) if lengths else 0
    results["cdr3_len_std"] = np.std(lengths) if lengths else 0

    # Save per-cluster details
    if consensus_clusters:
        rows = []
        tcr_to_row = dict(zip(df_epi["tcr_id"], df_epi.itertuples()))
        for cid, members in consensus_clusters.items():
            for tid in members:
                row = tcr_to_row.get(tid)
                if row:
                    rows.append({
                        "cluster_id": cid,
                        "tcr_id": tid,
                        "cdr3_beta": getattr(row, "cdr3_beta", ""),
                        "v_beta": getattr(row, "v_beta", ""),
                        "j_beta": getattr(row, "j_beta", ""),
                    })
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "cluster_members.tsv", sep="\t", index=False)

    return results


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()._raw
    clusterers = get_clusterers()
    log.info(f"Clusterers: {list(clusterers.keys())}")

    # Load benchmark
    df = pd.read_csv(BENCHMARK_DIR / "benchmark_main.tsv", sep="\t", dtype=str)
    df = df.rename(columns={"CDR3_beta": "cdr3_beta", "V_beta": "v_beta", "J_beta": "j_beta", "Epitope": "epitope"})

    # Top epitopes for case study
    case_epitopes = ["KLGGALQAK", "GILGFVFTL", "NLVPMVATV", "AVFDRKSDAK", "RAKFKQLL", "GLCTLVAML"]

    all_results = []
    for epitope in case_epitopes:
        df_epi = df[df["epitope"] == epitope].copy()
        if len(df_epi) < 20:
            log.info(f"Skipping {epitope}: only {len(df_epi)} TCRs")
            continue

        log.info(f"\n{'='*50}")
        log.info(f"Case study: {epitope} ({len(df_epi)} TCRs)")
        log.info(f"{'='*50}")

        r = run_case_study(epitope, df_epi, clusterers, config, OUT_DIR / epitope)
        all_results.append(r)

        log.info(f"  Consensus clusters: {r.get('consensus_n_clusters', 'N/A')}")
        log.info(f"  Top V genes: {r.get('top_v_genes', {})}")
        log.info(f"  CDR3 length: {r.get('cdr3_len_mean',0):.1f} ± {r.get('cdr3_len_std',0):.1f}")

    # Summary
    res = pd.DataFrame(all_results)
    res.to_csv(OUT_DIR / "case_study_summary.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("EXP5: CASE STUDY RESULTS")
    print("=" * 80)
    for r in all_results:
        print(f"\n{r['epitope']} ({r['n_tcrs']} TCRs):")
        print(f"  Consensus clusters: {r.get('consensus_n_clusters', 'N/A')}")
        print(f"  Top V: {r.get('top_v_genes', {})}")
        print(f"  Top 4-mers: {r.get('top_4mers', {})}")
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    run()
