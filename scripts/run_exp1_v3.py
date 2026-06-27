#!/usr/bin/env python3
"""Experiment 1: Cross-Benchmark Comparison on high-confidence v3 benchmark.

Runs ALL 7 single methods (HD, clusTCR, TCRdist3, GLIPH2, GIANA, TCRMatch, DeepTCR)
+ consensus modes + simple baselines on the paper-methodology filtered dataset.

Uses Epitope column as ground-truth labels for evaluation.
"""

import copy
import json
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")  # for clustcr import

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import (
    compute_all_metrics, per_epitope_pairwise_f1, per_epitope_metrics,
    paired_bootstrap_ci, wilcoxon_test,
)
from tcrconsensus.schema.records import ConsensusEdge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clusterer registry — ALL 7 METHODS
# ---------------------------------------------------------------------------

def get_all_clusterers():
    """Get all 7 available clusterers."""
    clusterers = {"hd_baseline": HDBaselineClusterer()}

    # 1. clusTCR
    try:
        from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
        clusterers["clustcr"] = ClusTCRWrapper()
        logger.info("  ✅ clusTCR registered")
    except Exception as e:
        logger.warning(f"  ❌ clusTCR unavailable: {e}")

    # 2. TCRdist3
    try:
        from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
        clusterers["tcrdist3"] = TCRDist3Wrapper()
        logger.info("  ✅ TCRdist3 registered")
    except Exception as e:
        logger.warning(f"  ❌ TCRdist3 unavailable: {e}")

    # 3. GLIPH2
    try:
        from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
        clusterers["gliph2"] = GLIPH2Wrapper()
        logger.info("  ✅ GLIPH2 registered")
    except Exception as e:
        logger.warning(f"  ❌ GLIPH2 unavailable: {e}")

    # 4. GIANA
    try:
        from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
        clusterers["giana"] = GIANAWrapper(giana_script="/home/jilin/DeepTCR/GIANA/GIANA4.1.py")
        logger.info("  ✅ GIANA registered")
    except Exception as e:
        logger.warning(f"  ❌ GIANA unavailable: {e}")

    # 5. TCRMatch
    try:
        from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
        clusterers["tcrmatch"] = TCRMatchWrapper(tcrmatch_bin="/home/jilin/DeepTCR/TCRMatch/tcrmatch")
        logger.info("  ✅ TCRMatch registered")
    except Exception as e:
        logger.warning(f"  ❌ TCRMatch unavailable: {e}")

    # 6. DeepTCR
    try:
        from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper
        clusterers["deeptcr"] = DeepTCRWrapper()
        logger.info("  ✅ DeepTCR registered")
    except Exception as e:
        logger.warning(f"  ❌ DeepTCR unavailable: {e}")

    return clusterers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clusters_to_labels(clusters, tcr_ids):
    """Convert cluster list to per-TCR label array."""
    label_map = {}
    for cluster in clusters:
        for mid in cluster.member_ids:
            label_map[mid] = cluster.cluster_id
    return np.array([label_map.get(tid, -1) for tid in tcr_ids])


def run_single_method(df, clusterer, workdir, config):
    """Run a single clusterer and return assignments."""
    result = clusterer.safe_execute(df, workdir, config)
    return result.assignments, result.status.value, result.runtime_seconds


def run_consensus(assignments, weights, mode, config, skip_refinement=False):
    """Run consensus clustering with given mode."""
    cfg = config.get("consensus", {})

    if mode == "conservative":
        clusters, edges = conservative_consensus(
            assignments, weights, **cfg.get("conservative", {}))
    elif mode == "coverage":
        clusters, edges = coverage_consensus(
            assignments, weights, **cfg.get("coverage", {}))
    else:
        clusters, edges = balanced_consensus(
            assignments, weights, **cfg.get("balanced", {}))

    if not skip_refinement:
        clusters = refine(clusters, edges, config)

    return clusters, edges


# ---------------------------------------------------------------------------
# Simple baselines
# ---------------------------------------------------------------------------

def run_majority_vote(df, all_assignments, workdir, config):
    """Majority vote: equal weights, balanced consensus."""
    methods = list(set(a.method for a in all_assignments))
    weights = {m: 1.0 / len(methods) for m in methods}
    clusters, edges = run_consensus(all_assignments, weights, "balanced", config)
    return clusters


def run_intersection_only(all_assignments):
    """Intersection: only link pairs supported by ALL methods."""
    method_set = set(a.method for a in all_assignments)
    n_methods = len(method_set)
    if n_methods < 2:
        return []

    # Count method support per pair
    method_clusters = defaultdict(lambda: defaultdict(set))
    for a in all_assignments:
        method_clusters[a.method][a.cluster_id].add(a.tcr_id)

    pair_support = defaultdict(set)
    for method, clusters in method_clusters.items():
        for cid, members in clusters.items():
            if len(members) < 2:
                continue
            members_list = sorted(members)
            for i in range(len(members_list)):
                for j in range(i + 1, len(members_list)):
                    pair_support[(members_list[i], members_list[j])].add(method)

    edges = []
    for (a, b), methods in pair_support.items():
        if len(methods) >= n_methods:
            edges.append(ConsensusEdge(
                tcr_id_a=a, tcr_id_b=b,
                method_support_count=len(methods),
                weighted_support=float(len(methods)),
                final_score=float(len(methods)),
            ))

    if not edges:
        return []

    graph = build_consensus_graph(edges, threshold=0.1)
    return connected_components_clustering(graph)


def run_union_only(all_assignments, config):
    """Union: link pairs from ANY single method."""
    methods = list(set(a.method for a in all_assignments))
    weights = {m: 1.0 for m in methods}
    clusters, edges = run_consensus(all_assignments, weights, "coverage", config)
    return clusters


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_method(pred_labels, true_labels, tcr_ids, n_total, dataset_name, method_name):
    """Evaluate a method's predictions against true labels."""
    valid_mask = np.array([
        p != -1 and str(p) != "-1" and str(p) != ""
        for p in pred_labels
    ], dtype=bool)

    if valid_mask.sum() < 2:
        logger.warning(f"  {method_name}: < 2 valid predictions, skipping evaluation")
        return None

    le_true = LabelEncoder()
    le_pred = LabelEncoder()

    true_str = np.array(true_labels)[valid_mask]
    pred_str = np.array(pred_labels)[valid_mask].astype(str)

    le_true.fit(np.unique(true_labels[valid_mask]))
    le_pred.fit(pred_str)

    true_enc = le_true.transform(true_labels[valid_mask])
    pred_enc = le_pred.transform(pred_str)

    metrics = compute_all_metrics(pred_enc, true_enc, n_total)
    metrics["dataset"] = dataset_name
    metrics["method"] = method_name

    # Per-epitope F1
    epi_f1 = per_epitope_pairwise_f1(pred_enc, true_enc)
    if epi_f1 is not None:
        epi_f1["dataset"] = dataset_name
        epi_f1["method"] = method_name

    return metrics, epi_f1


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(benchmark_path, output_dir):
    """Run cross-benchmark experiment on v3 high-confidence dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    cfg = config._raw

    # ------------------------------------------------------------------
    # Load benchmark data
    # ------------------------------------------------------------------
    logger.info(f"Loading benchmark: {benchmark_path}")
    df = pd.read_csv(benchmark_path, sep="\t", dtype=str)

    # Use Epitope as ground truth
    epitope_col = "Epitope"
    df = df[df[epitope_col].notna()].copy()
    df = df.reset_index(drop=True)

    n_total = len(df)
    tcr_ids = df["tcr_id"].values
    true_labels = df[epitope_col].values
    n_epitopes = df[epitope_col].nunique()
    n_pairs = df[["CDR3_beta", "CDR3_alpha"]].drop_duplicates().shape[0]

    logger.info(f"  Dataset: {n_total} rows, {n_pairs} unique pairs, {n_epitopes} epitopes")
    logger.info(f"  CDR3b unique: {df['CDR3_beta'].nunique()}, CDR3a unique: {df['CDR3_alpha'].nunique()}")

    # Normalize column names to lowercase for wrappers
    # (normalize() adds lowercase canonical cols that shadow uppercase originals)
    rename_lower = {}
    for col in df.columns:
        low = col.lower()
        if low != col and low in ["cdr3_alpha", "cdr3_beta", "v_alpha", "v_beta",
                                   "j_alpha", "j_beta", "tcr_id", "epitope"]:
            rename_lower[col] = low
    if rename_lower:
        logger.info(f"  Renaming columns to lowercase: {list(rename_lower.keys())}")
        df = df.rename(columns=rename_lower)
    df_norm = normalize(df.copy())

    # Filter to epitopes with >= 5 TCRs (statistical power requirement)
    epi_counts = pd.Series(true_labels).value_counts()
    valid_epis = set(epi_counts[epi_counts >= 5].index)
    logger.info(f"  Epitopes with >=5 TCRs: {len(valid_epis)}/{n_epitopes}")

    workdir = output_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Register all 7 clusterers
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("REGISTERING CLUSTERERS")
    logger.info("=" * 60)
    clusterers = get_all_clusterers()
    logger.info(f"Total registered: {len(clusterers)} methods: {list(clusterers.keys())}")

    all_results = []
    all_epi_f1 = []

    # ------------------------------------------------------------------
    # Step 1: Run all 7 individual methods
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: RUNNING 7 INDIVIDUAL METHODS")
    logger.info("=" * 60)

    ds_assignments = {}
    method_stats = []

    for mname, clusterer in clusterers.items():
        logger.info(f"\n--- {mname} ---")
        try:
            t0 = time.time()
            assignments, status, runtime = run_single_method(df_norm, clusterer, workdir, cfg)
            elapsed = time.time() - t0

            n_clusters = len(set(a.cluster_id for a in assignments))
            n_assigned = len(assignments)
            logger.info(f"  Result: {n_assigned} assignments, {n_clusters} clusters, status={status}, {elapsed:.1f}s")

            method_stats.append({
                "method": mname,
                "status": status,
                "assignments": n_assigned,
                "clusters": n_clusters,
                "runtime_s": round(elapsed, 1),
            })

            if status == "success" and assignments:
                ds_assignments[mname] = assignments

                # Build per-TCR label from assignments
                label_map = {}
                for a in assignments:
                    if a.tcr_id not in label_map:  # keep first cluster per TCR
                        label_map[a.tcr_id] = a.cluster_id
                pred_labels = np.array([label_map.get(tid, -1) for tid in tcr_ids])

                result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, "benchmark_v3", mname)
                if result:
                    metrics, epi_f1_df = result
                    metrics["runtime_s"] = elapsed
                    all_results.append(metrics)
                    if epi_f1_df is not None:
                        all_epi_f1.append(epi_f1_df)
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            method_stats.append({
                "method": mname,
                "status": "error",
                "error": str(e),
            })

    # ------------------------------------------------------------------
    # Print method summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INDIVIDUAL METHOD RESULTS")
    print("=" * 70)
    print(f"{'Method':<15} {'Status':<10} {'Clusters':>8} {'Assign':>8} {'Time(s)':>8}")
    print("-" * 70)
    for s in method_stats:
        print(f"{s['method']:<15} {s.get('status','?'):<10} {s.get('clusters',0):>8} {s.get('assignments',0):>8} {s.get('runtime_s',0):>8.1f}")

    # ------------------------------------------------------------------
    # Step 2: Run consensus modes
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: RUNNING CONSENSUS MODES")
    logger.info("=" * 60)

    if len(ds_assignments) >= 2:
        all_method_assignments = []
        for assignments in ds_assignments.values():
            all_method_assignments.extend(assignments)

        available_methods = list(ds_assignments.keys())
        logger.info(f"Consensus using {len(available_methods)} methods: {available_methods}")

        for mode in ["conservative", "balanced", "coverage"]:
            logger.info(f"\n--- consensus_{mode} ---")
            try:
                t0 = time.time()
                weights = compute_method_weights(available_methods, "balanced", cfg)
                clusters, edges = run_consensus(all_method_assignments, weights, mode, cfg)
                elapsed = time.time() - t0
                logger.info(f"  {len(clusters)} clusters, {elapsed:.1f}s")

                pred_labels = clusters_to_labels(clusters, tcr_ids)
                result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, "benchmark_v3", f"consensus_{mode}")
                if result:
                    metrics, epi_f1_df = result
                    metrics["runtime_s"] = elapsed
                    all_results.append(metrics)
                    if epi_f1_df is not None:
                        all_epi_f1.append(epi_f1_df)
            except Exception as e:
                logger.error(f"  consensus_{mode} FAILED: {e}")

        # ------------------------------------------------------------------
        # Step 3: Simple baselines
        # ------------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: RUNNING BASELINES")
        logger.info("=" * 60)

        # Majority vote
        logger.info(f"\n--- majority_vote ---")
        try:
            t0 = time.time()
            clusters = run_majority_vote(df_norm, all_method_assignments, workdir, cfg)
            elapsed = time.time() - t0
            pred_labels = clusters_to_labels(clusters, tcr_ids)
            result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, "benchmark_v3", "majority_vote")
            if result:
                metrics, epi_f1_df = result
                metrics["runtime_s"] = elapsed
                all_results.append(metrics)
                if epi_f1_df is not None:
                    all_epi_f1.append(epi_f1_df)
        except Exception as e:
            logger.error(f"  majority_vote FAILED: {e}")

        # Intersection only
        logger.info(f"\n--- intersection_only ---")
        try:
            t0 = time.time()
            clusters = run_intersection_only(all_method_assignments)
            elapsed = time.time() - t0
            logger.info(f"  {len(clusters)} clusters, {elapsed:.1f}s")
            pred_labels = clusters_to_labels(clusters, tcr_ids)
            result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, "benchmark_v3", "intersection_only")
            if result:
                metrics, epi_f1_df = result
                metrics["runtime_s"] = elapsed
                all_results.append(metrics)
                if epi_f1_df is not None:
                    all_epi_f1.append(epi_f1_df)
        except Exception as e:
            logger.error(f"  intersection_only FAILED: {e}")

        # Union only
        logger.info(f"\n--- union_only ---")
        try:
            t0 = time.time()
            clusters = run_union_only(all_method_assignments, cfg)
            elapsed = time.time() - t0
            logger.info(f"  {len(clusters)} clusters, {elapsed:.1f}s")
            pred_labels = clusters_to_labels(clusters, tcr_ids)
            result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, "benchmark_v3", "union_only")
            if result:
                metrics, epi_f1_df = result
                metrics["runtime_s"] = elapsed
                all_results.append(metrics)
                if epi_f1_df is not None:
                    all_epi_f1.append(epi_f1_df)
        except Exception as e:
            logger.error(f"  union_only FAILED: {e}")
    else:
        logger.warning(f"Only {len(ds_assignments)} methods succeeded — skipping consensus (need >= 2)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    if all_epi_f1:
        epi_f1_df = pd.concat(all_epi_f1, ignore_index=True)
    else:
        epi_f1_df = pd.DataFrame()

    results_df.to_csv(output_dir / "per_method_detail.tsv", sep="\t", index=False)
    if not epi_f1_df.empty:
        epi_f1_df.to_csv(output_dir / "per_epitope_f1.tsv", sep="\t", index=False)

    # Method stats
    pd.DataFrame(method_stats).to_csv(output_dir / "method_stats.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # Statistical tests: consensus vs strongest single method
    # ------------------------------------------------------------------
    stat_tests = []
    if not epi_f1_df.empty:
        single_methods = [m for m in epi_f1_df["method"].unique()
                         if not m.startswith("consensus")
                         and m not in ("majority_vote", "intersection_only", "union_only")]

        if single_methods:
            medians = {m: epi_f1_df[epi_f1_df["method"] == m]["f1"].median()
                      for m in single_methods if m in epi_f1_df["method"].values}
            if medians:
                strongest = max(medians, key=medians.get)
                logger.info(f"\nStrongest single method: {strongest} (median F1={medians[strongest]:.4f})")

                for mode in ["consensus_balanced", "consensus_conservative", "consensus_coverage"]:
                    if mode not in epi_f1_df["method"].values:
                        continue
                    cons_f1 = epi_f1_df[epi_f1_df["method"] == mode].set_index("epitope")["f1"]
                    best_f1 = epi_f1_df[epi_f1_df["method"] == strongest].set_index("epitope")["f1"]
                    common = cons_f1.index.intersection(best_f1.index)
                    if len(common) < 5:
                        continue

                    cons_vals = cons_f1.loc[common].values
                    best_vals = best_f1.loc[common].values
                    pb = paired_bootstrap_ci(cons_vals, best_vals)
                    wt = wilcoxon_test(cons_vals, best_vals)

                    stat_tests.append({
                        "dataset": "benchmark_v3",
                        "comparison": f"{mode} vs {strongest}",
                        "strongest_method": strongest,
                        "consensus_mode": mode,
                        "mean_delta_f1": pb["mean_diff"],
                        "ci_low": pb["ci_low"],
                        "ci_high": pb["ci_high"],
                        "p_positive": pb["p_positive"],
                        "wilcoxon_stat": wt["statistic"],
                        "wilcoxon_p": wt["p_value"],
                        "n_epitopes": len(common),
                    })

    stat_df = pd.DataFrame(stat_tests)
    if not stat_df.empty:
        stat_df.to_csv(output_dir / "statistical_tests.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # Print final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: CROSS-BENCHMARK RESULTS (v3 High-Confidence Dataset)")
    print("=" * 80)

    if not results_df.empty:
        print(f"\n{'Method':<25} {'ARI':>7} {'AMI':>7} {'Purity':>7} {'Sensitivity':>11} {'Retention':>9} {'F1':>7} {'Time(s)':>8}")
        print("-" * 90)
        sorted_res = results_df.sort_values("ari", ascending=False)
        for _, row in sorted_res.iterrows():
            print(f"{row['method']:<25} {row.get('ari',0):>7.4f} {row.get('ami',0):>7.4f} {row.get('purity',0):>7.4f} {row.get('sensitivity',0):>11.4f} {row.get('retention',0):>9.4f} {row.get('f1',0):>7.4f} {row.get('runtime_s',0):>8.1f}")

    print(f"\n--- Statistical Tests (consensus vs strongest single method) ---")
    if not stat_df.empty:
        print(stat_df.to_string(index=False))
    else:
        print("  No statistical tests (insufficient paired data)")

    print(f"\nResults saved to: {output_dir}")
    return results_df, epi_f1_df, stat_df, method_stats


if __name__ == "__main__":
    benchmark_path = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
    output_dir = "/home/jilin/DeepTCR/tcrconsensus/results/exp1_v3_benchmark"
    run_experiment(benchmark_path, output_dir)
