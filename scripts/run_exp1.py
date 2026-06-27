#!/usr/bin/env python3
"""Experiment 1: Cross-Benchmark Comparison.

Runs all single methods + consensus modes + simple baselines on VDJdb and McPAS.
Computes per-dataset and per-epitope metrics with statistical tests.
"""

import copy
import json
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcrconsensus.io.parser import load_file, normalize
from tcrconsensus.config import load_config
from tcrconsensus.profiling.profiler import profile as compute_profile
from tcrconsensus.selection.selector import select_methods
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering, community_clustering
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import (
    compute_all_metrics, per_epitope_pairwise_f1, per_epitope_metrics,
    paired_bootstrap_ci, wilcoxon_test,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clusterer registry
# ---------------------------------------------------------------------------

def get_all_clusterers():
    """Get all available clusterers."""
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    try:
        from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
        clusterers["clustcr"] = ClusTCRWrapper()
    except Exception as e:
        logger.warning(f"clustcr unavailable: {e}")
    try:
        from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
        clusterers["tcrdist3"] = TCRDist3Wrapper()
    except Exception as e:
        logger.warning(f"tcrdist3 unavailable: {e}")
    try:
        from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
        clusterers["gliph2"] = GLIPH2Wrapper()
    except Exception as e:
        logger.warning(f"gliph2 unavailable: {e}")
    try:
        from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
        clusterers["giana"] = GIANAWrapper(giana_script="/home/jilin/DeepTCR/GIANA/GIANA4.1.py")
    except Exception as e:
        logger.warning(f"giana unavailable: {e}")
    try:
        from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
        clusterers["tcrmatch"] = TCRMatchWrapper(tcrmatch_bin="/home/jilin/DeepTCR/TCRMatch/tcrmatch")
    except Exception as e:
        logger.warning(f"tcrmatch unavailable: {e}")
    try:
        from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper
        clusterers["deeptcr"] = DeepTCRWrapper()
    except Exception as e:
        logger.warning(f"deeptcr unavailable: {e}")
    return clusterers


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
            assignments, weights,
            **cfg.get("conservative", {}),
        )
    elif mode == "coverage":
        clusters, edges = coverage_consensus(
            assignments, weights,
            **cfg.get("coverage", {}),
        )
    else:  # balanced
        clusters, edges = balanced_consensus(
            assignments, weights,
            **cfg.get("balanced", {}),
        )

    if not skip_refinement:
        clusters = refine(clusters, edges, config)

    return clusters, edges


# ---------------------------------------------------------------------------
# Simple baselines
# ---------------------------------------------------------------------------

def run_majority_vote(df, all_assignments, workdir, config):
    """Majority vote: equal weights, balanced consensus."""
    methods = list(set(a.method for a in all_assignments))
    weights = {m: 1.0 for m in methods}
    # Normalize
    total = sum(weights.values())
    weights = {m: w / total * len(methods) for m, w in weights.items()}

    clusters, edges = run_consensus(all_assignments, weights, "balanced", config)
    return clusters


def run_intersection_only(df, all_assignments, workdir, config):
    """Intersection: only link pairs supported by ALL methods."""
    from tcrconsensus.schema.records import ConsensusEdge
    from collections import defaultdict
    from itertools import combinations

    method_set = set(a.method for a in all_assignments)
    n_methods = len(method_set)

    # Count method support per pair
    method_clusters = defaultdict(lambda: defaultdict(set))
    for a in all_assignments:
        method_clusters[a.method][a.cluster_id].add(a.tcr_id)

    pair_support = defaultdict(set)
    for method, clusters in method_clusters.items():
        for cid, members in clusters.items():
            if len(members) < 2:
                continue
            for x, y in combinations(sorted(members), 2):
                pair_support[(x, y)].add(method)

    # Only keep pairs supported by ALL methods
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


def run_union_only(df, all_assignments, workdir, config):
    """Union: link pairs from ANY single method (threshold=0)."""
    weights = {}
    methods = list(set(a.method for a in all_assignments))
    for m in methods:
        weights[m] = 1.0

    clusters, edges = run_consensus(all_assignments, weights, "coverage", config)
    return clusters


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def evaluate_method(pred_labels, true_labels_str, tcr_ids, n_total, dataset_name, method_name):
    """Evaluate a method's predictions against true labels."""
    # Filter to TCRs that have both pred and true labels
    # pred_labels may contain strings (cluster_ids) or -1 (unclustered)
    valid_mask = np.array([
        p != -1 and str(p) != "-1" and str(p) != ""
        for p in pred_labels
    ], dtype=bool)

    if valid_mask.sum() < 2:
        return None

    # Encode labels
    le_true = LabelEncoder()
    le_pred = LabelEncoder()

    true_str = true_labels_str[valid_mask]
    pred_str = pred_labels[valid_mask].astype(str)

    all_labels = np.concatenate([true_str, pred_str])
    le_true.fit(np.unique(true_str))
    le_pred.fit(np.unique(pred_str))

    true_enc = le_true.transform(true_str)
    pred_enc = le_pred.transform(pred_str)

    metrics = compute_all_metrics(pred_enc, true_enc, n_total)
    metrics["dataset"] = dataset_name
    metrics["method"] = method_name

    # Per-epitope F1
    epi_f1 = per_epitope_pairwise_f1(pred_enc, true_enc)
    epi_f1["dataset"] = dataset_name
    epi_f1["method"] = method_name

    return metrics, epi_f1


def run_experiment(data_dir, output_dir):
    """Run the full cross-benchmark experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "VDJdb": {
            "data": f"{data_dir}/vdjdb_filtered.tsv",
            "labels": f"{data_dir}/vdjdb_labels.tsv",
        },
        "McPAS": {
            "data": f"{data_dir}/mcpas_filtered.tsv",
            "labels": f"{data_dir}/mcpas_labels.tsv",
        },
    }

    config = load_config()
    cfg = config._raw

    clusterers = get_all_clusterers()
    logger.info(f"Available clusterers: {list(clusterers.keys())}")

    all_results = []
    all_epi_f1 = []
    method_assignments_cache = {}  # cache per dataset to avoid re-running

    for ds_name, ds_info in datasets.items():
        logger.info(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")

        # Load data
        df = pd.read_csv(ds_info["data"], sep="\t", dtype=str)
        labels_df = pd.read_csv(ds_info["labels"], sep="\t", dtype=str)

        # Merge labels
        df = df.merge(labels_df, on="tcr_id", how="inner", suffixes=("", "_label"))
        epitope_col = "epitope_label" if "epitope_label" in df.columns else "epitope"
        df = df[df[epitope_col].notna()].copy()
        df = df.reset_index(drop=True)

        n_total = len(df)
        tcr_ids = df["tcr_id"].values
        true_labels = df[epitope_col].values
        n_epitopes = df[epitope_col].nunique()

        logger.info(f"  {n_total} TCRs, {n_epitopes} epitopes")

        # Normalize for clustering
        df_norm = normalize(df.copy())

        # Filter to epitopes with >= 5 TCRs
        epi_counts = pd.Series(true_labels).value_counts()
        valid_epis = set(epi_counts[epi_counts >= 5].index)
        valid_mask = np.isin(true_labels, list(valid_epis))
        df_eval = df_norm.copy()
        df_eval["_true_label"] = true_labels

        workdir = output_dir / ds_name / "work"
        workdir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Run individual methods
        # ------------------------------------------------------------------
        ds_assignments = {}
        for mname, clusterer in clusterers.items():
            logger.info(f"  Running {mname}...")
            t0 = time.time()
            assignments, status, runtime = run_single_method(df_norm, clusterer, workdir, cfg)
            elapsed = time.time() - t0
            logger.info(f"    {mname}: {len(assignments)} assignments, status={status}, {elapsed:.1f}s")

            if status == "success" and assignments:
                ds_assignments[mname] = assignments

                # Evaluate single method
                pred = clusters_to_labels(
                    [type('C', (), {'member_ids': [], 'cluster_id': ''})()] * 0,  # dummy
                    tcr_ids
                )
                # Build per-TCR label from assignments
                label_map = {}
                for a in assignments:
                    if a.tcr_id not in label_map:
                        label_map[a.tcr_id] = a.cluster_id
                pred_labels = np.array([label_map.get(tid, -1) for tid in tcr_ids])

                result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, ds_name, mname)
                if result:
                    metrics, epi_f1_df = result
                    metrics["runtime_s"] = elapsed
                    all_results.append(metrics)
                    all_epi_f1.append(epi_f1_df)

        method_assignments_cache[ds_name] = ds_assignments

        # ------------------------------------------------------------------
        # Run consensus modes
        # ------------------------------------------------------------------
        if len(ds_assignments) < 2:
            logger.warning(f"  Only {len(ds_assignments)} methods succeeded, skipping consensus")
            continue

        all_method_assignments = []
        for assignments in ds_assignments.values():
            all_method_assignments.extend(assignments)

        available_methods = list(ds_assignments.keys())

        for mode in ["conservative", "balanced", "coverage"]:
            logger.info(f"  Running consensus ({mode})...")
            t0 = time.time()

            weights = compute_method_weights(available_methods, "balanced", cfg)
            clusters, edges = run_consensus(all_method_assignments, weights, mode, cfg)

            elapsed = time.time() - t0
            logger.info(f"    {len(clusters)} clusters, {elapsed:.1f}s")

            pred_labels = clusters_to_labels(clusters, tcr_ids)
            result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, ds_name, f"consensus_{mode}")
            if result:
                metrics, epi_f1_df = result
                metrics["runtime_s"] = elapsed
                all_results.append(metrics)
                all_epi_f1.append(epi_f1_df)

        # ------------------------------------------------------------------
        # Run simple baselines
        # ------------------------------------------------------------------

        # Majority vote
        logger.info(f"  Running majority_vote...")
        t0 = time.time()
        clusters = run_majority_vote(df_norm, all_method_assignments, workdir, cfg)
        elapsed = time.time() - t0
        pred_labels = clusters_to_labels(clusters, tcr_ids)
        result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, ds_name, "majority_vote")
        if result:
            metrics, epi_f1_df = result
            metrics["runtime_s"] = elapsed
            all_results.append(metrics)
            all_epi_f1.append(epi_f1_df)

        # Intersection only
        logger.info(f"  Running intersection_only...")
        t0 = time.time()
        clusters = run_intersection_only(df_norm, all_method_assignments, workdir, cfg)
        elapsed = time.time() - t0
        pred_labels = clusters_to_labels(clusters, tcr_ids)
        result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, ds_name, "intersection_only")
        if result:
            metrics, epi_f1_df = result
            metrics["runtime_s"] = elapsed
            all_results.append(metrics)
            all_epi_f1.append(epi_f1_df)

        # Union only
        logger.info(f"  Running union_only...")
        t0 = time.time()
        clusters = run_union_only(df_norm, all_method_assignments, workdir, cfg)
        elapsed = time.time() - t0
        pred_labels = clusters_to_labels(clusters, tcr_ids)
        result = evaluate_method(pred_labels, true_labels, tcr_ids, n_total, ds_name, "union_only")
        if result:
            metrics, epi_f1_df = result
            metrics["runtime_s"] = elapsed
            all_results.append(metrics)
            all_epi_f1.append(epi_f1_df)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    epi_f1_df = pd.concat(all_epi_f1, ignore_index=True)

    # Save per-dataset summary
    summary = results_df.groupby("dataset").apply(
        lambda g: g.sort_values("ari", ascending=False)
    ).reset_index(drop=True)

    results_df.to_csv(output_dir / "per_dataset_detail.tsv", sep="\t", index=False)
    epi_f1_df.to_csv(output_dir / "per_epitope_f1.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # Statistical tests: consensus vs strongest single method
    # ------------------------------------------------------------------
    stat_tests = []
    for ds_name in datasets:
        ds_epi = epi_f1_df[epi_f1_df["dataset"] == ds_name]
        methods_in_ds = ds_epi["method"].unique()

        # Find strongest single method by median F1
        single_methods = [m for m in methods_in_ds if not m.startswith("consensus") and m not in ("majority_vote", "intersection_only", "union_only")]
        if not single_methods:
            continue

        medians = {m: ds_epi[ds_epi["method"] == m]["f1"].median() for m in single_methods}
        strongest = max(medians, key=medians.get)

        # Compare each consensus mode vs strongest single method
        for mode in ["consensus_balanced", "consensus_conservative", "consensus_coverage"]:
            if mode not in methods_in_ds:
                continue

            cons_f1 = ds_epi[ds_epi["method"] == mode].set_index("epitope")["f1"]
            best_f1 = ds_epi[ds_epi["method"] == strongest].set_index("epitope")["f1"]

            # Align epitopes
            common = cons_f1.index.intersection(best_f1.index)
            if len(common) < 5:
                continue

            cons_vals = cons_f1.loc[common].values
            best_vals = best_f1.loc[common].values

            # Paired bootstrap CI
            pb = paired_bootstrap_ci(cons_vals, best_vals)

            # Wilcoxon
            wt = wilcoxon_test(cons_vals, best_vals)

            stat_tests.append({
                "dataset": ds_name,
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
    stat_df.to_csv(output_dir / "statistical_tests.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("EXPERIMENT 1: CROSS-BENCHMARK RESULTS")
    print("="*80)

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        ds_res = results_df[results_df["dataset"] == ds_name].sort_values("ari", ascending=False)
        print(ds_res[["method", "ari", "ami", "nmi", "purity", "sensitivity", "retention", "f1", "v_measure"]].to_string(index=False))

    print("\n--- Statistical Tests ---")
    if len(stat_df) > 0:
        print(stat_df.to_string(index=False))
    else:
        print("No statistical tests (insufficient paired data)")

    print(f"\nResults saved to: {output_dir}")
    return results_df, epi_f1_df, stat_df


if __name__ == "__main__":
    data_dir = "/home/jilin/DeepTCR/tcrconsensus/results/data"
    output_dir = "/home/jilin/DeepTCR/tcrconsensus/results/exp1_cross_benchmark"
    run_experiment(data_dir, output_dir)
