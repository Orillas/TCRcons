#!/usr/bin/env python3
"""Experiment 3: Component Ablation.

Tests marginal contribution of each component:
  1. Full consensus
  2. Equal weights
  3. Majority vote (equal + no refinement)
  4. Intersection-only
  5. Union-only
  6. No refinement
  7. Leave-one-method-out (per method)
  8. Random weights (5 seeds)
"""

import copy
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering, community_clustering
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_clusterers():
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    try:
        from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
        clusterers["clustcr"] = ClusTCRWrapper()
    except: pass
    try:
        from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
        clusterers["tcrdist3"] = TCRDist3Wrapper()
    except: pass
    try:
        from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
        clusterers["gliph2"] = GLIPH2Wrapper()
    except: pass
    try:
        from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper
        clusterers["deeptcr"] = DeepTCRWrapper()
    except: pass
    return clusterers


def clusters_to_labels(clusters, tcr_ids):
    label_map = {}
    for c in clusters:
        for mid in c.member_ids:
            label_map[mid] = c.cluster_id
    return np.array([label_map.get(tid, -1) for tid in tcr_ids])


def evaluate(pred_labels, true_labels, n_total):
    valid = np.array([str(p) not in ("-1", "") for p in pred_labels], dtype=bool)
    if valid.sum() < 2:
        return {}
    le_t = LabelEncoder()
    le_p = LabelEncoder()
    true_str = true_labels[valid]
    pred_str = pred_labels[valid].astype(str)
    le_t.fit(np.unique(true_str))
    le_p.fit(np.unique(pred_str))
    return compute_all_metrics(le_p.transform(pred_str), le_t.transform(true_str), n_total)


def run_all_clusterers(df_norm, clusterers, config, workdir):
    """Run all clusterers, return dict of method_name -> assignments."""
    results = {}
    for mname, clusterer in clusterers.items():
        r = clusterer.safe_execute(df_norm, workdir, config)
        if r.status.value == "success" and r.assignments:
            results[mname] = r.assignments
    return results


def run_consensus_with_weights(assignments_dict, weights, mode, config, skip_refinement=False):
    """Run consensus with custom weights."""
    all_a = []
    for a in assignments_dict.values():
        all_a.extend(a)

    if mode == "conservative":
        clusters, edges = conservative_consensus(all_a, weights)
    elif mode == "coverage":
        clusters, edges = coverage_consensus(all_a, weights)
    else:
        clusters, edges = balanced_consensus(all_a, weights)

    if not skip_refinement:
        clusters = refine(clusters, edges, config)
    return clusters


def run_experiment(data_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    cfg = config._raw
    clusterers = get_clusterers()

    # Use VDJdb as primary dataset
    vdj = pd.read_csv(f"{data_dir}/vdjdb_filtered.tsv", sep="\t", dtype=str)
    labels = pd.read_csv(f"{data_dir}/vdjdb_labels.tsv", sep="\t", dtype=str)
    df = vdj.merge(labels, on="tcr_id", how="inner", suffixes=("", "_label"))
    epitope_col = "epitope_label" if "epitope_label" in df.columns else "epitope"
    df = df[df[epitope_col].notna()].reset_index(drop=True)

    tcr_ids = df["tcr_id"].values
    true_labels = df[epitope_col].values
    n_total = len(df)
    df_norm = normalize(df.copy())

    logger.info(f"Dataset: {n_total} TCRs, {df[epitope_col].nunique()} epitopes")

    # Run all clusterers once
    workdir = output_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    logger.info("Running all clusterers...")
    assignments_dict = run_all_clusterers(df_norm, clusterers, cfg, workdir)
    logger.info(f"Methods succeeded: {list(assignments_dict.keys())}")

    if len(assignments_dict) < 2:
        logger.error("Need at least 2 methods for ablation")
        return

    available_methods = list(assignments_dict.keys())
    all_results = []

    # 1. Full consensus (default)
    logger.info("1. Full consensus (default weights)")
    weights = compute_method_weights(available_methods, "balanced", cfg)
    clusters = run_consensus_with_weights(assignments_dict, weights, "balanced", cfg)
    pred = clusters_to_labels(clusters, tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "full_consensus"
    all_results.append(m)

    # 2. Equal weights
    logger.info("2. Equal weights")
    equal_w = {m: 1.0 for m in available_methods}
    clusters = run_consensus_with_weights(assignments_dict, equal_w, "balanced", cfg)
    pred = clusters_to_labels(clusters, tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "equal_weights"
    all_results.append(m)

    # 3. Majority vote (equal + no refinement)
    logger.info("3. Majority vote (no refinement)")
    clusters = run_consensus_with_weights(assignments_dict, equal_w, "balanced", cfg, skip_refinement=True)
    pred = clusters_to_labels(clusters, tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "majority_vote_no_refine"
    all_results.append(m)

    # 4. Intersection-only (conservative, min_method_support = n_methods)
    logger.info("4. Intersection-only (conservative)")
    clusters = run_consensus_with_weights(assignments_dict, equal_w, "conservative", cfg)
    pred = clusters_to_labels(clusters, tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "intersection_only"
    all_results.append(m)

    # 5. Union-only (coverage, low threshold)
    logger.info("5. Union-only (coverage)")
    clusters = run_consensus_with_weights(assignments_dict, equal_w, "coverage", cfg)
    pred = clusters_to_labels(clusters, tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "union_only"
    all_results.append(m)

    # 6. No refinement
    logger.info("6. No refinement (default weights)")
    clusters = run_consensus_with_weights(assignments_dict, weights, "balanced", cfg, skip_refinement=True)
    pred = clusters_to_labels(clusters, tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "no_refinement"
    all_results.append(m)

    # 7. Leave-one-method-out
    lomo_results = []
    for held_out in available_methods:
        logger.info(f"7. Leave-one-out: remove {held_out}")
        remaining = {k: v for k, v in assignments_dict.items() if k != held_out}
        if len(remaining) < 2:
            continue
        rm = list(remaining.keys())
        w = compute_method_weights(rm, "balanced", cfg)
        clusters = run_consensus_with_weights(remaining, w, "balanced", cfg)
        pred = clusters_to_labels(clusters, tcr_ids)
        m = evaluate(pred, true_labels, n_total)
        m["condition"] = f"lomo_remove_{held_out}"
        m["held_out_method"] = held_out
        all_results.append(m)
        lomo_results.append(m)

    # 8. Random weights sanity check (5 seeds)
    rng = np.random.RandomState(42)
    for seed in range(5):
        logger.info(f"8. Random weights (seed={seed})")
        rng2 = np.random.RandomState(seed * 100)
        random_w = {m: rng2.uniform(0.1, 2.0) for m in available_methods}
        clusters = run_consensus_with_weights(assignments_dict, random_w, "balanced", cfg)
        pred = clusters_to_labels(clusters, tcr_ids)
        m = evaluate(pred, true_labels, n_total)
        m["condition"] = f"random_weights_seed{seed}"
        all_results.append(m)

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "ablation_results.tsv", sep="\t", index=False)

    # Save LOMO separately
    if lomo_results:
        pd.DataFrame(lomo_results).to_csv(output_dir / "leave_one_method_out.tsv", sep="\t", index=False)

    print("\n" + "="*80)
    print("EXPERIMENT 3: ABLATION RESULTS")
    print("="*80)
    print(results_df[["condition", "ari", "ami", "nmi", "purity", "sensitivity", "retention", "v_measure"]].to_string(index=False))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    run_experiment(
        "/home/jilin/DeepTCR/tcrconsensus/results/data",
        "/home/jilin/DeepTCR/tcrconsensus/results/exp3_ablation",
    )
