#!/usr/bin/env python3
"""Experiment 3: Component Ablation for majority_vote.

Tests what makes majority_vote work by ablating each component:
  1. full_pipeline     — majority_vote with refinement (the champion)
  2. no_refinement     — majority_vote without refinement
  3. no_leiden         — equal weights + connected_components only (skip Leiden)
  4. leave_one_out     — remove one method at a time, see ARI drop
  5. random_weights    — random weights (5 seeds) instead of equal
  6. single_methods    — each individual method alone
"""

import sys
import time
import logging
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))
from exp_shared import (
    get_all_clusterers, majority_vote_consensus,
    clusters_to_labels, assignments_to_labels,
    evaluate_clustering, run_single_method, load_benchmark_data,
)
from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
from tcrconsensus.refinement.refiner import refine

logger = logging.getLogger(__name__)


def run_exp3(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    cfg = config._raw
    clusterers = get_all_clusterers()
    logger.info(f"Clusterers: {list(clusterers.keys())}")

    # Load benchmark
    df_raw, df_norm = load_benchmark_data()
    epitope_col = "epitope" if "epitope" in df_raw.columns else "Epitope"
    tcr_ids = df_norm["tcr_id"].values
    true_labels = df_raw[epitope_col].values
    n_total = len(df_norm)

    logger.info(f"Dataset: {n_total} TCRs, {df_raw[epitope_col].nunique()} epitopes")

    # Run all methods once
    workdir = output_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    logger.info("Running all individual methods...")
    method_results = {}
    for mname, clusterer in clusterers.items():
        assigns, rt = run_single_method(clusterer, df_norm, workdir / mname, cfg)
        if assigns:
            method_results[mname] = assigns
            logger.info(f"  {mname}: {len(assigns)} assignments")

    available_methods = sorted(method_results.keys())
    logger.info(f"Available methods: {available_methods}")

    all_results = []

    # ---- 1. Individual methods as baselines ----
    logger.info("\n--- 1. Individual methods ---")
    for mname, assigns in method_results.items():
        pred = assignments_to_labels(assigns, tcr_ids)
        m = evaluate_clustering(pred, true_labels, n_total, mname)
        m["condition"] = mname
        all_results.append(m)

    # ---- 2. Full majority_vote pipeline (champion) ----
    logger.info("\n--- 2. Full majority_vote pipeline ---")
    all_a = []
    for a_list in method_results.values():
        all_a.extend(a_list)
    clusters, edges = majority_vote_consensus(all_a, cfg)
    pred = clusters_to_labels(clusters, tcr_ids)
    m = evaluate_clustering(pred, true_labels, n_total, "majority_vote")
    m["condition"] = "full_pipeline"
    all_results.append(m)

    # ---- 3. No refinement ----
    logger.info("--- 3. No refinement ---")
    clusters_nr, edges_nr = majority_vote_consensus(all_a, cfg, skip_refinement=True)
    pred = clusters_to_labels(clusters_nr, tcr_ids)
    m = evaluate_clustering(pred, true_labels, n_total, "majority_vote")
    m["condition"] = "no_refinement"
    all_results.append(m)

    # ---- 4. No Leiden (connected components only) ----
    logger.info("--- 4. No Leiden (connected components only) ---")
    methods_list = sorted(set(a.method for a in all_a))
    n_m = len(methods_list)
    weights = {m: 1.0 / n_m for m in methods_list}
    pair_edges = extract_pairwise_support(all_a, weights)
    graph = build_consensus_graph(pair_edges, threshold=0.3)
    clusters_cc = connected_components_clustering(graph)
    pred = clusters_to_labels(clusters_cc, tcr_ids)
    m = evaluate_clustering(pred, true_labels, n_total, "no_leiden")
    m["condition"] = "no_leiden_cc_only"
    all_results.append(m)

    # ---- 5. Leave-one-method-out ----
    logger.info("--- 5. Leave-one-method-out ---")
    for held_out in available_methods:
        remaining = {k: v for k, v in method_results.items() if k != held_out}
        if len(remaining) < 2:
            continue
        rem_a = []
        for a_list in remaining.values():
            rem_a.extend(a_list)
        clusters_lo, edges_lo = majority_vote_consensus(rem_a, cfg)
        pred = clusters_to_labels(clusters_lo, tcr_ids)
        m = evaluate_clustering(pred, true_labels, n_total, "majority_vote")
        m["condition"] = f"lomo_remove_{held_out}"
        m["held_out"] = held_out
        all_results.append(m)

    # ---- 6. Random weights (5 seeds) ----
    logger.info("--- 6. Random weights ---")
    for seed in range(5):
        rng = np.random.RandomState(seed * 100)
        random_w = {m: rng.uniform(0.1, 2.0) for m in methods_list}
        clusters_rw, edges_rw = balanced_consensus(all_a, random_w)
        clusters_rw = refine(clusters_rw, edges_rw, cfg)
        pred = clusters_to_labels(clusters_rw, tcr_ids)
        m = evaluate_clustering(pred, true_labels, n_total, "random_weights")
        m["condition"] = f"random_weights_s{seed}"
        all_results.append(m)

    # ---- 7. Threshold sensitivity ----
    logger.info("--- 7. Threshold sensitivity ---")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        graph_t = build_consensus_graph(pair_edges, threshold=thresh)
        from tcrconsensus.consensus.graph import community_clustering
        clusters_t = community_clustering(graph_t)
        if clusters_t:
            clusters_t = refine(clusters_t, pair_edges, cfg)
        pred = clusters_to_labels(clusters_t, tcr_ids)
        m = evaluate_clustering(pred, true_labels, n_total, "majority_vote")
        m["condition"] = f"threshold_{thresh}"
        all_results.append(m)

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "exp3_ablation_results.tsv", sep="\t", index=False)

    # LOMO separate
    lomo = [r for r in all_results if "lomo" in r.get("condition", "")]
    if lomo:
        pd.DataFrame(lomo).to_csv(output_dir / "leave_one_method_out.tsv", sep="\t", index=False)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT 3: COMPONENT ABLATION FOR MAJORITY_VOTE")
    print("="*80)

    # Summary by condition
    summary_cols = ["condition", "ari", "ami", "purity", "sensitivity", "retention", "f1", "v_measure"]
    available_cols = [c for c in summary_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False))

    # ARI ranking
    print("\n--- ARI Ranking ---")
    ranked = results_df.sort_values("ari", ascending=False)
    for _, row in ranked.head(15).iterrows():
        print(f"  {row['condition']:40s}  ARI={row['ari']:.4f}  Purity={row.get('purity',0):.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    run_exp3("/home/jilin/DeepTCR/tcrconsensus/results/exp3_mv_ablation")
