#!/usr/bin/env python3
"""Experiment 4: Leave-one-epitope-out generalization.

For each epitope with ≥20 TCRs, runs majority_vote and all individual methods
on that epitope's TCRs alone. Tests whether majority_vote generalizes to
individual epitopes, not just the aggregate dataset.

This is the key experiment for showing majority_vote's advantage:
- On small single-epitope datasets, individual methods may fail or over-cluster
- majority_vote aggregates complementary signals from multiple methods
"""

import sys
import time
import logging
import warnings
from pathlib import Path

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

logger = logging.getLogger(__name__)


def run_exp4(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    cfg = config._raw
    clusterers = get_all_clusterers()
    logger.info(f"Clusterers: {list(clusterers.keys())}")

    # Load benchmark
    df_raw, _ = load_benchmark_data()
    epitope_col = "epitope" if "epitope" in df_raw.columns else "Epitope"

    # Select epitopes with ≥15 TCRs
    epi_counts = df_raw[epitope_col].value_counts()
    test_epis = epi_counts[epi_counts >= 15].index.tolist()
    logger.info(f"Testing {len(test_epis)} epitopes with ≥15 TCRs")

    all_results = []

    for idx, epitope in enumerate(test_epis):
        n_epi = epi_counts[epitope]
        subset = df_raw[df_raw[epitope_col] == epitope].copy()

        logger.info(f"\n[{idx+1}/{len(test_epis)}] {epitope} ({n_epi} TCRs)")

        # For single-epitope evaluation, we need a multi-epitope context
        # because ground truth only has one label
        # Instead: evaluate how well the method groups TCRs correctly
        # Use a trick: include TCRs from 2-3 other epitopes as distractors

        # Pick 3 most different epitopes as distractors
        distractor_epis = [e for e in test_epis if e != epitope][:3]
        distractor_df = df_raw[df_raw[epitope_col].isin(distractor_epis)].copy()
        # Sample up to 50 TCRs per distractor epitope
        d_rows = []
        for de in distractor_epis:
            de_subset = distractor_df[distractor_df[epitope_col] == de]
            if len(de_subset) > 50:
                de_subset = de_subset.sample(50, random_state=42)
            d_rows.append(de_subset)
        distractor_df = pd.concat(d_rows, ignore_index=True)

        # Mix target + distractors
        mixed = pd.concat([
            subset[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
            distractor_df[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
        ], ignore_index=True)

        true_labels = mixed["epitope"].values
        n_total = len(mixed)

        rename_lower = {col: col.lower() for col in mixed.columns
                       if col.lower() != col and col.lower() in ["cdr3_beta","v_beta","j_beta","tcr_id","epitope"]}
        if rename_lower:
            mixed = mixed.rename(columns=rename_lower)
        mixed_norm = normalize(mixed.copy())
        tcr_ids = mixed_norm["tcr_id"].values

        workdir = output_dir / f"work/{epitope[:20]}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Run all individual methods
        method_assigns = {}
        for mname, clusterer in clusterers.items():
            assigns, rt = run_single_method(clusterer, mixed_norm, workdir / mname, cfg)
            if assigns:
                method_assigns[mname] = assigns
                pred = assignments_to_labels(assigns, tcr_ids)
                m = evaluate_clustering(pred, true_labels, n_total, mname)
                m.update({"target_epitope": epitope, "n_target": n_epi, "n_total": n_total})
                all_results.append(m)

        # Run majority_vote
        if len(method_assigns) >= 2:
            all_a = []
            for a_list in method_assigns.values():
                all_a.extend(a_list)
            t0 = time.time()
            clusters, edges = majority_vote_consensus(all_a, cfg)
            elapsed = time.time() - t0

            pred = clusters_to_labels(clusters, tcr_ids)
            m = evaluate_clustering(pred, true_labels, n_total, "majority_vote")
            m.update({
                "target_epitope": epitope, "n_target": n_epi, "n_total": n_total,
                "runtime_s": elapsed, "n_clusters": len(clusters),
            })
            all_results.append(m)

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "exp4_generalization_results.tsv", sep="\t", index=False)

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT 4: LEAVE-ONE-EPITOPE-OUT GENERALIZATION")
    print("="*80)

    # Per-epitope: majority_vote vs best single
    print("\n--- Per-epitope ARI: majority_vote vs best single method ---")
    for epitope in test_epis[:10]:
        sub = results_df[results_df["target_epitope"] == epitope]
        mv = sub[sub["method"] == "majority_vote"]
        single = sub[sub["method"] != "majority_vote"]
        if len(mv) > 0 and len(single) > 0:
            best_s = single.loc[single["ari"].idxmax()]
            mv_r = mv.iloc[0]
            delta = mv_r["ari"] - best_s["ari"]
            symbol = "✓" if delta > 0 else "✗"
            print(f"  {epitope:20s}: MV={mv_r['ari']:.4f} vs {best_s['method']:15s}={best_s['ari']:.4f}  Δ={delta:+.4f} {symbol}")

    # Aggregate comparison
    print("\n--- Aggregate metrics by method ---")
    agg = results_df.groupby("method").agg({
        "ari": ["mean", "std"],
        "purity": ["mean", "std"],
        "sensitivity": ["mean", "std"],
        "f1": ["mean", "std"],
    }).reset_index()
    print(agg.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    run_exp4("/home/jilin/DeepTCR/tcrconsensus/results/exp4_mv_generalization")
