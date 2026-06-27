#!/usr/bin/env python3
"""Experiment 4 Rerun: Leave-one-epitope-out with improved consensus.

Three strategies per epitope:
1. Improved: CC + empirical_weights + merge 0.6 (new method)
2. Control:  CC + equal_weights + merge 0.6
3. All 7 individual methods as baselines

Same design as original run_exp4_mv.py but uses improved consensus.
"""

import sys
import time
import logging
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

from exp_shared import (
    get_all_clusterers, clusters_to_labels, assignments_to_labels,
    evaluate_clustering, run_single_method, load_benchmark_data,
)
from tcrconsensus.io.parser import normalize
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.refinement.refiner import refine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consensus strategies
# ---------------------------------------------------------------------------

def consensus_improved(all_assignments, config):
    """Improved: CC + empirical weights + merge 0.6 (default)."""
    methods = sorted(set(a.method for a in all_assignments))
    weights = empirical_weights(methods)
    logger.info(f"  consensus_improved: {len(methods)} methods, empirical weights")
    for m in sorted(weights, key=weights.get, reverse=True):
        logger.info(f"    {m}: {weights[m]:.4f}")

    edges = extract_pairwise_support(all_assignments, weights)
    graph = build_consensus_graph(edges, threshold=0.3)
    clusters = connected_components_clustering(graph)
    if clusters:
        clusters = refine(clusters, edges, config)
    return clusters, edges


def consensus_control(all_assignments, config):
    """Control: CC + equal weights + merge 0.6 (default)."""
    methods = sorted(set(a.method for a in all_assignments))
    weights = {m: 1.0 / len(methods) for m in methods}
    logger.info(f"  consensus_control: {len(methods)} methods, equal weights = {1.0/len(methods):.4f}")

    edges = extract_pairwise_support(all_assignments, weights)
    graph = build_consensus_graph(edges, threshold=0.3)
    clusters = connected_components_clustering(graph)
    if clusters:
        clusters = refine(clusters, edges, config)
    return clusters, edges


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp4(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusterers = get_all_clusterers()
    logger.info(f"Clusterers: {list(clusterers.keys())}")

    # Load benchmark
    df_raw, _ = load_benchmark_data()
    epitope_col = "epitope"

    # Select epitopes with >= 15 TCRs
    epi_counts = df_raw[epitope_col].value_counts()
    test_epis = epi_counts[epi_counts >= 15].index.tolist()
    logger.info(f"Testing {len(test_epis)} epitopes with >= 15 TCRs")

    all_results = []

    for idx, epitope in enumerate(test_epis):
        n_epi = epi_counts[epitope]
        subset = df_raw[df_raw[epitope_col] == epitope].copy()

        logger.info(f"\n[{idx+1}/{len(test_epis)}] {epitope} ({n_epi} TCRs)")

        # Build mixed dataset: target + distractors from other epitopes
        distractor_epis = [e for e in test_epis if e != epitope][:3]
        d_rows = []
        for de in distractor_epis:
            de_subset = df_raw[df_raw[epitope_col] == de]
            if len(de_subset) > 50:
                de_subset = de_subset.sample(50, random_state=42)
            d_rows.append(de_subset)
        distractor_df = pd.concat(d_rows, ignore_index=True)

        mixed = pd.concat([
            subset[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
            distractor_df[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
        ], ignore_index=True)

        true_labels = mixed["epitope"].values
        n_total = len(mixed)

        rename_lower = {col: col.lower() for col in mixed.columns
                       if col.lower() != col and col.lower() in
                       ["cdr3_beta", "v_beta", "j_beta", "tcr_id", "epitope"]}
        if rename_lower:
            mixed = mixed.rename(columns=rename_lower)
        mixed_norm = normalize(mixed.copy())
        tcr_ids = mixed_norm["tcr_id"].values

        workdir = output_dir / f"work/{epitope[:20]}"
        workdir.mkdir(parents=True, exist_ok=True)

        config = {}  # use default refine config (merge=0.6)

        # === Run all individual methods ===
        method_assigns = {}
        for mname, clusterer in clusterers.items():
            assigns, rt = run_single_method(clusterer, mixed_norm, workdir / mname, config)
            if assigns:
                method_assigns[mname] = assigns
                pred = assignments_to_labels(assigns, tcr_ids)
                m = evaluate_clustering(pred, true_labels, n_total, mname)
                m.update({"target_epitope": epitope, "n_target": n_epi, "n_total": n_total})
                all_results.append(m)

        # === Strategy 1: Improved (CC + Empirical + merge 0.6) ===
        if len(method_assigns) >= 2:
            all_a = []
            for a_list in method_assigns.values():
                all_a.extend(a_list)

            t0 = time.time()
            clusters, edges = consensus_improved(all_a, config)
            elapsed = time.time() - t0

            pred = clusters_to_labels(clusters, tcr_ids)
            m = evaluate_clustering(pred, true_labels, n_total, "improved_cc_empirical")
            m.update({
                "target_epitope": epitope, "n_target": n_epi, "n_total": n_total,
                "runtime_s": elapsed, "n_clusters": len(clusters),
            })
            all_results.append(m)

            # === Strategy 2: Control (CC + Equal + merge 0.6) ===
            t0 = time.time()
            clusters2, edges2 = consensus_control(all_a, config)
            elapsed = time.time() - t0

            pred2 = clusters_to_labels(clusters2, tcr_ids)
            m2 = evaluate_clustering(pred2, true_labels, n_total, "control_cc_equal")
            m2.update({
                "target_epitope": epitope, "n_target": n_epi, "n_total": n_total,
                "runtime_s": elapsed, "n_clusters": len(clusters2),
            })
            all_results.append(m2)

    # === Save results ===
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "exp4_improved_results.tsv", sep="\t", index=False)

    # === Summary ===
    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT 4 RERUN: IMPROVED vs CONTROL vs INDIVIDUAL METHODS", flush=True)
    print("=" * 80, flush=True)

    # Per-epitope: improved vs control vs best single
    print("\n--- Per-epitope ARI: improved vs control vs best single ---", flush=True)
    for epitope in test_epis[:15]:
        sub = results_df[results_df["target_epitope"] == epitope]
        imp = sub[sub["method"] == "improved_cc_empirical"]
        ctrl = sub[sub["method"] == "control_cc_equal"]
        singles = sub[~sub["method"].isin(["improved_cc_empirical", "control_cc_equal"])]

        if len(imp) > 0 and len(singles) > 0:
            best_s = singles.loc[singles["ari"].idxmax()]
            imp_r = imp.iloc[0]
            ctrl_r = ctrl.iloc[0] if len(ctrl) > 0 else None
            d_imp = imp_r["ari"] - best_s["ari"]
            d_ctrl = ctrl_r["ari"] - best_s["ari"] if ctrl_r is not None else None
            s1 = "✓" if d_imp > 0 else "✗"
            ctrl_ari = f"{ctrl_r['ari']:.4f}" if ctrl_r is not None else "N/A"
            print(
                f"  {epitope:20s}: Improved={imp_r['ari']:.4f}  Control={ctrl_ari}  "
                f"BestSingle={best_s['method']:12s}={best_s['ari']:.4f}  "
                f"Δ_imp={d_imp:+.4f}{s1}",
                flush=True,
            )

    # Aggregate comparison
    print("\n--- Aggregate metrics by method ---", flush=True)
    agg = results_df.groupby("method").agg({
        "ari": ["mean", "std"],
        "purity": ["mean", "std"],
        "sensitivity": ["mean", "std"],
        "f1": ["mean", "std"],
    }).reset_index()
    print(agg.to_string(index=False), flush=True)

    # Improved vs Control paired comparison
    imp_data = results_df[results_df["method"] == "improved_cc_empirical"]
    ctrl_data = results_df[results_df["method"] == "control_cc_equal"]
    if len(imp_data) > 0 and len(ctrl_data) > 0:
        from scipy import stats as sp_stats
        ari_imp = imp_data["ari"].values
        ari_ctrl = ctrl_data["ari"].values
        wins = sum(1 for a, c in zip(ari_imp, ari_ctrl) if a > c)
        t_stat, p_val = sp_stats.ttest_rel(ari_imp, ari_ctrl)
        print(f"\n--- Improved vs Control paired t-test ---", flush=True)
        print(f"  Improved ARI: {np.mean(ari_imp):.4f} ± {np.std(ari_imp):.4f}", flush=True)
        print(f"  Control  ARI: {np.mean(ari_ctrl):.4f} ± {np.std(ari_ctrl):.4f}", flush=True)
        print(f"  Improved wins: {wins}/{len(ari_imp)} epitopes", flush=True)
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.6f}", flush=True)

    # Improved vs best single method per epitope
    print(f"\n--- Improved vs Best Single Method ---", flush=True)
    n_wins = 0
    n_total_epi = 0
    for epitope in test_epis:
        sub = results_df[results_df["target_epitope"] == epitope]
        imp = sub[sub["method"] == "improved_cc_empirical"]
        singles = sub[~sub["method"].isin(["improved_cc_empirical", "control_cc_equal"])]
        if len(imp) > 0 and len(singles) > 0:
            best_single_ari = singles["ari"].max()
            if imp.iloc[0]["ari"] >= best_single_ari:
                n_wins += 1
            n_total_epi += 1
    print(f"  Improved >= best single in {n_wins}/{n_total_epi} epitopes", flush=True)

    print(f"\nResults saved to: {output_dir}", flush=True)
    print("EXPERIMENT 4 RERUN COMPLETE", flush=True)


if __name__ == "__main__":
    run_exp4("/home/jilin/DeepTCR/tcrconsensus/results/exp4_improved")
