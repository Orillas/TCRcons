#!/usr/bin/env python3
"""Experiment 2: Background Robustness — majority_vote vs individual methods under noise.

Uses non-target-epitope TCRs from the same benchmark dataset as background.
No external data files needed.
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


def run_exp2(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    cfg = config._raw
    clusterers = get_all_clusterers()
    logger.info(f"Clusterers: {list(clusterers.keys())}")

    # Load benchmark signal TCRs
    df_raw, _ = load_benchmark_data()
    epitope_col = "epitope" if "epitope" in df_raw.columns else "Epitope"

    # Background pool: all TCRs NOT from target epitope (use other epitopes)
    all_epitopes = df_raw[epitope_col].value_counts()

    # Select top 5 epitopes as targets
    top_epis = all_epitopes[all_epitopes >= 20].head(5).index.tolist()
    logger.info(f"Target epitopes: {top_epis}")

    # Build background pool from all other epitopes
    bg_pool = df_raw[~df_raw[epitope_col].isin(top_epis)].copy()
    logger.info(f"Background pool: {len(bg_pool)} TCRs from other epitopes")

    ratios = [0, 5, 10, 50, 100, 500]
    rng = np.random.RandomState(42)
    all_results = []

    for epitope in top_epis:
        signal = df_raw[df_raw[epitope_col] == epitope].copy()
        n_signal = len(signal)
        if n_signal < 10:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Epitope: {epitope} ({n_signal} signal TCRs)")
        logger.info(f"{'='*60}")

        for ratio in ratios:
            if ratio == 0:
                mixed = signal[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]].copy()
            else:
                n_bg = min(int(n_signal * ratio), len(bg_pool))
                bg_sample = bg_pool.sample(n=n_bg, replace=False, random_state=rng).copy()
                mixed = pd.concat([
                    signal[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
                    bg_sample[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
                ], ignore_index=True)

            true_labels = mixed["epitope"].values
            n_total = len(mixed)

            # Normalize
            rename_lower = {col: col.lower() for col in mixed.columns
                           if col.lower() != col and col.lower() in ["cdr3_beta","v_beta","j_beta","tcr_id","epitope"]}
            if rename_lower:
                mixed = mixed.rename(columns=rename_lower)
            mixed_norm = normalize(mixed.copy())
            tcr_ids = mixed_norm["tcr_id"].values

            logger.info(f"  Ratio 1:{ratio} ({n_signal} signal + {n_total - n_signal} bg = {n_total})")

            workdir = output_dir / f"work/{epitope[:20]}_r{ratio}"
            workdir.mkdir(parents=True, exist_ok=True)

            # Run all individual methods
            method_assigns = {}
            for mname, clusterer in clusterers.items():
                assigns, rt = run_single_method(clusterer, mixed_norm, workdir / mname, cfg)
                if assigns:
                    method_assigns[mname] = assigns
                    pred = assignments_to_labels(assigns, tcr_ids)
                    m = evaluate_clustering(pred, true_labels, n_total, mname)
                    m.update({
                        "epitope": epitope,
                        "bg_ratio": ratio,
                        "n_signal": n_signal,
                        "n_background": n_total - n_signal,
                        "n_total": n_total,
                    })
                    # False recruitment: background TCRs that got clustered with target
                    bg_mask = true_labels != epitope
                    if bg_mask.sum() > 0:
                        bg_pred = pred[bg_mask]
                        clustered_bg = sum(1 for p in bg_pred if str(p) not in ("-1", ""))
                        m["false_recruitment_rate"] = clustered_bg / bg_mask.sum()
                        m["n_bg_clustered"] = clustered_bg
                    else:
                        m["false_recruitment_rate"] = 0.0
                        m["n_bg_clustered"] = 0
                    all_results.append(m)

            # Run majority_vote consensus
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
                    "epitope": epitope,
                    "bg_ratio": ratio,
                    "n_signal": n_signal,
                    "n_background": n_total - n_signal,
                    "n_total": n_total,
                    "runtime_s": elapsed,
                    "n_clusters": len(clusters),
                })
                bg_mask = true_labels != epitope
                if bg_mask.sum() > 0:
                    bg_pred = pred[bg_mask]
                    clustered_bg = sum(1 for p in bg_pred if str(p) not in ("-1", ""))
                    m["false_recruitment_rate"] = clustered_bg / bg_mask.sum()
                    m["n_bg_clustered"] = clustered_bg
                else:
                    m["false_recruitment_rate"] = 0.0
                    m["n_bg_clustered"] = 0
                all_results.append(m)

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "exp2_robustness_results.tsv", sep="\t", index=False)

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT 2: BACKGROUND ROBUSTNESS — majority_vote focus")
    print("="*80)

    pivot = results_df.groupby(["method", "bg_ratio"]).agg({
        "ari": "mean", "purity": "mean", "retention": "mean",
        "false_recruitment_rate": "mean",
    }).reset_index()
    print("\nARI × noise ratio:")
    print(pivot.to_string(index=False))

    print("\n--- majority_vote vs best single method per noise level ---")
    for ratio in ratios:
        sub = results_df[results_df["bg_ratio"] == ratio]
        mv = sub[sub["method"] == "majority_vote"]
        single = sub[sub["method"] != "majority_vote"]
        if len(mv) > 0 and len(single) > 0:
            best_single = single.loc[single["ari"].idxmax()]
            mv_row = mv.iloc[0]
            delta = mv_row["ari"] - best_single["ari"]
            print(f"  1:{ratio}: majority_vote ARI={mv_row['ari']:.4f} vs "
                  f"best_single={best_single['method']} ARI={best_single['ari']:.4f} "
                  f"(Δ={delta:+.4f})")

    print(f"\nResults saved to: {output_dir / 'exp2_robustness_results.tsv'}")


if __name__ == "__main__":
    run_exp2("/home/jilin/DeepTCR/tcrconsensus/results/exp2_mv_robustness")
