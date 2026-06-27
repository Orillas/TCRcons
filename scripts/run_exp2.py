#!/usr/bin/env python3
"""Experiment 2: Background Robustness Stress Test.

Injects unrelated background TCRs at various ratios and measures
how well consensus maintains labeled-cluster quality.
"""

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
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.consensus.weights import compute_method_weights
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


def run_clustering(df_norm, clusterers, config, workdir):
    """Run all clusterers + consensus balanced."""
    all_assignments = []
    method_status = {}
    for mname, clusterer in clusterers.items():
        result = clusterer.safe_execute(df_norm, workdir, config)
        if result.status.value == "success":
            all_assignments.extend(result.assignments)
            method_status[mname] = "ok"
        else:
            method_status[mname] = "failed"

    if len(all_assignments) < 2:
        return [], {}

    methods = list(set(a.method for a in all_assignments))
    weights = compute_method_weights(methods, "balanced", config)
    clusters, edges = balanced_consensus(all_assignments, weights)
    clusters = refine(clusters, edges, config)
    return clusters, method_status


def evaluate(pred_labels, true_labels, n_total):
    """Evaluate clustering vs true labels."""
    valid = np.array([
        str(p) not in ("-1", "") and str(t) not in ("BACKGROUND", "")
        for p, t in zip(pred_labels, true_labels)
    ], dtype=bool)
    if valid.sum() < 2:
        return {}

    le_t = LabelEncoder()
    le_p = LabelEncoder()
    true_str = true_labels[valid]
    pred_str = pred_labels[valid].astype(str)
    le_t.fit(np.unique(true_str))
    le_p.fit(np.unique(pred_str))
    t_enc = le_t.transform(true_str)
    p_enc = le_p.transform(pred_str)

    metrics = compute_all_metrics(p_enc, t_enc, n_total)

    # False recruitment rate
    bg_mask = true_labels == "BACKGROUND"
    if bg_mask.sum() > 0:
        bg_pred = pred_labels[bg_mask]
        clustered_bg = sum(1 for p in bg_pred if str(p) not in ("-1", ""))
        metrics["false_recruitment_rate"] = clustered_bg / bg_mask.sum()
        metrics["n_bg_clustered"] = float(clustered_bg)
        metrics["n_bg_total"] = float(bg_mask.sum())
    else:
        metrics["false_recruitment_rate"] = 0.0

    return metrics


def run_experiment(data_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    cfg = config._raw
    clusterers = get_clusterers()

    # Load signal TCRs (top epitopes from VDJdb)
    vdj = pd.read_csv(f"{data_dir}/vdjdb_filtered.tsv", sep="\t", dtype=str)
    epi_counts = vdj["epitope"].value_counts()
    top_epis = epi_counts.head(10).index.tolist()

    # Load background (VDJdb score=0)
    logger.info("Loading background TCRs from VDJdb score=0...")
    vdj_full = pd.read_csv("/home/jilin/DeepTCR/Data/VDJdb.tsv", sep="\t", dtype=str,
                            keep_default_na=False)
    vdj_full["Score"] = pd.to_numeric(vdj_full["Score"], errors="coerce").fillna(0).astype(int)
    bg = vdj_full[(vdj_full["Gene"] == "TRB") & (vdj_full["Score"] == 0)].copy()
    bg = bg.drop_duplicates(subset=["CDR3"]).head(100000).copy()
    bg["tcr_id"] = ["bg_" + str(i).zfill(6) for i in range(len(bg))]
    bg["epitope"] = "BACKGROUND"
    # Rename to match filtered data column names
    bg = bg.rename(columns={"CDR3": "cdr3_beta", "V": "v_beta", "J": "j_beta"})
    logger.info(f"  Background pool: {len(bg)} TCRs")

    ratios = [10, 100, 1000]  # signal:background
    all_results = []

    for epitope in top_epis:
        signal = vdj[vdj["epitope"] == epitope].copy()
        n_signal = len(signal)
        if n_signal < 10:
            continue

        logger.info(f"\nEpitope: {epitope} ({n_signal} signal TCRs)")

        for ratio in ratios:
            n_bg = int(n_signal * ratio)
            bg_sample = bg.sample(n=min(n_bg, len(bg)), replace=False).copy()

            # Merge
            mixed = pd.concat([
                signal[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
                bg_sample[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
            ], ignore_index=True)

            true_labels = mixed["epitope"].values
            n_total = len(mixed)
            df_norm = normalize(mixed.copy())

            workdir = output_dir / f"work/{epitope}_{ratio}"
            workdir.mkdir(parents=True, exist_ok=True)

            logger.info(f"  Ratio 1:{ratio} ({n_signal} signal + {len(bg_sample)} bg = {n_total})")

            # Run consensus
            t0 = time.time()
            clusters, status = run_clustering(df_norm, clusterers, cfg, workdir)
            elapsed = time.time() - t0

            pred = clusters_to_labels(clusters, df_norm["tcr_id"].values)
            metrics = evaluate(pred, true_labels, n_total)
            metrics.update({
                "epitope": epitope,
                "ratio": f"1:{ratio}",
                "n_signal": n_signal,
                "n_background": len(bg_sample),
                "n_total": n_total,
                "runtime_s": elapsed,
                "n_clusters": len(clusters),
            })
            all_results.append(metrics)

            # Also run HD baseline alone for comparison
            hd = HDBaselineClusterer()
            result = hd.safe_execute(df_norm, workdir, cfg)
            if result.status.value == "success":
                label_map = {}
                for a in result.assignments:
                    if a.tcr_id not in label_map:
                        label_map[a.tcr_id] = a.cluster_id
                pred_hd = np.array([label_map.get(tid, -1) for tid in df_norm["tcr_id"].values])
                hd_metrics = evaluate(pred_hd, true_labels, n_total)
                hd_metrics.update({
                    "epitope": epitope,
                    "ratio": f"1:{ratio}",
                    "n_signal": n_signal,
                    "n_background": len(bg_sample),
                    "n_total": n_total,
                    "method": "hd_baseline_only",
                })
                all_results.append(hd_metrics)

            metrics["method"] = "consensus_balanced"

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "robustness_results.tsv", sep="\t", index=False)

    print("\n" + "="*80)
    print("EXPERIMENT 2: BACKGROUND ROBUSTNESS RESULTS")
    print("="*80)
    pivot = results_df.groupby(["epitope", "ratio", "method"]).agg({
        "ari": "mean", "purity": "mean", "retention": "mean",
        "false_recruitment_rate": "mean", "n_clusters": "mean"
    }).reset_index()
    print(pivot.to_string(index=False))
    print(f"\nResults saved to: {output_dir / 'robustness_results.tsv'}")


if __name__ == "__main__":
    run_experiment(
        "/home/jilin/DeepTCR/tcrconsensus/results/data",
        "/home/jilin/DeepTCR/tcrconsensus/results/exp2_robustness",
    )
