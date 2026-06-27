#!/usr/bin/env python3
"""Experiment 4: Adaptive Recommendation Generalization.

Tests whether auto-selected mode performs better than fixed balanced
using leave-one-epitope-out evaluation.
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
from tcrconsensus.profiling.profiler import profile as compute_profile
from tcrconsensus.selection.selector import select_methods
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
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


def run_consensus(df_norm, clusterers, mode, config, workdir):
    all_a = []
    methods = []
    for mname, clusterer in clusterers.items():
        r = clusterer.safe_execute(df_norm, workdir, config)
        if r.status.value == "success" and r.assignments:
            all_a.extend(r.assignments)
            methods.append(mname)

    if len(all_a) < 2:
        return []

    weights = compute_method_weights(methods, "balanced", config)

    if mode == "conservative":
        clusters, edges = conservative_consensus(all_a, weights)
    elif mode == "coverage":
        clusters, edges = coverage_consensus(all_a, weights)
    else:
        clusters, edges = balanced_consensus(all_a, weights)

    return refine(clusters, edges, config)


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


def run_experiment(data_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    cfg = config._raw
    clusterers = get_clusterers()

    # Load VDJdb
    vdj = pd.read_csv(f"{data_dir}/vdjdb_filtered.tsv", sep="\t", dtype=str)
    labels = pd.read_csv(f"{data_dir}/vdjdb_labels.tsv", sep="\t", dtype=str)
    df = vdj.merge(labels, on="tcr_id", how="inner", suffixes=("", "_label"))
    epitope_col = "epitope_label" if "epitope_label" in df.columns else "epitope"
    df = df[df[epitope_col].notna()].reset_index(drop=True)

    epitopes = df[epitope_col].unique()
    logger.info(f"Total: {len(df)} TCRs, {len(epitopes)} epitopes")

    # Only test epitopes with >= 20 TCRs for statistical reliability
    epi_counts = df[epitope_col].value_counts()
    test_epis = epi_counts[epi_counts >= 20].index.tolist()
    logger.info(f"Testing {len(test_epis)} epitopes with >= 20 TCRs")

    all_results = []

    for held_out_epi in test_epis:
        n_held = (df[epitope_col] == held_out_epi).sum()
        test_df = df[df[epitope_col] == held_out_epi].copy()
        train_df = df[df[epitope_col] != held_out_epi].copy()

        logger.info(f"\nHeld-out: {held_out_epi} ({n_held} TCRs)")

        # Normalize test set
        test_norm = normalize(test_df.copy())
        tcr_ids = test_norm["tcr_id"].values
        true_labels = test_df[epitope_col].values
        n_total = len(test_norm)

        workdir = output_dir / f"work/{held_out_epi[:30]}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Strategy 1: Auto-selected mode via selector
        train_norm = normalize(train_df.copy())
        prof = compute_profile(train_norm, cfg)
        plan = select_methods(prof, "balanced", cfg)
        auto_mode = plan.consensus_mode.value

        t0 = time.time()
        clusters = run_consensus(test_norm, clusterers, auto_mode, cfg, workdir / "auto")
        elapsed = time.time() - t0

        pred = clusters_to_labels(clusters, tcr_ids)
        m = evaluate(pred, true_labels, n_total)
        m.update({
            "strategy": "auto_selected",
            "selected_mode": auto_mode,
            "held_out_epitope": held_out_epi,
            "n_test": n_total,
            "runtime_s": elapsed,
        })
        all_results.append(m)

        # Strategy 2: Fixed balanced mode
        t0 = time.time()
        clusters = run_consensus(test_norm, clusterers, "balanced", cfg, workdir / "balanced")
        elapsed = time.time() - t0

        pred = clusters_to_labels(clusters, tcr_ids)
        m = evaluate(pred, true_labels, n_total)
        m.update({
            "strategy": "fixed_balanced",
            "selected_mode": "balanced",
            "held_out_epitope": held_out_epi,
            "n_test": n_total,
            "runtime_s": elapsed,
        })
        all_results.append(m)

        # Strategy 3: Single best method (hd_baseline)
        hd = HDBaselineClusterer()
        r = hd.safe_execute(test_norm, workdir / "hd", cfg)
        if r.status.value == "success" and r.assignments:
            label_map = {}
            for a in r.assignments:
                if a.tcr_id not in label_map:
                    label_map[a.tcr_id] = a.cluster_id
            pred = np.array([label_map.get(tid, -1) for tid in tcr_ids])
            m = evaluate(pred, true_labels, n_total)
            m.update({
                "strategy": "single_hd_baseline",
                "selected_mode": "single",
                "held_out_epitope": held_out_epi,
                "n_test": n_total,
            })
            all_results.append(m)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "generalization_results.tsv", sep="\t", index=False)

    print("\n" + "="*80)
    print("EXPERIMENT 4: GENERALIZATION RESULTS")
    print("="*80)

    summary = results_df.groupby("strategy").agg({
        "ari": ["mean", "std"],
        "ami": ["mean", "std"],
        "purity": ["mean", "std"],
        "sensitivity": ["mean", "std"],
    }).reset_index()
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    run_experiment(
        "/home/jilin/DeepTCR/tcrconsensus/results/data",
        "/home/jilin/DeepTCR/tcrconsensus/results/exp4_generalization",
    )
