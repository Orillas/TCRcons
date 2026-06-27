#!/usr/bin/env python3
"""Exp3: Component Ablation on benchmark dataset.

Conditions: full/equal_weights/majority_vote/intersection/union/no_refinement/
            leave-one-method-out/random_weights.
"""

import sys, time, logging, copy
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/benchmark_data")
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/bench_exp3")


def get_clusterers():
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    for name, mod in [("clustcr", "clustcr_wrapper"), ("tcrdist3", "tcrdist3_wrapper"),
                      ("gliph2", "gliph2_wrapper"), ("giana", "giana_wrapper"),
                      ("tcrmatch", "tcrmatch_wrapper")]:
        try:
            m = __import__(f"tcrconsensus.clusterers.{mod}", fromlist=["X"])
            cls = getattr(m, [c for c in dir(m) if "Wrapper" in c][0])
            clusterers[name] = cls()
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
    le_t, le_p = LabelEncoder(), LabelEncoder()
    true_str = true_labels[valid]
    pred_str = pred_labels[valid].astype(str)
    le_t.fit(np.unique(true_str))
    le_p.fit(np.unique(pred_str))
    return compute_all_metrics(le_p.transform(pred_str), le_t.transform(true_str), n_total)


def run_consensus_with(assignments_dict, weights, mode, config, skip_refine=False):
    all_a = []
    for a in assignments_dict.values():
        all_a.extend(a)
    fn = {"conservative": conservative_consensus, "coverage": coverage_consensus}.get(mode, balanced_consensus)
    clusters, edges = fn(all_a, weights)
    if not skip_refine:
        clusters = refine(clusters, edges, config)
    return clusters


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()._raw
    clusterers = get_clusterers()
    log.info(f"Clusterers: {list(clusterers.keys())}")

    df = pd.read_csv(BENCHMARK_DIR / "benchmark_main.tsv", sep="\t", dtype=str)
    df = df.rename(columns={"CDR3_beta": "cdr3_beta", "V_beta": "v_beta", "J_beta": "j_beta", "Epitope": "epitope"})
    true_labels = df["epitope"].values
    tcr_ids = df["tcr_id"].values
    n_total = len(df)
    df_norm = normalize(df.copy())
    log.info(f"Dataset: {n_total} TCRs, {df['epitope'].nunique()} epitopes")

    workdir = OUT_DIR / "work"
    workdir.mkdir(exist_ok=True)

    # Run all clusterers once
    log.info("Running all clusterers...")
    assignments_dict = {}
    for mname, cl in clusterers.items():
        r = cl.safe_execute(df_norm, workdir / mname, config)
        if r.status.value == "success" and r.assignments:
            assignments_dict[mname] = r.assignments
            log.info(f"  {mname}: {len(r.assignments)} assignments")
    log.info(f"Methods succeeded: {list(assignments_dict.keys())}")

    if len(assignments_dict) < 2:
        log.error("Need >= 2 methods for ablation")
        return

    available = list(assignments_dict.keys())
    all_results = []

    # 1. Full consensus
    log.info("1. Full consensus")
    weights = compute_method_weights(available, "balanced", config)
    pred = clusters_to_labels(run_consensus_with(assignments_dict, weights, "balanced", config), tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "full_consensus"
    all_results.append(m)

    # 2. Equal weights
    log.info("2. Equal weights")
    eq_w = {m: 1.0 for m in available}
    pred = clusters_to_labels(run_consensus_with(assignments_dict, eq_w, "balanced", config), tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "equal_weights"
    all_results.append(m)

    # 3. Majority vote (no refinement)
    log.info("3. Majority vote (no refinement)")
    pred = clusters_to_labels(run_consensus_with(assignments_dict, eq_w, "balanced", config, skip_refine=True), tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "majority_vote_no_refine"
    all_results.append(m)

    # 4. Intersection-only (conservative)
    log.info("4. Intersection-only")
    pred = clusters_to_labels(run_consensus_with(assignments_dict, eq_w, "conservative", config), tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "intersection_only"
    all_results.append(m)

    # 5. Union-only (coverage)
    log.info("5. Union-only")
    pred = clusters_to_labels(run_consensus_with(assignments_dict, eq_w, "coverage", config), tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "union_only"
    all_results.append(m)

    # 6. No refinement
    log.info("6. No refinement")
    pred = clusters_to_labels(run_consensus_with(assignments_dict, weights, "balanced", config, skip_refine=True), tcr_ids)
    m = evaluate(pred, true_labels, n_total)
    m["condition"] = "no_refinement"
    all_results.append(m)

    # 7. Leave-one-method-out
    for held_out in available:
        log.info(f"7. LOMO: remove {held_out}")
        remaining = {k: v for k, v in assignments_dict.items() if k != held_out}
        if len(remaining) < 2:
            continue
        w = compute_method_weights(list(remaining.keys()), "balanced", config)
        pred = clusters_to_labels(run_consensus_with(remaining, w, "balanced", config), tcr_ids)
        m = evaluate(pred, true_labels, n_total)
        m["condition"] = f"lomo_remove_{held_out}"
        m["held_out_method"] = held_out
        all_results.append(m)

    # 8. Random weights (5 seeds)
    for seed in range(5):
        log.info(f"8. Random weights seed={seed}")
        rng = np.random.RandomState(seed * 100)
        rand_w = {m: rng.uniform(0.1, 2.0) for m in available}
        pred = clusters_to_labels(run_consensus_with(assignments_dict, rand_w, "balanced", config), tcr_ids)
        m = evaluate(pred, true_labels, n_total)
        m["condition"] = f"random_weights_seed{seed}"
        all_results.append(m)

    # Save
    res = pd.DataFrame(all_results)
    res.to_csv(OUT_DIR / "ablation_results.tsv", sep="\t", index=False)

    lomo = [r for r in all_results if "lomo" in r.get("condition", "")]
    if lomo:
        pd.DataFrame(lomo).to_csv(OUT_DIR / "leave_one_method_out.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("EXP3: ABLATION RESULTS")
    print("=" * 80)
    show = ["condition", "ari", "ami", "nmi", "purity", "sensitivity", "retention", "v_measure"]
    print(res[[c for c in show if c in res.columns]].to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    run()
