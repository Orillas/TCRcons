#!/usr/bin/env python3
"""Exp3: Component Ablation on core benchmark (4,779 pairs).

Tests marginal contribution of each consensus component:
  1. Full consensus  2. Equal weights  3. Majority vote (no refine)
  4. Intersection-only  5. Union-only  6. No refinement
  7. Leave-one-method-out  8. Random weights (5 seeds)
"""

import sys, time, logging
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
            m = __import__(f"tcrconsensus.clusterers.{mod}", fromlist=[name.title().replace("_","")+"Wrapper"])
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
    le_t.fit(np.unique(true_labels[valid]))
    le_p.fit(np.unique(pred_labels[valid].astype(str)))
    return compute_all_metrics(le_p.transform(pred_labels[valid].astype(str)),
                               le_t.transform(true_labels[valid]), n_total)


def run_consensus(assignments_dict, weights, mode, config, skip_refine=False):
    all_a = []
    for a in assignments_dict.values():
        all_a.extend(a)
    if mode == "conservative":
        clusters, edges = conservative_consensus(all_a, weights)
    elif mode == "coverage":
        clusters, edges = coverage_consensus(all_a, weights)
    else:
        clusters, edges = balanced_consensus(all_a, weights)
    if not skip_refine:
        clusters = refine(clusters, edges, config)
    return clusters


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()._raw
    clusterers = get_clusterers()
    log.info(f"Clusterers: {list(clusterers.keys())}")

    df = pd.read_csv(BENCHMARK_DIR / "benchmark_core_4779.tsv", sep="\t", dtype=str)
    rename_map = {"CDR3_beta": "cdr3_beta", "V_beta": "v_beta", "J_beta": "j_beta",
                  "CDR3_alpha": "cdr3_alpha", "V_alpha": "v_alpha", "J_alpha": "j_alpha",
                  "Epitope": "epitope"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    true_labels = df["epitope"].values
    tcr_ids = df["tcr_id"].values
    n_total = len(df)
    df_norm = normalize(df.copy())
    log.info(f"Dataset: {n_total} TCRs, {df['epitope'].nunique()} epitopes")

    workdir = OUT_DIR / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    # Run all clusterers once
    log.info("Running all clusterers...")
    assignments_dict = {}
    for mname, clusterer in clusterers.items():
        r = clusterer.safe_execute(df_norm, workdir, config)
        if r.status.value == "success" and r.assignments:
            assignments_dict[mname] = r.assignments
    log.info(f"Methods succeeded: {list(assignments_dict.keys())}")

    if len(assignments_dict) < 2:
        log.error("Need >= 2 methods for ablation")
        return

    available = list(assignments_dict.keys())
    all_results = []

    def record(condition, clusters):
        pred = clusters_to_labels(clusters, tcr_ids)
        m = evaluate(pred, true_labels, n_total)
        m["condition"] = condition
        all_results.append(m)
        log.info(f"  {condition}: ARI={m.get('ari',0):.4f}")

    # 1. Full consensus
    w = compute_method_weights(available, "balanced", config)
    record("full_consensus", run_consensus(assignments_dict, w, "balanced", config))

    # 2. Equal weights
    eq_w = {m: 1.0 for m in available}
    record("equal_weights", run_consensus(assignments_dict, eq_w, "balanced", config))

    # 3. Majority vote (no refine)
    record("majority_vote_no_refine", run_consensus(assignments_dict, eq_w, "balanced", config, skip_refine=True))

    # 4. Intersection-only (conservative)
    record("intersection_only", run_consensus(assignments_dict, eq_w, "conservative", config))

    # 5. Union-only (coverage)
    record("union_only", run_consensus(assignments_dict, eq_w, "coverage", config))

    # 6. No refinement
    record("no_refinement", run_consensus(assignments_dict, w, "balanced", config, skip_refine=True))

    # 7. Leave-one-method-out
    lomo = []
    for held in available:
        remaining = {k: v for k, v in assignments_dict.items() if k != held}
        if len(remaining) < 2:
            continue
        rm = list(remaining.keys())
        w2 = compute_method_weights(rm, "balanced", config)
        cond = f"lomo_remove_{held}"
        record(cond, run_consensus(remaining, w2, "balanced", config))
        lomo.append(all_results[-1])

    # 8. Random weights (5 seeds)
    for seed in range(5):
        rng = np.random.RandomState(seed * 100)
        rw = {m: rng.uniform(0.1, 2.0) for m in available}
        record(f"random_weights_seed{seed}", run_consensus(assignments_dict, rw, "balanced", config))

    res = pd.DataFrame(all_results)
    res.to_csv(OUT_DIR / "ablation_results.tsv", sep="\t", index=False)
    if lomo:
        pd.DataFrame(lomo).to_csv(OUT_DIR / "leave_one_method_out.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("EXP3: ABLATION RESULTS (core benchmark 4,779 pairs)")
    print("=" * 80)
    print(res[["condition", "ari", "ami", "nmi", "purity", "sensitivity", "retention"]].to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    run()
