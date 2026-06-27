#!/usr/bin/env python3
"""Threshold sweep for the consensus on the Donor1 stress test.

Reuses the per-(subset,rep) method assignments cached by stress_test.py
(cheap — no method recompute) and rebuilds the balanced consensus at
several thresholds, plus the conservative mode. Shows the precision /
noise-robustness operating points — the scenario-adaptive story.

Writes results/reproducibility/stress_test/stress_consensus_sweep.tsv.
"""
import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import yaml
import pandas as pd

TCRROOT = Path("/home/jilin/DeepTCR/tcrconsensus")
sys.path.insert(0, str(TCRROOT / "src"))
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
sys.path.insert(0, str(TCRROOT))

from tcrconsensus.io.parser import normalize
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus
from tcrconsensus.refinement.refiner import refine

SUBSET_DIR = Path(
    "/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1/subsets")
EPI_MAP_PATH = Path("/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_mapping.json")
OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test")
CACHE = OUT / "cache"

REPS = 3
MIN_SIZE = 2
METHODS = ["hd_baseline", "giana", "tcrmatch", "clustcr", "gliph2", "tcrdist3", "deeptcr"]
DETERMINISTIC = {"hd_baseline", "giana", "tcrmatch", "gliph2", "tcrdist3"}
THRESHOLDS = [0.3, 0.5, 0.6]


def stress_metrics(tid2cid, cdr3_by_tid, specific_set, epi_map, subset_total, specific_total,
                   min_size=MIN_SIZE):
    cid2 = defaultdict(list)
    for tid, cid in tid2cid.items():
        cid2[cid].append(tid)
    big = {c: m for c, m in cid2.items() if len(m) >= min_size}
    clustered = set()
    for m in big.values():
        clustered |= set(m)
    clustered_specific = sum(1 for t in clustered if cdr3_by_tid.get(t) in specific_set)
    retention_all = len(clustered) / subset_total if subset_total else 0.0
    retention_specific = clustered_specific / specific_total if specific_total else 0.0
    num = den = 0
    for members in big.values():
        rows = []
        for t in members:
            c = cdr3_by_tid.get(t)
            if c in epi_map:
                rows.extend(epi_map[c])
        if rows:
            num += max(Counter(rows).values())
            den += len(members)
    purity_epitope = num / den if den else 0.0
    snum = sden = 0
    for members in big.values():
        n_spec = sum(1 for t in members if cdr3_by_tid.get(t) in specific_set)
        if n_spec >= 1:
            snum += n_spec
            sden += len(members)
    specific_purity = snum / sden if sden else 0.0
    return {"retention_all": retention_all, "retention_specific": retention_specific,
            "purity_epitope": purity_epitope, "specific_purity": specific_purity,
            "n_clustered": len(clustered), "n_clusters": len(big)}


def labels_from_consensus(clusters):
    tid2cid = {}
    for c in clusters:
        for mid in c.member_ids:
            tid2cid[mid] = c.cluster_id
    return tid2cid


def main():
    config = yaml.safe_load(open(TCRROOT / "configs" / "default.yaml")) or {}
    epi_map = json.load(open(EPI_MAP_PATH))["cdr3_to_epitopes"]
    subsets = {i: pd.read_csv(SUBSET_DIR / f"subset_{i}.txt", sep="\t", dtype=str)["cdr3"].tolist()
               for i in range(1, 7)}
    specific_set = set(subsets[1])

    rows = []
    for subset_i in range(1, 7):
        cdr3 = subsets[subset_i]
        subset_total = len(cdr3)
        specific_total = sum(1 for c in cdr3 if c in specific_set)
        df_norm = normalize(pd.DataFrame({"cdr3_beta": cdr3}))
        cdr3_by_tid = dict(zip(df_norm["tcr_id"], df_norm["cdr3_beta"]))
        for rep in range(1, REPS + 1):
            method_assigns = {}
            for method in METHODS:
                eff_rep = 1 if method in DETERMINISTIC else rep
                cf = CACHE / f"s{subset_i}_{method}_r{eff_rep}.pkl"
                if cf.exists():
                    method_assigns[method] = pickle.load(open(cf, "rb"))
            all_assigns = [a for asn in method_assigns.values() for a in asn]
            if not all_assigns:
                continue
            n = len(method_assigns)
            weights = {m: 1.0 / n for m in method_assigns}

            for t in THRESHOLDS:
                try:
                    clusters, _ = balanced_consensus(all_assigns, weights, threshold=t)
                    if clusters:
                        clusters = refine(clusters, _, config)
                    tid2cid = labels_from_consensus(clusters)
                    m = stress_metrics(tid2cid, cdr3_by_tid, specific_set, epi_map,
                                       subset_total, specific_total)
                    rows.append({"subset": subset_i, "noise_frac": 1 - specific_total / subset_total,
                                 "rep": rep, "mode": f"balanced_t{t}", **m})
                    print(f"s{subset_i} r{rep} t{t}: ret_sp={m['retention_specific']:.3f} "
                          f"pur_sp={m['specific_purity']:.3f} pur_epi={m['purity_epitope']:.3f}",
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"s{subset_i} r{rep} t{t} FAILED: {e}", flush=True)

            try:
                clusters, _ = conservative_consensus(all_assigns, weights)
                if clusters:
                    clusters = refine(clusters, _, config)
                tid2cid = labels_from_consensus(clusters)
                m = stress_metrics(tid2cid, cdr3_by_tid, specific_set, epi_map,
                                   subset_total, specific_total)
                rows.append({"subset": subset_i, "noise_frac": 1 - specific_total / subset_total,
                             "rep": rep, "mode": "conservative", **m})
                print(f"s{subset_i} r{rep} conservative: ret_sp={m['retention_specific']:.3f} "
                      f"pur_sp={m['specific_purity']:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"s{subset_i} r{rep} conservative FAILED: {e}", flush=True)

    df = pd.DataFrame(rows)
    out = OUT / "stress_consensus_sweep.tsv"
    df.to_csv(out, sep="\t", index=False, float_format="%.5f")
    print(f"\nwrote {out}: {len(df)} rows")


if __name__ == "__main__":
    main()
