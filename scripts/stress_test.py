#!/usr/bin/env python3
"""Stress test (STRESS_TEST.md): i3-unit 10X Donor1 noise-spike-in robustness.

6 subsets (2876 -> 26561 seqs): fixed pool of 2876 antigen-specific CDR3b
+ increasing non-specific noise. Runs all single methods + balanced
consensus, 3 reps. Reports retention + purity as noise grows.

Metrics (faithful to the i3-unit benchmark + STRESS_TEST.md sec.3):
  retention_all       = |clustered in size>=2| / |subset|
  retention_specific  = |specific clustered| / |specific in subset|     (sec.3.1)
  purity_epitope      = sum(majority-epitope rows) / sum(size) over
                        clusters size>=2 with >=1 annotated member (melt,
                        replicates purity_function.R)
  specific_purity     = specific fraction within specificity-containing
                        clusters  (= 1 - noise contamination, sec.3.2)

Design: each (subset, method, rep) runs in an isolated subprocess with a
hard timeout; results are cached on disk so the run is resumable.
Deterministic methods reuse rep-1 across reps; stochastic methods (clustcr,
deeptcr) run all 3 reps. Writes results incrementally per subset.
"""
import sys
import json
import time
import pickle
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

import yaml
import pandas as pd

TCRROOT = Path("/home/jilin/DeepTCR/tcrconsensus")
sys.path.insert(0, str(TCRROOT / "src"))
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
sys.path.insert(0, str(TCRROOT))

from tcrconsensus.io.parser import normalize
from scripts.exp_shared import majority_vote_consensus

VENV_PY = "/home/jilin/DeepTCR/.venv/bin/python"
STRESS_ONE = TCRROOT / "scripts" / "stress_one.py"

BASE = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e")
SUBSET_DIR = BASE / "Data/10X/Donor1/subsets"
EPI_MAP_PATH = Path("/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_mapping.json")

OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test")
CACHE = OUT / "cache"

REPS = 3
MIN_SIZE = 2
METHODS = ["hd_baseline", "giana", "tcrmatch", "clustcr", "gliph2", "tcrdist3", "deeptcr"]
DETERMINISTIC = {"hd_baseline", "giana", "tcrmatch", "gliph2", "tcrdist3"}
TIMEOUTS = {
    "hd_baseline": 900, "giana": 1800, "tcrmatch": 2400, "clustcr": 2400,
    "gliph2": 2400, "tcrdist3": 5400, "deeptcr": 5400,
}
SUBSET_SIZE = {1: 2876, 2: 7613, 3: 12350, 4: 17087, 5: 21824, 6: 26561}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_subsets():
    return {i: pd.read_csv(SUBSET_DIR / f"subset_{i}.txt", sep="\t", dtype=str)["cdr3"].tolist()
            for i in range(1, 7)}


def ensure_method(subset_i, method, rep):
    """Run stress_one.py for this (subset, method, rep) if not cached. Returns pkl path or None."""
    cf = CACHE / f"s{subset_i}_{method}_r{rep}.pkl"
    if cf.exists():
        return cf
    timeout = TIMEOUTS.get(method, 2400)
    log(f"  run {method} s{subset_i} r{rep} (timeout {timeout}s)")
    t0 = time.time()
    try:
        p = subprocess.run(
            [VENV_PY, str(STRESS_ONE), str(subset_i), method, str(rep)],
            capture_output=True, text=True, timeout=timeout,
        )
        dt = time.time() - t0
        tail = (p.stdout or "").strip().splitlines()[-1:] or [(p.stderr or "").strip()[-200:]]
        log(f"    -> {dt:.0f}s  {' '.join(tail)}")
    except subprocess.TimeoutExpired:
        log(f"    -> TIMEOUT after {timeout}s ({method} s{subset_i} r{rep})")
    return cf if cf.exists() else None


def load_assignments(cf):
    try:
        return pickle.load(open(cf, "rb"))
    except Exception as e:  # noqa: BLE001
        log(f"    cache load failed {cf}: {e}")
        return []


def stress_metrics(tid2cid, cdr3_by_tid, specific_set, epi_map, subset_total, specific_total,
                   min_size=MIN_SIZE):
    cid2members = defaultdict(list)
    for tid, cid in tid2cid.items():
        cid2members[cid].append(tid)
    big = {cid: m for cid, m in cid2members.items() if len(m) >= min_size}
    clustered = set()
    for m in big.values():
        clustered |= set(m)

    clustered_specific = sum(1 for t in clustered if cdr3_by_tid.get(t) in specific_set)
    retention_all = len(clustered) / subset_total if subset_total else 0.0
    retention_specific = clustered_specific / specific_total if specific_total else 0.0

    # epitope purity (faithful melt): clusters size>=min_size with >=1 annotated member
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

    # specific purity (1 - contamination) over specificity-containing clusters
    snum = sden = 0
    for members in big.values():
        n_spec = sum(1 for t in members if cdr3_by_tid.get(t) in specific_set)
        if n_spec >= 1:
            snum += n_spec
            sden += len(members)
    specific_purity = snum / sden if sden else 0.0

    return {
        "retention_all": retention_all,
        "retention_specific": retention_specific,
        "purity_epitope": purity_epitope,
        "specific_purity": specific_purity,
        "n_clustered": len(clustered),
        "n_clusters": len(big),
    }


def labels_from_assignments(assigns):
    tid2cid = {}
    for a in assigns:
        tid2cid.setdefault(a.tcr_id, a.cluster_id)  # first assignment wins
    return tid2cid


def labels_from_consensus(clusters):
    tid2cid = {}
    for c in clusters:
        for mid in c.member_ids:
            tid2cid[mid] = c.cluster_id
    return tid2cid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (OUT / "work").mkdir(parents=True, exist_ok=True)

    subsets = load_subsets()
    epi_map = json.load(open(EPI_MAP_PATH))["cdr3_to_epitopes"]
    specific_set = set(subsets[1])  # subset_1 = the 2876 annotated specific pool
    log(f"specific pool (subset_1): {len(specific_set)} unique; epi_map: {len(epi_map)} annotated")

    config = yaml.safe_load(open(TCRROOT / "configs" / "default.yaml")) or {}
    tsv_path = OUT / "stress_results.tsv"
    rows = []

    for subset_i in range(1, 7):
        cdr3 = subsets[subset_i]
        subset_total = len(cdr3)
        specific_total = sum(1 for c in cdr3 if c in specific_set)
        df_norm = normalize(pd.DataFrame({"cdr3_beta": cdr3}))
        tcr_ids = df_norm["tcr_id"].tolist()
        cdr3_by_tid = dict(zip(tcr_ids, df_norm["cdr3_beta"]))
        log(f"=== subset {subset_i}: {subset_total} seqs, {specific_total} specific ===")

        # ensure rep1 method caches exist first (deterministic reuse for rep2/3)
        for rep in range(1, REPS + 1):
            method_assigns = {}
            for method in METHODS:
                eff_rep = 1 if method in DETERMINISTIC else rep
                cf = ensure_method(subset_i, method, eff_rep)
                if cf:
                    method_assigns[method] = load_assignments(cf)

            # single-method metrics
            for method, assigns in method_assigns.items():
                tid2cid = labels_from_assignments(assigns)
                m = stress_metrics(tid2cid, cdr3_by_tid, specific_set, epi_map,
                                   subset_total, specific_total)
                rows.append({"subset": subset_i, "subset_size": subset_total,
                             "noise_frac": 1 - specific_total / subset_total,
                             "rep": rep, "method": method, "kind": "single", **m})
                _append_tsv(tsv_path, rows[-1])

            # consensus
            all_assigns = [a for assigns in method_assigns.values() for a in assigns]
            methods_used = ",".join(sorted(method_assigns))
            if all_assigns:
                try:
                    clusters, _edges = majority_vote_consensus(all_assigns, config)
                    tid2cid = labels_from_consensus(clusters)
                    m = stress_metrics(tid2cid, cdr3_by_tid, specific_set, epi_map,
                                       subset_total, specific_total)
                    m["methods_used"] = methods_used
                    rows.append({"subset": subset_i, "subset_size": subset_total,
                                 "noise_frac": 1 - specific_total / subset_total,
                                 "rep": rep, "method": "consensus", "kind": "consensus", **m})
                    _append_tsv(tsv_path, rows[-1])
                    log(f"  consensus r{rep}: {len(method_assigns)} methods, "
                        f"ret_sp={m['retention_specific']:.3f} pur_sp={m['specific_purity']:.3f}")
                except Exception as e:  # noqa: BLE001
                    log(f"  consensus r{rep} FAILED: {e}")
            else:
                log(f"  consensus r{rep}: no method assignments available")

        # checkpoint after each subset
        _summarize(rows, OUT / "stress_summary.tsv")
        json.dump(rows, open(OUT / "stress_results.json", "w"), indent=1)
        log(f"--- subset {subset_i} done, checkpoint written ---")

    _summarize(rows, OUT / "stress_summary.tsv")
    json.dump(rows, open(OUT / "stress_results.json", "w"), indent=1)
    log(f"ALL DONE. {len(rows)} rows. results in {OUT}")


TSV_COLS = ["subset", "subset_size", "noise_frac", "rep", "method", "kind",
            "retention_all", "retention_specific", "purity_epitope", "specific_purity",
            "n_clustered", "n_clusters", "methods_used"]


def _append_tsv(path, row):
    import os
    write_header = not path.exists() or path.stat().st_size == 0
    extra_keys = [k for k in TSV_COLS if k in row]
    with open(path, "a") as f:
        if write_header:
            f.write("\t".join(TSV_COLS) + "\n")
        vals = []
        for k in TSV_COLS:
            v = row.get(k, "")
            vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
        f.write("\t".join(vals) + "\n")


def _summarize(rows, path):
    agg = defaultdict(list)
    for r in rows:
        agg[(r["subset"], r["noise_frac"], r["method"], r["kind"])].append(r)
    import statistics as st
    with open(path, "w") as f:
        f.write("subset\tnoise_frac\tmethod\tkind\tn_reps\t"
                "retention_specific_mean\tretention_specific_sd\t"
                "purity_epitope_mean\tspecific_purity_mean\tretention_all_mean\n")
        for (s, nf, mth, kind), rs in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][3], x[0][2])):
            def ms(key):
                v = [x[key] for x in rs if key in x]
                return st.mean(v) if v else float("nan")
            def sd(key):
                v = [x[key] for x in rs if key in x]
                return st.pstdev(v) if len(v) > 1 else 0.0
            f.write(f"{s}\t{nf:.4f}\t{mth}\t{kind}\t{len(rs)}\t"
                    f"{ms('retention_specific'):.4f}\t{sd('retention_specific'):.4f}\t"
                    f"{ms('purity_epitope'):.4f}\t{ms('specific_purity'):.4f}\t"
                    f"{ms('retention_all'):.4f}\n")


if __name__ == "__main__":
    main()
