#!/usr/bin/env python3
"""Stress test on RECONSTRUCTED Donor1 subsets (CDR3beta + Vbeta + Jbeta).

Controlled counterpart to stress_test.py: identical unique CDR3beta pool and
noise structure; the only difference is V/J genes restored (the signal the
CDR3beta-only test discarded). Tests whether consensus recovers when its
member methods (tcrdist3/gliph2) regain their V/J structural signal.

Separate cache + output from the CDR3beta-only run.
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
STRESS_ONE = TCRROOT / "scripts" / "stress_one_rich.py"

SUBSET_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/data/reconstruct_data")
EPI_MAP_PATH = Path("/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_mapping.json")

OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test_rich")
CACHE = OUT / "cache"

REPS = 3
MIN_SIZE = 2
METHODS = ["hd_baseline", "giana", "tcrmatch", "clustcr", "gliph2", "tcrdist3", "deeptcr"]
DETERMINISTIC = {"hd_baseline", "giana", "tcrmatch", "gliph2", "tcrdist3"}
TIMEOUTS = {"hd_baseline": 900, "giana": 1800, "tcrmatch": 2400, "clustcr": 2400,
            "gliph2": 2400, "tcrdist3": 5400, "deeptcr": 5400}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_subsets():
    return {i: pd.read_csv(SUBSET_DIR / f"subset_{i}.tsv", sep="\t", dtype=str)
            for i in range(1, 7)}


def ensure_method(subset_i, method, rep):
    cf = CACHE / f"s{subset_i}_{method}_r{rep}.pkl"
    if cf.exists():
        return cf
    timeout = TIMEOUTS.get(method, 2400)
    log(f"  run {method} s{subset_i} r{rep} (timeout {timeout}s)")
    t0 = time.time()
    try:
        p = subprocess.run(
            [VENV_PY, str(STRESS_ONE), str(subset_i), method, str(rep)],
            capture_output=True, text=True, timeout=timeout)
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


def labels_from_assignments(assigns):
    tid2cid = {}
    for a in assigns:
        tid2cid.setdefault(a.tcr_id, a.cluster_id)
    return tid2cid


def labels_from_consensus(clusters):
    tid2cid = {}
    for c in clusters:
        for mid in c.member_ids:
            tid2cid[mid] = c.cluster_id
    return tid2cid


TSV_COLS = ["subset", "subset_size", "noise_frac", "rep", "method", "kind",
            "retention_all", "retention_specific", "purity_epitope", "specific_purity",
            "n_clustered", "n_clusters", "methods_used"]


def _append_tsv(path, row):
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a") as f:
        if write_header:
            f.write("\t".join(TSV_COLS) + "\n")
        vals = []
        for k in TSV_COLS:
            v = row.get(k, "")
            vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
        f.write("\t".join(vals) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (OUT / "work").mkdir(parents=True, exist_ok=True)

    subsets = load_subsets()
    epi_map = json.load(open(EPI_MAP_PATH))["cdr3_to_epitopes"]
    specific_set = set(subsets[1]["cdr3_beta"])
    log(f"RICH run. specific pool (subset_1): {len(specific_set)} unique; "
        f"epi_map: {len(epi_map)} annotated")

    config = yaml.safe_load(open(TCRROOT / "configs" / "default.yaml")) or {}
    tsv_path = OUT / "stress_results.tsv"

    for subset_i in range(1, 7):
        sub_df = subsets[subset_i]
        cdr3 = sub_df["cdr3_beta"].tolist()
        subset_total = len(cdr3)
        specific_total = sum(1 for c in cdr3 if c in specific_set)
        df_norm = normalize(sub_df.copy())
        cdr3_by_tid = dict(zip(df_norm["tcr_id"], df_norm["cdr3_beta"]))
        log(f"=== subset {subset_i}: {subset_total} seqs, {specific_total} specific ===")

        for rep in range(1, REPS + 1):
            method_assigns = {}
            for method in METHODS:
                eff_rep = 1 if method in DETERMINISTIC else rep
                cf = ensure_method(subset_i, method, eff_rep)
                if cf:
                    method_assigns[method] = load_assignments(cf)

            for method, assigns in method_assigns.items():
                tid2cid = labels_from_assignments(assigns)
                m = stress_metrics(tid2cid, cdr3_by_tid, specific_set, epi_map,
                                   subset_total, specific_total)
                rows = {"subset": subset_i, "subset_size": subset_total,
                        "noise_frac": 1 - specific_total / subset_total,
                        "rep": rep, "method": method, "kind": "single", **m}
                _append_tsv(tsv_path, rows)

            all_assigns = [a for asn in method_assigns.values() for a in asn]
            methods_used = ",".join(sorted(method_assigns))
            if all_assigns:
                try:
                    clusters, _edges = majority_vote_consensus(all_assigns, config)
                    tid2cid = labels_from_consensus(clusters)
                    m = stress_metrics(tid2cid, cdr3_by_tid, specific_set, epi_map,
                                       subset_total, specific_total)
                    m["methods_used"] = methods_used
                    rows = {"subset": subset_i, "subset_size": subset_total,
                            "noise_frac": 1 - specific_total / subset_total,
                            "rep": rep, "method": "consensus", "kind": "consensus", **m}
                    _append_tsv(tsv_path, rows)
                    log(f"  consensus r{rep}: {len(method_assigns)} methods, "
                        f"ret_sp={m['retention_specific']:.3f} pur_sp={m['specific_purity']:.3f}")
                except Exception as e:  # noqa: BLE001
                    log(f"  consensus r{rep} FAILED: {e}")
            else:
                log(f"  consensus r{rep}: no method assignments available")

        json.dump({"subset": subset_i, "done": True}, open(OUT / f"checkpoint_s{subset_i}.json", "w"))
        log(f"--- subset {subset_i} done, checkpoint written ---")

    log(f"RICH RUN ALL DONE. results in {OUT}")


if __name__ == "__main__":
    main()
