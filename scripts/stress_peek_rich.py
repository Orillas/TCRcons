#!/usr/bin/env python3
"""Early peek: rich (V/J) subsets 1-2 vs CDR3beta-only, rep 1.

Runs missing method caches for rich s1/s2 (via stress_one_rich), builds
consensus, computes metrics, and prints a side-by-side vs the CDR3beta-only
results.tsv. Does NOT touch subsets 3-6 (left for the chained full run).
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
EPI = json.load(open("/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_mapping.json"))["cdr3_to_epitopes"]
RICH_OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test_rich")
CACHE = RICH_OUT / "cache"
BETA_TSV = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test/stress_results.tsv")

METHODS = ["hd_baseline", "giana", "tcrmatch", "clustcr", "gliph2", "tcrdist3", "deeptcr"]
TIMEOUTS = {"hd_baseline": 900, "giana": 1800, "tcrmatch": 2400, "clustcr": 2400,
            "gliph2": 2400, "tcrdist3": 5400, "deeptcr": 5400}
MIN_SIZE = 2


def metrics(tid2cid, cby, spec, total, sp_total):
    cid2 = defaultdict(list)
    for t, c in tid2cid.items():
        cid2[c].append(t)
    big = {c: m for c, m in cid2.items() if len(m) >= MIN_SIZE}
    clu = set()
    for m in big.values():
        clu |= set(m)
    cspec = sum(1 for t in clu if cby.get(t) in spec)
    num = den = 0
    for m in big.values():
        rows = [e for t in m for e in EPI.get(cby.get(t), [])]
        if rows:
            num += max(Counter(rows).values()); den += len(m)
    snum = sden = 0
    for m in big.values():
        ns = sum(1 for t in m if cby.get(t) in spec)
        if ns:
            snum += ns; sden += len(m)
    return {"ret_sp": cspec / sp_total if sp_total else 0,
            "pur_sp": snum / sden if sden else 0,
            "pur_epi": num / den if den else 0}


def labs_a(asn):
    d = {}
    for a in asn:
        d.setdefault(a.tcr_id, a.cluster_id)
    return d


def labs_c(cl):
    d = {}
    for c in cl:
        for m in c.member_ids:
            d[m] = c.cluster_id
    return d


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(TCRROOT / "configs" / "default.yaml")) or {}
    spec = set(pd.read_csv(SUBSET_DIR / "subset_1.tsv", sep="\t", dtype=str)["cdr3_beta"])

    # CDR3beta-only reference (mean over reps) for s1,s2
    beta = pd.read_csv(BETA_TSV, sep="\t")
    beta = beta[beta.subset.isin([1, 2])].groupby(["subset", "method"])[
        ["retention_specific", "specific_purity", "purity_epitope"]].mean()

    for s in [1, 2]:
        df = pd.read_csv(SUBSET_DIR / f"subset_{s}.tsv", sep="\t", dtype=str)
        cdr3 = df["cdr3_beta"].tolist()
        total = len(cdr3)
        sp_total = sum(1 for c in cdr3 if c in spec)
        dn = normalize(df.copy())
        cby = dict(zip(dn["tcr_id"], dn["cdr3_beta"]))
        print(f"\n========== rich subset {s} (noise {1 - sp_total/total:.2f}) ==========")
        m_assigns = {}
        for mth in METHODS:
            cf = CACHE / f"s{s}_{mth}_r1.pkl"
            if not cf.exists():
                print(f"  running {mth} s{s}...", flush=True)
                t0 = time.time()
                try:
                    subprocess.run([VENV_PY, str(STRESS_ONE), str(s), mth, "1"],
                                   capture_output=True, text=True, timeout=TIMEOUTS[mth])
                except subprocess.TimeoutExpired:
                    print(f"    TIMEOUT {mth}")
                print(f"    {time.time()-t0:.0f}s")
            if cf.exists():
                m_assigns[mth] = pickle.load(open(cf, "rb"))

        # single-method metrics
        print("%-12s %8s %8s %8s   | CDR3b-only: ret_sp pur_sp pur_epi" % ("method", "ret_sp", "pur_sp", "pur_epi"))
        for mth, asn in m_assigns.items():
            r = metrics(labs_a(asn), cby, spec, total, sp_total)
            b = beta.loc[(s, mth)] if (s, mth) in beta.index else None
            bs = "  |  %.3f  %.3f  %.3f" % (b.retention_specific, b.specific_purity, b.purity_epitope) if b is not None else ""
            print("%-12s %8.3f %8.3f %8.3f%s" % (mth, r["ret_sp"], r["pur_sp"], r["pur_epi"], bs))

        # consensus
        all_a = [a for asn in m_assigns.values() for a in asn]
        if all_a:
            cl, _ = majority_vote_consensus(all_a, cfg)
            r = metrics(labs_c(cl), cby, spec, total, sp_total)
            b = beta.loc[(s, "consensus")] if (s, "consensus") in beta.index else None
            bs = "  |  %.3f  %.3f  %.3f" % (b.retention_specific, b.specific_purity, b.purity_epitope) if b is not None else ""
            star = "  <- rich consensus"
            print("%-12s %8.3f %8.3f %8.3f%s%s" % ("consensus*", r["ret_sp"], r["pur_sp"], r["pur_epi"], bs, star))


if __name__ == "__main__":
    main()
