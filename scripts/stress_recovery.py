#!/usr/bin/env python3
"""F3 + Tier-2 recovery sweep on the Donor1 stress test.

Reuses the CDR3beta-only cached method assignments (no method recompute) and
rebuilds the consensus with F3 (permutation FDR threshold) and Tier-2 (signed
repulsion) turned on, to test whether they cut noise bridges and recover
consensus purity above the best single method.

Configs: baseline | +signed(0.9) | +signed(0.6) | +fdr | +fdr+signed(0.9)
         | +fdr+signed(0.6)
Reads:  results/reproducibility/stress_test/cache/  (CDR3beta-only)
Writes: results/reproducibility/stress_test/stress_recovery.tsv + log
"""
import sys
import json
import time
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
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.refinement.refiner import refine

SUBSET_DIR = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1/subsets")
EPI = json.load(open("/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_mapping.json"))["cdr3_to_epitopes"]
CACHE = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test/cache")
OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test")
BETA_TSV = OUT / "stress_results.tsv"

METHODS = ["hd_baseline", "giana", "tcrmatch", "clustcr", "gliph2", "tcrdist3", "deeptcr"]
MIN_SIZE = 2
CONFIGS = [
    ("baseline",        dict(use_signed=False, use_fdr_threshold=False)),
    ("signed_t0.9",     dict(use_signed=True,  use_fdr_threshold=False, high_purity_threshold=0.9)),
    ("signed_t0.6",     dict(use_signed=True,  use_fdr_threshold=False, high_purity_threshold=0.6)),
    ("fdr",             dict(use_signed=False, use_fdr_threshold=True,  target_fdr=0.05, null_permutations=50)),
    ("fdr+signed_t0.9", dict(use_signed=True,  use_fdr_threshold=True,  high_purity_threshold=0.9, target_fdr=0.05, null_permutations=50)),
    ("fdr+signed_t0.6", dict(use_signed=True,  use_fdr_threshold=True,  high_purity_threshold=0.6, target_fdr=0.05, null_permutations=50)),
]


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


def method_purity(asn, cby, spec):
    """Epitope-weighted purity of a method's own clusters -> for purity_lookup."""
    tid2cid = labs_a(asn)
    r = metrics(tid2cid, cby, spec, 1, 1)  # pur_epi is scale-free
    return r["pur_epi"]


def main():
    cfg = yaml.safe_load(open(TCRROOT / "configs" / "default.yaml")) or {}
    spec = set(pd.read_csv(SUBSET_DIR / "subset_1.txt", sep="\t", dtype=str)["cdr3"])

    # single-method + baseline-consensus reference (mean over reps)
    beta = pd.read_csv(BETA_TSV, sep="\t")
    beta = beta.groupby(["subset", "method"])[
        ["retention_specific", "specific_purity", "purity_epitope"]].mean()

    rows = []
    print("F3+Tier-2 recovery sweep (CDR3beta-only cache, rep1 assignments)")
    for s in range(1, 7):
        # need all 7 method caches for this subset
        caches = {m: CACHE / f"s{s}_{m}_r1.pkl" for m in METHODS}
        if not all(c.exists() for c in caches.values()):
            print(f"\nsubset {s}: not all methods cached, skip")
            continue
        cdr3 = pd.read_csv(SUBSET_DIR / f"subset_{s}.txt", sep="\t", dtype=str)["cdr3"].tolist()
        total = len(cdr3)
        sp_total = sum(1 for c in cdr3 if c in spec)
        dn = normalize(pd.DataFrame({"cdr3_beta": cdr3}))
        cby = dict(zip(dn["tcr_id"], dn["cdr3_beta"]))
        m_assigns = {m: pickle.load(open(caches[m], "rb")) for m in METHODS}
        all_a = [a for asn in m_assigns.values() for a in asn]
        pur_lookup = {m: method_purity(asn, cby, spec) for m, asn in m_assigns.items()}
        weights = {m: 1.0 / len(m_assigns) for m in m_assigns}
        nf = 1 - sp_total / total

        print(f"\n=== subset {s} (noise {nf:.2f}) | per-method pur_lookup: "
              + " ".join(f"{m}={pur_lookup[m]:.2f}" for m in METHODS) + " ===")
        print("%-16s %8s %8s %8s" % ("config", "ret_sp", "pur_sp", "pur_epi"))

        # best single reference
        sing = beta.loc[s].drop(index="consensus", errors="ignore") if "consensus" in getattr(beta.loc[s], "index", []) else beta.loc[s]
        best_sp = sing["specific_purity"].max()
        best_m = sing["specific_purity"].idxmax()

        for name, flags in CONFIGS:
            kw = dict(flags)
            if kw.get("use_signed"):
                kw["purity_lookup"] = pur_lookup
            t0 = time.time()
            try:
                cl, _ = balanced_consensus(all_a, weights, **kw)
                if cl:
                    cl = refine(cl, _, cfg)
                r = metrics(labs_c(cl), cby, spec, total, sp_total)
                dt = time.time() - t0
                print("%-16s %8.3f %8.3f %8.3f   (%.0fs)" % (name, r["ret_sp"], r["pur_sp"], r["pur_epi"], dt))
                rows.append({"subset": s, "noise_frac": nf, "config": name,
                             "retention_specific": r["ret_sp"], "specific_purity": r["pur_sp"],
                             "purity_epitope": r["pur_epi"]})
            except Exception as e:  # noqa: BLE001
                print("%-16s FAILED: %s" % (name, str(e)[:120]))

        print("  -> best single pur_sp = %.3f (%s)" % (best_sp, best_m))

    pd.DataFrame(rows).to_csv(OUT / "stress_recovery.tsv", sep="\t", index=False, float_format="%.4f")
    print("\nwrote", OUT / "stress_recovery.tsv", "| rows", len(rows))


if __name__ == "__main__":
    main()
