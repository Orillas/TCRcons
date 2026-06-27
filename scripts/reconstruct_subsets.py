#!/usr/bin/env python3
"""Rebuild the 6 Donor1 stress-test subsets with native Vbeta/Jbeta attached.

Controlled comparison vs the CDR3beta-only stress test: SAME unique CDR3beta
sequences + SAME noise structure, the ONLY difference is V/J genes restored
(stripped to gene level). Alpha chain is unavailable in this dataset
(beta-only 10x), so this is CDR3beta + Vbeta + Jbeta, not paired alphabeta.

Source: subsets/subset_i.txt  (unique CDR3beta, the noise-spike-in subsets)
V/J:    input/TCRdist3/subset_i.tsv  (CDR3b, TRBV, TRBJ, per-cell, 100% coverage)
For a CDR3beta with multiple cells/alleles, attach the majority V/J.

Writes: data/reconstruct_data/subset_i.tsv  (columns: cdr3_beta, v_beta, j_beta)
"""
from pathlib import Path
from collections import Counter

import pandas as pd

BASE = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1")
SUBSET_DIR = BASE / "subsets"
VJ_DIR = BASE / "input" / "TCRdist3"
OUT = Path("/home/jilin/DeepTCR/tcrconsensus/data/reconstruct_data")


def gene(x):
    """TRBV10-3*01 -> TRBV10-3 (drop allele)."""
    if pd.isna(x):
        return None
    return str(x).split("*")[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(1, 7):
        cdr3 = pd.read_csv(SUBSET_DIR / f"subset_{i}.txt", sep="\t", dtype=str)["cdr3"].tolist()
        td = pd.read_csv(VJ_DIR / f"subset_{i}.tsv", sep="\t", dtype=str)
        td["v"] = td["TRBV"].map(gene)
        td["j"] = td["TRBJ"].map(gene)
        # majority V/J per CDR3b
        majority = {}
        for cdr, g in td.groupby("CDR3b"):
            v = Counter(g["v"].dropna()).most_common(1)
            j = Counter(g["j"].dropna()).most_common(1)
            majority[cdr] = (v[0][0] if v else None, j[0][0] if j else None)

        rows = []
        miss_v = miss_j = 0
        for c in cdr3:
            v, j = majority.get(c, (None, None))
            if v is None:
                miss_v += 1
            if j is None:
                miss_j += 1
            rows.append({"cdr3_beta": c, "v_beta": v, "j_beta": j})
        df = pd.DataFrame(rows)
        out = OUT / f"subset_{i}.tsv"
        df.to_csv(out, sep="\t", index=False)
        print(f"subset_{i}: {len(df)} rows -> {out.name} | "
              f"V non-null {df['v_beta'].notna().mean()*100:.1f}% "
              f"J non-null {df['j_beta'].notna().mean()*100:.1f}% "
              f"(V miss {miss_v}, J miss {miss_j})")


if __name__ == "__main__":
    main()
