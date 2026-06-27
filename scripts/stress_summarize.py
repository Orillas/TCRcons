#!/usr/bin/env python3
"""Summarize the Donor1 stress-test results into a table + figure.

Reads results/reproducibility/stress_test/stress_results.tsv (per
subset/rep/method rows) and emits:
  - stress_table.tsv          mean over reps: retention_specific, purity_epitope,
                              specific_purity, retention_all per (subset, method)
  - stress_figure.png         retention_specific & specific_purity vs noise frac,
                              consensus vs single methods
"""
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test")
SRC = OUT / "stress_results.tsv"

METHOD_ORDER = ["hd_baseline", "giana", "tcrmatch", "clustcr", "gliph2", "tcrdist3",
                "deeptcr", "consensus"]
COLORS = {
    "consensus": "#d62728",
    "clustcr": "#1f77b4", "gliph2": "#ff7f0e", "tcrdist3": "#2ca02c",
    "deeptcr": "#9467bd", "tcrmatch": "#8c564b", "giana": "#e377c2",
    "hd_baseline": "#7f7f7f",
}


def main():
    df = pd.read_csv(SRC, sep="\t")
    # mean over reps
    keys = ["subset", "subset_size", "noise_frac", "method", "kind"]
    agg = df.groupby(keys, sort=False).agg(
        retention_specific=("retention_specific", "mean"),
        retention_specific_sd=("retention_specific", "std"),
        purity_epitope=("purity_epitope", "mean"),
        specific_purity=("specific_purity", "mean"),
        retention_all=("retention_all", "mean"),
        n_clustered=("n_clustered", "mean"),
        n_clusters=("n_clusters", "mean"),
        n_reps=("rep", "count"),
    ).reset_index()
    agg["method_order"] = agg["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    agg = agg.sort_values(["subset", "method_order"])
    agg.drop(columns=["method_order"]).to_csv(OUT / "stress_table.tsv", sep="\t",
                                               index=False, float_format="%.4f")
    print(f"wrote {OUT/'stress_table.tsv'}: {len(agg)} rows")
    print(agg[["subset", "noise_frac", "method", "retention_specific",
               "purity_epitope", "specific_purity"]].to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for col, metric, ylabel in [(0, "retention_specific", "retention (specific)"),
                                    (1, "specific_purity", "specific-cluster purity")]:
            ax = axes[col]
            for m in METHOD_ORDER:
                sub = agg[agg["method"] == m].sort_values("noise_frac")
                if sub.empty:
                    continue
                lw = 2.6 if m == "consensus" else 1.3
                ax.plot(sub["noise_frac"], sub[metric], marker="o", label=m,
                        color=COLORS.get(m), linewidth=lw,
                        markersize=5 if m == "consensus" else 4,
                        zorder=5 if m == "consensus" else 3)
            ax.set_xlabel("noise fraction")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.set_xlim(-0.03, 0.93)
            ax.grid(alpha=0.3)
            if col == 1:
                ax.set_ylim(0, 1.02)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle("Donor1 noise-spike-in stress test: consensus vs single methods",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / "stress_figure.png", dpi=140)
        print(f"wrote {OUT/'stress_figure.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
