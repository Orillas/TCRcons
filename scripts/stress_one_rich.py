#!/usr/bin/env python3
"""Run ONE clusterer on ONE RECONSTRUCTED (CDR3beta+Vbeta+Jbeta) subset.

Same as stress_one.py but reads the V/J-enriched subsets and writes to a
separate cache/output so it never collides with the CDR3beta-only run.

Usage:  stress_one_rich.py <subset_i> <method> <rep>
"""
import sys
import pickle
from pathlib import Path

import yaml
import pandas as pd

TCRROOT = Path("/home/jilin/DeepTCR/tcrconsensus")
sys.path.insert(0, str(TCRROOT / "src"))
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
sys.path.insert(0, str(TCRROOT))

from tcrconsensus.io.parser import normalize

SUBSET_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/data/reconstruct_data")
OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test_rich")
CACHE = OUT / "cache"
WORK = OUT / "work"

_METHOD_CLS = {
    "hd_baseline": ("tcrconsensus.clusterers.hd_baseline", "HDBaselineClusterer"),
    "clustcr": ("tcrconsensus.clusterers.clustcr_wrapper", "ClusTCRWrapper"),
    "tcrdist3": ("tcrconsensus.clusterers.tcrdist3_wrapper", "TCRDist3Wrapper"),
    "gliph2": ("tcrconsensus.clusterers.gliph2_wrapper", "GLIPH2Wrapper"),
    "giana": ("tcrconsensus.clusterers.giana_wrapper", "GIANAWrapper"),
    "tcrmatch": ("tcrconsensus.clusterers.tcrmatch_wrapper", "TCRMatchWrapper"),
    "deeptcr": ("tcrconsensus.clusterers.deeptcr_wrapper", "DeepTCRWrapper"),
}


def main():
    subset_i = int(sys.argv[1])
    method = sys.argv[2]
    rep = int(sys.argv[3])

    cf = CACHE / f"s{subset_i}_{method}_r{rep}.pkl"
    if cf.exists():
        print(f"CACHE_HIT {cf.name}", flush=True)
        return

    mod_name, cls_name = _METHOD_CLS[method]
    mod = __import__(mod_name, fromlist=[cls_name])
    clusterer = getattr(mod, cls_name)()

    df = pd.read_csv(SUBSET_DIR / f"subset_{subset_i}.tsv", sep="\t", dtype=str)
    df_norm = normalize(df.copy())  # has cdr3_beta, v_beta, j_beta

    workdir = WORK / f"s{subset_i}" / f"r{rep}" / method
    workdir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(open(TCRROOT / "configs" / "default.yaml")) or {}

    try:
        res = clusterer.safe_execute(df_norm, workdir, config)
    except Exception as e:  # noqa: BLE001
        print(f"ERR {method} s{subset_i} r{rep}: {e}", flush=True)
        return

    if getattr(res, "status", None) is not None and res.status.value == "success" and res.assignments:
        pickle.dump(list(res.assignments), open(cf, "wb"))
        print(f"OK {method} s{subset_i} r{rep}: {len(res.assignments)} assigns", flush=True)
    else:
        print(f"EMPTY {method} s{subset_i} r{rep}", flush=True)


if __name__ == "__main__":
    main()
