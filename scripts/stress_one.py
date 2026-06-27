#!/usr/bin/env python3
"""Run ONE clusterer on ONE subset of the i3-unit Donor1 stress test.

Invoked as a subprocess by stress_test.py for isolation + hard timeout
(avoids TF/fork issues, hangs don't poison the orchestrator).

Usage:  stress_one.py <subset_i> <method> <rep>
Writes: results/reproducibility/stress_test/cache/s{subset}_{method}_r{rep}.pkl
        (a pickled list[ClusterAssignment])  -- only on success & non-empty.
"""
import sys
import pickle
from pathlib import Path

import yaml
import pandas as pd

TCRROOT = Path("/home/jilin/DeepTCR/tcrconsensus")
sys.path.insert(0, str(TCRROOT / "src"))
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
sys.path.insert(0, str(TCRROOT))  # for `scripts.exp_shared`

from tcrconsensus.io.parser import normalize

SUBSET_DIR = Path(
    "/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/"
    "Data/10X/Donor1/subsets"
)
OUT = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility/stress_test")
CACHE = OUT / "cache"
WORK = OUT / "work"

# Construct only the needed clusterer (avoid importing DeepTCR/TF for cheap methods).
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

    cdr3 = pd.read_csv(SUBSET_DIR / f"subset_{subset_i}.txt", sep="\t", dtype=str)["cdr3"].tolist()
    df_norm = normalize(pd.DataFrame({"cdr3_beta": cdr3}))

    workdir = WORK / f"s{subset_i}" / f"r{rep}" / method
    workdir.mkdir(parents=True, exist_ok=True)
    # Plain dict config: wrappers/refine use config.get(...); Config object lacks .get.
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
