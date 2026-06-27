#!/usr/bin/env python3
"""Shared utilities for Exp2-5 experiments.

All experiments use majority_vote (equal weights + balanced_consensus + refinement)
as the core consensus method. Individual methods are compared as baselines.
"""

import sys
import time
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Ensure src is importable
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import compute_all_metrics
from tcrconsensus.schema.records import ClusterAssignment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# All 7 clusterers
# ---------------------------------------------------------------------------

def get_all_clusterers():
    """Return dict of all 7 clusterer wrappers."""
    clusterers = {}

    from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
    clusterers["hd_baseline"] = HDBaselineClusterer()

    try:
        from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
        clusterers["clustcr"] = ClusTCRWrapper()
    except Exception:
        pass

    try:
        from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
        clusterers["tcrdist3"] = TCRDist3Wrapper()
    except Exception:
        pass

    try:
        from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
        clusterers["gliph2"] = GLIPH2Wrapper()
    except Exception:
        pass

    try:
        from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
        clusterers["giana"] = GIANAWrapper()
    except Exception:
        pass

    try:
        from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
        clusterers["tcrmatch"] = TCRMatchWrapper()
    except Exception:
        pass

    try:
        from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper
        clusterers["deeptcr"] = DeepTCRWrapper()
    except Exception:
        pass



    return clusterers


# ---------------------------------------------------------------------------
# Majority vote consensus
# ---------------------------------------------------------------------------

def majority_vote_consensus(all_assignments, config, skip_refinement=False, use_empirical_weights=False):
    """Majority vote consensus.

    Args:
        use_empirical_weights: if True, weight methods by empirical performance
            (ARI, purity, noise robustness) instead of equal 1/N weights.
    """
    methods = sorted(set(a.method for a in all_assignments))
    n = len(methods)

    if use_empirical_weights:
        weights = empirical_weights(methods)
        logger.info(f"  majority_vote: {n} methods, empirical weights")
        for m in sorted(weights, key=weights.get, reverse=True):
            logger.info(f"    {m}: {weights[m]:.4f}")
    else:
        weights = {m: 1.0 / n for m in methods}
        logger.info(f"  majority_vote: {n} methods, equal weights = {1/n:.4f}")

    clusters, edges = balanced_consensus(all_assignments, weights)

    if not skip_refinement and clusters:
        clusters = refine(clusters, edges, config)

    return clusters, edges


# ---------------------------------------------------------------------------
# Run individual method
# ---------------------------------------------------------------------------

def run_single_method(clusterer, df_norm, workdir, config):
    """Run one clusterer, return assignments or empty list."""
    try:
        result = clusterer.safe_execute(df_norm, workdir, config)
        if result.status.value == "success" and result.assignments:
            return result.assignments, result.runtime_seconds
    except Exception as e:
        logger.warning(f"  Method failed: {e}")
    return [], 0.0


def run_all_methods(df_norm, clusterers, config, workdir):
    """Run all clusterers, return dict method_name -> (assignments, runtime)."""
    results = {}
    for mname, clusterer in clusterers.items():
        logger.info(f"  Running {mname}...")
        t0 = time.time()
        assigns, rt = run_single_method(clusterer, df_norm, workdir / mname, config)
        elapsed = time.time() - t0
        if assigns:
            results[mname] = (assigns, elapsed)
            logger.info(f"    {mname}: {len(assigns)} assignments, {elapsed:.1f}s")
        else:
            logger.info(f"    {mname}: FAILED or 0 assignments")
    return results


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def clusters_to_labels(clusters, tcr_ids):
    """Map consensus clusters to label array aligned with tcr_ids."""
    label_map = {}
    for c in clusters:
        for mid in c.member_ids:
            label_map[mid] = c.cluster_id
    return np.array([label_map.get(tid, -1) for tid in tcr_ids])


def assignments_to_labels(assignments, tcr_ids):
    """Map method assignments to label array aligned with tcr_ids."""
    label_map = {}
    for a in assignments:
        if a.tcr_id not in label_map:
            label_map[a.tcr_id] = a.cluster_id
    return np.array([label_map.get(tid, -1) for tid in tcr_ids])


def evaluate_clustering(pred_labels, true_labels, n_total, method_name=""):
    """Evaluate clustering, filtering out unclustered (-1)."""
    valid = np.array([str(p) not in ("-1", "") for p in pred_labels], dtype=bool)
    if valid.sum() < 2:
        return {"method": method_name, "ari": 0.0, "n_clustered": 0.0}

    le_t = LabelEncoder()
    le_p = LabelEncoder()
    true_str = true_labels[valid]
    pred_str = pred_labels[valid].astype(str)
    le_t.fit(np.unique(true_str))
    le_p.fit(np.unique(pred_str))
    t_enc = le_t.transform(true_str)
    p_enc = le_p.transform(pred_str)

    m = compute_all_metrics(p_enc, t_enc, n_total)
    m["method"] = method_name
    return m


def load_benchmark_data(benchmark_path="/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"):
    """Load the v3 CD8 benchmark dataset, normalized."""
    df = pd.read_csv(benchmark_path, sep="\t", dtype=str)

    # Rename uppercase columns to lowercase
    rename_lower = {}
    for col in df.columns:
        low = col.lower()
        if low != col and low in ["cdr3_alpha", "cdr3_beta", "v_alpha", "v_beta",
                                   "j_alpha", "j_beta", "tcr_id", "epitope"]:
            rename_lower[col] = low
    if rename_lower:
        df = df.rename(columns=rename_lower)

    df_norm = normalize(df.copy())
    return df, df_norm
