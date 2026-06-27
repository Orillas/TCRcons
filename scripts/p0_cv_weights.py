#!/usr/bin/env python3
"""P0 Cross-Validation Weight Learning.

Leave-one-subset-out CV to learn optimal weights.
Compare learned weights vs empirical priors.
If they converge, this validates the prior-based approach.

Strategy:
- For each of 6 subsets, train weights on the other 5 subsets
- Evaluate on held-out subset
- Compare learned weights to empirical priors

Weight learning via grid search over weight coefficients,
optimizing for NMI on labeled CDR3s.
"""

import sys, time, logging, json, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/scripts")
sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.consensus.weights import empirical_weights, EMPIRICAL_PRIORS
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
from tcrconsensus.refinement.refiner import refine
from exp_shared import (
    get_all_clusterers, run_all_methods, clusters_to_labels,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Paths
DATA_BASE = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1")
GLIPH2_DIR = DATA_BASE / "input" / "Gliph2"
LABEL_JSON = "/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_h5.json"
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments/cv_weights")
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(LABEL_JSON) as f:
    CDR3_EPI = json.load(f)["cdr3_to_epitopes"]

config_obj = load_config()
config = dict(config_obj.__dict__)
all_clusterers = get_all_clusterers()
clusterers = all_clusterers

SUBSETS = [1, 2, 3, 4, 5, 6]


def get_label(cdr3):
    if cdr3 in CDR3_EPI:
        epis = CDR3_EPI[cdr3]
        return epis[0] if len(epis) == 1 else "MULTI:" + ";".join(sorted(epis))
    return "BACKGROUND"


def evaluate_nmi(pred, true_labels, labeled_idx):
    """Evaluate NMI on labeled subset only."""
    lp = pred[labeled_idx]
    lt = true_labels[labeled_idx]
    clustered = np.array([str(p) not in ("-1", "") for p in lp])
    if clustered.sum() < 2:
        return 0.0
    return normalized_mutual_info_score(lt[clustered], lp[clustered].astype(str))


def run_consensus_with_weights(all_assigns, weight_dict, tcr_ids, do_refine=True):
    """Run consensus with specific weights, return predictions."""
    edges = extract_pairwise_support(all_assigns, weight_dict)
    graph = build_consensus_graph(edges, threshold=0.3)
    clusters = connected_components_clustering(graph)
    if do_refine and clusters:
        clusters = refine(clusters, edges, config)
    return clusters_to_labels(clusters, tcr_ids)


def weight_grid_search(all_assigns, tcr_ids, true_labels, labeled_idx, methods_list):
    """Search over weight coefficient combinations to find best NMI."""
    # Define coefficient grid: (ari_w, ami_w, purity_w, sensitivity_w, noise_w)
    # Keep sum = 1.0
    best_nmi = -1
    best_coeffs = None
    best_weights = None

    # Coarse grid
    for ari_w in [0.2, 0.3, 0.4, 0.5]:
        for pur_w in [0.2, 0.3, 0.4]:
            remaining = 1.0 - ari_w - pur_w
            if remaining < 0.1:
                continue
            for sens_w in [0.05, 0.1, 0.15]:
                noise_w = remaining - sens_w
                if noise_w < 0.05:
                    continue
                ami_w = 0  # simplified: fold ami into ari
                coeffs = {"ari": ari_w, "ami": ami_w, "purity": pur_w, "sensitivity": sens_w, "noise_robust": noise_w}

                # Compute weights with these coefficients
                raw = {}
                for m in methods_list:
                    prior = EMPIRICAL_PRIORS.get(m, {"purity": 0.5, "sensitivity": 0.5, "ari": 0.05, "ami": 0.1, "noise_robust": 0.5})
                    w = (coeffs["ari"] * prior.get("ari", 0.05)
                         + coeffs["ami"] * prior.get("ami", 0.1)
                         + coeffs["purity"] * prior.get("purity", 0.5)
                         + coeffs["sensitivity"] * prior.get("sensitivity", 0.5)
                         + coeffs["noise_robust"] * prior.get("noise_robust", 0.5))
                    raw[m] = max(w, 0.05)
                total = sum(raw.values())
                weights = {m: w / total for m, w in raw.items()}

                pred = run_consensus_with_weights(all_assigns, weights, tcr_ids, do_refine=False)
                nmi = evaluate_nmi(pred, true_labels, labeled_idx)

                if nmi > best_nmi:
                    best_nmi = nmi
                    best_coeffs = coeffs
                    best_weights = weights

    return best_nmi, best_coeffs, best_weights


# ── Pre-compute all subset data ──
print("=" * 78)
print("CROSS-VALIDATION WEIGHT LEARNING")
print("=" * 78)

print("\nPre-computing method outputs for all subsets...")
subset_data = {}
for sid in SUBSETS:
    print(f"\n  Subset {sid}...")
    df = pd.read_csv(GLIPH2_DIR / f"subset_{sid}.csv", sep="\t")
    df = df.rename(columns={"CDR3b": "cdr3_beta", "TRBV": "v_beta", "TRBJ": "j_beta"})
    df["tcr_id"] = df["cdr3_beta"]
    df["epitope"] = df["cdr3_beta"].apply(get_label)
    df = df[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]].drop_duplicates(subset=["cdr3_beta"])

    df_norm = normalize(df.copy())
    tcr_ids = df_norm["tcr_id"].values
    true_labels = df_norm["epitope"].values
    labeled_idx = np.array([tid in CDR3_EPI for tid in tcr_ids])

    method_results = run_all_methods(df_norm, clusterers, config, OUTDIR / f"subset_{sid}")

    all_assigns = []
    for mname, (assigns, rt) in method_results.items():
        all_assigns.extend(assigns)

    methods_list = sorted(set(a.method for a in all_assigns))

    subset_data[sid] = {
        "tcr_ids": tcr_ids,
        "true_labels": true_labels,
        "labeled_idx": labeled_idx,
        "all_assigns": all_assigns,
        "methods_list": methods_list,
    }
    print(f"    {len(tcr_ids)} TCRs, {labeled_idx.sum()} labeled, {len(all_assigns)} assignments, {len(methods_list)} methods")

# ── Empirical weights baseline ──
print(f"\n{'=' * 78}")
print("EMPIRICAL WEIGHTS (current approach)")
print(f"{'=' * 78}")

# Use subset 1 methods as reference
ref_methods = subset_data[1]["methods_list"]
emp_weights = empirical_weights(ref_methods)
print(f"\nEmpirical weights:")
for m, w in sorted(emp_weights.items(), key=lambda x: -x[1]):
    print(f"  {m}: {w:.4f}")

# Evaluate empirical weights on all subsets
print(f"\nEmpirical weights performance:")
emp_nmi_scores = {}
for sid in SUBSETS:
    sd = subset_data[sid]
    pred = run_consensus_with_weights(sd["all_assigns"], emp_weights, sd["tcr_ids"])
    nmi = evaluate_nmi(pred, sd["true_labels"], sd["labeled_idx"])
    emp_nmi_scores[sid] = nmi
    print(f"  Subset {sid}: NMI={nmi:.4f}")

# ── Leave-one-subset-out CV ──
print(f"\n{'=' * 78}")
print("LEAVE-ONE-SUBSET-OUT CROSS-VALIDATION")
print(f"{'=' * 78}")

cv_results = {}
for test_sid in SUBSETS:
    print(f"\n  Test subset: {test_sid}")
    train_sids = [s for s in SUBSETS if s != test_sid]

    # Find common methods across training subsets
    all_methods = set()
    for sid in train_sids:
        all_methods.update(subset_data[sid]["methods_list"])
    common_methods = sorted(all_methods)

    # Grid search: find best coefficients on training subsets
    # Simplified: evaluate on each train subset, average NMI
    best_avg_nmi = -1
    best_coeffs = None
    best_weights = None

    for ari_w in [0.2, 0.3, 0.4, 0.5]:
        for pur_w in [0.2, 0.3, 0.4]:
            remaining = 1.0 - ari_w - pur_w
            if remaining < 0.1:
                continue
            for sens_w in [0.05, 0.1, 0.15]:
                noise_w = remaining - sens_w
                if noise_w < 0.05:
                    continue
                coeffs = {"ari": ari_w, "ami": 0, "purity": pur_w, "sensitivity": sens_w, "noise_robust": noise_w}

                raw = {}
                for m in common_methods:
                    prior = EMPIRICAL_PRIORS.get(m, {"purity": 0.5, "sensitivity": 0.5, "ari": 0.05, "ami": 0.1, "noise_robust": 0.5})
                    w = (coeffs["ari"] * prior.get("ari", 0.05)
                         + coeffs["purity"] * prior.get("purity", 0.5)
                         + coeffs["sensitivity"] * prior.get("sensitivity", 0.5)
                         + coeffs["noise_robust"] * prior.get("noise_robust", 0.5))
                    raw[m] = max(w, 0.05)
                total = sum(raw.values())
                weights = {m: w / total for m, w in raw.items()}

                # Evaluate on each training subset
                nmis = []
                for sid in train_sids:
                    sd = subset_data[sid]
                    # Only use common methods
                    filtered_assigns = [a for a in sd["all_assigns"] if a.method in common_methods]
                    if not filtered_assigns:
                        continue
                    pred = run_consensus_with_weights(filtered_assigns, weights, sd["tcr_ids"], do_refine=False)
                    nmi = evaluate_nmi(pred, sd["true_labels"], sd["labeled_idx"])
                    nmis.append(nmi)

                avg_nmi = np.mean(nmis) if nmis else 0
                if avg_nmi > best_avg_nmi:
                    best_avg_nmi = avg_nmi
                    best_coeffs = coeffs
                    best_weights = weights

    # Evaluate best CV weights on test subset
    sd_test = subset_data[test_sid]
    filtered_assigns = [a for a in sd_test["all_assigns"] if a.method in common_methods]
    pred_cv = run_consensus_with_weights(filtered_assigns, best_weights, sd_test["tcr_ids"])
    nmi_cv = evaluate_nmi(pred_cv, sd_test["true_labels"], sd_test["labeled_idx"])

    # Also evaluate empirical weights on test
    pred_emp = run_consensus_with_weights(filtered_assigns, emp_weights, sd_test["tcr_ids"])
    nmi_emp = evaluate_nmi(pred_emp, sd_test["true_labels"], sd_test["labeled_idx"])

    print(f"    Best CV coefficients: {best_coeffs}")
    print(f"    CV-learned weights:")
    for m, w in sorted(best_weights.items(), key=lambda x: -x[1]):
        emp_w = emp_weights.get(m, 0)
        print(f"      {m}: CV={w:.4f}, Emp={emp_w:.4f}, diff={w-emp_w:+.4f}")
    print(f"    Test NMI: CV={nmi_cv:.4f}, Empirical={nmi_emp:.4f}, diff={nmi_cv-nmi_emp:+.4f}")

    cv_results[test_sid] = {
        "cv_nmi": nmi_cv,
        "empirical_nmi": nmi_emp,
        "cv_weights": best_weights,
        "cv_coefficients": best_coeffs,
        "train_avg_nmi": best_avg_nmi,
    }

# ── Summary ──
print(f"\n{'=' * 78}")
print("CV WEIGHT LEARNING SUMMARY")
print(f"{'=' * 78}")
print(f"{'Subset':>7s} {'CV_NMI':>8s} {'Emp_NMI':>8s} {'Delta':>8s}")
print("-" * 35)

cv_nmis = []
emp_nmis = []
for sid in SUBSETS:
    r = cv_results[sid]
    delta = r["cv_nmi"] - r["empirical_nmi"]
    print(f"{sid:7d} {r['cv_nmi']:8.4f} {r['empirical_nmi']:8.4f} {delta:+8.4f}")
    cv_nmis.append(r["cv_nmi"])
    emp_nmis.append(r["empirical_nmi"])

print(f"\n  Mean NMI: CV={np.mean(cv_nmis):.4f}, Empirical={np.mean(emp_nmis):.4f}")
print(f"  Mean Delta: {np.mean(cv_nmis) - np.mean(emp_nmis):+.4f}")

# Weight correlation
all_cv_w = []
all_emp_w = []
for sid in SUBSETS:
    r = cv_results[sid]
    for m in r["cv_weights"]:
        all_cv_w.append(r["cv_weights"][m])
        all_emp_w.append(emp_weights.get(m, 0))

corr = np.corrcoef(all_cv_w, all_emp_w)[0, 1] if len(all_cv_w) > 1 else 0
print(f"\n  Weight correlation (CV vs Empirical): {corr:.4f}")

if corr > 0.9:
    print("  -> CV-learned weights strongly correlate with empirical priors")
    print("     This validates the prior-based weighting approach")
elif corr > 0.7:
    print("  -> CV-learned weights moderately correlate with empirical priors")
    print("     Priors capture most of the learnable signal")
else:
    print("  -> CV-learned weights differ from empirical priors")
    print("     Consider using CV-learned weights instead")

with open(OUTDIR / "cv_weights_results.json", "w") as f:
    json.dump({
        "cv_results": cv_results,
        "empirical_weights": {m: float(w) for m, w in emp_weights.items()},
        "weight_correlation": float(corr),
        "mean_cv_nmi": float(np.mean(cv_nmis)),
        "mean_emp_nmi": float(np.mean(emp_nmis)),
    }, f, indent=2, default=str)

print(f"\nSaved to {OUTDIR / 'cv_weights_results.json'}")
print("CV Weight Learning Done!")
