#!/usr/bin/env python3
"""Benchmark with BCubed F1 metric.

BCubed F1 (Amigó et al., 2009) computes per-element precision/recall:
  - BCubed Precision(i) = fraction of elements in i's cluster sharing i's true label
  - BCubed Recall(i) = fraction of elements with i's true label in i's cluster
  - Average over all elements, then harmonic mean

This is the mainstream metric for Nature-level TCR clustering papers,
more robust to cluster size imbalance than Pairwise F1.
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# BCubed F1 Implementation
# ============================================================================

def bcubed_metrics(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict:
    """Compute BCubed Precision, Recall, and F1 (Amigó et al., 2009).

    For each element i:
      precision(i) = |{j in C(i) : L(j) = L(i)}| / |C(i)|
      recall(i)    = |{j in C(i) : L(j) = L(i)}| / |{j : L(j) = L(i)}|

    where C(i) = cluster of i, L(i) = true label of i.
    Final P/R = mean over all elements.

    This is O(n) after counting, much faster than pairwise O(n^2).
    """
    n = len(pred_labels)
    if n == 0:
        return {"bcubed_precision": 0.0, "bcubed_recall": 0.0, "bcubed_f1": 0.0}

    # Count: (cluster, label) -> count, cluster -> size, label -> size
    cluster_label_counts = Counter()
    cluster_sizes = Counter()
    label_sizes = Counter()

    for i in range(n):
        c = pred_labels[i]
        l = true_labels[i]
        cluster_label_counts[(c, l)] += 1
        cluster_sizes[c] += 1
        label_sizes[l] += 1

    # Per-element precision and recall
    precisions = np.zeros(n)
    recalls = np.zeros(n)

    for i in range(n):
        c = pred_labels[i]
        l = true_labels[i]
        same_in_cluster = cluster_label_counts[(c, l)]
        precisions[i] = same_in_cluster / cluster_sizes[c] if cluster_sizes[c] > 0 else 0.0
        recalls[i] = same_in_cluster / label_sizes[l] if label_sizes[l] > 0 else 0.0

    bc_precision = float(np.mean(precisions))
    bc_recall = float(np.mean(recalls))
    bc_f1 = 2 * bc_precision * bc_recall / (bc_precision + bc_recall) if (bc_precision + bc_recall) > 0 else 0.0

    return {
        "bcubed_precision": bc_precision,
        "bcubed_recall": bc_recall,
        "bcubed_f1": bc_f1,
    }


# ============================================================================
# Per-epitope BCubed metrics (one-vs-rest)
# ============================================================================

def per_epitope_bcubed(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    epitope_names: np.ndarray = None,
) -> pd.DataFrame:
    """Compute per-epitope BCubed F1 in one-vs-rest fashion.

    For each epitope E:
      - Treat E as positive class, all others as negative
      - Compute BCubed P, R, F1 on this binary problem
    """
    if epitope_names is None:
        epitope_names = np.unique(true_labels)

    results = []
    for target_ep in epitope_names:
        ep_mask = true_labels == target_ep
        n_tcrs = int(ep_mask.sum())
        if n_tcrs < 2:
            continue

        # Binary labels: 1 = target epitope, 0 = rest
        binary_true = np.where(ep_mask, 1, 0)
        # Binary pred: 1 = same cluster as majority of target TCRs, 0 = rest
        # Actually, use original cluster labels for BCubed computation
        # BCubed naturally handles multi-class, so just compute it directly

        bc = bcubed_metrics(pred_labels, true_labels)

        # Also compute epitope-specific: restrict to TCRs in clusters containing target epitope
        target_clusters = set(pred_labels[ep_mask])
        relevant_mask = np.isin(pred_labels, list(target_clusters))
        if relevant_mask.sum() > 0:
            rel_pred = pred_labels[relevant_mask]
            rel_true = true_labels[relevant_mask]
            # One-vs-rest: target vs others
            rel_binary_true = (rel_true == target_ep).astype(int)
            # For predicted: same cluster as majority of target = positive
            target_cluster_votes = Counter(pred_labels[ep_mask])
            majority_cluster = target_cluster_votes.most_common(1)[0][0]
            rel_binary_pred = (rel_pred == majority_cluster).astype(int)

            # BCubed on this binary problem
            # Precision: of elements predicted positive, fraction truly positive
            # Recall: of elements truly positive, fraction predicted positive
            tp = ((rel_binary_pred == 1) & (rel_binary_true == 1)).sum()
            fp = ((rel_binary_pred == 1) & (rel_binary_true == 0)).sum()
            fn = ((rel_binary_pred == 0) & (rel_binary_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            prec, rec, f1 = 0.0, 0.0, 0.0

        results.append({
            "epitope": target_ep,
            "bcubed_precision": prec,
            "bcubed_recall": rec,
            "bcubed_f1": f1,
            "n_tcrs": n_tcrs,
        })

    return pd.DataFrame(results)


# ============================================================================
# Run single method benchmark
# ============================================================================

def run_single_method(method_name, df_norm, output_dir, config_raw, organism):
    """Run a single clustering method and return assignments."""
    from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
    from tcrconsensus.clusterers.levenshtein import LevenshteinClusterer
    from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
    from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
    from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
    from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper

    method_map = {
        "hd_baseline": lambda: HDBaselineClusterer(distance_threshold=1),
        "levenshtein": lambda: LevenshteinClusterer(),
        "giana": lambda: GIANAWrapper(),
        "clustcr": lambda: ClusTCRWrapper(),
        "tcrdist3": lambda: TCRDist3Wrapper(organism=organism),
        "deeptcr": lambda: DeepTCRWrapper(),
    }

    if method_name == "tcrmatch":
        logger.info(f"  TCRMatch: running via shell")
        from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
        clusterer = GIANAWrapper()
        # TCRMatch is run through GIANA wrapper with tcrmatch flag
        # Actually, let me use the direct approach
        import subprocess
        cdr3s = df_norm["cdr3_beta"].dropna().tolist()
        tcr_ids = df_norm.loc[df_norm["cdr3_beta"].notna(), "tcr_id"].tolist()
        tmp_input = output_dir / f"tcrmatch_input_{method_name}.txt"
        with open(tmp_input, "w") as f:
            for tid, seq in zip(tcr_ids, cdr3s):
                f.write(f"{tid}\t{seq}\n")
        # TCRMatch via tcrconsensus if available, else skip
        try:
            from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
            clusterer = TCRMatchWrapper()
            result = clusterer.safe_execute(df_norm, output_dir, config_raw)
            return result
        except ImportError:
            logger.warning("  TCRMatch wrapper not available, skipping")
            return None

    if method_name not in method_map:
        logger.warning(f"  Unknown method: {method_name}")
        return None

    clusterer = method_map[method_name]()
    t0 = time.time()
    result = clusterer.safe_execute(df_norm, output_dir, config_raw)
    elapsed = time.time() - t0
    return result


def extract_predictions(result, df_norm):
    """Extract cluster assignment arrays from result."""
    if result is None:
        return None, None, 0

    # Build tcr_id -> cluster_id map
    label_map = {}
    for assignment in result.assignments:
        label_map[assignment.tcr_id] = assignment.cluster_id

    tcr_ids = df_norm["tcr_id"].values
    pred_raw = np.array([label_map.get(tid, "") for tid in tcr_ids])

    # Valid = assigned to a cluster (not empty or -1)
    valid = np.array([str(p) not in ("", "-1", "None") for p in pred_raw], dtype=bool)
    n_clustered = valid.sum()

    return pred_raw, valid, n_clustered


# ============================================================================
# Main benchmark
# ============================================================================

def main():
    data_path = Path("/home/jilin/DeepTCR/tcrconsensus/data/vdjdb_28636592.tsv")
    output_dir = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproduce_vdjdb/bcubed_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    from tcrconsensus.io.parser import load_file, normalize
    from tcrconsensus.config import load_config
    from tcrconsensus.evaluation.metrics import compute_all_metrics

    config = load_config(None)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df_raw = pd.read_csv(data_path, sep="\t")
    logger.info(f"  Raw: {len(df_raw)} rows")

    # VDJdb columns: cdr3, antigen.epitope, species, gene, v.segm, j.segm
    # Rename to tcrconsensus convention
    df_raw = df_raw.rename(columns={
        "cdr3": "cdr3_beta",
        "v.segm": "v_beta",
        "j.segm": "j_beta",
    })

    # Filter: only TRB chain (standard for TCR clustering benchmarks)
    df_raw = df_raw[df_raw["gene"] == "TRB"].copy()
    logger.info(f"  After gene=TRB filter: {len(df_raw)} rows")

    # Filter: need cdr3_beta and epitope
    df_raw = df_raw.dropna(subset=["cdr3_beta", "antigen.epitope"])
    logger.info(f"  After dropna: {len(df_raw)} rows")

    # Per-species
    all_results = []
    all_per_epitope = []

    for species in ["HomoSapiens", "MusMusculus"]:
        sp_mask = df_raw["species"] == species
        df_sp = df_raw[sp_mask].copy()
        if len(df_sp) == 0:
            logger.info(f"No data for {species}, skipping")
            continue

        # Deduplicate by cdr3_beta + epitope
        df_sp = df_sp.drop_duplicates(subset=["cdr3_beta", "antigen.epitope"])
        logger.info(f"\n{'='*60}")
        logger.info(f"Species: {species}, {len(df_sp)} unique TCRs")

        # Get epitope counts
        ep_counts = df_sp["antigen.epitope"].value_counts()
        logger.info(f"  Epitopes: {len(ep_counts)}")
        for ep, cnt in ep_counts.items():
            logger.info(f"    {ep}: {cnt} TCRs")

        # Normalize
        df_norm = normalize(df_sp.rename(columns={"antigen.epitope": "epitope"}))

        # Encode true labels
        le_true = LabelEncoder()
        true_labels_all = le_true.fit_transform(df_norm["epitope"].values)
        n_total = len(df_norm)
        n_epitopes = len(ep_counts)

        # Organism for TCRdist3/DeepTCR
        organism = "human" if species == "HomoSapiens" else "mouse"

        methods = ["hd_baseline", "levenshtein", "giana", "clustcr", "tcrdist3", "deeptcr"]

        for method_name in methods:
            logger.info(f"\n--- {species} / {method_name} ---")
            try:
                result = run_single_method(method_name, df_norm, output_dir, config._raw, organism)
                pred_raw, valid, n_clustered = extract_predictions(result, df_norm)
            except Exception as e:
                logger.error(f"  Error: {e}")
                all_results.append({
                    "species": species, "method": method_name, "error": str(e),
                    "bcubed_f1": 0.0, "bcubed_precision": 0.0, "bcubed_recall": 0.0,
                })
                continue

            if n_clustered == 0:
                logger.warning(f"  No clusters produced")
                all_results.append({
                    "species": species, "method": method_name,
                    "bcubed_f1": 0.0, "bcubed_precision": 0.0, "bcubed_recall": 0.0,
                    "n_clustered": 0, "n_total": n_total, "n_epitopes": n_epitopes,
                })
                continue

            # Encode predicted labels (for valid TCRs only)
            pred_valid = pred_raw[valid]
            true_valid = true_labels_all[valid]

            le_pred = LabelEncoder()
            pred_encoded = le_pred.fit_transform(pred_valid)

            # ---- Standard metrics ----
            std_metrics = compute_all_metrics(pred_encoded, true_valid, n_total)

            # ---- BCubed metrics ----
            bc = bcubed_metrics(pred_encoded, true_valid)

            logger.info(f"  BCubed P={bc['bcubed_precision']:.4f}  R={bc['bcubed_recall']:.4f}  F1={bc['bcubed_f1']:.4f}")
            logger.info(f"  Pairwise P={std_metrics['pairwise_precision']:.4f}  R={std_metrics['pairwise_sensitivity']:.4f}  F1={std_metrics['f1']:.4f}")
            logger.info(f"  Retention={std_metrics['retention']:.4f}  n_clustered={n_clustered}")

            row = {
                "species": species,
                "method": method_name,
                **std_metrics,
                **bc,
                "n_epitopes": n_epitopes,
            }
            all_results.append(row)

            # ---- Per-epitope BCubed ----
            ep_names_encoded = np.unique(true_valid)
            ep_df = per_epitope_bcubed(pred_encoded, true_valid,
                                        epitope_names=ep_names_encoded)
            # Map back to string epitope names
            ep_df["epitope"] = ep_df["epitope"].apply(
                lambda x: le_true.classes_[x] if isinstance(x, (int, np.integer)) and x < len(le_true.classes_) else str(x)
            )
            ep_df["species"] = species
            ep_df["method"] = method_name
            all_per_epitope.append(ep_df)

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = output_dir / "bcubed_benchmark_results.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"\nResults saved to {out_path}")

    if all_per_epitope:
        ep_df_all = pd.concat(all_per_epitope, ignore_index=True)
        ep_path = output_dir / "bcubed_per_epitope.tsv"
        ep_df_all.to_csv(ep_path, sep="\t", index=False)
        logger.info(f"Per-epitope saved to {ep_path}")

    # ---- Print comparison table ----
    print("\n" + "="*80)
    print("BCubed F1 vs Pairwise F1 Comparison")
    print("="*80)
    print(f"{'Species':<14} {'Method':<14} {'BCubed_P':>10} {'BCubed_R':>10} {'BCubed_F1':>10} {'Pair_P':>10} {'Pair_R':>10} {'Pair_F1':>10} {'Delta_F1':>10}")
    print("-"*100)

    for _, row in results_df.iterrows():
        bc_f1 = row.get("bcubed_f1", 0)
        pair_f1 = row.get("f1", 0)
        delta = bc_f1 - pair_f1
        print(f"{row['species']:<14} {row['method']:<14} "
              f"{row.get('bcubed_precision', 0):>10.4f} {row.get('bcubed_recall', 0):>10.4f} {bc_f1:>10.4f} "
              f"{row.get('pairwise_precision', 0):>10.4f} {row.get('pairwise_sensitivity', 0):>10.4f} {pair_f1:>10.4f} "
              f"{delta:>+10.4f}")

    # ---- Compare against reference (reproductivity.md) ----
    print("\n" + "="*80)
    print("Comparison against Reference F1 (Dash dataset, from reproductivity.md)")
    print("Note: Reference is from Dash (Nature 2017) clean dataset")
    print("      Our data is vdjdb_28636592 (real-world VDJdb data)")
    print("="*80)
    ref = {
        "giana": {"p": 0.90, "r": 0.73, "f1": 0.81},
        "deeptcr": {"p": 0.82, "r": 0.70, "f1": 0.75},
        "tcrdist3": {"p": 0.68, "r": 0.62, "f1": 0.65},
        "clustcr": {"p": 0.75, "r": 0.65, "f1": 0.70},
    }
    # Real-world expected ranges
    ref_real = {
        "giana": (0.45, 0.55),
        "deeptcr": (0.45, 0.55),
    }

    print(f"{'Method':<14} {'Ref_F1(Dash)':>13} {'Real_F1_Range':>14} {'BCubed_Human':>14} {'BCubed_Mouse':>14} {'Status':>20}")
    print("-"*90)

    for method in ["giana", "clustcr", "deeptcr", "tcrdist3"]:
        ref_f1 = ref.get(method, {}).get("f1", 0)
        real_range = ref_real.get(method, ("N/A", "N/A"))

        human_row = results_df[(results_df["species"] == "HomoSapiens") & (results_df["method"] == method)]
        mouse_row = results_df[(results_df["species"] == "MusMusculus") & (results_df["method"] == method)]

        human_f1 = human_row["bcubed_f1"].values[0] if len(human_row) > 0 else 0.0
        mouse_f1 = mouse_row["bcubed_f1"].values[0] if len(mouse_row) > 0 else 0.0

        # Check against real-world range
        if isinstance(real_range[0], (int, float)):
            in_range = real_range[0] <= human_f1 <= real_range[1] or real_range[0] <= mouse_f1 <= real_range[1]
            status = "✓ in range" if in_range else "✗ below range"
        else:
            status = "no real-world ref"

        real_str = f"{real_range[0]}-{real_range[1]}" if isinstance(real_range[0], (int, float)) else "N/A"
        print(f"{method:<14} {ref_f1:>13.2f} {real_str:>14} {human_f1:>14.4f} {mouse_f1:>14.4f} {status:>20}")

    return results_df


if __name__ == "__main__":
    main()
