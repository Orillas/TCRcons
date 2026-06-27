#!/usr/bin/env python3
"""Experiment 5: Biological Case Study with majority_vote.

Deep analysis of two well-studied epitopes:
  - GILGFVFTL (Influenza M1, HLA-A*02:01) — large, well-characterized
  - NLVPMVATV (CMV pp65, HLA-A*02:01) — large, well-characterized

For each epitope, shows:
  1. Cluster membership table with per-method support
  2. Method agreement heatmap
  3. V/J gene enrichment in consensus clusters
  4. CDR3 motif analysis
  5. Confidence calibration (predicted confidence vs actual purity)
  6. Comparison: majority_vote clusters vs single-method clusters
"""

import sys
import logging
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))
from exp_shared import (
    get_all_clusterers, majority_vote_consensus,
    clusters_to_labels, assignments_to_labels,
    evaluate_clustering, run_single_method, load_benchmark_data,
)
from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config

logger = logging.getLogger(__name__)


def compute_motif(sequences):
    """Position frequency matrix from aligned sequences."""
    if sequences is None or len(sequences) < 2:
        return {}
    max_len = max(len(s) for s in sequences)
    padded = [s.ljust(max_len, '-') for s in sequences]
    aas = "ACDEFGHIKLMNPQRSTVWY"
    pfm = {}
    for pos in range(max_len):
        counts = Counter(s[pos] for s in padded)
        total = sum(counts.get(aa, 0) for aa in aas)
        pfm[pos] = {aa: counts.get(aa, 0) / total if total > 0 else 0 for aa in aas}
    return pfm


def compute_vj_enrichment(cluster_df, background_df, gene_col):
    """Compute V/J gene fold enrichment in cluster vs background."""
    if gene_col not in cluster_df.columns or gene_col not in background_df.columns:
        return []
    bg_counts = background_df[gene_col].value_counts(normalize=True)
    cl_counts = cluster_df[gene_col].value_counts(normalize=True)
    enriched = []
    for gene in cl_counts.index:
        if gene in bg_counts.index and bg_counts[gene] > 0:
            fold = cl_counts[gene] / bg_counts[gene]
            enriched.append({
                "gene": gene,
                "cluster_freq": round(cl_counts[gene], 4),
                "background_freq": round(bg_counts[gene], 4),
                "fold_enrichment": round(fold, 2),
            })
    enriched.sort(key=lambda x: x["fold_enrichment"], reverse=True)
    return enriched[:15]


def run_case_study(epitope, df_raw, df_norm, output_dir, cfg, clusterers):
    """Run case study for one epitope."""
    epi_dir = output_dir / epitope
    epi_dir.mkdir(parents=True, exist_ok=True)

    epitope_col = "epitope" if "epitope" in df_raw.columns else "Epitope"
    subset_raw = df_raw[df_raw[epitope_col] == epitope].copy()
    logger.info(f"\n{'='*60}")
    logger.info(f"Case Study: {epitope} ({len(subset_raw)} TCRs)")
    logger.info(f"{'='*60}")

    if len(subset_raw) < 10:
        logger.warning(f"  Too few TCRs ({len(subset_raw)}), skipping")
        return

    # Normalize subset
    rename_lower = {col: col.lower() for col in subset_raw.columns
                   if col.lower() != col and col.lower() in ["cdr3_alpha","cdr3_beta","v_alpha","v_beta","j_alpha","j_beta","tcr_id","epitope"]}
    if rename_lower:
        subset_raw = subset_raw.rename(columns=rename_lower)
    subset_norm = normalize(subset_raw.copy())
    tcr_ids = subset_norm["tcr_id"].values
    true_labels = subset_raw[epitope_col].values
    n_total = len(subset_norm)

    workdir = epi_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    # Run all methods
    method_results = {}
    for mname, clusterer in clusterers.items():
        assigns, rt = run_single_method(clusterer, subset_norm, workdir / mname, cfg)
        if assigns:
            method_results[mname] = assigns
            logger.info(f"  {mname}: {len(assigns)} assignments")

    # Run majority_vote
    if len(method_results) >= 2:
        all_a = []
        for a_list in method_results.values():
            all_a.extend(a_list)
        clusters, edges = majority_vote_consensus(all_a, cfg)
        logger.info(f"  majority_vote: {len(clusters)} clusters")
    else:
        logger.warning("  Not enough methods for consensus")
        return

    # === 1. Cluster membership table with method support ===
    logger.info("  Building membership table...")
    # Build per-method cluster lookup
    method_member_map = {}
    for mname, assigns in method_results.items():
        method_member_map[mname] = set(a.tcr_id for a in assigns)

    method_cluster_map = {}
    for mname, assigns in method_results.items():
        m_map = {}
        for a in assigns:
            if a.tcr_id not in m_map:
                m_map[a.tcr_id] = a.cluster_id
        method_cluster_map[mname] = m_map

    membership_rows = []
    for cluster in clusters:
        for mid in cluster.member_ids:
            row_data = subset_norm[subset_norm["tcr_id"] == mid]
            cdr3b = row_data["cdr3_beta"].values[0] if len(row_data) > 0 else ""
            vb = row_data["v_beta"].values[0] if len(row_data) > 0 and "v_beta" in row_data.columns else ""
            jb = row_data["j_beta"].values[0] if len(row_data) > 0 and "j_beta" in row_data.columns else ""

            supporting = []
            for mname, m_map in method_cluster_map.items():
                if mid in m_map:
                    supporting.append(mname)

            is_core = mid in (cluster.core_member_ids or [])
            membership_rows.append({
                "tcr_id": mid,
                "cdr3_beta": cdr3b,
                "v_beta": vb,
                "j_beta": jb,
                "consensus_cluster": cluster.cluster_id,
                "n_cluster_members": len(cluster.member_ids),
                "cluster_confidence": round(cluster.cluster_confidence, 4),
                "is_core": is_core,
                "n_supporting_methods": len(supporting),
                "supporting_methods": ",".join(sorted(supporting)),
            })

    membership_df = pd.DataFrame(membership_rows)
    membership_df.to_csv(epi_dir / "cluster_membership.tsv", sep="\t", index=False)

    # === 2. Cluster summary ===
    summary_rows = []
    for cluster in clusters:
        if len(cluster.member_ids) < 2:
            continue
        member_epi = subset_raw[subset_raw["tcr_id"].isin(cluster.member_ids)][epitope_col].values
        if len(member_epi) == 0:
            continue
        majority_epi = Counter(member_epi).most_common(1)[0]
        epi_purity = majority_epi[1] / len(member_epi)

        seqs = subset_norm[subset_norm["tcr_id"].isin(cluster.member_ids)]["cdr3_beta"].dropna().values

        summary_rows.append({
            "cluster_id": cluster.cluster_id,
            "n_members": len(cluster.member_ids),
            "n_core": len(cluster.core_member_ids or []),
            "confidence": round(cluster.cluster_confidence, 4),
            "actual_purity": round(epi_purity, 4),
            "dominant_epitope": majority_epi[0],
            "cdr3_beta_length_median": int(np.median([len(s) for s in seqs])) if len(seqs) > 0 else 0,
            "cdr3_beta_examples": ";".join(list(seqs)[:3]) if len(seqs) > 0 else "",
        })

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df.to_csv(epi_dir / "cluster_summary.tsv", sep="\t", index=False)

    # === 3. V/J enrichment for top clusters ===
    background_df = subset_norm
    for cluster in clusters[:5]:
        if len(cluster.member_ids) < 3:
            continue
        cl_df = subset_norm[subset_norm["tcr_id"].isin(cluster.member_ids)]
        for gene_col in ["v_beta", "j_beta"]:
            enriched = compute_vj_enrichment(cl_df, background_df, gene_col)
            if enriched:
                pd.DataFrame(enriched).to_csv(
                    epi_dir / f"{cluster.cluster_id}_{gene_col}_enrichment.tsv",
                    sep="\t", index=False)

    # === 4. Motif analysis for top 3 clusters ===
    for cluster in clusters[:3]:
        if len(cluster.member_ids) < 3:
            continue
        seqs = subset_norm[subset_norm["tcr_id"].isin(cluster.member_ids)]["cdr3_beta"].dropna().values
        if len(seqs) is not None and len(seqs) >= 3:
            pfm = compute_motif(seqs)
            if pfm:
                motif_rows = []
                for pos, freqs in sorted(pfm.items()):
                    for aa, freq in sorted(freqs.items(), key=lambda x: -x[1]):
                        motif_rows.append({
                            "position": pos, "amino_acid": aa,
                            "frequency": round(freq, 4),
                        })
                pd.DataFrame(motif_rows).to_csv(
                    epi_dir / f"{cluster.cluster_id}_motif.tsv", sep="\t", index=False)

    # === 5. Confidence calibration ===
    calib_rows = []
    for cluster in clusters:
        if len(cluster.member_ids) < 2:
            continue
        member_epi = subset_raw[subset_raw["tcr_id"].isin(cluster.member_ids)][epitope_col].values
        if len(member_epi) == 0:
            continue
        majority = Counter(member_epi).most_common(1)[0]
        calib_rows.append({
            "cluster_id": cluster.cluster_id,
            "predicted_confidence": round(cluster.cluster_confidence, 4),
            "actual_purity": round(majority[1] / len(member_epi), 4),
            "n_members": len(cluster.member_ids),
            "dominant_epitope": majority[0],
        })
    if calib_rows:
        pd.DataFrame(calib_rows).to_csv(epi_dir / "confidence_calibration.tsv", sep="\t", index=False)

    # === 6. Method comparison for this epitope ===
    comparison_rows = []
    for mname, assigns in method_results.items():
        pred = assignments_to_labels(assigns, tcr_ids)
        m = evaluate_clustering(pred, true_labels, n_total, mname)
        comparison_rows.append(m)

    # Add majority_vote
    pred_mv = clusters_to_labels(clusters, tcr_ids)
    m_mv = evaluate_clustering(pred_mv, true_labels, n_total, "majority_vote")
    comparison_rows.append(m_mv)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(epi_dir / "method_comparison.tsv", sep="\t", index=False)

    # Print summary
    print(f"\n  --- {epitope} Summary ---")
    print(f"  {len(clusters)} clusters from {len(subset_raw)} TCRs")
    print(f"  Method comparison:")
    comp_cols = ["method", "ari", "purity", "sensitivity", "retention", "f1"]
    available = [c for c in comp_cols if c in comparison_df.columns]
    print(comparison_df[available].to_string(index=False))

    if len(summary_df) > 0:
        print(f"\n  Top clusters:")
        for _, row in summary_df.head(5).iterrows():
            print(f"    {row['cluster_id']}: {row['n_members']} members, "
                  f"purity={row['actual_purity']:.3f}, "
                  f"confidence={row['confidence']:.3f}")


def run_exp5(output_dir):
    output_dir = Path(output_dir)

    config = load_config()
    cfg = config._raw
    clusterers = get_all_clusterers()
    logger.info(f"Clusterers: {list(clusterers.keys())}")

    df_raw, df_norm = load_benchmark_data()

    # Run case studies for 2 well-studied epitopes
    for epitope in ["GILGFVFTL", "NLVPMVATV"]:
        run_case_study(epitope, df_raw, df_norm, output_dir, cfg, clusterers)

    print("\n" + "="*80)
    print("EXPERIMENT 5: BIOLOGICAL CASE STUDY COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    run_exp5("/home/jilin/DeepTCR/tcrconsensus/results/exp5_mv_case_study")
