#!/usr/bin/env python3
"""Experiment 5: Biological Case Study.

Deep analysis of GILGFVFTL (Flu-M1) and NLVPMVATV (CMV-pp65).
Shows cluster membership, method support, motif, V/J enrichment, confidence.
"""

import sys
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import compute_all_metrics, per_epitope_pairwise_f1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_clusterers():
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    try:
        from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
        clusterers["clustcr"] = ClusTCRWrapper()
    except: pass
    try:
        from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
        clusterers["tcrdist3"] = TCRDist3Wrapper()
    except: pass
    try:
        from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
        clusterers["gliph2"] = GLIPH2Wrapper()
    except: pass
    try:
        from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper
        clusterers["deeptcr"] = DeepTCRWrapper()
    except: pass
    return clusterers


def extract_method_support(clusters, all_assignments, tcr_ids):
    """For each cluster, find which methods support it."""
    # Build method -> cluster assignments
    method_clusters = {}
    for a in all_assignments:
        if a.method not in method_clusters:
            method_clusters[a.method] = {}
        if a.cluster_id not in method_clusters[a.method]:
            method_clusters[a.method][a.cluster_id] = set()
        method_clusters[a.method][a.cluster_id].add(a.tcr_id)

    cluster_info = []
    for cluster in clusters:
        members = set(cluster.member_ids)
        supporting = {}
        for method, cids in method_clusters.items():
            for cid, members_m in cids.items():
                overlap = members & members_m
                if len(overlap) >= 2:
                    supporting[method] = {
                        "cluster_id": cid,
                        "overlap": len(overlap),
                        "pct": len(overlap) / len(members) * 100 if members else 0,
                    }

        cluster_info.append({
            "cluster_id": cluster.cluster_id,
            "n_members": len(members),
            "n_core": len(cluster.core_member_ids) if cluster.core_member_ids else 0,
            "n_peripheral": len(cluster.peripheral_member_ids) if cluster.peripheral_member_ids else 0,
            "confidence": cluster.cluster_confidence,
            "supporting_methods": list(supporting.keys()),
            "n_supporting_methods": len(supporting),
            "method_details": supporting,
        })

    return cluster_info


def compute_motif(sequences):
    """Compute simple position frequency matrix from aligned sequences."""
    if not sequences:
        return {}

    # Pad to same length
    max_len = max(len(s) for s in sequences)
    padded = [s.ljust(max_len, '-') for s in sequences]

    aas = "ACDEFGHIKLMNPQRSTVWY-"
    pfm = {}
    for pos in range(max_len):
        counts = Counter(s[pos] for s in padded)
        pfm[pos] = {aa: counts.get(aa, 0) for aa in aas if aa != '-'}

    return pfm


def compute_vj_enrichment(df, cluster_members, background_df):
    """Compute V/J gene enrichment in cluster vs background."""
    cluster_df = df[df["tcr_id"].isin(cluster_members)]

    results = {}
    for gene_col in ["v_beta", "j_beta"]:
        if gene_col not in cluster_df.columns:
            continue

        bg_counts = background_df[gene_col].value_counts(normalize=True)
        cl_counts = cluster_df[gene_col].value_counts(normalize=True)

        enriched = []
        for gene in cl_counts.index:
            if gene in bg_counts.index and bg_counts[gene] > 0:
                fold = cl_counts[gene] / bg_counts[gene]
                enriched.append({
                    "gene": gene,
                    "cluster_freq": cl_counts[gene],
                    "background_freq": bg_counts[gene],
                    "fold_enrichment": fold,
                })

        enriched.sort(key=lambda x: x["fold_enrichment"], reverse=True)
        results[gene_col] = enriched[:10]

    return results


def run_case_study(epitope, df, labels, output_dir, config, cfg):
    """Run case study for a single epitope."""
    output_dir = Path(output_dir) / epitope
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}\nCase Study: {epitope}\n{'='*60}")

    # Filter to this epitope
    epitope_col = "epitope_label" if "epitope_label" in df.columns else "epitope"
    subset = df[df[epitope_col] == epitope].copy()
    logger.info(f"  {len(subset)} TCRs for epitope {epitope}")

    df_norm = normalize(subset.copy())
    tcr_ids = df_norm["tcr_id"].values
    true_labels = subset[epitope_col].values

    workdir = output_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    # Run all clusterers
    clusterers = get_clusterers()
    all_assignments = []
    method_results = {}
    for mname, clusterer in clusterers.items():
        r = clusterer.safe_execute(df_norm, workdir, cfg)
        if r.status.value == "success" and r.assignments:
            all_assignments.extend(r.assignments)
            method_results[mname] = r.assignments
            logger.info(f"  {mname}: {len(r.assignments)} assignments")

    if len(all_assignments) < 2:
        logger.warning(f"  Not enough assignments for {epitope}")
        return

    # Run consensus
    methods = list(set(a.method for a in all_assignments))
    weights = compute_method_weights(methods, "balanced", cfg)
    clusters, edges = balanced_consensus(all_assignments, weights)
    clusters = refine(clusters, edges, cfg)

    logger.info(f"  Consensus: {len(clusters)} clusters")

    # Extract method support
    cluster_info = extract_method_support(clusters, all_assignments, tcr_ids)

    # Cluster membership table
    membership_rows = []
    for ci, cluster in enumerate(clusters):
        for mid in cluster.member_ids:
            row_data = df_norm[df_norm["tcr_id"] == mid].iloc[0] if len(df_norm[df_norm["tcr_id"] == mid]) > 0 else None
            supporting = []
            for mname, assigns in method_results.items():
                if any(a.tcr_id == mid for a in assigns):
                    supporting.append(mname)

            membership_rows.append({
                "tcr_id": mid,
                "cdr3_beta": row_data["cdr3_beta"] if row_data is not None else "",
                "v_beta": row_data["v_beta"] if row_data is not None else "",
                "j_beta": row_data["j_beta"] if row_data is not None else "",
                "cluster_id": cluster.cluster_id,
                "n_cluster_members": len(cluster.member_ids),
                "cluster_confidence": cluster.cluster_confidence,
                "is_core": mid in (cluster.core_member_ids or []),
                "supporting_methods": ",".join(supporting),
                "n_supporting": len(supporting),
            })

    membership_df = pd.DataFrame(membership_rows)
    membership_df.to_csv(output_dir / "cluster_membership.tsv", sep="\t", index=False)

    # Cluster summary
    info_df = pd.DataFrame(cluster_info)
    info_df.to_csv(output_dir / "cluster_summary.tsv", sep="\t", index=False)

    # V/J enrichment (use all VDJdb as background)
    for cluster in clusters[:5]:  # top 5 clusters
        vj = compute_vj_enrichment(subset, cluster.member_ids, df)
        for gene_col, enriched in vj.items():
            if enriched:
                pd.DataFrame(enriched).to_csv(
                    output_dir / f"{cluster.cluster_id}_{gene_col}_enrichment.tsv",
                    sep="\t", index=False
                )

    # Motif analysis per cluster (top 3)
    for cluster in clusters[:3]:
        members = cluster.member_ids
        seqs = df_norm[df_norm["tcr_id"].isin(members)]["cdr3_beta"].dropna().values
        if len(seqs) >= 3:
            pfm = compute_motif(seqs)
            # Save as simple frequency table
            motif_rows = []
            for pos, counts in pfm.items():
                total = sum(counts.values())
                for aa, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                    motif_rows.append({
                        "position": pos,
                        "amino_acid": aa,
                        "count": cnt,
                        "frequency": cnt / total if total > 0 else 0,
                    })
            pd.DataFrame(motif_rows).to_csv(
                output_dir / f"{cluster.cluster_id}_motif.tsv",
                sep="\t", index=False
            )

    # Confidence calibration data
    calib_rows = []
    for cluster in clusters:
        members = cluster.member_ids
        if len(members) < 2:
            continue
        # Check how many are same epitope
        member_epitopes = subset[subset["tcr_id"].isin(members)][epitope_col].values
        if len(member_epitopes) == 0:
            continue
        majority = Counter(member_epitopes).most_common(1)[0]
        purity_val = majority[1] / len(member_epitopes)
        calib_rows.append({
            "cluster_id": cluster.cluster_id,
            "confidence": cluster.cluster_confidence,
            "purity": purity_val,
            "n_members": len(members),
            "dominant_epitope": majority[0],
        })

    if calib_rows:
        pd.DataFrame(calib_rows).to_csv(output_dir / "confidence_calibration.tsv", sep="\t", index=False)

    # Print summary
    print(f"\n  --- {epitope} Summary ---")
    print(f"  {len(clusters)} clusters from {len(subset)} TCRs")
    print(f"  Top clusters:")
    for ci in cluster_info[:5]:
        print(f"    {ci['cluster_id']}: {ci['n_members']} members, "
              f"confidence={ci['confidence']:.3f}, "
              f"supported by {ci['supporting_methods']}")

    return cluster_info


def run_experiment(data_dir, output_dir):
    output_dir = Path(output_dir)

    config = load_config()
    cfg = config._raw

    # Load VDJdb
    vdj = pd.read_csv(f"{data_dir}/vdjdb_filtered.tsv", sep="\t", dtype=str)
    labels = pd.read_csv(f"{data_dir}/vdjdb_labels.tsv", sep="\t", dtype=str)
    df = vdj.merge(labels, on="tcr_id", how="inner", suffixes=("", "_label"))

    # Run case studies
    for epitope in ["GILGFVFTL", "NLVPMVATV"]:
        run_case_study(epitope, df, labels, output_dir, config, cfg)

    # Also load McPAS and run case studies
    mcpas = pd.read_csv(f"{data_dir}/mcpas_filtered.tsv", sep="\t", dtype=str)
    mcpas_labels = pd.read_csv(f"{data_dir}/mcpas_labels.tsv", sep="\t", dtype=str)
    mcpas_df = mcpas.merge(mcpas_labels, on="tcr_id", how="inner", suffixes=("", "_label"))

    for epitope in ["GILGFVFTL", "NLVPMVATV"]:
        run_case_study(epitope + "_McPAS", mcpas_df, mcpas_labels, output_dir, config, cfg)

    print("\n" + "="*80)
    print("EXPERIMENT 5: CASE STUDY COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    run_experiment(
        "/home/jilin/DeepTCR/tcrconsensus/results/data",
        "/home/jilin/DeepTCR/tcrconsensus/results/exp5_case_study",
    )
