#!/usr/bin/env python3
"""P0-2 Biological Case Study: GILGFVFTL and ELAGIGILTV epitopes.

Shows what consensus finds that individual methods miss:
1. Cluster composition: which CDR3s are in consensus clusters for each epitope
2. Motif analysis: sequence patterns in consensus clusters
3. V/J gene enrichment: TRBV/TRBJ over-representation
4. Unique discoveries: CDR3s found only by consensus (missed by all individuals)
"""

import sys, json, logging
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
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.refinement.refiner import refine
from exp_shared import get_all_clusterers, run_all_methods, clusters_to_labels

# Paths
DATA_BASE = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1")
GLIPH2_DIR = DATA_BASE / "input" / "Gliph2"
LABEL_JSON = "/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_h5.json"
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments/case_study")
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(LABEL_JSON) as f:
    CDR3_EPI = json.load(f)["cdr3_to_epitopes"]

config_obj = load_config()
config = dict(config_obj.__dict__)
all_clusterers = get_all_clusterers()
clusterers = all_clusterers

TARGET_EPITOPES = ["GILGFVFTL", "ELAGIGILTV"]


def get_cdr3_epitope(cdr3):
    if cdr3 in CDR3_EPI:
        epis = CDR3_EPI[cdr3]
        return epis[0] if len(epis) == 1 else "MULTI:" + ";".join(sorted(epis))
    return None


def motif_from_sequences(seqs):
    """Extract position-specific frequency matrix from aligned CDR3 sequences."""
    if not seqs:
        return {}
    # Pad to same length
    max_len = max(len(s) for s in seqs)
    padded = [s.ljust(max_len, "X") for s in seqs]
    # Position frequencies
    aa_order = sorted(set("".join(seqs)))
    psfm = {}
    for pos in range(max_len):
        col = [p[pos] for p in padded]
        freq = Counter(col)
        total = len(col)
        psfm[pos] = {aa: count / total for aa, count in freq.most_common()}
    return psfm


def sequence_similarity(s1, s2):
    """Simple normalized Hamming similarity for same-length sequences."""
    if len(s1) != len(s2):
        return 0.0
    return sum(a == b for a, b in zip(s1, s2)) / len(s1)


def find_consensus_only_cdr3s(consensus_clusters, individual_preds, target_epitope_cdr3s):
    """Find CDR3s in consensus clusters that NO individual method clustered together with the target epitope."""
    # For each target CDR3 in a consensus cluster, check if any individual method
    # also put it in a cluster with another target CDR3
    consensus_only = []

    # Build individual method cluster maps: method -> cluster_id -> set of CDR3s
    method_clusters = defaultdict(lambda: defaultdict(set))
    for method_name, pred in individual_preds.items():
        for i, cid in enumerate(pred):
            if cid != -1:
                method_clusters[method_name][cid].add(i)

    for cluster in consensus_clusters:
        members = cluster.member_ids
        # Check if this cluster has target epitope members
        target_members = [m for m in members if m in target_epitope_cdr3s]
        if not target_members:
            continue

        # For each non-target member, check if any individual method also clustered it with a target member
        non_target = [m for m in members if m not in target_epitope_cdr3s]
        for nt in non_target:
            found_by_individual = False
            for method_name, clusters in method_clusters.items():
                # Find cluster containing this non-target CDR3
                for cid, member_indices in clusters.items():
                    # This is by index, need to convert...
                    pass  # We'll use a different approach below
            consensus_only.append(nt)

    return consensus_only


# ── Load data ──
print("=" * 78)
print("BIOLOGICAL CASE STUDY: GILGFVFTL & ELAGIGILTV")
print("=" * 78)

# Use subset 1 (most labeled CDR3s)
df = pd.read_csv(GLIPH2_DIR / "subset_1.csv", sep="\t")
df = df.rename(columns={"CDR3b": "cdr3_beta", "TRBV": "v_beta", "TRBJ": "j_beta"})
df["tcr_id"] = df["cdr3_beta"]

def get_label(cdr3):
    if cdr3 in CDR3_EPI:
        epis = CDR3_EPI[cdr3]
        return epis[0] if len(epis) == 1 else "MULTI:" + ";".join(sorted(epis))
    return "BACKGROUND"

df["epitope"] = df["cdr3_beta"].apply(get_label)
df = df[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]].drop_duplicates(subset=["cdr3_beta"])

df_norm = normalize(df.copy())
true_labels = df_norm["epitope"].values
tcr_ids = df_norm["tcr_id"].values
n = len(tcr_ids)

# Build CDR3 -> V/J gene lookup
cdr3_to_v = dict(zip(df_norm["cdr3_beta"], df_norm["v_beta"]))
cdr3_to_j = dict(zip(df_norm["cdr3_beta"], df_norm["j_beta"]))

# Build CDR3 -> epitope lookup
cdr3_to_epi = {}
for i in range(n):
    epi = get_cdr3_epitope(tcr_ids[i])
    if epi:
        cdr3_to_epi[tcr_ids[i]] = epi

# Get target CDR3s for each epitope
epitope_cdr3s = defaultdict(set)
for cdr3, epi in cdr3_to_epi.items():
    if epi in TARGET_EPITOPES:
        epitope_cdr3s[epi].add(cdr3)

for epi in TARGET_EPITOPES:
    print(f"\n  {epi}: {len(epitope_cdr3s[epi])} CDR3s in subset 1")

# Run all methods
print("\nRunning methods...")
method_results = run_all_methods(df_norm, clusterers, config, OUTDIR / "subset_1")
all_assigns = []
for mname, (assigns, rt) in method_results.items():
    all_assigns.extend(assigns)

# Run consensus
methods_list = sorted(set(a.method for a in all_assigns))
weights = empirical_weights(methods_list)
clusters, edges = balanced_consensus(all_assigns, weights)
clusters = refine(clusters, edges, config)

print(f"\nConsensus: {len(clusters)} clusters from {len(methods_list)} methods")

# ── Analyze each target epitope ──
case_study_results = {}

for target_epi in TARGET_EPITOPES:
    print(f"\n{'=' * 78}")
    print(f"CASE STUDY: {target_epi}")
    print(f"{'=' * 78}")

    target_cdr3s = epitope_cdr3s[target_epi]
    if not target_cdr3s:
        print(f"  No CDR3s for {target_epi}, skipping")
        continue

    # Find consensus clusters containing target epitope CDR3s
    target_clusters = []
    for c in clusters:
        target_in_cluster = [m for m in c.member_ids if m in target_cdr3s]
        if target_in_cluster:
            target_clusters.append({
                "cluster_id": c.cluster_id,
                "confidence": c.cluster_confidence,
                "total_members": len(c.member_ids),
                "target_members": target_in_cluster,
                "n_target": len(target_in_cluster),
                "all_members": c.member_ids,
                "core_members": c.core_member_ids,
            })

    print(f"\n  Consensus clusters containing {target_epi} CDR3s: {len(target_clusters)}")
    for tc in sorted(target_clusters, key=lambda x: -x["n_target"])[:10]:
        purity = tc["n_target"] / tc["total_members"] if tc["total_members"] > 0 else 0
        print(f"    {tc['cluster_id']}: {tc['n_target']}/{tc['total_members']} target "
              f"(purity={purity:.3f}, conf={tc['confidence']:.3f})")

    # ── Per-method comparison ──
    print(f"\n  Per-method comparison for {target_epi}:")
    print(f"  {'Method':<15s} {'N_tgt_cls':>10s} {'N_tgt_ret':>10s} {'Best_pur':>10s}")

    method_target_stats = {}
    for mname, (assigns, rt) in method_results.items():
        # Build method cluster map
        method_cl = defaultdict(set)
        for a in assigns:
            method_cl[a.cluster_id].add(a.tcr_id)

        # Find clusters with target CDR3s
        method_target_clusters = []
        retained = set()
        for cid, members in method_cl.items():
            target_in = members & target_cdr3s
            if target_in:
                purity = len(target_in) / len(members)
                method_target_clusters.append({
                    "n_target": len(target_in),
                    "total": len(members),
                    "purity": purity,
                })
                retained.update(target_in)

        if method_target_clusters:
            best_pur = max(c["purity"] for c in method_target_clusters)
            method_target_stats[mname] = {
                "n_target_clusters": len(method_target_clusters),
                "n_retained": len(retained),
                "best_purity": best_pur,
            }
            print(f"  {mname:<15s} {len(method_target_clusters):10d} {len(retained):10d} {best_pur:10.3f}")
        else:
            method_target_stats[mname] = {"n_target_clusters": 0, "n_retained": 0, "best_purity": 0}
            print(f"  {mname:<15s} {0:10d} {0:10d} {0:10.3f}")

    # Consensus stats
    cons_retained = set()
    for tc in target_clusters:
        cons_retained.update(tc["target_members"])
    best_cons_pur = max((tc["n_target"] / tc["total_members"] for tc in target_clusters), default=0)
    print(f"  {'CONSENSUS':<15s} {len(target_clusters):10d} {len(cons_retained):10d} {best_cons_pur:10.3f}")

    # ── Unique consensus discoveries ──
    # CDR3s in consensus target clusters but NOT in any individual method's target cluster
    individual_retained = set()
    for mname, stats in method_target_stats.items():
        if stats["n_retained"] > 0:
            assigns, _ = method_results[mname]
            method_cl = defaultdict(set)
            for a in assigns:
                method_cl[a.cluster_id].add(a.tcr_id)
            for cid, members in method_cl.items():
                if members & target_cdr3s:
                    individual_retained.update(members)

    consensus_all_in_target = set()
    for tc in target_clusters:
        consensus_all_in_target.update(tc["all_members"])

    # Consensus-only = in consensus target cluster but not in any individual target cluster
    consensus_only = consensus_all_in_target - individual_retained
    # Filter to only non-target (background) CDR3s that consensus recruited
    consensus_only_labeled = [c for c in consensus_only if c in cdr3_to_epi]
    consensus_only_bg = [c for c in consensus_only if c not in cdr3_to_epi]

    print(f"\n  Unique consensus discoveries (not in any individual target cluster):")
    print(f"    Total unique members: {len(consensus_only)}")
    print(f"    Labeled CDR3s: {len(consensus_only_labeled)}")
    print(f"    Background CDR3s: {len(consensus_only_bg)}")

    if consensus_only_labeled:
        print(f"    Labeled CDR3 epitopes:")
        for c in sorted(consensus_only_labeled)[:20]:
            epi = cdr3_to_epi.get(c, "UNKNOWN")
            print(f"      {c} ({epi})")

    # ── Motif analysis for top consensus cluster ──
    print(f"\n  Motif analysis (top consensus cluster):")
    if target_clusters:
        top_cluster = max(target_clusters, key=lambda x: x["n_target"])
        target_seqs = top_cluster["target_members"]
        all_seqs = top_cluster["all_members"]

        print(f"    Cluster {top_cluster['cluster_id']}: {top_cluster['n_target']} target / {top_cluster['total_members']} total")
        print(f"    Target CDR3 sequences (first 15):")
        for s in sorted(target_seqs)[:15]:
            print(f"      {s}")

        # Length distribution
        lens = [len(s) for s in target_seqs]
        print(f"    Length distribution: mean={np.mean(lens):.1f}, range={min(lens)}-{max(lens)}")

        # Most common length
        common_len = Counter(lens).most_common(1)[0][0]
        same_len_seqs = [s for s in target_seqs if len(s) == common_len]
        if len(same_len_seqs) >= 3:
            psfm = motif_from_sequences(same_len_seqs)
            print(f"    Motif at length {common_len} ({len(same_len_seqs)} sequences):")
            for pos in sorted(psfm.keys()):
                top3 = sorted(psfm[pos].items(), key=lambda x: -x[1])[:3]
                motif_str = " ".join(f"{aa}:{freq:.2f}" for aa, freq in top3)
                print(f"      Pos {pos}: {motif_str}")

    # ── V/J gene enrichment ──
    print(f"\n  V/J gene enrichment for {target_epi}:")
    target_v_genes = [cdr3_to_v.get(c, "Unknown") for c in target_cdr3s if cdr3_to_v.get(c)]
    target_j_genes = [cdr3_to_j.get(c, "Unknown") for c in target_cdr3s if cdr3_to_j.get(c)]

    # Background V/J
    all_v_genes = [cdr3_to_v.get(c, "Unknown") for c in tcr_ids if cdr3_to_v.get(c)]
    all_j_genes = [cdr3_to_j.get(c, "Unknown") for c in tcr_ids if cdr3_to_j.get(c)]

    if target_v_genes:
        print(f"    Top V genes in {target_epi} clusters:")
        v_counts = Counter(target_v_genes)
        v_bg = Counter(all_v_genes)
        n_target = len(target_v_genes)
        n_bg = len(all_v_genes)
        for vg, cnt in v_counts.most_common(5):
            bg_cnt = v_bg.get(vg, 0)
            enrich = (cnt / n_target) / (bg_cnt / n_bg) if bg_cnt > 0 and n_bg > 0 else 0
            print(f"      {vg}: {cnt}/{n_target} ({cnt/n_target:.3f}), enrichment={enrich:.2f}x")

    if target_j_genes:
        print(f"    Top J genes in {target_epi} clusters:")
        j_counts = Counter(target_j_genes)
        j_bg = Counter(all_j_genes)
        for jg, cnt in j_counts.most_common(5):
            bg_cnt = j_bg.get(jg, 0)
            enrich = (cnt / n_target) / (bg_cnt / n_bg) if bg_cnt > 0 and n_bg > 0 else 0
            print(f"      {jg}: {cnt}/{n_target} ({cnt/n_target:.3f}), enrichment={enrich:.2f}x")

    # Save case study data
    case_study_results[target_epi] = {
        "n_target_cdr3s": len(target_cdr3s),
        "n_consensus_clusters": len(target_clusters),
        "n_consensus_retained": len(cons_retained),
        "best_consensus_purity": best_cons_pur,
        "n_unique_consensus_members": len(consensus_only),
        "n_unique_labeled": len(consensus_only_labeled),
        "top_clusters": [
            {
                "cluster_id": tc["cluster_id"],
                "n_target": tc["n_target"],
                "total": tc["total_members"],
                "purity": tc["n_target"] / tc["total_members"],
                "confidence": tc["confidence"],
                "target_cdr3s": sorted(tc["target_members"])[:20],
                "core_cdr3s": tc["core_members"][:10] if tc["core_members"] else [],
            }
            for tc in sorted(target_clusters, key=lambda x: -x["n_target"])[:5]
        ],
        "individual_methods": method_target_stats,
    }

with open(OUTDIR / "case_study_results.json", "w") as f:
    json.dump(case_study_results, f, indent=2, default=str)

print(f"\nSaved to {OUTDIR / 'case_study_results.json'}")
print("Case Study Done!")
