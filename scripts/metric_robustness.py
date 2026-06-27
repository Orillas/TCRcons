"""
Compare reproducibility of different clustering metrics under code version change.
Metrics: ARI, AMI, NMI, V-measure, Homogeneity, Completeness, Purity, Purity_90, Consistency, F1
"""
import sys
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score,
    normalized_mutual_info_score, v_measure_score,
    homogeneity_score, completeness_score, fowlkes_mallows_score,
    f1_score
)
from clustcr.input.vdjdb import parse_vdjdb
from clustcr.clustering.clustering import Clustering
from clustcr.clustering.tools import create_edgelist

# ============ Standard Clustering Metrics ============

def compute_ari(labels_true, labels_pred):
    """Adjusted Rand Index - corrects for chance"""
    return adjusted_rand_score(labels_true, labels_pred)

def compute_ami(labels_true, labels_pred):
    """Adjusted Mutual Information - corrects for chance"""
    return adjusted_mutual_info_score(labels_true, labels_pred)

def compute_nmi(labels_true, labels_pred):
    """Normalized Mutual Information"""
    return normalized_mutual_info_score(labels_true, labels_pred)

def compute_vmeasure(labels_true, labels_pred):
    """V-measure (harmonic mean of homogeneity and completeness)"""
    return v_measure_score(labels_true, labels_pred)

def compute_homogeneity(labels_true, labels_pred):
    """Each cluster contains only members of a single class"""
    return homogeneity_score(labels_true, labels_pred)

def compute_completeness(labels_true, labels_pred):
    """All members of a given class are assigned to the same cluster"""
    return completeness_score(labels_true, labels_pred)

def compute_fowlkes_mallows(labels_true, labels_pred):
    """Fowlkes-Mallows index (geometric mean of precision and recall)"""
    return fowlkes_mallows_score(labels_true, labels_pred)

def compute_purity(labels_true, labels_pred):
    """Weighted majority class accuracy"""
    df = pd.DataFrame({"true": labels_true, "pred": labels_pred})
    total = len(df)
    correct = 0
    for cluster in df["pred"].unique():
        subset = df[df["pred"] == cluster]
        majority_class = subset["true"].value_counts().idxmax()
        correct += (subset["true"] == majority_class).sum()
    return correct / total

def compute_purity_90(labels_true, labels_pred):
    """Fraction of clusters with purity >= 0.9"""
    df = pd.DataFrame({"true": labels_true, "pred": labels_pred})
    n_clusters = df["pred"].nunique()
    if n_clusters == 0:
        return 0.0
    high_purity = 0
    for cluster in df["pred"].unique():
        subset = df[df["pred"] == cluster]
        majority_frac = subset["true"].value_counts().iloc[0] / len(subset)
        if majority_frac >= 0.9:
            high_purity += 1
    return high_purity / n_clusters

def compute_consistency(labels_true, labels_pred):
    """Greedy diagonal matching (clusTCR's method)"""
    df = pd.DataFrame({"true": labels_true, "pred": labels_pred})
    df["count"] = 1
    conf = pd.pivot_table(df, values="count", index="true", columns="pred", aggfunc=np.sum, fill_value=0)

    def rec_max(mat):
        if mat.empty:
            return 0
        high = mat.max().max()
        col = mat.max().idxmax()
        row = mat[col].idxmax()
        if len(mat.index) > 1 and len(mat.columns) > 1:
            high = high + rec_max(mat.drop(row, axis=0).drop(col, axis=1))
        return high

    return rec_max(conf) / len(df)

# ============ Main Experiment ============

def get_chain_data(q):
    vdjdb = parse_vdjdb("./clustcr/input/vdjdb/vdjdb_full.txt", q=q)
    epitopes = vdjdb.drop(columns=["cdr3.alpha", "v.alpha"]).dropna().drop_duplicates()
    epitopes = epitopes.rename(columns={"cdr3.beta":"CDR3","v.beta":"V","antigen.epitope":"Epitope"})
    chain = epitopes.drop(columns="Epitope").drop_duplicates().reset_index(drop=True)
    return chain, epitopes

def compute_all_metrics(labels_true, labels_pred):
    return {
        "ARI": compute_ari(labels_true, labels_pred),
        "AMI": compute_ami(labels_true, labels_pred),
        "NMI": compute_nmi(labels_true, labels_pred),
        "V-measure": compute_vmeasure(labels_true, labels_pred),
        "Homogeneity": compute_homogeneity(labels_true, labels_pred),
        "Completeness": compute_completeness(labels_true, labels_pred),
        "Fowlkes-Mallows": compute_fowlkins_mallows(labels_true, labels_pred),
        "Purity": compute_purity(labels_true, labels_pred),
        "Purity_90": compute_purity_90(labels_true, labels_pred),
        "Consistency": compute_consistency(labels_true, labels_pred),
    }

def get_labels(cluster_df, epi_df):
    """Merge cluster assignments with epitope labels, return aligned arrays"""
    merged = pd.merge(
        epi_df.rename(columns={"CDR3":"junction_aa","Epitope":"epitope"}),
        cluster_df,
        on="junction_aa"
    )
    return merged["epitope"].values, merged["cluster"].values

print("=" * 80)
print("METRIC ROBUSTNESS TEST: Old vs New Code Behavior")
print("=" * 80)

all_results = []

for q in [0, 1, 2]:
    chain_data, epitope_data = get_chain_data(q)
    cdr3_input = chain_data.CDR3.drop_duplicates()
    epi_metrics = epitope_data.drop(columns=["V","subject","count"]).drop_duplicates()

    # NEW code output
    result = Clustering(n_cpus=8).fit(cdr3_input)
    new_output = result.clusters_df[["junction_aa","cluster"]].copy()

    # SIMULATE OLD code: add lost sequences as clusters (by pre-cluster)
    # First, get FAISS pre-clusters to group lost sequences properly
    from clustcr.modules.faiss_clustering import FaissClustering
    faiss_obj = FaissClustering(n_cpus=8)
    profiles = faiss_obj.train(cdr3_input)
    D, I = faiss_obj.kmeans.index.search(profiles, 1)
    pre_labels = I.flatten()

    # Build pre-cluster membership
    precluster_map = defaultdict(list)
    for i, label in enumerate(pre_labels):
        precluster_map[int(label)].append(cdr3_input.iloc[i])

    # Identify which CDR3s are in the new output
    new_cdr3s = set(new_output["junction_aa"])
    max_cluster = new_output["cluster"].max()

    # For each pre-cluster, add lost sequences as a single cluster (old behavior)
    old_rows = []
    cluster_offset = max_cluster + 1
    for pc_label, seqs in precluster_map.items():
        lost_in_pc = [s for s in seqs if s not in new_cdr3s]
        if lost_in_pc:
            for s in lost_in_pc:
                old_rows.append({"junction_aa": s, "cluster": cluster_offset})
            cluster_offset += 1

    old_output = pd.concat([new_output, pd.DataFrame(old_rows)], ignore_index=True)

    # Compute metrics for both versions
    epi_r = epi_metrics.rename(columns={"CDR3":"junction_aa","Epitope":"epitope"})

    # New code metrics
    merged_new = pd.merge(epi_r, new_output, on="junction_aa")
    y_true_new = merged_new["epitope"].values
    y_pred_new = merged_new["cluster"].values

    # Old code metrics
    merged_old = pd.merge(epi_r, old_output, on="junction_aa")
    y_true_old = merged_old["epitope"].values
    y_pred_old = merged_old["cluster"].values

    # Compute all metrics
    metric_names = ["ARI", "AMI", "NMI", "V-measure", "Homogeneity",
                    "Completeness", "Fowlkes-Mallows", "Purity", "Purity_90", "Consistency"]
    funcs = [compute_ari, compute_ami, compute_nmi, compute_vmeasure,
             compute_homogeneity, compute_completeness, compute_fowlkes_mallows,
             compute_purity, compute_purity_90, compute_consistency]

    print(f"\n{'='*70}")
    print(f"q={q}: {len(cdr3_input)} CDR3s, NEW={len(new_cdr3s)}, OLD_sim={len(set(old_output['junction_aa']))}")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'OLD (sim)':>12} {'NEW':>12} {'Abs Diff':>12} {'Rel Dev%':>12}")
    print("-" * 70)

    for name, func in zip(metric_names, funcs):
        old_val = func(y_true_old, y_pred_old)
        new_val = func(y_true_new, y_pred_new)
        abs_diff = new_val - old_val
        avg = (old_val + new_val) / 2
        rel_dev = abs(abs_diff) / max(avg, 1e-10) * 100
        print(f"{name:<20} {old_val:>12.4f} {new_val:>12.4f} {abs_diff:>+12.4f} {rel_dev:>11.1f}%")
        all_results.append({
            "q": q, "metric": name,
            "old_sim": old_val, "new": new_val,
            "abs_diff": abs_diff, "rel_dev": rel_dev
        })

# Summary
print(f"\n{'='*80}")
print("SUMMARY: Average Relative Deviation Across q=0,1,2")
print(f"{'='*80}")
results_df = pd.DataFrame(all_results)
summary = results_df.groupby("metric")["rel_dev"].mean().sort_values()
print(f"{'Metric':<20} {'Avg Rel Dev%':>15}")
print("-" * 40)
for metric, dev in summary.items():
    bar = "█" * int(dev / 2)
    tag = " ← MORE ROBUST" if dev < 10 else (" ← MODERATE" if dev < 30 else " ← SENSITIVE")
    print(f"{metric:<20} {dev:>12.1f}%  {bar}{tag}")
