import sys
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import markov_clustering as mcl
from clustcr.input.vdjdb import parse_vdjdb
from clustcr.modules.faiss_clustering import FaissClustering
from clustcr.clustering.tools import create_edgelist

# Paper exact data loading
vdjdb = parse_vdjdb("./clustcr/input/vdjdb/vdjdb_full.txt", q=1)
epitopes = vdjdb.drop(columns=["cdr3.alpha", "v.alpha"]).dropna().drop_duplicates()
epitopes = epitopes.rename(columns={"cdr3.beta":"CDR3","v.beta":"V","antigen.epitope":"Epitope"})
chain = epitopes.drop(columns="Epitope").drop_duplicates().reset_index(drop=True)
cdr3_input = chain.CDR3.drop_duplicates()
epi_metrics = epitopes.drop(columns=["V","subject","count"]).drop_duplicates()
epi_renamed = epi_metrics.rename(columns={"CDR3":"junction_aa","Epitope":"epitope"})

print("=" * 70)
print("DEEP DIVE: Quantifying Each Code Change")
print("=" * 70)

# Step 1: FAISS pre-clustering
faiss = FaissClustering(n_cpus=8)
profiles = faiss.train(cdr3_input)
profiles = faiss.profiles
pre_labels = faiss.cluster(cdr3_input, is_profile=True)

preclusters = defaultdict(list)
for i, label in enumerate(pre_labels):
    preclusters[int(label)].append(cdr3_input.iloc[i])

n_pre = len(preclusters)

# Step 2: Classify each pre-cluster
n_edgeless = 0
n_sparse = 0
n_dense = 0
seqs_edgeless = 0
seqs_sparse = 0
seqs_dense = 0
seqs_mcl_kept = 0

for pc_label, seqs in preclusters.items():
    edges = create_edgelist(seqs)
    if not edges or len(edges.strip()) == 0:
        n_edgeless += 1
        seqs_edgeless += len(seqs)
    else:
        try:
            G = nx.parse_adjlist(edges, nodetype=str)
            mat = nx.to_scipy_sparse_array(G)
            result = mcl.run_mcl(mat, inflation=1.2, expansion=2)
            mcl_output = mcl.get_clusters(result)
            kept = sum(len(cluster) for cluster in mcl_output)
            n_dense += 1
            seqs_dense += len(seqs)
            seqs_mcl_kept += kept
        except (nx.NetworkXError, ValueError):
            n_sparse += 1
            seqs_sparse += len(seqs)

total = seqs_edgeless + seqs_sparse + seqs_dense

print("\n--- Pre-cluster Classification ---")
print("Total pre-clusters: %d" % n_pre)
print("  Edgeless (no HD=1 pairs): %d clusters, %d seqs (%.1f%%)" % (n_edgeless, seqs_edgeless, seqs_edgeless/total*100))
print("  Sparse (MCL fails):       %d clusters, %d seqs (%.1f%%)" % (n_sparse, seqs_sparse, seqs_sparse/total*100))
print("  Dense (MCL succeeds):     %d clusters, %d seqs (%.1f%%)" % (n_dense, seqs_dense, seqs_dense/total*100))
print("  MCL output from dense:    %d seqs (%.1f%% of dense input)" % (seqs_mcl_kept, seqs_mcl_kept/max(1,seqs_dense)*100))

print("\n--- OLD vs NEW Code Behavior ---")
old_kept = seqs_edgeless + seqs_sparse + seqs_mcl_kept
new_kept = seqs_mcl_kept
print("OLD code keeps: %d seqs (%.1f%%)" % (old_kept, old_kept/total*100))
print("  - Edgeless preserved as single clusters: %d seqs" % (seqs_edgeless + seqs_sparse))
print("  - MCL output: %d seqs" % seqs_mcl_kept)
print("NEW code keeps: %d seqs (%.1f%%)" % (new_kept, new_kept/total*100))
print("  - Edgeless DROPPED: %d seqs LOST" % (seqs_edgeless + seqs_sparse))
print("  - MCL output: %d seqs" % seqs_mcl_kept)
print("NET LOSS from code change: %d seqs (%.1f%%)" % (seqs_edgeless + seqs_sparse, (seqs_edgeless + seqs_sparse)/total*100))

# Step 3: Purity of edgeless clusters
print("\n--- Purity Analysis of Edgeless Pre-clusters ---")
cdr3_to_epi = {}
for _, row in epi_renamed.iterrows():
    cdr3_to_epi[row["junction_aa"]] = row["epitope"]

edgeless_purities = []
for pc_label, seqs in preclusters.items():
    edges = create_edgelist(seqs)
    if not edges or len(edges.strip()) == 0:
        epis = [cdr3_to_epi.get(s, None) for s in seqs]
        epis_valid = [e for e in epis if e is not None]
        if epis_valid:
            counts = Counter(epis_valid)
            purity = counts.most_common(1)[0][1] / len(epis_valid)
            edgeless_purities.append((len(seqs), purity, counts.most_common(1)[0][0]))

if edgeless_purities:
    sizes_ep = [x[0] for x in edgeless_purities]
    pures_ep = [x[1] for x in edgeless_purities]
    print("Edgeless clusters with labels: %d" % len(edgeless_purities))
    print("  Size range: %d-%d" % (min(sizes_ep), max(sizes_ep)))
    print("  Mean purity: %.3f" % np.mean(pures_ep))
    high_p = sum(1 for p in pures_ep if p >= 0.9)
    print("  High purity (>0.9): %d/%d (%.1f%%)" % (high_p, len(pures_ep), high_p/len(pures_ep)*100))
    print("  -> Dropping these HIGH-PURITY clusters hurts Consistency!")
else:
    print("No edgeless clusters with epitope labels")

# Step 4: Simulate OLD code behavior
print("\n--- Simulating OLD Code (preserving edgeless clusters) ---")
from clustcr.clustering.clustering import Clustering, ClusteringResult

# Get FAISS pre-clusters via actual Clustering (which calls _faiss)
result_new = Clustering(n_cpus=8).fit(cdr3_input)

# Build old-style output: current MCL output + preserved edgeless clusters
new_output = result_new.clusters_df[["junction_aa", "cluster"]].copy()

# Find sequences NOT in new output (these are the lost edgeless/sparse ones)
new_cdr3s = set(new_output["junction_aa"])
all_cdr3s = set(cdr3_input)
lost_cdr3s = all_cdr3s - new_cdr3s

# Add lost sequences as individual clusters (simulating old code)
max_cluster = new_output["cluster"].max()
old_rows = []
for i, seq in enumerate(lost_cdr3s):
    old_rows.append({"junction_aa": seq, "cluster": max_cluster + i + 1})

old_output = pd.concat([new_output, pd.DataFrame(old_rows)], ignore_index=True)

# Calculate metrics for old-style output
from clustcr.clustering.metrics import Metrics
metrics_new = Metrics(new_output, epi_renamed).summary()
metrics_old = Metrics(old_output, epi_renamed).summary()

print("\n--- Metric Comparison: OLD vs NEW behavior ---")
print("%-15s %12s %12s %12s" % ("Metric", "OLD (sim)", "NEW (actual)", "Published"))
print("-" * 55)
pub_vals = {"retention": 0.2363, "purity": 0.8581, "purity_90": 0.7265, "consistency": 0.3614}
for m in ["retention", "purity", "purity_90", "consistency"]:
    old_v = metrics_old[metrics_old["metrics"]==m]["actual"].values[0]
    new_v = metrics_new[metrics_new["metrics"]==m]["actual"].values[0]
    print("%-15s %12.4f %12.4f %12.4f" % (m, old_v, new_v, pub_vals[m]))

print("\nOLD simulation has %d clusters (%d from MCL + %d preserved singletons)" % (
    old_output["cluster"].nunique(),
    new_output["cluster"].nunique(),
    len(lost_cdr3s)))
