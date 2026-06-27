import sys
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import markov_clustering as mcl
from clustcr.input.vdjdb import parse_vdjdb
from clustcr.clustering.clustering import Clustering, ClusteringResult
from clustcr.clustering.tools import create_edgelist
from clustcr.clustering.metrics import Metrics

# Paper exact data loading
vdjdb = parse_vdjdb("./clustcr/input/vdjdb/vdjdb_full.txt", q=1)
epitopes = vdjdb.drop(columns=["cdr3.alpha", "v.alpha"]).dropna().drop_duplicates()
epitopes = epitopes.rename(columns={"cdr3.beta":"CDR3","v.beta":"V","antigen.epitope":"Epitope"})
chain = epitopes.drop(columns="Epitope").drop_duplicates().reset_index(drop=True)
cdr3_input = chain.CDR3.drop_duplicates()
epi_metrics = epitopes.drop(columns=["V","subject","count"]).drop_duplicates()
epi_renamed = epi_metrics.rename(columns={"CDR3":"junction_aa","Epitope":"epitope"})

print("=" * 70)
print("DEEP DIVE: Why does the new code perform worse?")
print("=" * 70)

# Run actual clustering (NEW code)
result = Clustering(n_cpus=8).fit(cdr3_input)
new_output = result.clusters_df[["junction_aa", "cluster"]].copy()
new_cdr3s = set(new_output["junction_aa"])
all_cdr3s = set(cdr3_input)
lost_cdr3s = all_cdr3s - new_cdr3s

print("\nInput: %d CDR3s" % len(all_cdr3s))
print("NEW code output: %d CDR3s (%d lost = %.1f%%)" % (
    len(new_cdr3s), len(lost_cdr3s), len(lost_cdr3s)/len(all_cdr3s)*100))

# Classify lost sequences: why were they lost?
# Get FAISS pre-cluster assignments by re-running FAISS
from clustcr.modules.faiss_clustering import FaissClustering
faiss_obj = FaissClustering(n_cpus=8)
profiles = faiss_obj.train(cdr3_input)

# Cluster using profiles
D, I = faiss_obj.kmeans.index.search(profiles, 1)
pre_labels = I.flatten()

preclusters = defaultdict(list)
for i, label in enumerate(pre_labels):
    preclusters[int(label)].append(cdr3_input.iloc[i])

# For each pre-cluster, classify: edgeless / sparse / dense
print("\n--- Pre-cluster Analysis ---")
n_edgeless = 0; n_sparse = 0; n_dense = 0
seqs_edgeless = 0; seqs_sparse = 0; seqs_dense = 0
seqs_mcl_kept = 0

edgeless_seqs = []  # sequences in edgeless clusters

for pc_label, seqs in preclusters.items():
    edges = create_edgelist(seqs)
    if not edges or len(edges) == 0:
        n_edgeless += 1
        seqs_edgeless += len(seqs)
        edgeless_seqs.extend(seqs)
    else:
        try:
            G = nx.parse_adjlist(edges, nodetype=str)
            mat = nx.to_scipy_sparse_array(G)
            result_mcl = mcl.run_mcl(mat, inflation=1.2, expansion=2)
            mcl_output = mcl.get_clusters(result_mcl)
            kept = sum(len(cluster) for cluster in mcl_output)
            n_dense += 1
            seqs_dense += len(seqs)
            seqs_mcl_kept += kept
        except (nx.NetworkXError, ValueError):
            n_sparse += 1
            seqs_sparse += len(seqs)
            edgeless_seqs.extend(seqs)  # sparse = also lost in new code

total = seqs_edgeless + seqs_sparse + seqs_dense
print("Total pre-clusters: %d" % len(preclusters))
print("  Edgeless: %d clusters, %d seqs (%.1f%%)" % (n_edgeless, seqs_edgeless, seqs_edgeless/total*100))
print("  Sparse:   %d clusters, %d seqs (%.1f%%)" % (n_sparse, seqs_sparse, seqs_sparse/total*100))
print("  Dense:    %d clusters, %d seqs (%.1f%%)" % (n_dense, seqs_dense, seqs_dense/total*100))
print("  MCL keeps %d/%d from dense (%.1f%%)" % (seqs_mcl_kept, seqs_dense, seqs_mcl_kept/max(1,seqs_dense)*100))

# OLD code: preserve edgeless + sparse, MCL output from dense
old_kept = seqs_edgeless + seqs_sparse + seqs_mcl_kept
new_kept = seqs_mcl_kept
print("\nOLD code total: %d seqs (%.1f%%)" % (old_kept, old_kept/total*100))
print("NEW code total: %d seqs (%.1f%%)" % (new_kept, new_kept/total*100))
print("Extra loss:     %d seqs" % (seqs_edgeless + seqs_sparse))

# Purity of edgeless clusters
print("\n--- Purity of Edgeless+Sparse Clusters ---")
cdr3_to_epi = dict(zip(epi_renamed["junction_aa"], epi_renamed["epitope"]))

pure_counts = []
for pc_label, seqs in preclusters.items():
    edges = create_edgelist(seqs)
    if edges:
        try:
            G = nx.parse_adjlist(edges, nodetype=str)
            mat = nx.to_scipy_sparse_array(G)
            mcl.run_mcl(mat, inflation=1.2, expansion=2)
        except:
            pass  # sparse cluster
        continue
    # edgeless cluster
    epis = [cdr3_to_epi.get(s) for s in seqs if cdr3_to_epi.get(s)]
    if epis:
        counts = Counter(epis)
        purity = counts.most_common(1)[0][1] / len(epis)
        pure_counts.append((len(seqs), purity, counts.most_common(1)[0][0]))

if pure_counts:
    sizes_ep = [x[0] for x in pure_counts]
    pures_ep = [x[1] for x in pure_counts]
    print("Edgeless clusters with labels: %d" % len(pure_counts))
    print("  Mean purity: %.3f" % np.mean(pures_ep))
    hp = sum(1 for p in pures_ep if p >= 0.9)
    print("  Purity >= 0.9: %d/%d (%.1f%%)" % (hp, len(pures_ep), hp/len(pure_counts)*100))
    print("  Top epitopes in edgeless clusters:")
    top = Counter([x[2] for x in pure_counts]).most_common(5)
    for epi, cnt in top:
        print("    %s: %d clusters" % (epi, cnt))

# Simulate OLD code: add lost seqs as singletons
print("\n--- Simulating OLD Code Behavior ---")
max_cluster = new_output["cluster"].max()
old_rows = []
for i, seq in enumerate(lost_cdr3s):
    old_rows.append({"junction_aa": seq, "cluster": max_cluster + i + 1})
old_output = pd.concat([new_output, pd.DataFrame(old_rows)], ignore_index=True)

metrics_new = Metrics(new_output, epi_renamed).summary()
metrics_old = Metrics(old_output, epi_renamed).summary()

pub = {"retention": 0.2363, "purity": 0.8581, "purity_90": 0.7265, "consistency": 0.3614}

print("\n%-15s %12s %12s %12s %12s" % ("Metric", "OLD (sim)", "NEW", "Published", "OLD closer?"))
print("-" * 65)
for m in ["retention", "purity", "purity_90", "consistency"]:
    o = metrics_old[metrics_old["metrics"]==m]["actual"].values[0]
    n = metrics_new[metrics_new["metrics"]==m]["actual"].values[0]
    p = pub[m]
    old_dev = abs(o - p)
    new_dev = abs(n - p)
    closer = "YES" if old_dev < new_dev else "no"
    print("%-15s %12.4f %12.4f %12.4f %12s" % (m, o, n, p, closer))

print("\nClusters: OLD=%d (MCL:%d + singletons:%d), NEW=%d" % (
    old_output["cluster"].nunique(),
    new_output["cluster"].nunique(),
    len(lost_cdr3s),
    new_output["cluster"].nunique()))
