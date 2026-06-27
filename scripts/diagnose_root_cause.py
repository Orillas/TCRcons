import sys
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
import pandas as pd
import numpy as np
from clustcr.input.vdjdb import parse_vdjdb
from clustcr.clustering.clustering import Clustering
from clustcr.modules.faiss_clustering import FaissClustering
from clustcr.clustering.tools import create_edgelist

# Paper's data loading
vdjdb = parse_vdjdb('./clustcr/input/vdjdb/vdjdb_full.txt', q=1)
epitopes = vdjdb.drop(columns=['cdr3.alpha', 'v.alpha']).dropna().drop_duplicates()
epitopes = epitopes.rename(columns={'cdr3.beta':'CDR3','v.beta':'V','antigen.epitope':'Epitope'})
chain = epitopes.drop(columns='Epitope').drop_duplicates().reset_index(drop=True)
cdr3_input = chain.CDR3.drop_duplicates()

print("=" * 70)
print("DIAGNOSIS: What happens in the MCL second step?")
print("=" * 70)
print(f"\nInput: {len(cdr3_input)} CDR3 sequences")

# Step 1: Run FAISS pre-clustering (same for old and new code)
faiss = FaissClustering(n_cpus=8)
profiles = faiss.train(cdr3_input)
pre_labels = faiss.cluster(cdr3_input, is_profile=True)

# Count pre-clusters
preclusters = {}
for i, label in enumerate(pre_labels):
    seq = cdr3_input.iloc[i]
    label = int(label)
    if label not in preclusters:
        preclusters[label] = []
    preclusters[label].append(seq)

print(f"FAISS pre-clusters: {len(preclusters)}")
print(f"Pre-cluster sizes: min={min(len(v) for v in preclusters.values()])}, "
      f"max={max(len(v) for v in preclusters.values()])}, "
      f"median={np.median([len(v) for v in preclusters.values()])}")

# Step 2: Check how many pre-clusters have HD=1 edges
import networkx as nx
no_edges_count = 0
sequences_lost = 0
sequences_kept = 0

for label, seqs in preclusters.items():
    edges = create_edgelist(seqs)
    if not edges or len(edges.strip()) == 0:
        no_edges_count += 1
        sequences_lost += len(seqs)
    else:
        try:
            G = nx.parse_adjlist(edges, nodetype=str)
            sequences_kept += len(seqs)
        except nx.NetworkXError:
            no_edges_count += 1
            sequences_lost += len(seqs)

print(f"\nPre-clusters with NO HD=1 edges: {no_edges_count}")
print(f"Sequences in edgeless pre-clusters: {sequences_lost}")
print(f"Sequences in connected pre-clusters: {sequences_kept}")
print(f"Total: {sequences_lost + sequences_kept}")
print(f"\nOLD code behavior: keeps ALL {sequences_lost + sequences_kept} sequences")
print(f"NEW code behavior: loses {sequences_lost} sequences (only keeps {sequences_kept})")
print(f"Difference: {sequences_lost / (sequences_lost + sequences_kept) * 100:.1f}% of sequences lost")

# Now check: how many pre-clusters are singletons (size=1)?
singletons = sum(1 for v in preclusters.values() if len(v) == 1)
singleton_seqs = sum(len(v) for v in preclusters.values() if len(v) == 1)
print(f"\nSingleton pre-clusters: {singletons} (containing {singleton_seqs} sequences)")

# Compare with actual Clustering output
print("\n" + "=" * 70)
print("Actual Clustering output vs input")
print("=" * 70)
result = Clustering(n_cpus=8).fit(cdr3_input)
output_cdr3 = result.clusters_df.junction_aa.unique()
print(f"Input CDR3s: {len(cdr3_input)}")
print(f"Output CDR3s: {len(output_cdr3)}")
print(f"Lost: {len(cdr3_input) - len(output_cdr3)} ({(len(cdr3_input) - len(output_cdr3))/len(cdr3_input)*100:.1f}%)")

# Check retention directly
epi_renamed = epitopes.drop(columns=['V','subject','count']).drop_duplicates()
epi_renamed = epi_renamed.rename(columns={'CDR3':'junction_aa','Epitope':'epitope'})
metrics = result.metrics(epi_renamed).summary()
ret = metrics[metrics['metrics']=='retention']['actual'].values[0]
print(f"\nRetention: {ret:.4f}")
print(f"Expected if no loss: {len(output_cdr3)}/{len(epi_renamed.junction_aa.unique())} = {len(output_cdr3)/len(epi_renamed.junction_aa.unique()):.4f}")
