import sys
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
import pandas as pd
import numpy as np
from clustcr.input.vdjdb import parse_vdjdb
from clustcr.clustering.clustering import Clustering

def get_chain_data(q):
    vdjdb = parse_vdjdb("./clustcr/input/vdjdb/vdjdb_full.txt", q=q)
    epitopes = vdjdb.drop(columns=["cdr3.alpha", "v.alpha"]).dropna().drop_duplicates()
    epitopes = epitopes.rename(columns={"cdr3.beta":"CDR3","v.beta":"V","antigen.epitope":"Epitope"})
    chain = epitopes.drop(columns="Epitope").drop_duplicates().reset_index(drop=True)
    return chain, epitopes

published = {
    "ClusTCR": {
        0: {"retention": 0.2517, "purity": 0.5871, "purity_90": 0.4089, "consistency": 0.1300},
        1: {"retention": 0.2363, "purity": 0.8581, "purity_90": 0.7265, "consistency": 0.3614},
        2: {"retention": 0.2623, "purity": 0.9286, "purity_90": 0.9038, "consistency": 0.4796},
    }
}

print("=" * 80)
print("COMPREHENSIVE COMPARISON: Published vs Current Code")
print("=" * 80)

for q in [0, 1, 2]:
    chain_data, epitope_data = get_chain_data(q)
    cdr3_input = chain_data.CDR3.drop_duplicates()
    epi_metrics = epitope_data.drop(columns=["V","subject","count"]).drop_duplicates()
    epi_renamed = epi_metrics.rename(columns={"CDR3":"junction_aa","Epitope":"epitope"})

    result = Clustering(n_cpus=8).fit(cdr3_input)
    metrics = result.metrics(epi_renamed).summary()

    ret = metrics[metrics["metrics"]=="retention"]["actual"].values[0]
    pur = metrics[metrics["metrics"]=="purity"]["actual"].values[0]
    p90 = metrics[metrics["metrics"]=="purity_90"]["actual"].values[0]
    con = metrics[metrics["metrics"]=="consistency"]["actual"].values[0]

    pub = published["ClusTCR"][q]

    n_in = len(cdr3_input)
    n_out = len(result.clusters_df.junction_aa.unique())
    print(f"\nq={q}: {n_in} CDR3s input -> {n_out} output ({n_out/n_in*100:.1f}% retained)")
    print(f"{'Metric':<15} {'Published':>12} {'Reproduced':>12} {'Abs Diff':>12} {'Rel Dev%':>12}")
    print("-" * 65)
    for metric_name, pub_val, our_val in [
        ("Retention", pub["retention"], ret),
        ("Purity", pub["purity"], pur),
        ("Purity_90", pub["purity_90"], p90),
        ("Consistency", pub["consistency"], con),
    ]:
        abs_diff = our_val - pub_val
        rel_dev = abs(abs_diff) / pub_val * 100
        print(f"{metric_name:<15} {pub_val:>12.4f} {our_val:>12.4f} {abs_diff:>+12.4f} {rel_dev:>11.1f}%")

# Consistency sensitivity analysis
print("\n" + "=" * 80)
print("CONSISTENCY SENSITIVITY ANALYSIS (q=1)")
print("=" * 80)

q = 1
chain_data, epitope_data = get_chain_data(q)
cdr3_input = chain_data.CDR3.drop_duplicates()
epi_metrics = epitope_data.drop(columns=["V","subject","count"]).drop_duplicates()
epi_renamed = epi_metrics.rename(columns={"CDR3":"junction_aa","Epitope":"epitope"})

result = Clustering(n_cpus=8).fit(cdr3_input)
gt = pd.merge(epi_renamed, result.clusters_df, on="junction_aa")
gt["count"] = 1
conf = pd.pivot_table(gt, values="count", index=gt["epitope"], columns=gt["cluster"], aggfunc=np.sum, fill_value=0)

print(f"Confusion matrix: {conf.shape[0]} epitopes x {conf.shape[1]} clusters")
print(f"Total entries: {conf.values.sum()}")
print(f"Non-zero entries: {(conf.values > 0).sum()}")
sparsity = 1 - (conf.values > 0).sum() / (conf.shape[0] * conf.shape[1])
print(f"Sparsity: {sparsity:.4f}")
print(f"Max value: {conf.values.max()}")
print(f"Sum: {conf.values.sum()}")

# Show how greedy rec_max works step by step
print("\nGreedy rec_max trace (first 5 steps):")
mat = conf.copy()
for step in range(5):
    if mat.empty:
        break
    high = mat.max().max()
    col = mat.max().idxmax()
    row = mat[col].idxmax()
    print(f"  Step {step+1}: max={high:.0f}, epitope={row}, cluster={col}")
    if len(mat.index) > 1 and len(mat.columns) > 1:
        mat = mat.drop(row, axis=0).drop(col, axis=1)
    else:
        break

print("\n" + "=" * 80)
print("CODE VERSION CHANGES SINCE PAPER (key commits)")
print("=" * 80)
print("1. c774ffd (2021-04-29): Fixed retention calculation bug")
print("   OLD: len(nodelist) / len(epidata.CDR3.unique())")
print("   NEW: len(nodelist.CDR3.unique()) / len(epidata.CDR3.unique())")
print()
print("2. 266980a (2022-08-03): Column name refactoring")
print("   CDR3 -> junction_aa, Epitope -> epitope")
print()
print("3. MCL_from_preclusters: Error handling changed")
print("   OLD: try/except catches NetworkXError, keeps edgeless pre-clusters")
print("   NEW: NetworkXError caught inside MCL, returns EMPTY DataFrame")
print("   EFFECT: Sequences in edgeless pre-clusters are DROPPED")
