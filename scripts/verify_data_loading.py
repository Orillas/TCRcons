import sys
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
import pandas as pd
import numpy as np
from clustcr.input.vdjdb import parse_vdjdb
from clustcr.clustering.clustering import Clustering

def get_chain_data(chain_name, q=0):
    """Paper's EXACT data loading function"""
    vdjdb = parse_vdjdb('./clustcr/input/vdjdb/vdjdb_full.txt', q=q)
    if chain_name == 'beta':
        epitopes = vdjdb.drop(columns=['cdr3.alpha', 'v.alpha']).dropna().drop_duplicates()
        epitopes = epitopes.rename(columns={'cdr3.beta':'CDR3','v.beta':'V','antigen.epitope':'Epitope'})
        chain = epitopes.drop(columns='Epitope').drop_duplicates().reset_index(drop=True)
    return chain, epitopes

# === Step 1: Diagnose data loading difference ===
print("=" * 70)
print("STEP 1: Why do data sizes differ?")
print("=" * 70)

vdjdb = parse_vdjdb('./clustcr/input/vdjdb/vdjdb_full.txt', q=1)
print(f"\nparsed vdjdb shape: {vdjdb.shape}")
print(f"columns: {list(vdjdb.columns)}")

# Paper's approach: drop alpha columns, then dropna
paper_epitopes = vdjdb.drop(columns=['cdr3.alpha', 'v.alpha']).dropna().drop_duplicates()
print(f"\nPaper approach (drop alpha cols, then dropna):")
print(f"  shape: {paper_epitopes.shape}")
print(f"  unique CDR3: {paper_epitopes['cdr3.beta'].nunique()}")

# Our approach: directly extract beta columns
our_epitopes = vdjdb[['cdr3.beta','v.beta','antigen.epitope']].dropna().drop_duplicates()
print(f"\nOur approach (extract beta cols, dropna):")
print(f"  shape: {our_epitopes.shape}")
print(f"  unique CDR3: {our_epitopes['cdr3.beta'].nunique()}")

# Find the difference
paper_cdr3s = set(paper_epitopes['cdr3.beta'].drop_duplicates())
our_cdr3s = set(our_epitopes['cdr3.beta'].drop_duplicates())
only_in_ours = our_cdr3s - paper_cdr3s
only_in_paper = paper_cdr3s - our_cdr3s
print(f"\nCDR3s only in ours: {len(only_in_ours)}")
print(f"CDR3s only in paper: {len(only_in_paper)}")

# Which column has NaN that causes the difference?
remaining_cols = vdjdb.drop(columns=['cdr3.alpha', 'v.alpha']).columns
print(f"\nRemaining columns after dropping alpha: {list(remaining_cols)}")
for col in remaining_cols:
    nan_count = vdjdb[col].isna().sum()
    print(f"  {col}: {nan_count} NaN values")

# Check which rows are in our data but NOT in paper's data
our_indices = set(our_epitopes.index)
paper_indices = set(paper_epitopes.index)
diff_mask = vdjdb.drop(columns=['cdr3.alpha', 'v.alpha']).isna().any(axis=1) & vdjdb[['cdr3.beta','v.beta','antigen.epitope']].notna().all(axis=1)
diff_rows = vdjdb[diff_mask]
print(f"\nRows with valid beta data but NaN in other columns: {len(diff_rows)}")
if len(diff_rows) > 0:
    print("NaN columns per row:")
    for idx, row in diff_rows.head(10).iterrows():
        nans = [c for c in remaining_cols if pd.isna(row[c])]
        print(f"  row {idx}: NaN in {nans}")

# === Step 2: Run paper's exact code and compare ===
print("\n" + "=" * 70)
print("STEP 2: Run paper's exact code")
print("=" * 70)

for q in [0, 1, 2]:
    chain_data, epitope_data = get_chain_data('beta', q=q)
    cdr3_input = chain_data.CDR3.drop_duplicates()
    epi_metrics = epitope_data.drop(columns=['V','subject','count']).drop_duplicates()

    print(f"\nq={q}: {len(cdr3_input)} CDR3s, {len(epi_metrics)} epitope pairs, {epi_metrics.Epitope.nunique()} epitopes")

    result = Clustering(n_cpus=8).fit(cdr3_input)
    metrics = result.metrics(epi_metrics).summary()

    ret = metrics[metrics['metrics']=='retention']['actual'].values[0]
    pur = metrics[metrics['metrics']=='purity']['actual'].values[0]
    p90 = metrics[metrics['metrics']=='purity_90']['actual'].values[0]
    con = metrics[metrics['metrics']=='consistency']['actual'].values[0]
    print(f"  Ret={ret:.4f} Pur={pur:.4f} P90={p90:.4f} Con={con:.4f}")

# Published results for comparison
print("\n--- Published results (ClusTCR) ---")
pub = {
    0: {'retention': 0.2517, 'purity': 0.5871, 'purity_90': 0.4089, 'consistency': 0.1300},
    1: {'retention': 0.2363, 'purity': 0.8581, 'purity_90': 0.7265, 'consistency': 0.3614},
    2: {'retention': 0.2623, 'purity': 0.9286, 'purity_90': 0.9038, 'consistency': 0.4796},
}
for q, vals in pub.items():
    print(f"  q={q}: Ret={vals['retention']:.4f} Pur={vals['purity']:.4f} P90={vals['purity_90']:.4f} Con={vals['consistency']:.4f}")
