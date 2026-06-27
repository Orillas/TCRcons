import pandas as pd
df = pd.read_csv("results/benchmark_data/benchmark_core_4779.tsv", sep="\t", dtype=str)
print(f"Total rows: {len(df)}")
pairs = df.drop_duplicates(subset=["CDR3_alpha","CDR3_beta"])
print(f"Unique TRA/TRB pairs: {len(pairs)}")
print(f"Unique CDR3_beta: {df.CDR3_beta.nunique()}")
print(f"Unique CDR3_alpha: {df.CDR3_alpha.nunique()}")
combined = pd.concat([df.CDR3_alpha.dropna(), df.CDR3_beta.dropna()])
print(f"Unique CDR3 (combined alpha+beta): {combined.nunique()}")
dup = df.groupby("CDR3_beta").size()
multi = dup[dup > 1]
print(f"CDR3_beta appearing >1 time: {len(multi)} sequences, {multi.sum()} total rows")

# Show example of duplicate CDR3_beta with different tcr_ids
example_seqs = multi.head(3).index.tolist()
example_df = df[df.CDR3_beta.isin(example_seqs)].sort_values("CDR3_beta")
print("\nExample: same CDR3_beta, different tcr_ids:")
cols = ["CDR3_beta","CDR3_alpha","Epitope","tcr_id"]
print(example_df[cols].to_string(index=False))
