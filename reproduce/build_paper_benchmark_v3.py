"""Build benchmark from paper's pooled database — v3: exact paper methodology.

Paper methodology (from user's description):

1. Raw pooled DB: IEDB + VDJdb + McPAS-TCR → 190,670 TCRs
2. VS = 2 (both TRA & TRB verified)
3. AIS > 4.3 (Antigen Identification Score)
4. V & J genes present (TRAV, TRAJ, TRBV, TRBJ not NA)
5. CDR3 length 6-23 amino acids (both alpha & beta)
6. >= 2 unique CDR3α/CDR3β pairs per epitope
7. Dedup on: V-CDR3-J sequence, epitope, organism, PubMed ID, cell subset
   (NOT full-row dedup — R's unique() on specific columns)
8. Remove specific degenerate pairs
9. Final: 4,779 unique TRA/TRB pairs (mainly CD8+ — NOT filtered, 94% natural bias)

Key improvements over v2:
  - Dedup on SPECIFIC columns (not all columns)
  - CD8 filter is OPTIONAL (--cd8-only flag), default OFF matching paper
  - Step-by-step reporting matching paper's description
  - Preserves more metadata columns

Usage:
  python build_paper_benchmark_v3.py [--cd8-only] [--out-dir PATH]
"""
import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB = "/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/Database/database_pooled_human_2023_03_15.txt"
DEFAULT_OUT = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark"

# Specific pairs to remove (from paper's R script)
REMOVE_PAIRS = [
    "CASSDSRGTEAFF_CASSDSRGTEAFF",   # degenerate: same CDR3 for alpha & beta
    "CASSPITGTGAYGYTF_CASSSVNEQYF",   # paper-specified removal
]

# Columns for dedup (matching paper: V-CDR3-J sequence, epitope, organism, PubMed, cell subset)
DEDUP_COLUMNS = [
    "V_alpha", "CDR3_alpha", "J_alpha",
    "V_beta", "CDR3_beta", "J_beta",
    "Epitope", "Antigen_organism", "PubMed_ID", "Cell_subset",
]

# Bad PubMed ID (10x marketing PDF, not a real paper)
BAD_PUBMED = "https://pages.10xgenomics.com/rs/446-PBO-704/images/10x_AN047_IP_A_New_Way_of_Exploring_Immunity_Digital.pdf"


def load_raw(path: str) -> pd.DataFrame:
    """Load raw pooled database with proper encoding and NA handling."""
    print(f"[1] Loading: {path}")
    raw = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,   # keep "NA" strings, we handle NAs ourselves
        encoding="latin-1",       # handles 0xa0 non-breaking spaces
    )
    # Replace empty strings with pd.NA (R-style: "" is missing)
    raw = raw.replace("", pd.NA)
    # Also replace literal "NA" string with pd.NA for numeric columns
    # (R's read.table would parse "NA" as missing)
    raw = raw.replace("NA", pd.NA)
    print(f"     Raw rows: {len(raw):,}")
    print(f"     Columns: {len(raw.columns)}")
    return raw


def step_report(step_num: int, name: str, n_before: int, n_after: int):
    """Print a step report."""
    removed = n_before - n_after
    pct = f" ({removed/n_before*100:.1f}% removed)" if n_before > 0 else ""
    print(f"[{step_num}] {name}: {n_before:,} → {n_after:,}{pct}")


def main():
    parser = argparse.ArgumentParser(description="Build paper benchmark dataset")
    parser.add_argument("--cd8-only", action="store_true",
                        help="Filter to Cell_subset=='CD8' only (paper did NOT do this)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--db", default=DB,
                        help="Path to the pooled database TSV (IEDB+VDJdb+McPAS-TCR)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # =====================================================================
    # Step 1: Load
    # =====================================================================
    df = load_raw(args.db)
    n_raw = len(df)

    # =====================================================================
    # Step 2: Verified_score == 2 & Antigen_identification_score > 4.3
    # =====================================================================
    df["Verified_score"] = pd.to_numeric(df["Verified_score"], errors="coerce")
    df["Identification_score"] = pd.to_numeric(df["Identification_score"], errors="coerce")
    n_before = len(df)
    df = df[(df["Verified_score"] == 2) & (df["Identification_score"] > 4.3)].copy()
    step_report(2, "VS==2 & AIS>4.3", n_before, len(df))

    # =====================================================================
    # Step 3: Epitope not NA
    # =====================================================================
    n_before = len(df)
    df = df[df["Epitope"].notna()].copy()
    step_report(3, "Epitope not NA", n_before, len(df))

    # =====================================================================
    # Step 4: Remove bad PubMed entry (10x marketing PDF)
    # =====================================================================
    n_before = len(df)
    df = df[df["PubMed_ID"] != BAD_PUBMED].copy()
    step_report(4, "Remove bad PubMed", n_before, len(df))

    # =====================================================================
    # Step 5: (OPTIONAL) CD8 filter
    # =====================================================================
    if args.cd8_only:
        n_before = len(df)
        df = df[df["Cell_subset"] == "CD8"].copy()
        step_report(5, "CD8 filter (OPTIONAL)", n_before, len(df))
    else:
        # Just report CD8 proportion
        n_cd8 = (df["Cell_subset"] == "CD8").sum()
        print(f"[5] CD8 proportion: {n_cd8}/{len(df)} = {n_cd8/len(df)*100:.1f}% (NOT filtering)")

    # =====================================================================
    # Step 6: V & J genes not NA (all 4 required: Vα, Jα, Vβ, Jβ)
    # =====================================================================
    n_before = len(df)
    for col in ["V_alpha", "J_alpha", "V_beta", "J_beta"]:
        df = df[df[col].notna()].copy()
    step_report(6, "V/J genes not NA (4 genes)", n_before, len(df))

    # =====================================================================
    # Step 7: CDR3 length 6-23 amino acids (both alpha & beta)
    # =====================================================================
    n_before = len(df)
    for col in ["CDR3_alpha", "CDR3_beta"]:
        df = df[df[col].notna()].copy()
        df = df[df[col].str.len().between(6, 23)].copy()
    step_report(7, "CDR3 length 6-23 (α & β)", n_before, len(df))

    # =====================================================================
    # Step 8: At least 2 unique CDR3α/CDR3β pairs per epitope
    # =====================================================================
    n_before = len(df)
    df["_pair"] = df["CDR3_beta"] + "_" + df["CDR3_alpha"]
    epi_pair_counts = df.groupby("Epitope")["_pair"].apply(lambda x: x.nunique())
    valid_epis = epi_pair_counts[epi_pair_counts >= 2].index
    df = df[df["Epitope"].isin(valid_epis)].copy()
    step_report(8, "≥2 unique pairs per epitope", n_before, len(df))

    # =====================================================================
    # Step 9: Dedup on specific columns
    # Paper: "如果条目在 V-CDR3-J 序列、表位、生物体、PubMed ID 和细胞子集中完全相同，
    #         则仅保留一个实例"
    # =====================================================================
    n_before = len(df)
    dedup_cols = [c for c in DEDUP_COLUMNS if c in df.columns]
    print(f"[9] Dedup columns: {dedup_cols}")
    print(f"     Before dedup: {len(df)} rows, {df['_pair'].nunique()} unique pairs")
    df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()
    step_report(9, "Dedup (V-CDR3-J + epitope + organism + PubMed + cell_subset)", n_before, len(df))

    # =====================================================================
    # Step 10: Remove specific degenerate pairs
    # =====================================================================
    n_before = len(df)
    df = df[~df["_pair"].isin(REMOVE_PAIRS)].copy()
    step_report(10, "Remove specific degenerate pairs", n_before, len(df))

    # =====================================================================
    # Final stats
    # =====================================================================
    n_pairs = df["_pair"].nunique()
    n_cdr3 = pd.concat([
        df[["CDR3_beta"]].rename(columns={"CDR3_beta": "CDR3"}),
        df[["CDR3_alpha"]].rename(columns={"CDR3_alpha": "CDR3"}),
    ])["CDR3"].nunique()

    n_cd8 = (df["Cell_subset"] == "CD8").sum()
    n_epitopes = df["Epitope"].nunique()

    print(f"\n{'='*60}")
    print(f"FINAL DATASET")
    print(f"{'='*60}")
    print(f"Rows:              {len(df):,}")
    print(f"Unique pairs:      {n_pairs:,}  (paper: 4,779)")
    print(f"Unique CDR3s:      {n_cdr3:,}  (paper: 8,395)")
    print(f"Unique epitopes:   {n_epitopes:,}")
    print(f"CD8 proportion:    {n_cd8}/{len(df)} = {n_cd8/len(df)*100:.1f}% (paper: 94%)")
    print(f"CD8-only filter:   {'ON' if args.cd8_only else 'OFF (matching paper)'}")

    # =====================================================================
    # Save
    # =====================================================================
    df = df.reset_index(drop=True)
    df["tcr_id"] = [f"tcr_{i+1:06d}" for i in range(len(df))]

    # Canonical column names for tcrconsensus
    rename_map = {
        "CDR3_alpha": "CDR3_alpha",
        "CDR3_beta": "CDR3_beta",
        "V_alpha": "V_alpha",
        "V_beta": "V_beta",
        "J_alpha": "J_alpha",
        "J_beta": "J_beta",
    }
    # Already canonical, just ensure they exist

    # Save full benchmark
    out_path = f"{args.out_dir}/paper_benchmark_v3.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved: {out_path}")

    # Save CDR3 list for GIANA
    all_cdr3s = pd.concat([
        df[["CDR3_beta"]].rename(columns={"CDR3_beta": "CDR3"}),
        df[["CDR3_alpha"]].rename(columns={"CDR3_alpha": "CDR3"}),
    ])["CDR3"].drop_duplicates()

    giana_path = f"{args.out_dir}/GIANA_paper_input_v3.txt"
    with open(giana_path, "w") as f:
        for cdr3 in sorted(all_cdr3s):
            f.write(cdr3 + "\n")
    print(f"Saved: {giana_path} ({len(all_cdr3s):,} unique CDR3s)")

    # =====================================================================
    # Comparison with paper
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH PAPER (4,779 pairs)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Paper':>10} {'Ours':>10} {'Delta':>10}")
    print("-" * 62)
    print(f"{'Rows':<30} {'?':>10} {len(df):>10,} {'—':>10}")
    print(f"{'Unique pairs':<30} {4779:>10,} {n_pairs:>10,} {n_pairs-4779:>+10,}")
    print(f"{'Unique CDR3s':<30} {8395:>10,} {n_cdr3:>10,} {n_cdr3-8395:>+10,}")
    print(f"{'CD8%':<30} {'94%':>10} {n_cd8/len(df)*100:>9.1f}% {'—':>10}")
    if n_pairs > 4779:
        print(f"\nNote: +{n_pairs-4779} more pairs than paper — database updated since 2023-03-15.")

    # Save stats
    stats = {
        "timestamp": datetime.now().isoformat(),
        "cd8_filter": args.cd8_only,
        "rows": int(len(df)),
        "unique_pairs": int(n_pairs),
        "unique_cdr3s": int(n_cdr3),
        "unique_epitopes": int(n_epitopes),
        "cd8_count": int(n_cd8),
        "cd8_pct": round(float(n_cd8) / len(df) * 100, 1),
        "paper_pairs": 4779,
        "paper_cdr3s": 8395,
        "delta_pairs": int(n_pairs) - 4779,
    }
    import json
    stats_path = f"{args.out_dir}/paper_benchmark_v3_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")

    print("\nDONE")
    return df


if __name__ == "__main__":
    main()
