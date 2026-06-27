#!/usr/bin/env python3
"""Prepare benchmark dataset from pooled database.

Applies filters per Database.md:
  1. VS=2 (both chains verified), AIS>4.3
  2. CDR3 length 6-23 aa
  3. Epitope >= 2 unique TRA/TRB pairs
  4. Dedup on V-CDR3-J + Epitope + Organism + PubMed + Cell_subset

Also prepares 10X noise subsets with signal labels.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCH_DIR = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data")
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/benchmark_data")


def prepare_main_dataset():
    """Prepare high-confidence benchmark dataset."""
    logger.info("Loading pooled database...")
    raw = pd.read_csv(
        BENCH_DIR / "Database" / "database_pooled_human_2023_03_15.txt",
        sep="\t", dtype=str, encoding="latin-1"
    )
    logger.info(f"Raw: {len(raw)} rows")

    # 1. VS=2, AIS>4.3
    raw["VS"] = pd.to_numeric(raw["Verified_score"], errors="coerce")
    raw["AIS"] = pd.to_numeric(raw["Identification_score"], errors="coerce")
    df = raw[(raw["VS"] == 2) & (raw["AIS"] > 4.3)].copy()
    logger.info(f"After VS=2 + AIS>4.3: {len(df)} rows")

    # 2. CDR3 length 6-23
    df["cdr3b_len"] = df["CDR3_beta"].str.len().astype(int)
    df = df[(df["cdr3b_len"] >= 6) & (df["cdr3b_len"] <= 23)]
    df["cdr3a_len"] = df["CDR3_alpha"].str.len()
    df["cdr3a_len"] = pd.to_numeric(df["cdr3a_len"], errors="coerce")
    df = df[(df["cdr3a_len"] >= 6) & (df["cdr3a_len"] <= 23)]
    logger.info(f"After CDR3 length filter: {len(df)} rows")

    # 3. Require V and J gene info
    df = df[(df["V_beta"] != "") & (df["J_beta"] != "")].copy()
    logger.info(f"After V/J filter: {len(df)} rows")

    # 4. Epitope >= 2 unique pairs
    epi_counts = df.groupby("Epitope").size()
    valid_epis = epi_counts[epi_counts >= 2].index
    df = df[df["Epitope"].isin(valid_epis)]
    logger.info(f"After epitope>=2: {len(df)} rows, {df['Epitope'].nunique()} epitopes")

    # 5. Dedup
    dedup_cols = ["V_alpha", "J_alpha", "CDR3_alpha", "V_beta", "J_beta", "CDR3_beta",
                  "Epitope", "Antigen_organism", "PubMed_ID", "Cell_subset"]
    df = df.drop_duplicates(subset=dedup_cols)
    logger.info(f"After dedup: {len(df)} rows, {df['Epitope'].nunique()} epitopes")

    # Generate tcr_id
    df["tcr_id"] = ["tcr_" + str(i).zfill(6) for i in range(len(df))]

    # Save main benchmark
    df.to_csv(OUT_DIR / "benchmark_main.tsv", sep="\t", index=False)

    # Save labels file (tcr_id -> epitope mapping)
    labels = df[["tcr_id", "Epitope"]].rename(columns={"Epitope": "epitope"})
    labels.to_csv(OUT_DIR / "benchmark_labels.tsv", sep="\t", index=False)

    # Save per-epitope stats
    epi_stats = df.groupby("Epitope").agg(
        n_tcrs=("tcr_id", "count"),
        n_organisms=("Antigen_organism", "nunique"),
        top_organism=("Antigen_organism", lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else "unknown"),
    ).sort_values("n_tcrs", ascending=False)
    epi_stats.to_csv(OUT_DIR / "epitope_stats.tsv", sep="\t")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK DATASET SUMMARY")
    print("=" * 60)
    print(f"Total TCRs: {len(df)}")
    print(f"Unique epitopes: {df['Epitope'].nunique()}")
    print(f"Unique organisms: {df['Antigen_organism'].nunique()}")
    print(f"Top 15 epitopes:")
    for epi, cnt in df["Epitope"].value_counts().head(15).items():
        org = df[df["Epitope"] == epi]["Antigen_organism"].value_counts().index[0]
        print(f"  {epi}: {cnt} ({org})")

    return df


def prepare_10x_subsets():
    """Prepare 10X noise subsets with signal labels."""
    logger.info("\nPreparing 10X subsets...")

    # Load signal labels from DeepTCR TrainData
    train = pd.read_csv(
        BENCH_DIR / "10X" / "Donor1" / "input" / "DeepTCR" / "TrainData" / "TrainData_DeepTCR.tsv",
        sep="\t", dtype=str
    )
    signal_cdr3_to_epi = dict(zip(train["CDR3_beta"], train["Epitope"]))
    logger.info(f"Signal CDR3s: {len(signal_cdr3_to_epi)}, epitopes: {train['Epitope'].nunique()}")

    all_subsets = []
    for i in range(1, 7):
        sub = pd.read_csv(BENCH_DIR / "10X" / "Donor1" / "subsets" / f"subset_{i}.txt",
                          sep="\t", dtype=str)
        sub["subset"] = i
        sub["is_signal"] = sub["cdr3"].isin(signal_cdr3_to_epi)
        sub["epitope"] = sub["cdr3"].map(signal_cdr3_to_epi).fillna("BACKGROUND")
        sub["tcr_id"] = [f"10x_s{i}_{j:05d}" for j in range(len(sub))]

        n_sig = sub["is_signal"].sum()
        n_bg = len(sub) - n_sig
        pct_bg = n_bg / len(sub) * 100

        logger.info(f"  subset_{i}: {len(sub)} total, {n_sig} signal, {n_bg} bg ({pct_bg:.1f}% noise)")

        sub.to_csv(OUT_DIR / f"10x_subset_{i}.tsv", sep="\t", index=False)
        all_subsets.append(sub)

    # Summary
    print("\n" + "=" * 60)
    print("10X SUBSETS SUMMARY")
    print("=" * 60)
    for sub in all_subsets:
        i = sub["subset"].iloc[0]
        n_sig = sub["is_signal"].sum()
        print(f"  subset_{i}: {len(sub)} total, {n_sig} signal, {len(sub)-n_sig} bg")

    return all_subsets


def prepare_per_epitope_datasets(df):
    """Save individual epitope datasets for case study."""
    epi_dir = OUT_DIR / "per_epitope"
    epi_dir.mkdir(exist_ok=True)

    # Top epitopes with >= 50 TCRs
    epi_counts = df["Epitope"].value_counts()
    top_epis = epi_counts[epi_counts >= 50].index.tolist()

    for epi in top_epis:
        epi_df = df[df["Epitope"] == epi].copy()
        epi_df.to_csv(epi_dir / f"{epi}.tsv", sep="\t", index=False)

    logger.info(f"\nSaved {len(top_epis)} per-epitope datasets (>= 50 TCRs)")
    print(f"\nPer-epitope datasets: {len(top_epis)}")
    for epi in top_epis:
        print(f"  {epi}: {epi_counts[epi]} TCRs")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_main_dataset()
    prepare_10x_subsets()
    prepare_per_epitope_datasets(df)
    print(f"\nAll data saved to: {OUT_DIR}")
