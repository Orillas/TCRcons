#!/usr/bin/env python3
"""Prepare VDJdb and McPAS-TCR data for tcrconsensus experiments.

Filters:
  VDJdb: Gene=TRB, Score>=1, epitope with >=5 TCRs
  McPAS: has CDR3.beta.aa + has Epitope.peptide

Outputs:
  vdjdb_filtered.tsv  — normalized TCR data
  vdjdb_labels.tsv    — tcr_id, epitope columns
  mcpas_filtered.tsv  — normalized TCR data
  mcpas_labels.tsv    — tcr_id, epitope columns
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def prepare_vdjdb(input_path: str, output_dir: str):
    """Filter and normalize VDJdb data."""
    print("Loading VDJdb...")
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)

    # Filter: TRB only, Score >= 1
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0).astype(int)
    df = df[(df["Gene"] == "TRB") & (df["Score"] >= 1)].copy()
    print(f"  After TRB + Score>=1 filter: {len(df)} rows")

    # Extract subject_id from Meta JSON
    def extract_subject(meta_str):
        try:
            meta = json.loads(meta_str)
            return meta.get("subject.id", "") or meta.get("subject_id", "")
        except (json.JSONDecodeError, TypeError):
            return ""

    df["subject_id"] = df["Meta"].apply(extract_subject)

    # Deduplicate by CDR3+V+J (keep first occurrence per subject)
    df = df.drop_duplicates(subset=["CDR3", "V", "J"]).copy()
    print(f"  After dedup: {len(df)} rows")

    # Filter epitopes with >= 5 TCRs
    epi_counts = df["Epitope"].value_counts()
    valid_epis = set(epi_counts[epi_counts >= 5].index)
    df = df[df["Epitope"].isin(valid_epis)].copy()
    print(f"  After epitope>=5 filter: {len(df)} rows, {len(valid_epis)} epitopes")

    # Generate tcr_id
    df["tcr_id"] = ["vdj_" + str(i).zfill(6) for i in range(len(df))]

    # Build output data
    out_data = pd.DataFrame({
        "tcr_id": df["tcr_id"].values,
        "cdr3_beta": df["CDR3"].values,
        "v_beta": df["V"].values,
        "j_beta": df["J"].values,
        "epitope": df["Epitope"].values,
        "species": df["Species"].values,
        "mhc": df["MHC A"].values,
        "subject_id": df["subject_id"].values,
        "score": df["Score"].values,
    })

    # Build labels
    labels = out_data[["tcr_id", "epitope"]].copy()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "vdjdb_filtered.tsv"
    labels_path = out_dir / "vdjdb_labels.tsv"

    out_data.to_csv(data_path, sep="\t", index=False)
    labels.to_csv(labels_path, sep="\t", index=False)

    print(f"  Written: {data_path} ({len(out_data)} rows)")
    print(f"  Written: {labels_path}")

    # Stats
    epi_dist = out_data["epitope"].value_counts()
    print(f"  Epitope size: min={epi_dist.min()}, median={epi_dist.median()}, max={epi_dist.max()}")
    print(f"  Top 5 epitopes:")
    for e, c in epi_dist.head(5).items():
        print(f"    {e}: {c}")

    return out_data


def prepare_mcpas(input_path: str, output_dir: str):
    """Filter and normalize McPAS-TCR data."""
    print("\nLoading McPAS-TCR...")
    df = pd.read_csv(input_path, dtype=str, keep_default_na=True, na_values=["NA", ""])

    # Filter: has CDR3.beta.aa + has Epitope.peptide
    df = df.dropna(subset=["CDR3.beta.aa", "Epitope.peptide"]).copy()
    print(f"  After CDR3beta + Epitope filter: {len(df)} rows")

    # Deduplicate
    df = df.drop_duplicates(subset=["CDR3.beta.aa", "TRBV"]).copy()
    print(f"  After dedup: {len(df)} rows")

    # Filter epitopes with >= 5 TCRs
    epi_counts = df["Epitope.peptide"].value_counts()
    valid_epis = set(epi_counts[epi_counts >= 5].index)
    df = df[df["Epitope.peptide"].isin(valid_epis)].copy()
    print(f"  After epitope>=5 filter: {len(df)} rows, {len(valid_epis)} epitopes")

    # Generate tcr_id
    df["tcr_id"] = ["mcpas_" + str(i).zfill(6) for i in range(len(df))]

    # Build output - handle TRBV format (e.g., "TRBV13-2" -> keep as-is)
    v_beta = df["TRBV"].fillna("").values
    j_beta = df["TRBJ"].fillna("").values

    out_data = pd.DataFrame({
        "tcr_id": df["tcr_id"].values,
        "cdr3_beta": df["CDR3.beta.aa"].values,
        "v_beta": v_beta,
        "j_beta": j_beta,
        "epitope": df["Epitope.peptide"].values,
        "species": df["Species"].fillna("").values,
        "mhc": df["MHC"].fillna("").values,
        "subject_id": df["PubMed.ID"].fillna("").values,
        "pathology": df["Pathology"].fillna("").values,
    })

    labels = out_data[["tcr_id", "epitope"]].copy()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "mcpas_filtered.tsv"
    labels_path = out_dir / "mcpas_labels.tsv"

    out_data.to_csv(data_path, sep="\t", index=False)
    labels.to_csv(labels_path, sep="\t", index=False)

    print(f"  Written: {data_path} ({len(out_data)} rows)")
    print(f"  Written: {labels_path}")

    epi_dist = out_data["epitope"].value_counts()
    print(f"  Epitope size: min={epi_dist.min()}, median={epi_dist.median()}, max={epi_dist.max()}")
    print(f"  Top 5 epitopes:")
    for e, c in epi_dist.head(5).items():
        print(f"    {e}: {c}")

    return out_data


if __name__ == "__main__":
    data_dir = "/home/jilin/DeepTCR/Data"
    output_dir = "/home/jilin/DeepTCR/tcrconsensus/results/data"

    vdj = prepare_vdjdb(f"{data_dir}/VDJdb.tsv", output_dir)
    mcpas = prepare_mcpas(f"{data_dir}/McPAS-TCR.csv", output_dir)

    print(f"\n=== Summary ===")
    print(f"VDJdb filtered: {len(vdj)} TCRs, {vdj['epitope'].nunique()} epitopes")
    print(f"McPAS filtered: {len(mcpas)} TCRs, {mcpas['epitope'].nunique()} epitopes")
