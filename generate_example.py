"""Generate synthetic example TCR dataset for tcrconsensus.

Produces:
  - examples/synthetic_tcrs.tsv   (TCR table, ~200 CDR3β sequences, 5 epitopes)
  - examples/synthetic_labels.tsv  (epitope labels)
"""

import os
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

AA = list("ACDEFGHIKLMNPQRSTVWY")
V_GENES = [
    "TRBV2", "TRBV3-1", "TRBV4-1", "TRBV5-1", "TRBV6-1",
    "TRBV7-2", "TRBV9", "TRBV10-1", "TRBV11-2", "TRBV12-3",
    "TRBV13", "TRBV14", "TRBV15", "TRBV18", "TRBV19",
    "TRBV20-1", "TRBV24-1", "TRBV25-1", "TRBV27", "TRBV28",
    "TRBV29-1", "TRBV30",
]
J_GENES = ["TRBJ1-1", "TRBJ1-2", "TRBJ1-3", "TRBJ1-4", "TRBJ1-5",
           "TRBJ2-1", "TRBJ2-2", "TRBJ2-3", "TRBJ2-5", "TRBJ2-7"]
EPITOPES = ["GILGFVFTL", "NLVPMVATV", "ELAGIGILTV", "LLFGYPVYV", "TPQDLNTML"]

OUT_DIR = "examples"


def random_cdr3(length: int = 14) -> str:
    """Generate a random CDR3β sequence starting with C."""
    return "C" + "".join(random.choices(AA, k=length - 1))


def mutate_step(seq: str, n_mutations: int = 1) -> str:
    """Introduce n_mutations random point mutations."""
    s = list(seq)
    for _ in range(n_mutations):
        pos = random.randint(1, len(s) - 2)  # avoid C/F/Y ends
        s[pos] = random.choice([a for a in AA if a != s[pos]])
    return "".join(s)


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []

    for epi_idx, epitope in enumerate(EPITOPES):
        # Each epitope has 2-5 seed TCRs that form a "specificity group"
        n_seeds = random.randint(2, 5)
        seed_cdr3s = [random_cdr3(random.randint(12, 16)) for _ in range(n_seeds)]

        # Each seed generates 6-12 mutated variants (same specificity)
        for seed in seed_cdr3s:
            n_variants = random.randint(6, 12)
            v_gene = random.choice(V_GENES)
            j_gene = random.choice(J_GENES)

            # The seed itself
            rows.append({
                "tcr_id": f"tcr_{epi_idx}_{len(rows):04d}",
                "cdr3_beta": seed,
                "v_beta": v_gene,
                "j_beta": j_gene,
                "chain_mode": "beta_only",
                "count": random.randint(1, 20),
                "subject_id": f"subj_{random.randint(1, 5):02d}",
            })

            for _ in range(n_variants):
                variant = mutate_step(seed, n_mutations=random.randint(1, 3))
                rows.append({
                    "tcr_id": f"tcr_{epi_idx}_{len(rows):04d}",
                    "cdr3_beta": variant,
                    "v_beta": v_gene,
                    "j_beta": j_gene,
                    "chain_mode": "beta_only",
                    "count": random.randint(1, 15),
                    "subject_id": f"subj_{random.randint(1, 5):02d}",
                })

    # Add some noise sequences (no clear epitope)
    for i in range(30):
        rows.append({
            "tcr_id": f"noise_{i:04d}",
            "cdr3_beta": random_cdr3(random.randint(10, 18)),
            "v_beta": random.choice(V_GENES),
            "j_beta": random.choice(J_GENES),
            "chain_mode": "beta_only",
            "count": random.randint(1, 5),
            "subject_id": f"subj_{random.randint(1, 5):02d}",
        })

    tcr_df = pd.DataFrame(rows).drop_duplicates(subset=["cdr3_beta"]).reset_index(drop=True)

    # Create labels table
    labels = []
    for _, row in tcr_df.iterrows():
        if "noise_" in str(row["tcr_id"]):
            labels.append({"tcr_id": row["tcr_id"], "epitope": "noise"})
        else:
            epi_idx = int(row["tcr_id"].split("_")[1])
            labels.append({"tcr_id": row["tcr_id"], "epitope": EPITOPES[epi_idx]})

    labels_df = pd.DataFrame(labels)

    tcr_path = os.path.join(OUT_DIR, "synthetic_tcrs.tsv")
    labels_path = os.path.join(OUT_DIR, "synthetic_labels.tsv")

    tcr_df.to_csv(tcr_path, sep="\t", index=False)
    labels_df.to_csv(labels_path, sep="\t", index=False)

    print(f"Generated {len(tcr_df)} TCRs, {len(labels_df)} labels")
    print(f"  Epitopes: {labels_df['epitope'].value_counts().to_dict()}")
    print(f"  TCR file: {tcr_path}")
    print(f"  Labels file: {labels_path}")
    return tcr_df, labels_df


if __name__ == "__main__":
    generate()
