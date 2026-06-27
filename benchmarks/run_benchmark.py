#!/usr/bin/env python3
"""Benchmark runner for tcrconsensus.

Usage:
    python run_benchmark.py [--methods hd_baseline] [--output benchmark_output]
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcrconsensus.io.parser import load_file, normalize
from tcrconsensus.io.writer import ensure_run_dir
from tcrconsensus.profiling.profiler import profile as compute_profile
from tcrconsensus.selection.selector import select_methods
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import compute_all_metrics
from tcrconsensus.config import load_config
import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_tcrs=200, n_epitopes=5, noise_frac=0.2, seed=42):
    """Generate synthetic TCR dataset with known epitope labels."""
    rng = np.random.RandomState(seed)

    # Amino acids (excluding C start and F/W end which we add)
    aas = "ACDEFGHIKLMNPQRSTVWY"

    epitopes = [f"EP_{i:03d}" for i in range(n_epitopes)]
    records = []

    for ep_idx, epitope in enumerate(epitopes):
        # Each epitope has a "seed" CDR3 pattern
        seed_len = rng.randint(12, 18)
        # Generate seed sequence
        seed = "C" + "".join(rng.choice(list(aas), seed_len - 2)) + rng.choice(["F"])

        n_signal = n_tcrs // n_epitopes
        for i in range(n_signal):
            # Mutate seed by 0-2 positions
            seq = list(seed)
            n_mutations = rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            for _ in range(n_mutations):
                pos = rng.randint(1, len(seq) - 1)
                seq[pos] = rng.choice(list(aas))
            records.append({
                "tcr_id": f"tcr_{len(records):06d}",
                "cdr3_beta": "".join(seq),
                "v_beta": f"TRBV{rng.randint(1, 30):02d}",
                "j_beta": f"TRBJ{rng.randint(1, 7):02d}-{rng.randint(1, 4)}",
                "epitope": epitope,
                "subject_id": f"S{rng.randint(1, 10):02d}",
                "count": rng.randint(1, 50),
            })

    # Add noise (random sequences, no epitope label)
    n_noise = int(len(records) * noise_frac / (1 - noise_frac))
    for i in range(n_noise):
        seq_len = rng.randint(8, 20)
        seq = "C" + "".join(rng.choice(list(aas), seq_len - 2)) + rng.choice(["F"])
        records.append({
            "tcr_id": f"tcr_{len(records):06d}",
            "cdr3_beta": seq,
            "v_beta": f"TRBV{rng.randint(1, 30):02d}",
            "j_beta": f"TRBJ{rng.randint(1, 7):02d}-{rng.randint(1, 4)}",
            "epitope": "NOISE",
            "subject_id": f"S{rng.randint(1, 10):02d}",
            "count": rng.randint(1, 5),
        })

    return pd.DataFrame(records)


def run_benchmark(methods, output_dir, config_path=None):
    """Run benchmark with specified methods."""
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset()
    data_path = output_dir / "synthetic_data.tsv"
    df.to_csv(data_path, sep="\t", index=False)
    print(f"  {len(df)} TCRs written to {data_path}")

    # Normalize
    df_norm = normalize(df)

    # True labels
    true_labels = df_norm["epitope"].values
    n_total = len(df_norm)

    # Profile
    prof = compute_profile(df_norm, config._raw)
    print(f"\nProfile:")
    print(f"  TCRs: {prof.n_tcrs}")
    print(f"  Noise score: {prof.background_noise_score:.3f}")
    print(f"  VJ completeness: {prof.vj_completeness:.2f}")
    print(f"  Repertoire type: {prof.repertoire_type.value}")

    # Run each consensus mode
    modes = ["conservative", "balanced", "coverage"]
    all_results = []

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")

        # Run clusterers
        clusterer_map = {"hd_baseline": HDBaselineClusterer(distance_threshold=1)}
        all_assignments = []
        method_info = []

        for mname in methods:
            if mname not in clusterer_map:
                print(f"  Skipping {mname} (not available)")
                continue
            clusterer = clusterer_map[mname]
            result = clusterer.safe_execute(df_norm, output_dir / "work", config._raw)
            all_assignments.extend(result.assignments)
            method_info.append({
                "method": mname,
                "status": result.status.value,
                "assignments": len(result.assignments),
                "runtime_s": result.runtime_seconds,
            })
            print(f"  {mname}: {len(result.assignments)} assignments in {result.runtime_seconds:.3f}s")

        if not all_assignments:
            print("  No assignments, skipping")
            continue

        # Compute weights
        weights = compute_method_weights(methods, "balanced", config._raw)

        # Consensus
        if mode == "conservative":
            clusters, edges = conservative_consensus(all_assignments, weights)
        elif mode == "coverage":
            clusters, edges = coverage_consensus(all_assignments, weights)
        else:
            clusters, edges = balanced_consensus(all_assignments, weights)

        # Refine
        clusters = refine(clusters, edges, config._raw)

        print(f"  Clusters: {len(clusters)}")

        # Evaluate
        tcr_ids = df_norm["tcr_id"].values
        label_map = {}
        for cluster in clusters:
            for mid in cluster.member_ids:
                label_map[mid] = cluster.cluster_id
        pred_labels = np.array([label_map.get(tid, "unclustered") for tid in tcr_ids])

        # Filter to TCRs with true labels (exclude NOISE)
        valid = true_labels != "NOISE"
        if valid.sum() > 0:
            # Encode labels for metrics
            from sklearn.preprocessing import LabelEncoder
            le_pred = LabelEncoder()
            le_true = LabelEncoder()
            pred_encoded = le_pred.fit_transform(pred_labels[valid])
            true_encoded = le_true.fit_transform(true_labels[valid])

            metrics = compute_all_metrics(pred_encoded, true_encoded, n_total)
            print(f"  Metrics:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

            all_results.append({
                "mode": mode,
                "n_clusters": len(clusters),
                **metrics,
            })

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "benchmark_results.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    print(f"\nResults saved to {results_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tcrconsensus")
    parser.add_argument("--methods", default="hd_baseline", help="Comma-separated methods")
    parser.add_argument("--output", default="benchmark_output", help="Output directory")
    parser.add_argument("--config", default=None, help="Config YAML path")
    args = parser.parse_args()

    methods = args.methods.split(",")
    run_benchmark(methods, args.output, args.config)
