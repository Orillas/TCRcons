# reproduce/ — reproducing the manuscript benchmark

Minimal, curated entry points to reproduce the headline result of the
manuscript: **tcrconsensus vs. the seven individual clustering methods on the
high-confidence paired αβ-TCR benchmark**. These two scripts are self-contained
and configurable; the full experiment/ablation suite that produced every figure
is preserved in the project history (see *Full experiment suite* below).

## Prerequisites

```bash
# the package + tcrdist3
pip install "tcrconsensus[tcrdist3]"

# DeepTCR (two-step workflow — recommended for reproducible pinned deps)
pip install --no-deps "DeepTCR @ git+https://github.com/sidhomj/DeepTCR.git@3930ca05a987c7cc621b4f2ecfd740e2d62799d8"
pip install -r requirements/deeptcr-pinned.txt

# external binaries (subprocess wrappers) must be on PATH: GLIPH2, GIANA, TCRMatch
```

Run from the repository root so the `results/` defaults resolve.

## Data source

The benchmark is built from a **pooled TCR-epitope database** combining
**IEDB + VDJdb + McPAS-TCR**, distributed as part of the
[i3-unit TCR Unsupervised Benchmark](https://github.com/s175111/i3-unit-TCR_Unsupervised_Benchmark)
(`Data/Database/database_pooled_human_2023_03_15.txt`). Download that file and
pass its path to the builder.

## Step 1 — build the benchmark

```bash
python reproduce/build_paper_benchmark_v3.py \
    --db /path/to/database_pooled_human_2023_03_15.txt \
    --out-dir results/paper_benchmark
```

Applies the paper methodology (verified by the docstring of the script):

1. pooled IEDB + VDJdb + McPAS-TCR
2. `Verified_score == 2` (both TRA & TRB verified)
3. `Antigen_identification_score > 4.3`
4. V/J genes present (TRAV/TRAJ/TRBV/TRBJ)
5. CDR3 length 6–23 aa (both chains)
6. ≥ 2 unique CDR3α/CDR3β pairs per epitope
7. dedup on V-CDR3-J + epitope + organism + PubMed + cell subset
8. remove specific degenerate pairs

Expected: **≈ 4,779 unique TRA/TRB pairs** (the script prints a side-by-side
comparison against the paper's counts). The CD8 filter is OFF by default,
matching the paper (≈94% natural CD8 bias).

Output: `results/paper_benchmark/paper_benchmark_v3.tsv` (+ a CDR3 list for
GIANA and a stats JSON).

## Step 2 — run the headline comparison

```bash
# point at the benchmark built in Step 1 (cd8 subset) and run all 7 methods + consensus
TCR_BENCHMARK=results/paper_benchmark/paper_benchmark_v3_cd8.tsv \
TCR_OUT_DIR=results/full_comparison \
TCR_FIG_DIR=results/figures \
python reproduce/run_full_comparison.py
```

Runs each of the seven methods once (`seed=42`) and the tcrconsensus consensus
(connected components + empirical weights + refinement), then computes
ARI / Purity / AMI / Sensitivity / Precision / F1 / Retention for each, prints a
summary table, and writes JSON + figures (bar / radar / heatmap / table) to
`TCR_FIG_DIR`.

All three paths are environment-configurable (`TCR_BENCHMARK`, `TCR_OUT_DIR`,
`TCR_FIG_DIR`); defaults are repo-relative.

## Expected result

tcrconsensus should lead on the headline metrics versus the best individual
method on this clean paired-αβ benchmark — see the manuscript and its results
table for the reference numbers. Method-level numbers vary slightly with backend
versions (notably DeepTCR's VAE and clusTCR's FAISS), as documented in the
paper's reproducibility section.

## Full experiment suite

The complete set of analyses (per-experiment benchmarks `run_bench_exp{1-5}_v2`,
ablations `p0_*`, the stress-test family, reproducibility/seed-stability, and
figure generators) lived under `scripts/` and is preserved in the project
history — recover with, e.g.:

```bash
git checkout <release-tag>^ -- scripts/        # inspect a historical copy
```

They were kept out of the public release to keep the repository focused on the
installable software; the two scripts above are the canonical path to reproduce
the central benchmark.
