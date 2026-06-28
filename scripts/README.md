# scripts/ — experiment & reproducibility entry points

These scripts drive the analyses reported in the manuscript. They are standalone
entry points (run with the project venv); none are imported by the `tcrconsensus`
package. Datasets under `data/` / `results/` are not shipped — regenerate them via
the data-build scripts below.

## Data preparation
| Script | Purpose |
|---|---|
| `prepare_data.py` | Filter VDJdb / McPAS-TCR for experiments |
| `prepare_benchmark.py` | Build benchmark per `Database.md` (VS=2, AIS>4.3, CDR3 len 6–23) |
| `build_paper_benchmark_v3.py` | Pooled-database benchmark, exact paper methodology (4,779 TRA/TRB pairs) |
| `reconstruct_subsets.py` | Rebuild the 6 Donor1 stress subsets with native Vβ/Jβ attached |

## Primary results — `p0` suite
`run_all_p0.sh` orchestrates the core paper analyses; each `p0_*.py` is runnable
standalone. `exp_shared.py` holds shared utilities.
| Script | Analysis |
|---|---|
| `p0_whole_dataset.py` | Whole-dataset evaluation + GIANA (primary result) |
| `p0_algorithm_ablation.py` | Co-association matrix × clustering-strategy ablation (Occam's-razor argument) |
| `p0_leave_one_out.py` | Leave-one-method-out ablation |
| `p0_cv_weights.py` | Cross-validated weight learning vs empirical priors |
| `p0_intersection_union.py` | Intersection / Union / Random consensus baselines |
| `p0_case_study.py` | Biological case study (GILGFVFTL, ELAGIGILTV) |
| `p0_stress_test_10x.py` | 10X Donor1 background stress test (STRESS_TEST.md) |
| `p0_improved_stress.py` | Multi-metric improved stress test |
| `p0_background_tcr.py` | Background-TCR robustness |

## Core benchmark experiments (4,779-pair dataset)
One canonical script per experiment (the `*_v2` family targets the 4,779-pair
core benchmark, matching the reference metric table).
| Script | Experiment |
|---|---|
| `run_bench_exp1_v2.py` | Cross-benchmark comparison |
| `run_bench_exp2_v2.py` | Background-robustness stress test |
| `run_bench_exp3_v2.py` | Component ablation |
| `run_bench_exp4_v2.py` | Adaptive-recommendation generalization (leave-one-epitope-out) |
| `run_bench_exp5_v2.py` | Biological case study |
| `run_full_comparison.py` | tcrconsensus vs 7 individual methods, full dataset |
| `run_vdjdb_validation.py` | Independent external validation (VDJdb, per-species splits) |

## Reproducibility
| Script | Purpose |
|---|---|
| `reproducibility.py` | Before/after method-improvement comparison (5 seeds) |
| `reproducibility_test.py` | Seed-stability test of the ARI improvement |
| `reproduce_clustcr_paper.py` | Reproduce clusTCR paper (Valkiers et al., 2021) benchmark |
| `metric_robustness.py` | Clustering-metric stability under code-version change |

## Stress test (Donor1 noise spike-in)
`stress_test.py` is the canonical runner; `stress_one*.py` are its per-cell
subprocess workers (invoked for isolation + hard timeouts).
| Script | Purpose |
|---|---|
| `stress_test.py` / `stress_one.py` | CDR3β-only stress test (STRESS_TEST.md) |
| `stress_test_rich.py` / `stress_one_rich.py` | V/J-enriched controlled counterpart |
| `stress_recovery.py` | F3 (permutation FDR) + Tier-2 recovery sweep |
| `stress_consensus_sweep.py` | Consensus-threshold sweep |
| `stress_summarize.py` | Results → table + figure |
| `stress_notes.py` | Results → `notes.html` report |
| `stress_test_final.py` | Epitope-level background variant |

## Figures
`gen_final_figures.py`, `fig2_stress_test.py`, `fig_subset1_vis.py`,
`fig_subset1_clusters.py`, `plot_reproducibility.py`, `merge_and_visualize.py`.
