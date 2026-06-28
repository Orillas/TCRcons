# TCR Consensus Clustering

**Scenario-adaptive TCR specificity consensus clustering framework.**

Combines multiple TCR clustering methods (clusTCR, GLIPH2, tcrdist3, Hamming distance baseline) via weighted co-association consensus. Automatically profiles input data, selects optimal method combinations, and produces refined consensus clusters with confidence scores.

## Installation

```bash
# Core (no external clusterers)
pip install .

# With clusTCR + tcrdist3
pip install ".[clusterers]"

# Development
pip install ".[dev]"
```

### Requirements

- Python ≥ 3.10
- pandas, numpy, click, pyyaml, networkx, scipy, scikit-learn, matplotlib

### Clustering Methods

tcrconsensus aggregates seven TCR clustering methods. `hd_baseline` is built in;
the rest are optional backends — install the ones you intend to use.

| Method | Backend | Install |
|--------|---------|---------|
| Hamming baseline | built-in | — |
| clusTCR | Python package | `pip install clustcr` |
| tcrdist3 | Python package | `pip install tcrdist3` |
| DeepTCR | Python package (TensorFlow; optional GPU) | `pip install DeepTCR` |
| GLIPH2 | external binary (subprocess) | on `PATH` |
| GIANA | external binary (subprocess) | on `PATH` |
| TCRMatch | external binary (subprocess) | on `PATH` |

## Quick Start

### Python API

```python
from tcrconsensus import TCRConsensus

# Auto mode: profile → select methods → cluster → consensus → refine
model = TCRConsensus(objective="balanced")
result = model.fit_predict("input.tsv")

print(f"Clusters: {len(result.clusters)}")
print(f"Recommendation: {result.recommendation.recommended_mode}")
for c in result.clusters:
    print(f"  {c.cluster_id}: {len(c.member_ids)} members, confidence={c.cluster_confidence:.2f}")
```

### CLI

```bash
# Profile dataset
tcrconsensus profile input.tsv

# Run full pipeline
tcrconsensus run input.tsv --methods hd_baseline,clustcr --mode balanced -o output/

# Auto mode (profile → select → cluster)
tcrconsensus auto input.tsv --objective high_purity -o output/

# Benchmark evaluation
tcrconsensus benchmark input.tsv -o bench_output/
```

## Input Formats

Auto-detected from file headers:

| Format | Key Columns |
|--------|-------------|
| AIRR TSV | `sequence_id`, `cdr3`, `v_call`, `j_call` |
| VDJdb | `CDR3`, `V`, `J`, `Epitope`, `MHC` |
| Custom CSV/TSV | `cdr3_beta`, `v_beta`, `j_beta`, `epitope` |

Custom column mapping supported via config.

## Configuration

Layered YAML config: `default.yaml` → preset → user override.

```yaml
# user_config.yaml
consensus:
  balanced:
    threshold: 0.4
    algorithm: leiden
    resolution: 1.2
refinement:
  filter:
    min_member_confidence: 0.15
```

```bash
tcrconsensus run input.tsv --config user_config.yaml
```

### Built-in Presets

| Preset | Effect |
|--------|--------|
| `high_purity` | Higher consensus threshold, conservative merging |
| `noise_robust` | Noise-aware weighting, conservative mode |
| `fast_screening` | Max 2 methods, faster runtime |

## Objectives

| Objective | Strategy |
|-----------|----------|
| `balanced` | Equal weight purity/sensitivity/noise/speed |
| `high_purity` | Conservative consensus, high-purity methods |
| `high_recall` | Coverage mode, sensitivity-focused methods |
| `noise_robust` | Noise-aware weights, robust methods |
| `fast_screening` | Minimal methods, fast execution |

## Consensus Modes

| Mode | Algorithm | Use Case |
|------|-----------|----------|
| **conservative** | Connected components with k-method threshold | High confidence, fewer but reliable clusters |
| **balanced** | Leiden/Louvain community detection | Trade-off between precision and recall |
| **coverage** | Union of all method links | Maximum recall, comprehensive clusters |

## Pipeline Architecture

```
Input → Normalize → Profile → Select Methods → Run Clusterers
                                                    ↓
Report ← Refine ← Consensus ← Weighted Co-association ←┘
```

1. **IO** — Parse AIRR/VDJdb/custom, normalize to canonical schema
2. **Profiling** — Compute noise, VJ completeness, repertoire type
3. **Selection** — Rule-based method selection from profile + objective
4. **Clustering** — Run each method, collect assignments
5. **Consensus** — Weighted pairwise co-association → graph clustering
6. **Refinement** — Split/merge/filter clusters, label core vs peripheral
7. **Evaluation** — Purity, sensitivity, F1, ARI, NMI, retention
8. **Reporting** — JSON + Markdown + matplotlib figures

## Output Structure

```
output/
├── run_20260603_120000/
│   ├── input/
│   ├── normalized/tcr_table.tsv
│   ├── profile/profile.json
│   ├── plan/run_plan.json
│   ├── methods/
│   │   ├── hd_baseline/
│   │   └── clustcr/
│   ├── consensus/
│   │   ├── pairwise_consensus_scores.tsv
│   │   ├── clusters.tsv
│   │   └── cluster_members.tsv
│   ├── refinement/
│   ├── evaluation/
│   ├── reports/
│   │   ├── report.json
│   │   ├── report.md
│   │   ├── method_runtime.png
│   │   └── metrics_summary.png
│   └── artifact_manifest.json
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Retention | Fraction of input TCRs in any cluster |
| Purity | Fraction of correctly assigned per cluster |
| Sensitivity | Mean same-epitope co-clustering rate |
| F1 | Harmonic mean of purity and sensitivity |
| ARI | Adjusted Rand Index |
| NMI | Normalized Mutual Information |

## Development

```bash
pip install ".[dev]"
pytest tests/ -v
pytest tests/ --cov=tcrconsensus
```

## Reproducibility & data

This repository ships the software, the test suite, a minimal `examples/`
dataset, and the default configuration. The full benchmark datasets and the
experiment scripts that produced the manuscript figures are maintained
separately — see the accompanying publication and its data-availability
statement for benchmark acquisition. A self-contained example ships in the repo:

```bash
tcrconsensus run examples/synthetic_tcrs.tsv --mode balanced -o out/
```

## Citation

If you use tcrconsensus, please cite:

```bibtex
@software{tcrconsensus,
  title  = {tcrconsensus: Scenario-adaptive TCR specificity consensus clustering},
  author = {{TCR-Consensus Team}},
  year   = {2026},
  url    = {https://github.com/Orillas/TCRcons}
}
```

A versioned DOI (Zenodo) and the full author list will be added here upon
deposit. See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 TCR-Consensus Team.
