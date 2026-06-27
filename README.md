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

### External Tools (optional)

| Tool | Install | Notes |
|------|---------|-------|
| clusTCR | `pip install clustcr` | Python package |
| tcrdist3 | `pip install tcrdist3` | Python package |
| GLIPH2 | Binary on PATH | Subprocess wrapper |

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

## License

MIT
