# TCR Consensus Clustering

**Scenario-adaptive TCR specificity consensus clustering framework.**

Combines seven TCR clustering methods (clusTCR, GLIPH2, tcrdist3, GIANA, TCRMatch, DeepTCR, Hamming baseline) via weighted co-association consensus, plus a built-in Levenshtein edit-distance baseline. Automatically profiles input data, selects optimal method combinations, and produces refined consensus clusters with confidence scores.

## Installation

```bash
# Core (built-in: hd_baseline + levenshtein)
pip install .

# Add the two cleanly pip-installable backends (tcrdist3, DeepTCR)
pip install ".[clusterers]"

# Or install a single backend
pip install ".[tcrdist3]"   # or:  pip install ".[deeptcr]"

# Development
pip install ".[dev]"
```

> A single `pip install ".[clusterers]"` covers four of the eight methods
> (`hd_baseline`, `levenshtein`, `tcrdist3`, `DeepTCR`). `clusTCR`, `GLIPH2`,
> `GIANA`, and `TCRMatch` cannot be pip-installed — see
> [Manual backend setup](#manual-backend-setup).

### Requirements

- Python ≥ 3.10
- pandas, numpy, click, pyyaml, networkx, scipy, scikit-learn, matplotlib

### Clustering Methods

tcrconsensus aggregates seven TCR clustering methods, plus a built-in edit-distance
baseline. `hd_baseline` and `levenshtein` ship with the package (no install). Of the
remaining six, only **tcrdist3** and **DeepTCR** install cleanly via pip —
`pip install ".[clusterers]"` brings in both. The other four (`clusTCR`, `GLIPH2`,
`GIANA`, `TCRMatch`) cannot be pip-installed and must be set up manually
(see [Manual backend setup](#manual-backend-setup)).

| Method | Type | Install source | Wrapper resolution |
|--------|------|----------------|--------------------|
| Hamming baseline | built-in (pure Python) | — | always available |
| Levenshtein baseline | built-in (pure Python) | — | always available |
| tcrdist3 | Python pkg (needs `parasail`, C) | `pip install ".[tcrdist3]"` | `import tcrdist3` |
| DeepTCR | Python pkg (TensorFlow; GPU optional) | `pip install ".[deeptcr]"` | `import DeepTCR` |
| clusTCR | Python pkg, **not on PyPI** | clone + install (manual) | `import clustcr` |
| GLIPH2 | `irtools` binary + v2.0 ref DB | upstream binary (manual) | `TCR_GLIPH2_LIB` → `PATH` |
| GIANA | standalone `GIANA4.1.py` script | `github.com/s175573/GIANA` | `TCR_GIANA_SCRIPT` → `PATH` |
| TCRMatch | C++ binary | upstream binary (manual) | `TCR_TCRMATCH_BIN` → `PATH` |

For every external method the wrapper resolves its binary/script as:
constructor argument → `TCR_*` environment variable → `PATH` (`shutil.which`),
and raises a clear, actionable error if none is found.

> **DeepTCR build caveat:** DeepTCR ships as an sdist whose `setup.py` calls
> `nvidia-smi` during install. On a GPU-less host `pip install ".[deeptcr]"` can
> therefore fail at build time — install on a CUDA machine, or pre-install
> TensorFlow and run `pip install DeepTCR --no-deps`.

#### Manual backend setup

**clusTCR** is not published to PyPI, and its `setup.py` pins `scipy==1.8`, which
conflicts with tcrconsensus's `scipy>=1.9`. Install from source **without**
re-pinning scipy:

```bash
pip install --no-deps "clustcr @ git+https://github.com/svalkiers/clusTCR.git"
```

**GLIPH2 / TCRMatch** are compiled binaries (GLIPH2: Huang *et al.*, Nat.
Biotechnol. 2020; TCRMatch: Li *et al.*). Build or obtain the upstream binary,
then point the wrapper at it via the env vars above. **GIANA** is a Python
script — clone `github.com/s175573/GIANA` and set `TCR_GIANA_SCRIPT` to
`GIANA4.1.py`. TCRMatch additionally needs an IEDB database
(`TCR_TCRMATCH_IEDB`); if unset it falls back to self-comparison.

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
