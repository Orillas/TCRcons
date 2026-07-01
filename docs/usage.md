# Usage Guide

## Installation

```bash
# Core package (includes hd_baseline, levenshtein)
pip install .

# tcrdist3
pip install ".[tcrdist3]"

# DeepTCR (two-step workflow — recommended)
pip install --no-deps "DeepTCR @ git+https://github.com/sidhomj/DeepTCR.git@3930ca05a987c7cc621b4f2ecfd740e2d62799d8"
pip install -r requirements/deeptcr-pinned.txt

# Development dependencies
pip install ".[dev]"
```

## Quick Start: Python API

```python
from tcrconsensus import TCRConsensus

# Simplest usage
model = TCRConsensus()
result = model.fit_predict("my_tcr_data.tsv")

# Inspect results
print(f"Clusters found: {len(result.clusters)}")
for cluster in result.clusters:
    print(f"  {cluster.cluster_id}: {len(cluster.member_ids)} TCRs "
          f"(core={len(cluster.core_member_ids)}, "
          f"confidence={cluster.cluster_confidence:.2f})")
```

## Input Data Preparation

### AIRR Format

Standard AIRR TSV with columns: `sequence_id`, `cdr3`, `v_call`, `j_call`, `clone_count`, `clone_frequency`.

### VDJdb Format

VDJdb export with columns: `CDR3`, `V`, `J`, `Epitope`, `MHC`, `Species`.

### Custom Format

Any TSV/CSV with recognizable column names:

| Accepted Names | Maps To |
|---------------|---------|
| `cdr3`, `cdr3_beta` | CDR3β sequence |
| `cdr3_alpha` | CDR3α sequence |
| `v_gene`, `v_beta` | Vβ gene |
| `j_gene`, `j_beta` | Jβ gene |
| `epitope` | Epitope label |
| `subject` | Subject ID |

```python
from tcrconsensus.io.parser import load_file, normalize

# Auto-detect format
df = load_file("data.tsv", fmt="auto")
df = normalize(df)  # canonical schema
```

## Choosing an Objective

| Objective | When to Use |
|-----------|-------------|
| `balanced` | General purpose, unknown data quality |
| `high_purity` | Need reliable clusters, OK with missing some TCRs |
| `high_recall` | Must capture all related TCRs, OK with some noise |
| `noise_robust` | High background noise in bulk repertoires |
| `fast_screening` | Quick scan, limited compute |

```python
# High purity for curated database analysis
model = TCRConsensus(objective="high_purity")

# Noise robust for bulk repertoire
model = TCRConsensus(objective="noise_robust")
```

## Choosing a Consensus Mode

Control with `mode` parameter or let `mode="auto"` decide.

| Mode | Behavior |
|------|----------|
| `conservative` | Only links TCR pairs supported by ≥2 methods |
| `balanced` | Community detection on weighted co-association graph |
| `coverage` | Links pairs from any single method (max recall) |
| `auto` | Uses profile recommendation |

```python
model = TCRConsensus(mode="conservative")
result = model.fit_predict("input.tsv")
```

## Custom Configuration

Create a YAML config to override defaults:

```yaml
# my_config.yaml
consensus:
  balanced:
    threshold: 0.4
    algorithm: leiden
    resolution: 1.2

refinement:
  filter:
    min_member_confidence: 0.15
  confidence:
    core_threshold: 0.5
    peripheral_threshold: 0.2
```

```python
model = TCRConsensus(config_path="my_config.yaml")
```

Or via CLI:

```bash
tcrconsensus run input.tsv --config my_config.yaml
```

## Selecting Specific Methods

```python
# Only use HD baseline
result = model.fit_predict("input.tsv", methods=["hd_baseline"])

# Multiple methods (requires installed dependencies)
result = model.fit_predict("input.tsv", methods=["hd_baseline", "clustcr", "tcrdist3"])
```

## Working with Results

### Access Clusters

```python
for cluster in result.clusters:
    print(f"Cluster {cluster.cluster_id}")
    print(f"  Total members: {len(cluster.member_ids)}")
    print(f"  Core members: {len(cluster.core_member_ids)}")
    print(f"  Peripheral members: {len(cluster.peripheral_member_ids)}")
    print(f"  Confidence: {cluster.cluster_confidence:.3f}")
    print(f"  Supporting methods: {cluster.supporting_methods}")
```

### Access Pairwise Edges

```python
for edge in result.edges[:10]:
    print(f"{edge.tcr_id_a} <-> {edge.tcr_id_b}: "
          f"support={edge.method_support_count}, "
          f"score={edge.final_score:.3f}")
```

### Dataset Profile

```python
prof = result.profile
print(f"TCRs: {prof.n_tcrs}")
print(f"Chain mode: {prof.chain_mode.value}")
print(f"Noise score: {prof.background_noise_score:.3f}")
print(f"VJ completeness: {prof.vj_completeness:.2f}")
print(f"Repertoire type: {prof.repertoire_type.value}")
```

## CLI Usage

### Profile a Dataset

```bash
tcrconsensus profile input.tsv
```

Output:
```
TCRs: 15000
Chain mode: beta_only
V/J completeness: 0.87
Noise score: 0.432
Repertoire type: bulk
Unique ratio: 0.956
```

### Run Full Pipeline

```bash
# Auto mode
tcrconsensus auto input.tsv -o results/

# Specific methods and mode
tcrconsensus run input.tsv --methods hd_baseline,clustcr --mode balanced -o results/

# With config
tcrconsensus run input.tsv --config config.yaml -o results/
```

### Benchmark

```bash
tcrconsensus benchmark input.tsv -o benchmark_results/
```

## Evaluation with Ground Truth

```python
from tcrconsensus.evaluation import compute_all_metrics
import numpy as np

# pred_labels and true_labels from your analysis
metrics = compute_all_metrics(pred_labels, true_labels, n_total=len(df))
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
```

## Output Files

After running, the output directory contains:

```
output/
└── run_YYYYMMDD_HHMMSS/
    ├── normalized/tcr_table.tsv       # Canonical input
    ├── profile/profile.json           # Dataset statistics
    ├── plan/run_plan.json             # Method selection
    ├── methods/                       # Per-method results
    │   ├── hd_baseline/
    │   │   ├── normalized_output.tsv
    │   │   └── runtime_metadata.json
    │   └── clustcr/
    ├── consensus/
    │   ├── pairwise_consensus_scores.tsv
    │   ├── clusters.tsv
    │   └── cluster_members.tsv
    ├── reports/
    │   ├── report.json               # Full structured report
    │   ├── report.md                 # Human-readable summary
    │   ├── method_runtime.png        # Runtime comparison chart
    │   └── metrics_summary.png       # Metrics bar chart
    └── artifact_manifest.json        # List of all output files
```
