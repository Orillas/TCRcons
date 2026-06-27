# API Reference

## Top-Level API

### `TCRConsensus`

Main entry point for TCR consensus clustering.

```python
from tcrconsensus import TCRConsensus, Result

model = TCRConsensus(
    objective="balanced",    # balanced | high_purity | high_recall | noise_robust | fast_screening
    mode="auto",             # auto | conservative | balanced | coverage
    config_path=None,        # optional YAML config
    output_dir="tcrconsensus_output",
)
result = model.fit_predict("input.tsv", methods=["hd_baseline"])
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objective` | `str` | `"balanced"` | Optimization objective |
| `mode` | `str` | `"auto"` | Consensus mode; `"auto"` uses recommended mode from profiling |
| `config_path` | `str \| None` | `None` | Path to user YAML config |
| `output_dir` | `str` | `"tcrconsensus_output"` | Base output directory |

**Methods:**

#### `fit_predict(input_path, methods=None) -> Result`

Run full pipeline: load → profile → select → cluster → consensus → refine.

- `input_path`: Path to TCR data file (AIRR/VDJdb/custom TSV)
- `methods`: Optional list of method names to override auto-selection

Returns a `Result` dataclass.

### `Result`

Pipeline result container.

| Field | Type | Description |
|-------|------|-------------|
| `clusters` | `list[ConsensusCluster]` | Final consensus clusters |
| `edges` | `list[ConsensusEdge]` | Pairwise consensus edges |
| `profile` | `DatasetProfile` | Dataset statistics |
| `run_plan` | `RunPlan` | Methods and parameters used |
| `recommendation` | `Recommendation` | Auto-generated recommendation |
| `metrics` | `dict[str, float]` | Evaluation metrics (if labels available) |
| `report` | `dict` | Full report data |
| `run_dir` | `str` | Output directory path |

---

## Schema (`tcrconsensus.schema`)

### Data Classes

| Class | Description |
|-------|-------------|
| `TCRRecord` | Single TCR sequence record with chain, CDR3, V/J genes |
| `DatasetProfile` | Computed dataset statistics (noise, completeness, type) |
| `RunPlan` | Selected methods, consensus mode, parameters |
| `ClusterAssignment` | Single TCR → cluster mapping from one method |
| `ConsensusEdge` | Pairwise co-association score between two TCRs |
| `ConsensusCluster` | Final cluster with members, confidence, core/peripheral labels |
| `Recommendation` | Auto-generated method/mode recommendation |

### Enums

| Enum | Values |
|------|--------|
| `ChainMode` | `alpha_only`, `beta_only`, `paired_ab` |
| `RepertoireType` | `bulk`, `single_cell`, `antigen_enriched`, `curated_db` |
| `Objective` | `high_purity`, `balanced`, `high_recall`, `noise_robust`, `fast_screening` |
| `ConsensusMode` | `conservative`, `balanced`, `coverage` |
| `MemberLabel` | `core`, `peripheral`, `low_confidence` |
| `MethodStatus` | `success`, `failed`, `skipped` |

### Validation

```python
from tcrconsensus.schema.validation import validate_tcr_record, validate_cdr3_basic
```

- `validate_tcr_record(record) -> list[str]`: Returns error messages (empty = valid)
- `validate_cdr3_basic(seq) -> bool`: Quick CDR3 format check

---

## IO (`tcrconsensus.io`)

### Parser

```python
from tcrconsensus.io.parser import load_file, normalize, to_records, detect_format
```

| Function | Description |
|----------|-------------|
| `load_file(path, fmt="auto")` | Load any format, return DataFrame |
| `normalize(df)` | Canonical schema, fill defaults, infer chain mode |
| `to_records(df)` | Convert DataFrame → `list[TCRRecord]` |
| `detect_format(path)` | Auto-detect: `"airr"` / `"vdjdb"` / `"custom"` |

### Writer

```python
from tcrconsensus.io.writer import ensure_run_dir, write_normalized, write_artifact_manifest
```

| Function | Description |
|----------|-------------|
| `ensure_run_dir(base_dir, run_name)` | Create run directory with subdirs |
| `write_normalized(df, run_dir)` | Write normalized TSV |
| `write_profile(profile, run_dir)` | Write profile JSON |
| `write_method_output(name, assignments, raw, meta, run_dir)` | Write per-method artifacts |
| `write_consensus_edges(edges, run_dir)` | Write pairwise scores TSV |
| `write_consensus_clusters(clusters, run_dir)` | Write cluster assignments |
| `write_artifact_manifest(run_dir)` | Write manifest of all outputs |

---

## Clusterers (`tcrconsensus.clusterers`)

### Base Classes

```python
from tcrconsensus.clusterers import BaseClusterer, ClustererResult
```

- `BaseClusterer`: ABC with `prepare_input()`, `run()`, `parse_output()`, `normalize()`, `safe_execute()`
- `ClustererResult`: method name, assignments, runtime, status

### Built-in Clusterers

| Class | Name | Description |
|-------|------|-------------|
| `HDBaselineClusterer` | `hd_baseline` | Hamming distance clustering (pure Python) |

**HDBaselineClusterer parameters:**
- `distance_threshold: int = 1` — Max Hamming distance for same cluster
- `min_cluster_size: int = 2` — Minimum members per cluster

### Wrapper Clusterers (require external tools)

| Class | Name | Dependency |
|-------|------|-----------|
| `ClusTCRWrapper` | `clustcr` | `pip install clustcr` |
| `TCRDist3Wrapper` | `tcrdist3` | `pip install tcrdist3` |
| `GLIPH2Wrapper` | `gliph2` | GLIPH2 binary on PATH |

---

## Consensus (`tcrconsensus.consensus`)

### Modes

```python
from tcrconsensus.consensus import conservative_consensus, balanced_consensus, coverage_consensus
```

| Function | Algorithm | Use Case |
|----------|-----------|----------|
| `conservative_consensus(assignments, weights, min_method_support=2)` | Connected components | High confidence |
| `balanced_consensus(assignments, weights, threshold=0.3, algorithm="leiden")` | Leiden/Louvain | General purpose |
| `coverage_consensus(assignments, weights, threshold=0.1, resolution=0.8)` | Low-threshold community | Maximum recall |

All return `tuple[list[ConsensusCluster], list[ConsensusEdge]]`.

### Weights

```python
from tcrconsensus.consensus import compute_method_weights
```

`compute_method_weights(methods, scenario, config)` — Compute per-method weights from config priors and scenario coefficients.

### Graph

```python
from tcrconsensus.consensus import build_consensus_graph, connected_components_clustering, community_clustering
```

### Co-association

```python
from tcrconsensus.consensus import extract_pairwise_support, build_coassociation_matrix
```

---

## Profiling (`tcrconsensus.profiling`)

```python
from tcrconsensus.profiling import profile
```

`profile(df, config) -> DatasetProfile` — Compute noise, VJ completeness, CDR3 stats, repertoire type.

---

## Selection (`tcrconsensus.selection`)

```python
from tcrconsensus.selection import select_methods
```

`select_methods(profile, objective, config, available_methods) -> RunPlan`

---

## Refinement (`tcrconsensus.refinement`)

```python
from tcrconsensus.refinement import refine
```

`refine(clusters, edges, config) -> list[ConsensusCluster]`

Pipeline: score confidence → split incoherent → merge redundant → filter weak → label core/peripheral → recompute confidence.

---

## Evaluation (`tcrconsensus.evaluation`)

### Metrics

```python
from tcrconsensus.evaluation import retention, purity, sensitivity, f1_score, ari, nmi, compute_all_metrics
```

### Benchmark

```python
from tcrconsensus.evaluation import BenchmarkRunner
```

| Method | Description |
|--------|-------------|
| `run_single_dataset(input_path, labels_path, output_dir)` | Full benchmark on one dataset |
| `run_noise_stress_test(signal_path, background_path, noise_levels)` | Noise robustness test |
| `run_ablation(input_path, output_dir)` | Remove components one at a time |

---

## Configuration (`tcrconsensus.config`)

```python
from tcrconsensus.config import load_config, Config, config_to_dict
```

`load_config(user_yaml, preset, package_dir) -> Config`

Layer order: `default.yaml` → preset → user YAML. Deep merge.

---

## Reporting (`tcrconsensus.reporting`)

```python
from tcrconsensus.reporting import generate_report, write_json_report, write_markdown_report, generate_figures
```
