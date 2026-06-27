# Architecture

## Overview

tcrconsensus implements a **scenario-adaptive TCR specificity consensus clustering** framework. Multiple TCR clustering methods run independently, their results are combined via weighted co-association consensus, and clusters are refined through split/merge/filter/label operations.

## Pipeline

```
┌─────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  Input   │───>│ Normalize │───>│  Profile │───>│  Select  │
│ (TSV/CSV)│    │  (IO)     │    │          │    │ Methods  │
└─────────┘    └───────────┘    └──────────┘    └──────────┘
                                                       │
                                                       ▼
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  Report  │<───│  Refine   │<───│ Consensus│<───│ Cluster  │
│          │    │           │    │  Engine  │    │ (N ways) │
└──────────┘    └───────────┘    └──────────┘    └──────────┘
```

## Module Structure

```
src/tcrconsensus/
├── __init__.py          # TCRConsensus class + Result (top-level API)
├── config.py            # Layered YAML config with deep merge
├── schema/
│   ├── records.py       # 7 dataclasses + 6 enums (data contracts)
│   └── validation.py    # CDR3 regex, V/J gene format validation
├── io/
│   ├── parser.py        # AIRR/VDJdb/custom loading, auto-detect
│   └── writer.py        # Run directory, artifacts, manifest
├── profiling/
│   └── profiler.py      # Noise estimation, repertoire type inference
├── selection/
│   └── selector.py      # Rule-based scenario → method mapping
├── clusterers/
│   ├── base.py          # BaseClusterer ABC + safe_execute()
│   ├── hd_baseline.py   # Hamming distance (pure Python, always available)
│   ├── clustcr_wrapper.py   # clusTCR adapter
│   ├── gliph2_wrapper.py    # GLIPH2 subprocess adapter
│   └── tcrdist3_wrapper.py  # tcrdist3 adapter
├── consensus/
│   ├── coassociation.py # Pairwise co-association extraction
│   ├── graph.py         # NetworkX graph + community detection
│   ├── modes.py         # Conservative / Balanced / Coverage strategies
│   └── weights.py       # Method weight from priors + scenario
├── refinement/
│   └── refiner.py       # Split / Merge / Filter / Label pipeline
├── evaluation/
│   ├── metrics.py       # Retention, Purity, Sensitivity, F1, ARI, NMI
│   └── benchmark.py     # BenchmarkRunner: single, noise stress, ablation
├── reporting/
│   └── report.py        # JSON + Markdown + matplotlib figures
└── cli/
    └── main.py          # Click CLI: profile | run | auto | benchmark
```

## Data Flow

### 1. Input → Normalization

Raw TCR files (AIRR/VDJdb/custom) are loaded into a pandas DataFrame and normalized to a canonical schema with 15 columns: `tcr_id`, `chain_mode`, `cdr3_alpha`, `cdr3_beta`, `v_alpha`, `j_alpha`, `v_beta`, `j_beta`, `subject_id`, `sample_id`, `epitope`, `hla`, `count`, `frequency`, `source_dataset`.

### 2. Profiling

The normalized DataFrame is analyzed to produce a `DatasetProfile`:
- **VJ completeness**: fraction of records with V and J gene annotations
- **Noise score**: weighted combination of singleton fraction, low-frequency fraction, V/J skewness, and sequence density
- **Repertoire type**: inferred from size, label availability, and structure (bulk / antigen_enriched / curated_db / single_cell)
- **CDR3 length statistics**: mean, std, min, max

### 3. Method Selection

Rule-based mapping from profile + objective → method set + consensus mode:

```
profile → classify_scenario() → lookup config rules → RunPlan
```

Scenarios: `bulk_noisy_beta`, `antigen_enriched`, `high_purity`, `high_recall`.

### 4. Clustering

Each selected method runs via `BaseClusterer.safe_execute()`:
1. `prepare_input()` — transform canonical TCR table to method-specific format
2. `run()` — execute clustering
3. `normalize()` — convert to `list[ClusterAssignment]`

Failed methods are caught and excluded from consensus (status = FAILED, weight = 0).

### 5. Consensus Engine

**Co-association extraction**: For each method, all TCR pairs co-clustered get a support link. Accumulated across methods with weights:

```
pair_score(a,b) = Σ w_m  for each method m that clusters a and b together
```

**Three modes:**

| Mode | Graph Construction | Clustering |
|------|-------------------|------------|
| Conservative | Filter edges by `min_method_support ≥ k` | Connected components |
| Balanced | Filter by `threshold` on weighted score | Leiden/Louvain community detection |
| Coverage | Low threshold (`0.1`) | Leiden with low resolution (`0.8`) |

**Method weights** computed from config priors:
```
w_m = a·purity + b·sensitivity + c·noise_robust + d·speed
```
Coefficients `(a,b,c,d)` selected by objective/scenario.

### 6. Refinement

Sequential operations on consensus clusters:

1. **Score confidence**: mean pairwise edge score within cluster
2. **Split**: clusters with low internal consensus → connected components
3. **Merge**: clusters with high cross-cluster association → combine
4. **Filter**: remove members with low average edge score
5. **Label**: classify members as core (≥0.6) or peripheral (≥0.3)
6. **Recompute confidence**

### 7. Reporting

Output artifacts written to structured run directory:
- `report.json` — full structured report
- `report.md` — human-readable markdown
- `method_runtime.png` — runtime comparison chart
- `metrics_summary.png` — metric bar chart
- `artifact_manifest.json` — provenance

## Extending with New Clusterers

Implement `BaseClusterer`:

```python
from tcrconsensus.clusterers.base import BaseClusterer, ClustererResult

class MyClusterer(BaseClusterer):
    name = "my_method"

    def prepare_input(self, tcr_table, config):
        return tcr_table[["tcr_id", "cdr3_beta"]]

    def run(self, prepared_input, workdir):
        # Your clustering logic
        return clusters_dict

    def parse_output(self, workdir):
        return {}

    def normalize(self, raw_output):
        return [ClusterAssignment(method=self.name, tcr_id=tid, cluster_id=cid)
                for cid, members in raw_output.items() for tid in members]
```

Register in `TCRConsensus._get_clusterers()` and add to config weights.

## Design Decisions

- **Dataclass schema** over Pydantic: lighter dependency, sufficient for scientific computing
- **ABC + safe_execute()**: isolated failure handling — one method failing doesn't crash the pipeline
- **Layered YAML config**: reproducibility + flexibility without code changes
- **networkx graph**: standard graph library, community detection fallback chain (Leiden → Louvain → connected components)
- **pandas DataFrame**: canonical internal representation, efficient for IO and aggregation
