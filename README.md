# TCR Consensus Clustering

[![CI](https://github.com/Orillas/TCRcons/actions/workflows/ci.yml/badge.svg)](https://github.com/Orillas/TCRcons/actions/workflows/ci.yml)

**Scenario-adaptive TCR specificity consensus clustering framework.**

Combines seven TCR clustering methods (clusTCR, GLIPH2, tcrdist3, GIANA, TCRMatch, DeepTCR, Hamming baseline) via weighted co-association consensus, plus a built-in Levenshtein edit-distance baseline. Automatically profiles input data, selects optimal method combinations, and produces refined consensus clusters with confidence scores.

## Installation

tcrconsensus is **not yet on PyPI** — install from GitHub. This guide uses
[**uv**](https://docs.astral.sh/uv/) for fast, isolated, reproducible installs;
plain `pip` works too (see [No uv?](#no-uv-plain-pip) at the end).

The eight clustering methods are unlocked in tiers: install only what you need.

### Prerequisites

| Requirement | Needed for | Notes |
|---|---|---|
| uv | environment + installs | `curl -LsSf https://astral.sh/uv/install.sh \| sh` — also fetches Python |
| git | source install + `install-backends` | |
| C/C++ toolchain (gcc/g++) | tcrdist3 (`parasail`), TCRMatch (`make`) | parasail ships **Linux/Intel-mac wheels**; on **Apple-Silicon mac** it builds from source → `brew install autoconf automake libtool` |
| CUDA toolkit | DeepTCR on GPU | optional; CPU works without it |

### 0 · Create a project-local environment with uv

The virtual environment lives **inside the project directory** — no conflicts
with other projects or the system Python:

```bash
git clone https://github.com/Orillas/TCRcons.git && cd TCRcons       # project root
uv venv --python 3.10                                                  # creates .venv/ right here
source .venv/bin/activate                                       # activate venv 
```

All following `uv pip install` and `uv run` commands target this `.venv`
automatically — no activation required.

> **Project isolation** — everything stays together: source code `.venv`,
> installed backends (`$VIRTUAL_ENV/tcrconsensus/backends`), and any
> configuration. Delete the project directory and nothing is left behind.

### 1 · Core install — built-in methods (`hd_baseline`, `levenshtein`)

```bash
uv pip install .          # installs tcrconsensus into the project-local .venv
```

This is enough to run the consensus engine with the two built-in baselines.

If you prefer to install directly from GitHub **without** a local clone:

```bash
uv pip install git+https://github.com/Orillas/TCRcons.git
```

This is enough to run the consensus engine with the two built-in baselines.

### 2 · Optional Python backends — extras (`tcrdist3`, `DeepTCR`)

From a local clone:

```bash
uv pip install ".[tcrdist3]"     # tcrdist3 (see note below)
```

**DeepTCR** — use the two-step workflow (recommended) for a reproducible pinned
environment (TF 2.15.1, Keras 2.15.0, numpy 1.23.5):

```bash
pip install --no-deps "DeepTCR @ git+https://github.com/sidhomj/DeepTCR.git@3930ca05a987c7cc621b4f2ecfd740e2d62799d8"
pip install -r requirements/deeptcr-pinned.txt
```

Or the equivalent convenience script:

```bash
bash scripts/install-deeptcr-repro.sh            # pip
UV=1 bash scripts/install-deeptcr-repro.sh       # uv
DRY_RUN=1 bash scripts/install-deeptcr-repro.sh  # preview only
```

> **Why two steps?** The published PyPI sdist of DeepTCR probes `nvidia-smi` at
> build time and raises `FileNotFoundError` on any host without an NVIDIA driver.
> Installing from the **GitHub source** with `--no-deps` avoids this probe, then
> the pinned requirements provide a verified dependency set. See
> `requirements/deeptcr-pinned.txt` for the full version table.
>
> **Alternative — `.[deeptcr]` extra:** `uv pip install ".[deeptcr]"` installs
> DeepTCR from its GitHub source and lets DeepTCR manage its own dependencies
> (TensorFlow 2.12.0 on Linux, no numpy pin). This is simpler but **not
> reproducible** — version drift may affect results.

Without a local clone:

```bash
uv pip install "tcrconsensus[tcrdist3] @ git+https://github.com/Orillas/TCRcons.git"
# DeepTCR: still use the two-step workflow above before installing tcrconsensus
```

### 3 · External binary backends — `install-backends` (`GLIPH2`, `GIANA`, `TCRMatch`)

These three carry **non-commercial licenses** (GLIPH2 `irtools`: academic-use,
bundled inside clusTCR; GIANA: UT Southwestern, academic-research-only; TCRMatch:
Non-Profit OSL 3.0) and ship as binaries/scripts + reference data, so they
**cannot be pip-installed or bundled**. The `install-backends` helper clones and
builds them **on your machine** — you pull directly from upstream, the
license-clean path; tcrconsensus never redistributes them:

```bash
uv run tcrconsensus install-backends --giana        # clone github.com/s175573/GIANA   (pure Python)
uv run tcrconsensus install-backends --tcrmatch     # clone github.com/IEDB/TCRMatch → make → IEDB ref data
uv run tcrconsensus install-backends --gliph2       # clone github.com/svalkiers/clusTCR for irtools + ref
uv run tcrconsensus install-backends --all          # all three
uv run tcrconsensus install-backends --dir /opt/tcr # custom backends dir (default: $VIRTUAL_ENV/tcrconsensus/backends or ~/.local/share/tcrconsensus/backends)
uv run tcrconsensus install-backends --dry-run      # print the commands without running
```

After install the wrappers **auto-discover** the backends directory
(`$TCRCONS_BACKEND_DIR`, or `$VIRTUAL_ENV/tcrconsensus/backends`,
or `~/.local/share/tcrconsensus/backends`) — **no
`TCR_*` environment variables required**. Needs: `git` (all three), `g++`/OpenMP
(TCRMatch), and network access (all, for the upstream fetch). `irtools` is
**Linux-only**.

> **GLIPH2 detail** — the `irtools` binary + GLIPH2 v2.0 reference files live in
> a full clusTCR *source checkout* (`clustcr/modules/gliph2/lib/`); a
> pip-installed clusTCR does **not** ship them, which is why `--gliph2` clones
> clusTCR rather than relying on the installed package.

### 4 · clusTCR (not on PyPI)

clusTCR is not published to PyPI, and its `setup.py` pins `scipy==1.8`, which
conflicts with tcrconsensus's `scipy>=1.9`. Install from source **without**
re-pinning scipy, then install its runtime dependencies:

```bash
uv pip install --no-deps "clustcr @ git+https://github.com/svalkiers/clusTCR.git"
uv pip install markov-clustering faiss-cpu==1.7.4
```

> **Why the separate faiss step?** Recent `faiss-cpu` (≥1.13) requires
> `numpy>=2`, which conflicts with the numpy 1.23.5 used by DeepTCR's pinned
> environment. `faiss-cpu==1.7.4` is compatible with both numpy 1.x and
> clusTCR's FAISS integration.

### 5 · Verify what is installed

```bash
uv run python -c "from tcrconsensus import TCRConsensus; print(TCRConsensus(mode='auto').available_methods)"
```

### One-shot full setup — all 8 methods

```bash
git clone https://github.com/Orillas/TCRcons.git && cd TCRcons
uv venv --python 3.10
source .venv/bin/activate
uv pip install .                                                                       # core
uv pip install ".[tcrdist3]"                                                           # +tcrdist3
pip install --no-deps "DeepTCR @ git+https://github.com/sidhomj/DeepTCR.git@3930ca05a987c7cc621b4f2ecfd740e2d62799d8"
pip install -r requirements/deeptcr-pinned.txt                                         # +deeptcr (reproducible)
uv pip install --no-deps "clustcr @ git+https://github.com/svalkiers/clusTCR.git"     # +clustcr (no deps)
uv pip install markov-clustering faiss-cpu==1.7.4                                      # clusTCR runtime deps
uv run tcrconsensus install-backends --all                                            # +gliph2/giana/tcrmatch
```

### Docker

```bash
docker build -t tcrconsensus .
docker run -it --rm -v "$PWD/data:/data" tcrconsensus run /data/input.tsv -o /data/output
```

The image installs core + the `[tcrdist3]` extra. DeepTCR and clusTCR are left
as commented steps in the `Dockerfile` (heavy / non-PyPI); GLIPH2 / GIANA /
TCRMatch need `install-backends` inside the container if required.

### Development

```bash
uv pip install -e ".[dev]"   # pytest, pytest-cov, build, ruff
uv run pytest -q
```

### No uv? (plain pip)

Prefer regular `pip`? Create a venv and drop the `uv ` prefix — every command
above is the same, just `pip …` instead of `uv pip …`, and `python …` /
`tcrconsensus …` instead of `uv run …`:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install .                                          # core
pip install ".[tcrdist3]"                              # +tcrdist3
pip install --no-deps "DeepTCR @ git+https://github.com/sidhomj/DeepTCR.git@3930ca05a987c7cc621b4f2ecfd740e2d62799d8"
pip install -r requirements/deeptcr-pinned.txt         # +deeptcr (reproducible)
```

### Clustering methods at a glance

| Method | Type | How to get it | Wrapper resolution |
|---|---|---|---|
| Hamming baseline | built-in (pure Python) | core install | always available |
| Levenshtein baseline | built-in (pure Python) | core install | always available |
| tcrdist3 | Python pkg (`parasail`, C) | `.[tcrdist3]` | `import tcrdist3` |
| DeepTCR | Python pkg (TensorFlow) | `.[deeptcr]` | `import DeepTCR` |
| clusTCR | Python pkg, **not on PyPI** | manual `--no-deps` git install | `import clustcr` |
| GLIPH2 | `irtools` binary + ref DB | `install-backends --gliph2` | backends dir → `TCR_GLIPH2_LIB` → `PATH` |
| GIANA | `GIANA4.1.py` script | `install-backends --giana` | backends dir → `TCR_GIANA_SCRIPT` → `PATH` |
| TCRMatch | C++ binary + IEDB data | `install-backends --tcrmatch` | backends dir → `TCR_TCRMATCH_BIN` → `PATH` |

For every external method the wrapper resolves its binary/script as:
**constructor argument → `TCR_*` env var → backends dir → `PATH`**, and raises a
clear, actionable error if none is found.

## Quick Start

### Python API

```python
from tcrconsensus import TCRConsensus

# Fast mode (default): cheap methods on full data, expensive only where needed
model = TCRConsensus()
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

| Preset | Effect on method selection |
|--------|---------------------------|
| `fast_screening` **(default)** | Broad method pool, tiered execution handles efficiency |
| `balanced` | Equally weighted purity/sensitivity/noise/speed |
| `high_purity` | Conservative methods, prioritise precision over recall |
| `high_recall` | Sensitivity-focused methods, minimise false negatives |
| `noise_robust` | Methods resilient to background noise |

## Objectives

The **objective** controls which clustering methods are selected and how they
are weighted. It does **not** affect whether methods run on all data — tiered
execution (cheap methods first) is always enabled regardless of objective.

| Objective | Selected methods | Weighting | Best for |
|-----------|-----------------|-----------|---------|
| `fast_screening` **(default)** | hd_baseline, levenshtein, giana, tcrdist3, gliph2, tcrmatch | speed | General use. Tiered execution keeps it fast on easy data |
| `balanced` | All 8 methods | equal | Full coverage with compute budget |
| `high_purity` | clustcr, gliph2, tcrmatch, hd_baseline | purity | Minimise false positives |
| `high_recall` | deeptcr, giana, tcrdist3, hd_baseline | recall | Minimise false negatives |
| `noise_robust` | tcrdist3, gliph2, hd_baseline | noise-robust | Noisy bulk repertoires |

## Consensus Modes

| Mode | Algorithm | Use Case |
|------|-----------|----------|
| **conservative** | Connected components with k-method threshold | High confidence, fewer but reliable clusters |
| **balanced** | Leiden/Louvain community detection | Trade-off between precision and recall |
| **coverage** | Union of all method links | Maximum recall, comprehensive clusters |

## Pipeline Architecture

```
Input → Normalize → Profile → Select Methods
                                         │
                                         ▼
                   ┌── Tiered Execution ──┐
                   │                      │
                   │ ① cheap methods      │  (hd_baseline, levenshtein,
                   │    run on FULL data  │   clustcr, gliph2, giana, tcrmatch)
                   │                      │
                   │ ② detect divergent   │
                   │    TCRs — cheap       │
                   │    methods disagree   │
                   │         │             │
                   │    divergent ≥ 20     │
                   │    && < total × 0.95? │
                   │    ┌─ yes ─┐ no ─┐   │
                   │    │                │   │
                   │ ③ expensive       full │
                   │    methods         fall-│
                   │    on SUB-         back │
                   │    SET             │   │
                   └────────┬──────────┘   │
                            │              │
                            ▼              │
                   Weighted Co-association ◄┘
                            │
                            ▼
                   Consensus (balanced)
                            │
                            ▼
                   Refine → Report
```

1. **IO** — Parse AIRR/VDJdb/custom, normalize to canonical schema
2. **Profiling** — Compute noise, VJ completeness, repertoire type
3. **Selection** — Rule-based method selection from profile + objective
4. **Tiered Clustering** ★ — Cheap methods run on full data; expensive
   methods (tcrdist3, DeepTCR) are only invoked on TCRs where cheap
   methods disagree, saving substantial O(n²) compute
5. **Consensus** — Weighted pairwise co-association → Leiden graph clustering
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
uv pip install -e ".[dev]"
uv run pytest tests/ -v
uv run pytest tests/ --cov=tcrconsensus
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
  author = {Zhang, Jilin},
  year   = {2026},
  doi    = {10.5281/zenodo.21094480},
  url    = {https://github.com/Orillas/TCRcons}
}
```

A DOI is available at [10.5281/zenodo.21094480](https://doi.org/10.5281/zenodo.21094480).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 TCR-Consensus Team.
