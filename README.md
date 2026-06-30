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

### 0 · Create an isolated environment with uv

```bash
uv venv --python 3.10                         # creates .venv with Python 3.10
# (optional) source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
```

All following `uv pip install` and `uv run` commands target this `.venv`
automatically — no activation required.

### 1 · Core install — built-in methods (`hd_baseline`, `levenshtein`)

```bash
# direct from GitHub, no clone:
uv pip install git+https://github.com/Orillas/TCRcons.git
# …or clone and install locally:
git clone https://github.com/Orillas/TCRcons.git && cd TCRcons && uv pip install .
```

This is enough to run the consensus engine with the two built-in baselines.

### 2 · Optional Python backends — extras (`tcrdist3`, `DeepTCR`)

```bash
# both, from a git URL:
uv pip install "tcrconsensus[clusterers] @ git+https://github.com/Orillas/TCRcons.git"
# …or from a local clone:
uv pip install ".[clusterers]"   # tcrdist3 + DeepTCR
uv pip install ".[tcrdist3]"     # tcrdist3 only
uv pip install ".[deeptcr]"      # DeepTCR only
```

| Extra | Installs | Notes |
|---|---|---|
| `tcrdist3` | `tcrdist3>=0.3` (+ `parasail`) | imports as **`tcrdist`**; **no `parasail` wheel on Apple-Silicon mac** → `brew install autoconf automake libtool` to build it |
| `deeptcr` | `DeepTCR` from **git source** | installed from GitHub (not the PyPI sdist — see caveat); DeepTCR pins its own TensorFlow |
| `clusterers` | both of the above | umbrella |

> **DeepTCR** — the extra installs DeepTCR from its **GitHub source**
> (`git+https://github.com/sidhomj/DeepTCR.git`), **not** the PyPI sdist. The
> published sdist's `setup.py` probes `nvidia-smi` at build time and raises
> `FileNotFoundError` on any host without an NVIDIA driver; the GitHub `setup.py`
> has no such probe, so the git URL installs cleanly on CPU/non-CUDA hosts too.
> DeepTCR pins **its own** stack on every platform (`tensorflow==2.12.0` on
> Linux; `tensorflow-macos==2.12.0` + `tensorflow-metal` on Apple Silicon), so
> the extra does not re-pin TensorFlow — let DeepTCR manage it.
>
> For **reproducible installs** with pinned dependency versions matching a
> verified working DeepTCR environment (TF 2.15.1, Keras 2.15.0, etc.), use
> the convenience script:
> ```bash
> bash scripts/install-deeptcr-repro.sh            # pip
> UV=1 bash scripts/install-deeptcr-repro.sh       # uv
> DRY_RUN=1 bash scripts/install-deeptcr-repro.sh  # preview only
> ```
> Or the equivalent two-step workflow:
> ```bash
> pip install --no-deps ".[deeptcr]"
> pip install -r requirements/deeptcr-pinned.txt
> ```
> See `requirements/deeptcr-pinned.txt` for the full version table.

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
re-pinning scipy:

```bash
uv pip install --no-deps "clustcr @ git+https://github.com/svalkiers/clusTCR.git"
```

### 5 · Verify what is installed

```bash
uv run python -c "from tcrconsensus import TCRConsensus; print(TCRConsensus(mode='auto').available_methods)"
```

### One-shot full setup — all 8 methods

```bash
uv venv --python 3.10
git clone https://github.com/Orillas/TCRcons.git && cd TCRcons
uv pip install ".[clusterers]"                                                        # core + tcrdist3 + deeptcr
uv pip install --no-deps "clustcr @ git+https://github.com/svalkiers/clusTCR.git"     # +clustcr
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
pip install ".[clusterers]"
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
  author = {{TCR-Consensus Team}},
  year   = {2026},
  url    = {https://github.com/Orillas/TCRcons}
}
```

A versioned DOI (Zenodo) and the full author list will be added here upon
deposit. See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 TCR-Consensus Team.
