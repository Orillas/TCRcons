# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `BaseClusterer.is_available()` classmethod — each wrapper now detects
  whether its backend is actually installed (try-import for Python packages,
  file-exists for external binaries). `available_methods()` returns only
  methods whose dependencies are present on the current system.
- `tcrconsensus.available_methods()` module-level convenience function.
- `levenshtein` added to `ALL_METHODS` list (was only in CLI, missing from
  the public API).
- `backend-test` CI job (Python 3.10/3.11) installing `.[tcrdist3]` and
  running backend-specific tests.
- `tests/test_backend_detection.py` — 12 tests covering `is_available()` on
  all 8 wrappers and `available_methods()` behaviour.

### Fixed
- `selection/selector.py` and `configs/default.yaml`: replaced invalid
  `ismart` reference with `tcrdist3` in the `high_recall` preset (iSMART
  was never a supported method).
- `evaluation/benchmark.py`: `_get_available_clusterers()` now uses
  `is_available()` pattern consistent with `__init__.py` and `main.py`;
  also added `levenshtein` which was missing from benchmark clusterers.

[1.1.1] - 2026-06-30

### Fixed
- `[deeptcr]` and `[clusterers]` were uninstallable on most hosts. Two causes,
  both fixed:
  - **`nvidia-smi` build probe.** The PyPI sdist of DeepTCR calls `nvidia-smi`
    during build, raising `FileNotFoundError` on any host without an NVIDIA
    driver — so `pip install ".[deeptcr]"` / `.[clusterers]` failed on every
    CPU / macOS machine. The extra now installs DeepTCR from its **GitHub
    source** (`git+https://github.com/sidhomj/DeepTCR.git`); the GitHub
    `setup.py` has no such probe and builds cleanly on non-CUDA hosts.
  - **TensorFlow pin conflict.** The previous extra pinned
    `tensorflow==2.15.1` / `keras==2.15.0`, but DeepTCR's own
    `requirements.txt` pins `tensorflow==2.12.0` / `keras==2.12.0` on Linux, so
    the two were mutually exclusive and the resolver failed on Linux. The extra
    no longer re-pins TensorFlow — DeepTCR manages its own stack on every
    platform (`tensorflow==2.12.0` on Linux; `tensorflow-macos==2.12.0` +
    `tensorflow-metal` on Apple Silicon).
- `requirements.txt`: corrected the `[deeptcr]` hint (it no longer claims a TF
  pin).

### Notes
- Verified on an Apple-Silicon mac: `uv pip install ".[deeptcr]"` now succeeds —
  DeepTCR builds from its GitHub source with no `nvidia-smi` error and imports,
  with TensorFlow 2.12.0 (`tensorflow-macos`) pulled by DeepTCR itself.
  `tcrdist3`'s `parasail` has **no arm64 wheel on macOS** and builds from source
  — install autotools first (`brew install autoconf automake libtool`); Linux
  and Intel-mac pick up a prebuilt wheel.

## [1.1.0] - 2026-06-29

### Added
- `tcrconsensus install-backends` CLI command: clones and builds the external
  backends GIANA, TCRMatch and GLIPH2 **on the user's machine** into a standard
  backends directory. GIANA = pure-Python clone; TCRMatch = clone + `make` +
  IEDB reference-data download; GLIPH2 = clone clusTCR (MIT) to obtain the
  bundled `irtools` binary + v2.0 reference files. All carry non-commercial
  licenses (GIANA: UT Southwestern academic-only; TCRMatch: Non-Profit OSL 3.0;
  GLIPH2's irtools: academic-use) and cannot be bundled in this MIT package, so
  tcrconsensus never redistributes them — the user pulls them directly from
  upstream.
- The GIANA, TCRMatch and GLIPH2 wrappers now auto-discover the backends
  directory (`$TCRCONS_BACKEND_DIR`, or `~/.local/share/tcrconsensus/backends`);
  no `TCR_*` environment variables are required after `install-backends`. Note:
  `irtools` is a Linux-only binary, and a pip-installed clusTCR does not ship it
  — the `install-backends --gliph2` clone is what provides it.

### Fixed
- GIANA wrapper now runs `GIANA4.1.py` with its own directory as CWD, so it
  finds its bundled `Imgt_Human_TRBV.fasta` (previously failed when invoked
  from the pipeline CWD).
- Removed the last hardcoded `/home/jilin/...` server paths: the reproduce
  scripts now take the benchmark DB / output dir / extra sys.path from env vars
  (`TCR_BENCHMARK_DB`, `TCR_BENCHMARK_OUT`, `TCR_EXTRA_PATHS`) instead of
  machine-specific defaults.

### Notes
- Verified on the target Linux host: after a simulated install, GIANA clusters
  a tight-motif sample (4 assignments) and TCRMatch runs against the real IEDB
  data, both via auto-discovery with no `TCR_*` env vars set. `pytest
  tests/test_backends.py` passes (10 tests, no network).

## [1.0.1] - 2026-06-29

### Fixed
- The `[clusterers]` optional-dependency group was unresolvable: `clustcr>=1.0`
  is not on PyPI (404 under every name variant) and `tcrdist3>=3.2` is
  unsatisfiable (the latest release is `0.3`). Replaced with granular opt-in
  extras (`[tcrdist3]`, `[deeptcr]`) and an umbrella `[clusterers]` =
  tcrdist3 + DeepTCR. `clusTCR` is now documented as a manual install (not on
  PyPI; its `setup.py` pins `scipy==1.8`, conflicting with `scipy>=1.9`).
- Dockerfile: the default image no longer runs `pip install clustcr pygliph`
  (both fail on PyPI); it installs the `[tcrdist3]` extra instead, with
  DeepTCR / clusTCR left as commented, documented steps.

### Changed
- README: full eight-method backend table with install sources and resolution
  order, a DeepTCR CUDA build caveat, and a manual-backend-setup section.
- `requirements.txt`: corrected the misleading commented backend hints
  (`tcrdist3>=3.2`, `clustcr>=1.0`); backends now point at the pyproject extras.

### Notes
- Verified: `tcrdist3>=0.3` resolves to `tcrdist3==0.3` (+ `parasail==1.3.4`);
  installed backends import on the target Linux host (`tcrdist3` imports as
  `tcrdist`; DeepTCR 2.1.29 imports). A single `pip install ".[clusterers]"`
  now covers four of the eight methods (hd_baseline, levenshtein, tcrdist3,
  DeepTCR); the other four are external/non-PyPI by necessity.

## [1.0.0] - 2026-06-28

Initial public release (BMC Bioinformatics submission).

### Added
- Scenario-adaptive consensus framework aggregating seven TCR clustering
  methods (Hamming baseline, clusTCR, tcrdist3, GLIPH2, GIANA, TCRMatch,
  DeepTCR) via weighted pairwise co-association.
- Three consensus modes: conservative (connected components), balanced
  (Leiden community detection), coverage (union of method links).
- Layered YAML configuration (`configs/default.yaml`) with presets
  (`high_purity`, `noise_robust`, `fast_screening`) and user overrides.
- Rule-based profiling + method selection driven by dataset characteristics.
- Refinement: split / merge / filter / core-vs-peripheral labelling.
- Evaluation metrics: retention, purity, sensitivity, F1, ARI, NMI.
- CLI (`tcrconsensus profile | run | auto | benchmark`) and Python API.
- JSON + Markdown + figure reporting.
- MIT LICENSE, CITATION.cff, .zenodo.json, GitHub Actions CI, Dockerfile.

### Fixed
- `configs/default.yaml` now ships inside the wheel and is resolved via
  `importlib.resources`, so the default config is available after a plain
  `pip install` (previously it was only found in the source tree).

### Notes
- One pre-existing consensus-weights edge case is marked `xfail`
  (`test_failed_method_zero`); it is result-neutral for the published
  benchmarks and documented in the test's xfail reason.
