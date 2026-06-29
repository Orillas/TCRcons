# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

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
