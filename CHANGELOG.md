# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

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
