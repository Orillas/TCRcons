"""Benchmark runner for evaluation experiments."""

from __future__ import annotations

import copy
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ..io.parser import load_file, normalize
from ..profiling.profiler import profile as compute_profile
from ..selection.selector import select_methods
from ..clusterers.hd_baseline import HDBaselineClusterer
from ..clusterers.levenshtein import LevenshteinClusterer
from ..consensus.modes import balanced_consensus, conservative_consensus
from ..consensus.weights import compute_method_weights
from ..refinement.refiner import refine
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrate benchmark experiments."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def run_single_dataset(
        self,
        input_path: str,
        labels_path: str | None = None,
        output_dir: str = "benchmark_output",
        methods: list[str] | None = None,
        consensus_mode: str = "balanced",
    ) -> dict[str, Any]:
        """Run benchmark on a single dataset."""
        df = normalize(load_file(input_path))
        n_total = len(df)

        # Get true labels if available
        true_labels = None
        if labels_path:
            labels_df = pd.read_csv(labels_path, sep="\t")
            label_map = dict(zip(labels_df["tcr_id"], labels_df["epitope"]))
            true_labels = df["tcr_id"].map(label_map).values

        # Profile and select
        prof = compute_profile(df, self.config)
        plan = select_methods(prof, "balanced", self.config, methods)

        # Run clusterers
        workdir = Path(output_dir) / "work"
        workdir.mkdir(parents=True, exist_ok=True)

        all_assignments = []
        clusterer_map = _get_available_clusterers()

        for method_name in plan.selected_methods:
            if method_name not in clusterer_map:
                logger.warning(f"Method {method_name} not available, skipping")
                continue
            clusterer = clusterer_map[method_name]
            result = clusterer.safe_execute(df, workdir, self.config)
            all_assignments.extend(result.assignments)

        # Consensus
        weights = compute_method_weights(
            plan.selected_methods, plan.weighting_profile, self.config
        )

        if consensus_mode == "conservative":
            clusters, edges = conservative_consensus(
                all_assignments, weights,
                **self.config.get("consensus", {}).get("conservative", {}),
            )
        else:
            clusters, edges = balanced_consensus(
                all_assignments, weights,
                **self.config.get("consensus", {}).get("balanced", {}),
            )

        # Refine
        clusters = refine(clusters, edges, self.config)

        # Evaluate
        results = {"n_clusters": len(clusters), "n_total": n_total}

        if true_labels is not None:
            pred_labels = _clusters_to_labels(clusters, df["tcr_id"].values)
            valid = pd.notna(true_labels) & pd.notna(pred_labels)
            if valid.sum() > 0:
                # Encode labels to ints for sklearn compatibility
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                t_str = true_labels[valid].astype(str)
                p_str = pred_labels[valid].astype(str)
                all_labels = np.concatenate([t_str, p_str])
                le.fit(all_labels)
                encoded_true = le.transform(t_str)
                encoded_pred = le.transform(p_str)
                metrics = compute_all_metrics(
                    encoded_pred,
                    encoded_true,
                    n_total,
                )
                results.update(metrics)

        return results

    def run_noise_stress_test(
        self,
        signal_path: str,
        background_path: str,
        noise_levels: list[float] | None = None,
        output_dir: str = "noise_stress_output",
    ) -> pd.DataFrame:
        """Run noise stress test at multiple noise levels."""
        noise_levels = noise_levels or [0.1, 0.25, 0.5, 0.75, 0.9]
        signal_df = normalize(load_file(signal_path))
        background_df = normalize(load_file(background_path))

        all_results = []
        for noise_frac in noise_levels:
            n_noise = int(len(signal_df) * noise_frac / (1 - noise_frac))
            noise_sample = background_df.sample(
                n=min(n_noise, len(background_df)), replace=True
            )
            mixed = pd.concat([signal_df, noise_sample], ignore_index=True)
            mixed = mixed.drop_duplicates(subset=["tcr_id"]).reset_index(drop=True)

            tmp_path = Path(output_dir) / f"mixed_{noise_frac:.2f}.tsv"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            mixed.to_csv(tmp_path, sep="\t", index=False)

            result = self.run_single_dataset(
                str(tmp_path),
                output_dir=output_dir,
            )
            result["noise_level"] = noise_frac
            all_results.append(result)

        return pd.DataFrame(all_results)

    def run_ablation(
        self,
        input_path: str,
        output_dir: str = "ablation_output",
    ) -> pd.DataFrame:
        """Run ablation study: remove components one at a time."""
        ablation_configs = [
            ("full", {}),
            ("no_weighting", {"consensus": {"weights": {"global_priors": {}}}}),
            ("no_vj", {}),
            ("no_refinement", {"refinement": {"skip": True}}),
        ]

        results = []
        for name, override in ablation_configs:
            cfg = copy.deepcopy(self.config)
            cfg.update(override)
            runner = BenchmarkRunner(cfg)
            result = runner.run_single_dataset(input_path, output_dir=output_dir)
            result["ablation"] = name
            results.append(result)

        return pd.DataFrame(results)

    def run_leave_one_epitope_out(
        self,
        input_path: str,
        labels_path: str,
        output_dir: str = "loeo_output",
        methods: list[str] | None = None,
    ) -> pd.DataFrame:
        """Leave-one-epitope-out cross-validation.

        For each epitope, hold it out as the test set,
        train on remaining epitopes, and evaluate clustering on test.
        """
        df = normalize(load_file(input_path))
        labels_df = pd.read_csv(labels_path, sep="\t")
        df = df.merge(labels_df, on="tcr_id", how="inner")
        epitopes = df["epitope"].dropna().unique()

        if len(epitopes) < 2:
            logger.warning("Need at least 2 epitopes for LOEO")
            return pd.DataFrame()

        all_results = []
        for test_ep in epitopes:
            train = df[df["epitope"] != test_ep]
            test = df[df["epitope"] == test_ep]

            if len(test) < 2:
                continue

            tmpdir = Path(tempfile.mkdtemp())
            train_path = tmpdir / "train.tsv"
            train_labels = tmpdir / "train_labels.tsv"

            train.to_csv(train_path, sep="\t", index=False)
            train[["tcr_id", "epitope"]].to_csv(train_labels, sep="\t", index=False)

            try:
                result = self.run_single_dataset(
                    str(train_path),
                    labels_path=str(train_labels),
                    output_dir=str(tmpdir / "out"),
                    methods=methods,
                )
                result["held_out_epitope"] = test_ep
                result["test_size"] = len(test)
                all_results.append(result)
            except Exception as e:
                logger.warning(f"LOEO failed for epitope {test_ep}: {e}")

        return pd.DataFrame(all_results)

    def run_leave_one_dataset_out(
        self,
        dataset_paths: list[str],
        labels_paths: list[str] | None = None,
        output_dir: str = "lodo_output",
        methods: list[str] | None = None,
    ) -> pd.DataFrame:
        """Leave-one-dataset-out cross-validation.

        For each dataset, hold it out as test, train consensus weights
        on remaining datasets, and evaluate on the held-out dataset.
        """
        n_datasets = len(dataset_paths)
        if n_datasets < 2:
            logger.warning("Need at least 2 datasets for LODO")
            return pd.DataFrame()

        all_results = []
        for holdout_idx in range(n_datasets):
            test_path = dataset_paths[holdout_idx]
            test_labels = labels_paths[holdout_idx] if labels_paths else None

            train_dfs = []
            for i in range(n_datasets):
                if i == holdout_idx:
                    continue
                train_dfs.append(normalize(load_file(dataset_paths[i])))

            train_df = pd.concat(train_dfs, ignore_index=True)
            train_df = train_df.drop_duplicates(subset=["tcr_id"]).reset_index(drop=True)

            tmpdir = Path(tempfile.mkdtemp())
            train_path = tmpdir / "train.tsv"
            train_df.to_csv(train_path, sep="\t", index=False)

            train_labels_path = None
            if labels_paths:
                all_train_labels = []
                for i in range(n_datasets):
                    if i == holdout_idx:
                        continue
                    all_train_labels.append(pd.read_csv(labels_paths[i], sep="\t"))
                train_labels_df = pd.concat(all_train_labels, ignore_index=True)
                train_labels_path = tmpdir / "train_labels.tsv"
                train_labels_df.to_csv(train_labels_path, sep="\t", index=False)

            try:
                result = self.run_single_dataset(
                    str(train_path),
                    labels_path=str(train_labels_path) if train_labels_path else None,
                    output_dir=str(tmpdir / "out"),
                    methods=methods,
                )
                result["held_out_dataset"] = str(test_path)
                all_results.append(result)

                test_result = self.run_single_dataset(
                    test_path,
                    labels_path=test_labels,
                    output_dir=str(tmpdir / "out_test"),
                    methods=methods,
                )
                test_result["held_out_dataset"] = f"{test_path}__test_eval"
                all_results.append(test_result)

            except Exception as e:
                logger.warning(f"LODO failed for dataset {test_path}: {e}")

        return pd.DataFrame(all_results)

    def run_tradeoff_curve(
        self,
        input_path: str,
        labels_path: str | None = None,
        output_dir: str = "tradeoff_output",
        methods: list[str] | None = None,
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate purity vs sensitivity tradeoff curve by sweeping
        consensus threshold.

        Varies the consensus graph threshold to trace out the purity/
        retention/sensitivity tradeoff curve.
        """
        thresholds = thresholds or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        all_results = []
        for thr in thresholds:
            cfg = copy.deepcopy(self.config)
            cfg.setdefault("consensus", {})
            cfg["consensus"].setdefault("balanced", {})
            cfg["consensus"]["balanced"]["threshold"] = thr

            runner = BenchmarkRunner(cfg)
            result = runner.run_single_dataset(
                input_path,
                labels_path=labels_path,
                output_dir=output_dir,
                methods=methods,
            )
            result["threshold"] = thr
            all_results.append(result)

        return pd.DataFrame(all_results)

    def run_full_benchmark(
        self,
        input_path: str,
        labels_path: str | None = None,
        dataset_paths: list[str] | None = None,
        dataset_labels: list[str] | None = None,
        output_dir: str = "full_benchmark",
        methods: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run complete benchmark suite: single-dataset, noise stress,
        ablation, tradeoff, and cross-validation if possible.

        Returns a dict mapping experiment name -> results DataFrame.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = {}

        logger.info("Running single dataset benchmark")
        results["single_dataset"] = self.run_single_dataset(
            input_path, labels_path, str(out / "single"), methods
        )

        if labels_path:
            logger.info("Running tradeoff curve")
            results["tradeoff_curve"] = self.run_tradeoff_curve(
                input_path, labels_path, str(out / "tradeoff"), methods
            )

        logger.info("Running ablation study")
        results["ablation"] = self.run_ablation(
            input_path, str(out / "ablation")
        )

        if labels_path:
            df = normalize(load_file(input_path))
            labels_df = pd.read_csv(labels_path, sep="\t")
            df = df.merge(labels_df, on="tcr_id", how="inner")
            n_epitopes = df["epitope"].dropna().nunique()
            if n_epitopes >= 3:
                logger.info(f"Running LOEO ({n_epitopes} epitopes)")
                results["loeo"] = self.run_leave_one_epitope_out(
                    input_path, labels_path, str(out / "loeo"), methods
                )

        if dataset_paths and len(dataset_paths) >= 2:
            logger.info(f"Running LODO ({len(dataset_paths)} datasets)")
            results["lodo"] = self.run_leave_one_dataset_out(
                dataset_paths, dataset_labels, str(out / "lodo"), methods
            )

        logger.info("Running noise stress test")
        try:
            noise_path = _generate_synthetic_noise(input_path, out / "noise_bg.tsv", n=500)
            results["noise_stress"] = self.run_noise_stress_test(
                input_path, str(noise_path),
                output_dir=str(out / "noise_stress"),
            )
        except Exception as e:
            logger.warning(f"Noise stress test skipped: {e}")

        for name, res in results.items():
            if isinstance(res, pd.DataFrame):
                res.to_csv(out / f"{name}_results.tsv", sep="\t", index=False)
            elif isinstance(res, dict):
                with open(out / f"{name}_results.json", "w") as f:
                    json.dump(res, f, indent=2, default=str)

        summary = _make_benchmark_summary(results)
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return results


def _get_available_clusterers() -> dict:
    """Return dict of available clusterer instances (uses is_available())."""
    from ..clusterers.levenshtein import LevenshteinClusterer
    clusterers = {"hd_baseline": HDBaselineClusterer(), "levenshtein": LevenshteinClusterer()}

    from ..clusterers.clustcr_wrapper import ClusTCRWrapper
    if ClusTCRWrapper.is_available():
        clusterers["clustcr"] = ClusTCRWrapper()

    from ..clusterers.tcrdist3_wrapper import TCRDist3Wrapper
    if TCRDist3Wrapper.is_available():
        clusterers["tcrdist3"] = TCRDist3Wrapper()

    from ..clusterers.gliph2_wrapper import GLIPH2Wrapper
    if GLIPH2Wrapper.is_available():
        clusterers["gliph2"] = GLIPH2Wrapper()

    from ..clusterers.giana_wrapper import GIANAWrapper
    if GIANAWrapper.is_available():
        clusterers["giana"] = GIANAWrapper()

    from ..clusterers.tcrmatch_wrapper import TCRMatchWrapper
    if TCRMatchWrapper.is_available():
        clusterers["tcrmatch"] = TCRMatchWrapper()

    from ..clusterers.deeptcr_wrapper import DeepTCRWrapper
    if DeepTCRWrapper.is_available():
        clusterers["deeptcr"] = DeepTCRWrapper()

    return clusterers


def _clusters_to_labels(
    clusters: list, tcr_ids: np.ndarray
) -> np.ndarray:
    """Convert cluster list to per-TCR label array."""
    label_map = {}
    for cluster in clusters:
        for mid in cluster.member_ids:
            label_map[mid] = cluster.cluster_id
    return np.array([label_map.get(tid, -1) for tid in tcr_ids])


def _generate_synthetic_noise(input_path: str, output_path: Path, n: int = 500) -> Path:
    """Generate synthetic noise TCRs by shuffling CDR3 sequences."""
    df = normalize(load_file(input_path))
    cdr3_col = "cdr3_beta" if "cdr3_beta" in df.columns else "cdr3_alpha"
    seqs = df[cdr3_col].dropna().values

    noise_seqs = []
    for _ in range(min(n, len(seqs) * 3)):
        s = seqs[np.random.randint(len(seqs))]
        if len(s) >= 4:
            pos = np.random.randint(0, len(s))
            aa_list = list(s)
            aa_list[pos] = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
            noise_seqs.append("".join(aa_list))
        else:
            noise_seqs.append(s)

    noise_df = pd.DataFrame({
        "tcr_id": [f"noise_{i:06d}" for i in range(len(noise_seqs))],
        cdr3_col: noise_seqs,
        "chain_mode": "beta_only",
        "count": 1,
    })
    noise_df.to_csv(output_path, sep="\t", index=False)
    return output_path


def _make_benchmark_summary(results: dict) -> dict:
    """Create readable summary from benchmark results."""
    summary = {}
    for name, res in results.items():
        if isinstance(res, pd.DataFrame) and not res.empty:
            summary[name] = {
                "n_experiments": len(res),
                "columns": res.columns.tolist(),
            }
        elif isinstance(res, dict):
            flat = {}
            for k, v in res.items():
                try:
                    flat[k] = float(v)
                except (TypeError, ValueError):
                    flat[k] = str(v)
            summary[name] = flat
    return summary
