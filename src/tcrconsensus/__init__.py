"""TCR Consensus Clustering Framework — top-level API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .io.parser import load_file, normalize
from .io.writer import ensure_run_dir, write_normalized, write_artifact_manifest
from .config import load_config, Config
from .profiling.profiler import profile as compute_profile
from .selection.selector import select_methods
from .clusterers.hd_baseline import HDBaselineClusterer
from .consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from .consensus.weights import compute_method_weights
from .refinement.refiner import refine
from .reporting.report import generate_report
from .schema.records import (
    DatasetProfile,
    RunPlan,
    ConsensusCluster,
    ConsensusEdge,
    Recommendation,
)


# All supported clustering methods
ALL_METHODS: list[str] = [
    "hd_baseline", "giana", "gliph2", "clustcr",
    "tcrmatch", "tcrdist3", "deeptcr",
]


@dataclass
class Result:
    """Container for pipeline results."""

    clusters: list[ConsensusCluster] = field(default_factory=list)
    edges: list[ConsensusEdge] = field(default_factory=list)
    profile: Optional[DatasetProfile] = None
    run_plan: Optional[RunPlan] = None
    recommendation: Optional[Recommendation] = None
    metrics: dict[str, float] = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    run_dir: Optional[str] = None


class TCRConsensus:
    """Main entry point for TCR consensus clustering.

    Usage:
        model = TCRConsensus(objective="balanced", mode="auto")
        result = model.fit_predict("input.tsv")
    """

    def __init__(
        self,
        objective: str = "balanced",
        mode: str = "auto",
        config_path: str | None = None,
        output_dir: str = "tcrconsensus_output",
    ):
        self.objective = objective
        self.mode = mode
        self.config = load_config(config_path)
        self.output_dir = output_dir

    @property
    def available_methods(self) -> list[str]:
        """Methods actually importable on this system."""
        return list(self._get_clusterers(ALL_METHODS).keys())

    def fit_predict(
        self,
        input_path: str,
        methods: list[str] | None = None,
    ) -> Result:
        """Run full pipeline and return Result."""
        cfg = self.config._raw

        # Load
        df = normalize(load_file(input_path))

        # Profile
        prof = compute_profile(df, cfg)

        # Select
        plan = select_methods(prof, self.objective, cfg, methods)

        # Resolve mode
        if self.mode == "auto":
            consensus_mode = plan.consensus_mode.value
        else:
            consensus_mode = self.mode

        # Run clusterers
        available = self._get_clusterers(plan.selected_methods)
        all_assignments = []
        method_results = []

        for method_name, clusterer in available.items():
            result = clusterer.safe_execute(df, Path(self.output_dir), cfg)
            all_assignments.extend(result.assignments)
            method_results.append({
                "method": method_name,
                "status": result.status.value,
                "n_assignments": len(result.assignments),
                "runtime_seconds": result.runtime_seconds,
            })

        # Consensus — only pass kwargs each mode actually accepts
        weights = compute_method_weights(
            plan.selected_methods, plan.weighting_profile, cfg
        )

        if consensus_mode == "conservative":
            cons_cfg = cfg.get("consensus", {}).get("conservative", {})
            conservative_kw = {}
            if "min_method_support" in cons_cfg:
                conservative_kw["min_method_support"] = cons_cfg["min_method_support"]
            if "threshold" in cons_cfg:
                conservative_kw["threshold"] = cons_cfg["threshold"]
            clusters, edges = conservative_consensus(
                all_assignments, weights, **conservative_kw,
            )
        elif consensus_mode == "coverage":
            cov_cfg = cfg.get("consensus", {}).get("coverage", {})
            coverage_kw = {}
            if "threshold" in cov_cfg:
                coverage_kw["threshold"] = cov_cfg["threshold"]
            clusters, edges = coverage_consensus(
                all_assignments, weights, **coverage_kw,
            )
        else:
            bal_cfg = cfg.get("consensus", {}).get("balanced", {})
            balanced_kw = {}
            if "threshold" in bal_cfg:
                balanced_kw["threshold"] = bal_cfg["threshold"]
            clusters, edges = balanced_consensus(
                all_assignments, weights, **balanced_kw,
            )

        # Refine
        clusters = refine(clusters, edges, cfg)

        # Recommendation
        rec = Recommendation(
            scenario=prof.repertoire_type.value,
            recommended_mode=plan.consensus_mode,
            recommended_methods=plan.selected_methods,
            confidence=prof.vj_completeness,
            justification=f"Based on {prof.n_tcrs} TCRs, noise={prof.background_noise_score:.2f}",
        )

        return Result(
            clusters=clusters,
            edges=edges,
            profile=prof,
            run_plan=plan,
            recommendation=rec,
            metrics={},
            report={"method_results": method_results},
            run_dir=self.output_dir,
        )

    @staticmethod
    def _get_clusterers(methods: list[str]) -> dict:
        """Instantiate available clusterers."""
        clusterers: dict[str, Any] = {"hd_baseline": HDBaselineClusterer()}
        try:
            from .clusterers.clustcr_wrapper import ClusTCRWrapper
            clusterers["clustcr"] = ClusTCRWrapper()
        except Exception:
            pass
        try:
            from .clusterers.tcrdist3_wrapper import TCRDist3Wrapper
            clusterers["tcrdist3"] = TCRDist3Wrapper()
        except Exception:
            pass
        try:
            from .clusterers.gliph2_wrapper import GLIPH2Wrapper
            clusterers["gliph2"] = GLIPH2Wrapper()
        except Exception:
            pass
        try:
            from .clusterers.giana_wrapper import GIANAWrapper
            clusterers["giana"] = GIANAWrapper()
        except Exception:
            pass
        try:
            from .clusterers.tcrmatch_wrapper import TCRMatchWrapper
            clusterers["tcrmatch"] = TCRMatchWrapper()
        except Exception:
            pass
        try:
            from .clusterers.deeptcr_wrapper import DeepTCRWrapper
            clusterers["deeptcr"] = DeepTCRWrapper()
        except Exception:
            pass
        return {k: v for k, v in clusterers.items() if k in methods}
