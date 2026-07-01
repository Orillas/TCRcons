"""CLI entry point for tcrconsensus."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from ..io.parser import load_file, normalize
from ..io.writer import (
    ensure_run_dir,
    write_normalized,
    write_profile,
    write_run_plan,
    write_consensus_edges,
    write_consensus_clusters,
    write_cluster_members,
    write_artifact_manifest,
    write_method_output,
)
from ..config import load_config
from ..profiling.profiler import profile as compute_profile
from ..selection.selector import select_methods
from ..selection.tiered import execute_tiered, split_methods_by_tier
from ..clusterers.hd_baseline import HDBaselineClusterer
from ..consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from ..consensus.weights import compute_method_weights
from ..refinement.refiner import refine
from ..reporting.report import (
    generate_report,
    write_json_report,
    write_markdown_report,
    generate_figures,
)
from ..evaluation.benchmark import BenchmarkRunner
from ..visualization import generate_cluster_visualizations

logger = logging.getLogger(__name__)


def _get_clusterers(methods: list[str]) -> dict:
    """Instantiate available clusterers."""
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    try:
        from ..clusterers.levenshtein import LevenshteinClusterer
        clusterers["levenshtein"] = LevenshteinClusterer()
    except Exception:
        pass
    try:
        from ..clusterers.clustcr_wrapper import ClusTCRWrapper
        clusterers["clustcr"] = ClusTCRWrapper()
    except Exception:
        pass
    try:
        from ..clusterers.tcrdist3_wrapper import TCRDist3Wrapper
        clusterers["tcrdist3"] = TCRDist3Wrapper()
    except Exception:
        pass
    try:
        from ..clusterers.gliph2_wrapper import GLIPH2Wrapper
        clusterers["gliph2"] = GLIPH2Wrapper()
    except Exception:
        pass
    try:
        from ..clusterers.giana_wrapper import GIANAWrapper
        clusterers["giana"] = GIANAWrapper()
    except Exception:
        pass
    try:
        from ..clusterers.tcrmatch_wrapper import TCRMatchWrapper
        clusterers["tcrmatch"] = TCRMatchWrapper()
    except Exception:
        pass
    try:
        from ..clusterers.deeptcr_wrapper import DeepTCRWrapper
        clusterers["deeptcr"] = DeepTCRWrapper()
    except Exception:
        pass
    return {k: v for k, v in clusterers.items() if k in methods}


def _run_pipeline(
    input_path: str,
    output_dir: str,
    objective: str = "balanced",
    methods: list[str] | None = None,
    consensus_mode: str | None = None,
    config_path: str | None = None,
) -> dict:
    """Full pipeline execution."""
    config = load_config(config_path)
    cfg_dict = config._raw

    # Load and normalize
    df = normalize(load_file(input_path))

    # Setup run directory
    run_dir = ensure_run_dir(output_dir)
    write_normalized(df, run_dir)

    # Profile
    prof = compute_profile(df, cfg_dict)
    prof_dict = {
        "n_tcrs": prof.n_tcrs,
        "chain_mode": prof.chain_mode.value,
        "vj_completeness": prof.vj_completeness,
        "background_noise_score": prof.background_noise_score,
        "repertoire_type": prof.repertoire_type.value,
        "label_availability": prof.label_availability,
        "unique_ratio": prof.unique_ratio,
        "clone_expansion_score": prof.clone_expansion_score,
    }
    write_profile(prof_dict, run_dir)

    # Select methods
    plan = select_methods(prof, objective, cfg_dict, methods)
    plan_dict = {
        "objective": plan.objective.value,
        "selected_methods": plan.selected_methods,
        "consensus_mode": plan.consensus_mode.value,
        "weighting_profile": plan.weighting_profile,
    }
    write_run_plan(plan_dict, run_dir)

    # Run clusterers (tiered or flat)
    all_tcr_ids = [str(tid) for tid in df["tcr_id"].tolist()]

    if plan.use_tiered:
        # Tiered execution: cheap methods on full data, expensive on divergent subset
        cheap_methods, expensive_methods = split_methods_by_tier(
            plan.selected_methods, cfg_dict.get("tiered", {}).get("tiers")
        )
        if cheap_methods and expensive_methods:
            logger.info(
                f"Tiered: {len(cheap_methods)} cheap ({', '.join(cheap_methods)}) "
                f"on full data, {len(expensive_methods)} expensive "
                f"({', '.join(expensive_methods)}) on divergent subset"
            )
        else:
            logger.warning(
                "Tiered execution requested but no cheap/expensive split; "
                "falling back to flat execution"
            )
            plan.use_tiered = False

    if not plan.use_tiered:
        # Flat execution (original behavior)
        available = _get_clusterers(plan.selected_methods)
        all_assignments = []
        method_results = []
        tier_stats = None

        for method_name, clusterer in available.items():
            workdir = run_dir / "methods" / method_name
            workdir.mkdir(parents=True, exist_ok=True)
            result = clusterer.safe_execute(df, workdir, cfg_dict)
            all_assignments.extend(result.assignments)
            method_results.append({
                "method": method_name,
                "status": result.status.value,
                "n_assignments": len(result.assignments),
                "runtime_seconds": result.runtime_seconds,
                "error": result.error_message,
            })
            if result.assignments:
                import pandas as pd
                assign_df = pd.DataFrame([a.__dict__ for a in result.assignments])
                write_method_output(method_name, assign_df, result.raw_output, {
                    "runtime_seconds": result.runtime_seconds,
                    "status": result.status.value,
                }, run_dir)
    else:
        # Tiered execution
        def _clusterer_factory(mname):
            avail = _get_clusterers([mname])
            return avail.get(mname)

        all_assignments, tier_stats = execute_tiered(
            df=df,
            all_tcr_ids=all_tcr_ids,
            cheap_methods=cheap_methods,
            expensive_methods=expensive_methods,
            clusterer_factory=_clusterer_factory,
            workdir=run_dir,
            config=cfg_dict,
        )
        method_results = tier_stats.get("method_results", [])

        # Write per-method outputs for tiered run
        for mr in method_results:
            mname = mr["method"]
            m_asgn = [a for a in all_assignments if a.method == mname]
            if m_asgn:
                import pandas as pd
                assign_df = pd.DataFrame([a.__dict__ for a in m_asgn])
                write_method_output(mname, assign_df, None, {
                    "runtime_seconds": mr.get("runtime_seconds", 0.0),
                    "status": mr.get("status", "success"),
                }, run_dir)

        # Store tier stats for reporting
        plan_dict["tiered_stats"] = tier_stats

    # Consensus
    weights = compute_method_weights(
        plan.selected_methods, plan.weighting_profile, cfg_dict
    )
    mode = consensus_mode or plan.consensus_mode.value

    if mode == "conservative":
        clusters, edges = conservative_consensus(
            all_assignments, weights,
            **cfg_dict.get("consensus", {}).get("conservative", {}),
        )
    elif mode == "coverage":
        clusters, edges = coverage_consensus(
            all_assignments, weights,
            **cfg_dict.get("consensus", {}).get("coverage", {}),
        )
    else:
        clusters, edges = balanced_consensus(
            all_assignments, weights,
            **cfg_dict.get("consensus", {}).get("balanced", {}),
        )

    # Refine
    clusters = refine(clusters, edges, cfg_dict)

    # === Post-clustering visualization ===
    try:
        edge_dicts = []
        if edges:
            edge_dicts = [e.__dict__ for e in edges]
        assign_dicts = [a.__dict__ for a in all_assignments]
        viz_dir = run_dir / "reports" / "figures"
        fmt = cfg_dict.get("reporting", {}).get("figure_formats", ["png"])
        viz_paths = generate_cluster_visualizations(
            df=df,
            clusters=[{
                "cluster_id": c.cluster_id,
                "member_ids": c.member_ids,
                "core_member_ids": c.core_member_ids,
                "peripheral_member_ids": c.peripheral_member_ids,
                "cluster_confidence": c.cluster_confidence,
            } for c in clusters],
            edges=edge_dicts,
            assignments=assign_dicts,
            output_dir=viz_dir,
            formats=fmt,
        )
        logger.info(f"Generated {len(viz_paths)} cluster visualizations")
    except Exception as e:
        logger.warning(f"Visualization failed (non-fatal): {e}")

    # Write outputs
    if edges:
        import pandas as pd
        edge_df = pd.DataFrame([e.__dict__ for e in edges])
        write_consensus_edges(edge_df, run_dir)

    cluster_dicts = []
    for c in clusters:
        cluster_dicts.append({
            "cluster_id": c.cluster_id,
            "member_ids": c.member_ids,
            "core_member_ids": c.core_member_ids,
            "peripheral_member_ids": c.peripheral_member_ids,
            "cluster_confidence": c.cluster_confidence,
        })

    write_cluster_members(cluster_dicts, run_dir)

    import pandas as pd
    cluster_rows = []
    for c in clusters:
        for mid in c.member_ids:
            cluster_rows.append({
                "cluster_id": c.cluster_id,
                "tcr_id": mid,
                "confidence": c.cluster_confidence,
            })
    if cluster_rows:
        write_consensus_clusters(pd.DataFrame(cluster_rows), run_dir)

    # Report
    rec_methods = plan.selected_methods
    tier_stats_for_report = plan_dict.get("tiered_stats") if plan.use_tiered else None
    report = generate_report(
        run_dir=run_dir,
        profile=prof_dict,
        run_plan=plan_dict,
        clusters=cluster_dicts,
        metrics={},
        recommendation={"recommended_mode": mode,
                       "recommended_methods": rec_methods},
        method_results=method_results,
        tiered_stats=tier_stats_for_report,
    )
    write_json_report(report, run_dir / "reports" / "report.json")
    write_markdown_report(report, run_dir / "reports" / "report.md")

    fmt = cfg_dict.get("reporting", {}).get("figure_formats", ["png"])
    generate_figures(report, run_dir / "reports", fmt)

    write_artifact_manifest(run_dir)

    return report


@click.group()
@click.version_option(version=None, package_name="tcrconsensus", message="%(package)s %(version)s")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """TCR Consensus Clustering Framework."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@cli.command()
@click.argument("input_path")
@click.option("--output", "-o", default=".", help="Output directory")
def profile(input_path, output):
    """Profile a TCR dataset without clustering."""
    config = load_config()
    df = normalize(load_file(input_path))
    prof = compute_profile(df, config._raw)

    click.echo(f"TCRs: {prof.n_tcrs}")
    click.echo(f"Chain mode: {prof.chain_mode.value}")
    click.echo(f"V/J completeness: {prof.vj_completeness:.2f}")
    click.echo(f"Noise score: {prof.background_noise_score:.3f}")
    click.echo(f"Repertoire type: {prof.repertoire_type.value}")
    click.echo(f"Unique ratio: {prof.unique_ratio:.3f}")


@cli.command()
@click.argument("input_path")
@click.option("--mode", default="balanced", help="Consensus mode: balanced | conservative")
@click.option("--methods", default=None, help="Comma-separated methods")
@click.option("--objective", default="balanced", help="Objective: balanced | high_purity | high_recall | noise_robust")
@click.option("--output", "-o", default="tcrconsensus_output", help="Output directory")
@click.option("--config", "config_path", default=None, help="YAML config file")
def run(input_path, mode, methods, objective, output, config_path):
    """Run full clustering pipeline."""
    method_list = methods.split(",") if methods else None
    report = _run_pipeline(input_path, output, objective, method_list, mode, config_path)
    click.echo(f"Done. {report['summary']['n_clusters']} clusters written to {output}")


@cli.command()
@click.argument("input_path")
@click.option("--objective", default="balanced", help="Objective")
@click.option("--output", "-o", default="tcrconsensus_output", help="Output directory")
@click.option("--config", "config_path", default=None, help="YAML config file")
def auto(input_path, objective, output, config_path):
    """Auto mode: profile → select → cluster → report."""
    report = _run_pipeline(input_path, output, objective, config_path=config_path)
    click.echo(f"Done. {report['summary']['n_clusters']} clusters written to {output}")


@cli.command()
@click.argument("input_path")
@click.option("--output", "-o", default="benchmark_output", help="Output directory")
def benchmark(input_path, output):
    """Run benchmark evaluation."""
    runner = BenchmarkRunner()
    result = runner.run_single_dataset(input_path, output_dir=output)
    click.echo(json.dumps(result, indent=2, default=str))


@cli.command("install-backends")
@click.option("--giana", is_flag=True, help="Install GIANA (clone github.com/s175573/GIANA).")
@click.option("--tcrmatch", is_flag=True, help="Install TCRMatch (clone + make + IEDB data).")
@click.option("--gliph2", is_flag=True, help="Install GLIPH2 (clone clusTCR for the irtools binary + reference).")
@click.option("--all", "all_", is_flag=True, help="Install all supported backends (default).")
@click.option(
    "--dir", "dir_",
    default=None,
    help="Backends directory (default: $TCRCONS_BACKEND_DIR or "
         "$VIRTUAL_ENV/tcrconsensus/backends or "
         "~/.local/share/tcrconsensus/backends).",
)
@click.option("--force", is_flag=True, help="Reinstall even if already present.")
@click.option("--dry-run", is_flag=True, help="Print install commands without executing them.")
def install_backends(giana, tcrmatch, gliph2, all_, dir_, force, dry_run):
    """Download and build external clustering backends on this machine.

    GIANA, TCRMatch and GLIPH2 use non-commercial licenses (GIANA:
    academic-only; TCRMatch: Non-Profit OSL 3.0; GLIPH2's irtools: academic-use,
    bundled inside clusTCR) and ship as source / compiled binaries + reference
    data, so they cannot be bundled in the pip package. This command fetches and
    builds them locally (you, the user, pull directly from upstream — the
    license-clean path). After install, the wrappers discover them automatically;
    no environment variables are required.

    Requires: git and network access. TCRMatch additionally needs g++ (OpenMP);
    GLIPH2's irtools is a Linux-only binary.
    """
    from ..backends import backends_dir, install_giana, install_gliph2, install_tcrmatch

    if not (giana or tcrmatch or gliph2 or all_):
        all_ = True
    targets = []
    if all_ or giana:
        targets.append("giana")
    if all_ or tcrmatch:
        targets.append("tcrmatch")
    if all_ or gliph2:
        targets.append("gliph2")

    base = backends_dir(dir_)
    click.echo(f"Backends directory: {base}")
    if dry_run:
        click.echo("(dry-run: commands below are printed, not executed)")

    failures = 0
    if "giana" in targets:
        try:
            script = install_giana(base, force=force, dry_run=dry_run)
            click.echo(f"  GIANA script: {script}")
        except Exception as e:  # noqa: BLE001 - report, don't abort the whole run
            click.echo(f"  GIANA FAILED: {e}", err=True)
            failures += 1
    if "tcrmatch" in targets:
        try:
            binary, iedb = install_tcrmatch(base, force=force, dry_run=dry_run)
            click.echo(f"  TCRMatch binary:   {binary}")
            click.echo(f"  TCRMatch IEDB data: {iedb}")
        except Exception as e:  # noqa: BLE001
            click.echo(f"  TCRMatch FAILED: {e}", err=True)
            failures += 1
    if "gliph2" in targets:
        try:
            lib = install_gliph2(base, force=force, dry_run=dry_run)
            click.echo(f"  GLIPH2 lib (irtools+ref): {lib}")
        except Exception as e:  # noqa: BLE001
            click.echo(f"  GLIPH2 FAILED: {e}", err=True)
            failures += 1

    if not dry_run and failures == 0 and targets:
        click.echo(
            "\nDone. The GIANA/TCRMatch wrappers will auto-discover this "
            "directory; no TCR_* env vars are needed."
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
