"""Report generation — JSON, markdown, figures."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    run_dir: str | Path,
    profile: dict | None = None,
    run_plan: dict | None = None,
    clusters: list[dict] | None = None,
    metrics: dict[str, float] | None = None,
    recommendation: dict | None = None,
    method_results: list[dict] | None = None,
    tiered_stats: dict | None = None,
) -> dict:
    """Generate structured report dict."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "profile": profile or {},
        "run_plan": run_plan or {},
        "summary": {
            "n_clusters": len(clusters) if clusters else 0,
            "metrics": metrics or {},
            "recommendation": recommendation or {},
        },
        "method_results": method_results or [],
        "tiered_stats": tiered_stats or {},
    }
    return report


def write_json_report(report: dict, path: Path) -> Path:
    """Write report as JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def write_markdown_report(report: dict, path: Path) -> Path:
    """Write report as Markdown."""
    lines = ["# TCR Consensus Clustering Report\n"]
    lines.append(f"**Generated:** {report.get('timestamp', 'N/A')}\n")

    # Profile
    profile = report.get("profile", {})
    if profile:
        lines.append("## Dataset Profile\n")
        lines.append(f"- **TCRs:** {profile.get('n_tcrs', 'N/A')}")
        lines.append(f"- **Chain mode:** {profile.get('chain_mode', 'N/A')}")
        lines.append(f"- **V/J completeness:** {profile.get('vj_completeness', 0):.2f}")
        lines.append(f"- **Noise score:** {profile.get('background_noise_score', 0):.3f}")
        lines.append(f"- **Repertoire type:** {profile.get('repertoire_type', 'N/A')}")
        lines.append("")

    # Run plan
    plan = report.get("run_plan", {})
    if plan:
        lines.append("## Run Plan\n")
        lines.append(f"- **Objective:** {plan.get('objective', 'N/A')}")
        lines.append(f"- **Methods:** {', '.join(plan.get('selected_methods', []))}")
        lines.append(f"- **Consensus mode:** {plan.get('consensus_mode', 'N/A')}")
        lines.append("")

    # Summary
    summary = report.get("summary", {})
    metrics = summary.get("metrics", {})
    if metrics:
        lines.append("## Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    # Clusters
    n_clusters = summary.get("n_clusters", 0)
    lines.append(f"## Clusters\n")
    lines.append(f"**Total clusters:** {n_clusters}\n")

    # Recommendation
    rec = summary.get("recommendation", {})
    if rec:
        lines.append("## Recommendation\n")
        lines.append(f"- **Mode:** {rec.get('recommended_mode', 'N/A')}")
        lines.append(f"- **Methods:** {', '.join(rec.get('recommended_methods', []))}")
        lines.append(f"- **Confidence:** {rec.get('confidence', 0):.2f}")
        if rec.get("justification"):
            lines.append(f"- **Justification:** {rec['justification']}")
        lines.append("")

    path.write_text("\n".join(lines))
    return path


def generate_figures(
    report: dict,
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Generate standard figures from report data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping figures")
        return []

    formats = formats or ["png"]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    metrics = report.get("summary", {}).get("metrics", {})
    method_results = report.get("method_results", [])

    # Method comparison bar chart
    if method_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = [r.get("method", "") for r in method_results]
        runtimes = [r.get("runtime_seconds", 0) for r in method_results]
        ax.barh(methods, runtimes)
        ax.set_xlabel("Runtime (seconds)")
        ax.set_title("Method Runtime Comparison")
        for fmt in formats:
            p = output_dir / f"method_runtime.{fmt}"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            generated.append(p)
        plt.close(fig)

    # Metrics radar/bar
    if metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        metric_names = [k for k in metrics if isinstance(metrics[k], float)]
        metric_vals = [metrics[k] for k in metric_names]
        ax.bar(metric_names, metric_vals)
        ax.set_ylabel("Score")
        ax.set_title("Clustering Metrics")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        for fmt in formats:
            p = output_dir / f"metrics_summary.{fmt}"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            generated.append(p)
        plt.close(fig)

    return generated
