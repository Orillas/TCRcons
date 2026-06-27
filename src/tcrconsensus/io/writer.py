"""Output writers for TCR Consensus artifacts."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_run_dir(base_dir: str, run_name: str | None = None) -> Path:
    """Create run directory with standard subdirectories."""
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / run_name
    subdirs = [
        "input", "normalized", "profile", "plan",
        "methods", "consensus", "refinement",
        "evaluation", "reports", "logs",
    ]
    for sub in subdirs:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_normalized(df: pd.DataFrame, run_dir: Path) -> Path:
    """Write normalized TCR table."""
    path = run_dir / "normalized" / "tcr_table.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


def write_profile(profile: dict, run_dir: Path) -> Path:
    """Write dataset profile as JSON."""
    path = run_dir / "profile" / "profile.json"
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    return path


def write_run_plan(plan: dict, run_dir: Path) -> Path:
    """Write run plan as JSON."""
    path = run_dir / "plan" / "run_plan.json"
    with open(path, "w") as f:
        json.dump(plan, f, indent=2, default=str)
    return path


def write_method_output(
    method_name: str,
    assignments: pd.DataFrame,
    raw_output: Any,
    metadata: dict,
    run_dir: Path,
) -> Path:
    """Write method-specific outputs."""
    method_dir = run_dir / "methods" / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    # Normalized assignments
    assign_path = method_dir / "normalized_output.tsv"
    assignments.to_csv(assign_path, sep="\t", index=False)

    # Raw output
    raw_path = method_dir / "raw_output"
    if isinstance(raw_output, pd.DataFrame):
        raw_path = raw_path.with_suffix(".tsv")
        raw_output.to_csv(raw_path, sep="\t", index=False)
    elif isinstance(raw_output, str):
        raw_path.write_text(raw_output)
    else:
        raw_path.write_text(str(raw_output))

    # Metadata
    meta_path = method_dir / "runtime_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return assign_path


def write_consensus_edges(edges: pd.DataFrame, run_dir: Path) -> Path:
    """Write consensus edges."""
    path = run_dir / "consensus" / "pairwise_consensus_scores.tsv"
    edges.to_csv(path, sep="\t", index=False)
    return path


def write_consensus_clusters(clusters: pd.DataFrame, run_dir: Path) -> Path:
    """Write consensus cluster assignments."""
    path = run_dir / "consensus" / "clusters.tsv"
    clusters.to_csv(path, sep="\t", index=False)
    return path


def write_cluster_members(clusters: list[dict], run_dir: Path) -> Path:
    """Write cluster membership table."""
    rows = []
    for c in clusters:
        for mid in c.get("member_ids", []):
            label = "core" if mid in c.get("core_member_ids", []) else "peripheral"
            rows.append({
                "cluster_id": c["cluster_id"],
                "tcr_id": mid,
                "label": label,
                "cluster_confidence": c.get("cluster_confidence", 0.0),
            })
    path = run_dir / "consensus" / "cluster_members.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def write_artifact_manifest(run_dir: Path) -> Path:
    """Write manifest of all artifacts in run directory."""
    artifacts = []
    for root, dirs, files in os.walk(run_dir):
        for fname in files:
            fpath = Path(root) / fname
            rel = fpath.relative_to(run_dir)
            artifacts.append({"path": str(rel), "size_bytes": fpath.stat().st_size})

    path = run_dir / "artifact_manifest.json"
    with open(path, "w") as f:
        json.dump(artifacts, f, indent=2)
    return path
