"""Hamming Distance baseline clusterer — pure Python, no external deps."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import BaseClusterer
from ..schema.records import ClusterAssignment

logger = logging.getLogger(__name__)


class HDBaselineClusterer(BaseClusterer):
    """Cluster TCRs by CDR3 Hamming distance with simple threshold."""

    name = "hd_baseline"

    def __init__(self, distance_threshold: int = 1, min_cluster_size: int = 2):
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size

    def prepare_input(self, tcr_table: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Extract CDR3 sequences for clustering."""
        chain = config.get("chain", "beta")
        col = f"cdr3_{chain}"
        if col not in tcr_table.columns:
            col = "cdr3_beta"
        df = tcr_table[["tcr_id", col]].dropna(subset=[col]).copy()
        df = df.rename(columns={col: "cdr3"})
        # Filter to same-length sequences for Hamming
        df["seq_len"] = df["cdr3"].str.len()
        return df

    def run(self, prepared_input: pd.DataFrame, workdir: Path) -> dict:
        """Run Hamming distance clustering per length group."""
        groups = prepared_input.groupby("seq_len")
        clusters: dict[str, list[str]] = {}
        cluster_id = 0

        for seq_len, group in groups:
            seqs = group["cdr3"].values
            ids = group["tcr_id"].values

            if len(seqs) < self.min_cluster_size:
                continue

            # Build adjacency by Hamming distance
            n = len(seqs)
            visited = [False] * n

            for i in range(n):
                if visited[i]:
                    continue
                members = [ids[i]]
                visited[i] = True

                for j in range(i + 1, n):
                    if visited[j]:
                        continue
                    if _hamming(seqs[i], seqs[j]) <= self.distance_threshold:
                        members.append(ids[j])
                        visited[j] = True

                if len(members) >= self.min_cluster_size:
                    cid = f"hd_{cluster_id:04d}"
                    clusters[cid] = members
                    cluster_id += 1

        return {"clusters": clusters}

    def parse_output(self, workdir: Path) -> dict:
        """No-op: run() returns dict directly."""
        return {}

    def normalize(self, raw_output: dict) -> list[ClusterAssignment]:
        """Convert clusters dict to ClusterAssignment list."""
        assignments = []
        for cluster_id, members in raw_output.get("clusters", {}).items():
            for tcr_id in members:
                assignments.append(
                    ClusterAssignment(
                        method=self.name,
                        tcr_id=tcr_id,
                        cluster_id=cluster_id,
                        membership_score=1.0,
                    )
                )
        return assignments


def _hamming(s1: str, s2: str) -> int:
    """Compute Hamming distance between two equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))
