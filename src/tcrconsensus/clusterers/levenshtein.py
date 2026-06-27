"""Levenshtein Distance (Edit Distance) clusterer — pure Python, no external deps.

Clusters TCRs by pairwise Levenshtein distance on CDR3β sequences using
connected components (sequences within edit distance threshold are linked).
Unlike Hamming distance, this works on sequences of different lengths.
"""

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


class LevenshteinClusterer(BaseClusterer):
    """Cluster TCRs by CDR3 Levenshtein (edit) distance.

    Uses Union-Find to build connected components: sequences with
    edit distance <= threshold are merged into the same cluster.

    Args:
        distance_threshold: max edit distance to link two sequences (default 1).
        min_cluster_size: minimum members to form a cluster (default 2).
    """

    name = "levenshtein"

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
        return df

    def run(self, prepared_input: pd.DataFrame, workdir: Path) -> dict:
        """Run Levenshtein distance clustering with Union-Find."""
        seqs = prepared_input["cdr3"].values
        ids = prepared_input["tcr_id"].values
        n = len(seqs)

        if n < self.min_cluster_size:
            return {"clusters": {}}

        # Union-Find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Pre-group by length for efficiency: only compare sequences
        # within length difference <= threshold
        len_groups = defaultdict(list)
        for i in range(n):
            len_groups[len(seqs[i])].append(i)

        # Compare all pairs within reachable length groups
        all_lengths = sorted(len_groups.keys())
        checked = set()

        for li, length in enumerate(all_lengths):
            indices_i = len_groups[length]
            # Check this length group and nearby lengths
            for lj in range(li, len(all_lengths)):
                other_length = all_lengths[lj]
                if other_length - length > self.distance_threshold:
                    break
                indices_j = len_groups[other_length]
                for i in indices_i:
                    for j in indices_j:
                        if i >= j:
                            continue
                        pair = (i, j)
                        if pair in checked:
                            continue
                        checked.add(pair)
                        if _levenshtein(seqs[i], seqs[j]) <= self.distance_threshold:
                            union(i, j)

        # Collect clusters
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        clusters: dict[str, list[str]] = {}
        cluster_id = 0
        for root, members in groups.items():
            if len(members) >= self.min_cluster_size:
                cid = f"lev_{cluster_id:04d}"
                clusters[cid] = [ids[m] for m in members]
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


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings.

    Uses dynamic programming with O(min(m,n)) space.
    """
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insert
                prev_row[j + 1] + 1,   # delete
                prev_row[j] + cost,    # substitute
            ))
        prev_row = curr_row

    return prev_row[-1]
