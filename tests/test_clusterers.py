"""Tests for clusterer modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.clusterers.base import BaseClusterer


class TestHDBaseline:
    def test_cluster(self, sample_tcr_df, tmp_dir):
        clusterer = HDBaselineClusterer(distance_threshold=1, min_cluster_size=2)
        result = clusterer.safe_execute(sample_tcr_df, tmp_dir)
        assert result.status.value == "success"
        assert len(result.assignments) > 0

    def test_prepare_input(self, sample_tcr_df):
        clusterer = HDBaselineClusterer()
        prepared = clusterer.prepare_input(sample_tcr_df, {})
        assert "cdr3" in prepared.columns
        assert len(prepared) > 0

    def test_normalize(self):
        raw = {
            "clusters": {
                "hd_0000": ["t1", "t2", "t3"],
                "hd_0001": ["t4", "t5"],
            }
        }
        clusterer = HDBaselineClusterer()
        assignments = clusterer.normalize(raw)
        assert len(assignments) == 5
        assert assignments[0].method == "hd_baseline"
        assert assignments[0].cluster_id == "hd_0000"

    def test_small_input(self, tmp_dir):
        df = pd.DataFrame({
            "tcr_id": ["t1", "t2"],
            "cdr3_beta": ["CASSLAPGATNEKLFF", "CASSLAPGATNEKLFX"],  # Hamming=1
        })
        clusterer = HDBaselineClusterer(distance_threshold=1, min_cluster_size=2)
        result = clusterer.safe_execute(df, tmp_dir)
        assert len(result.assignments) == 2

    def test_no_clusters_below_min(self, tmp_dir):
        df = pd.DataFrame({
            "tcr_id": ["t1", "t2"],
            "cdr3_beta": ["CASSLAPGATNEKLFF", "CASSQETQYF"],  # different length
        })
        clusterer = HDBaselineClusterer(distance_threshold=1, min_cluster_size=2)
        result = clusterer.safe_execute(df, tmp_dir)
        assert len(result.assignments) == 0  # different lengths = different groups, solo = no cluster


class TestBaseClusterer:
    def test_hdbaseline_is_base(self):
        clusterer = HDBaselineClusterer()
        assert isinstance(clusterer, BaseClusterer)
        assert clusterer.name == "hd_baseline"
