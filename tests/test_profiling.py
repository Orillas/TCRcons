"""Tests for profiling and selection modules."""

import pytest
import pandas as pd

from tcrconsensus.profiling.profiler import profile as compute_profile
from tcrconsensus.selection.selector import select_methods
from tcrconsensus.schema.records import Objective


class TestProfiler:
    def test_profile(self, sample_tcr_df):
        prof = compute_profile(sample_tcr_df)
        assert prof.n_tcrs == 20
        assert prof.chain_mode.value == "beta_only"
        assert 0 <= prof.vj_completeness <= 1
        assert 0 <= prof.background_noise_score <= 1

    def test_empty_df(self):
        df = pd.DataFrame(columns=[
            "tcr_id", "chain_mode", "cdr3_alpha", "cdr3_beta",
            "v_alpha", "j_alpha", "v_beta", "j_beta",
            "subject_id", "sample_id", "epitope", "hla",
            "count", "frequency", "source_dataset",
        ])
        prof = compute_profile(df)
        assert prof.n_tcrs == 0


class TestSelector:
    def test_select_balanced(self, sample_tcr_df):
        prof = compute_profile(sample_tcr_df)
        plan = select_methods(prof, Objective.BALANCED)
        assert len(plan.selected_methods) > 0
        assert "hd_baseline" in plan.selected_methods

    def test_select_with_available(self, sample_tcr_df):
        prof = compute_profile(sample_tcr_df)
        plan = select_methods(prof, Objective.BALANCED, available_methods=["hd_baseline"])
        assert plan.selected_methods == ["hd_baseline"]

    def test_high_purity(self, sample_tcr_df):
        prof = compute_profile(sample_tcr_df)
        plan = select_methods(prof, Objective.HIGH_PURITY, available_methods=["hd_baseline", "clustcr"])
        assert len(plan.selected_methods) > 0
