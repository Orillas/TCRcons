"""Tests for IO module."""

import pytest
import pandas as pd
from pathlib import Path

from tcrconsensus.io.parser import (
    load_file, normalize, to_records, detect_format, load_custom,
)
from tcrconsensus.io.writer import (
    ensure_run_dir, write_normalized, write_artifact_manifest,
)


class TestParser:
    def test_load_custom_tsv(self, sample_tsv):
        df = load_custom(sample_tsv)
        assert len(df) == 10
        assert "cdr3_beta" in df.columns

    def test_load_file_auto(self, sample_tsv):
        df = load_file(sample_tsv, fmt="auto")
        assert len(df) == 10

    def test_normalize(self, sample_tsv):
        df = load_file(sample_tsv)
        normed = normalize(df)
        assert "tcr_id" in normed.columns
        assert "chain_mode" in normed.columns
        assert "cdr3_beta" in normed.columns
        assert normed["chain_mode"].mode().iloc[0] == "beta_only"

    def test_normalize_fills_defaults(self, sample_tsv):
        df = load_file(sample_tsv)
        normed = normalize(df)
        assert (normed["count"] > 0).all()

    def test_to_records(self, sample_tsv):
        df = load_file(sample_tsv)
        normed = normalize(df)
        records = to_records(normed)
        assert len(records) == len(normed)
        assert records[0].tcr_id.startswith("tcr_")

    def test_detect_format(self, sample_tsv):
        fmt = detect_format(sample_tsv)
        assert fmt == "custom"


class TestWriter:
    def test_ensure_run_dir(self, tmp_dir):
        run_dir = ensure_run_dir(str(tmp_dir), "test_run")
        assert run_dir.exists()
        assert (run_dir / "methods").exists()
        assert (run_dir / "consensus").exists()
        assert (run_dir / "reports").exists()

    def test_write_normalized(self, tmp_path):
        df = pd.DataFrame({"tcr_id": ["t1"], "cdr3_beta": ["CASSLAPGATNEKLFF"]})
        path = tmp_path / "norm.tsv"
        from tcrconsensus.io.writer import write_normalized
        # Need a run_dir-like structure
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "normalized").mkdir()
        result = write_normalized(df, run_dir)
        assert result.exists()

    def test_artifact_manifest(self, tmp_dir):
        run_dir = ensure_run_dir(str(tmp_dir), "manifest_test")
        manifest = write_artifact_manifest(run_dir)
        assert manifest.exists()
