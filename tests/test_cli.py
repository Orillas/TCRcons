"""Tests for CLI."""

import pytest
from click.testing import CliRunner
from tcrconsensus.cli.main import cli


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "TCR Consensus" in result.output

    def test_profile_command(self, sample_tsv, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["profile", sample_tsv])
        assert result.exit_code == 0
        assert "TCRs:" in result.output

    def test_run_command(self, sample_tsv, tmp_path):
        runner = CliRunner()
        output = str(tmp_path / "output")
        result = runner.invoke(cli, [
            "run", sample_tsv,
            "--methods", "hd_baseline",
            "--output", output,
        ])
        assert result.exit_code == 0
