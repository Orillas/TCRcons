"""Tests for backend detection on a minimal (core-only) install.

Verifies that ``is_available()`` and ``available_methods()`` correctly
report only the in-tree clusterers when no optional backends are installed.
"""

from __future__ import annotations

import pytest

from tcrconsensus import TCRConsensus, ALL_METHODS, available_methods
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.clusterers.levenshtein import LevenshteinClusterer


class TestIsAvailable:
    """is_available() on each wrapper — core methods always True,
    external backends False under a minimal install."""

    def test_hd_baseline_always_available(self):
        assert HDBaselineClusterer.is_available() is True

    def test_levenshtein_always_available(self):
        assert LevenshteinClusterer.is_available() is True

    def test_clustcr_not_available(self):
        from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
        assert ClusTCRWrapper.is_available() is False

    def test_tcrdist3_not_available(self):
        from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
        assert TCRDist3Wrapper.is_available() is False

    def test_deeptcr_not_available(self):
        from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper
        assert DeepTCRWrapper.is_available() is False

    def test_giana_not_available(self):
        from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
        assert GIANAWrapper.is_available() is False

    def test_gliph2_not_available(self):
        from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
        assert GLIPH2Wrapper.is_available() is False

    def test_tcrmatch_not_available(self):
        from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
        assert TCRMatchWrapper.is_available() is False


class TestAvailableMethods:
    """available_methods() returns only core methods on minimal install."""

    def test_module_level_function(self):
        methods = available_methods()
        assert isinstance(methods, list)
        assert "hd_baseline" in methods
        assert "levenshtein" in methods

    def test_only_core_methods_returned(self):
        """Under a core-only install, only the two in-tree methods appear."""
        methods = available_methods()
        for ext in ("giana", "gliph2", "clustcr", "tcrmatch", "tcrdist3", "deeptcr"):
            assert ext not in methods, f"{ext} should not be available"

    def test_tcrconsensus_instance(self):
        model = TCRConsensus()
        assert model.available_methods == available_methods()

    def test_all_methods_constant_contains_everything(self):
        """ALL_METHODS is the full list regardless of install."""
        for m in ("hd_baseline", "levenshtein", "giana", "gliph2",
                  "clustcr", "tcrmatch", "tcrdist3", "deeptcr"):
            assert m in ALL_METHODS
