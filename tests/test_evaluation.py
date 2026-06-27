"""Tests for evaluation metrics."""

import pytest
import numpy as np

from tcrconsensus.evaluation.metrics import (
    retention, purity, sensitivity, f1_score, ari, nmi, compute_all_metrics,
)


class TestRetention:
    def test_full(self):
        assert retention(100, 100) == 1.0

    def test_half(self):
        assert retention(50, 100) == 0.5

    def test_zero_total(self):
        assert retention(0, 0) == 0.0


class TestPurity:
    def test_perfect(self):
        pred = np.array([0, 0, 1, 1, 2, 2])
        true = np.array([0, 0, 1, 1, 2, 2])
        assert purity(pred, true) == 1.0

    def test_partial(self):
        pred = np.array([0, 0, 0, 1, 1, 1])
        true = np.array([0, 0, 1, 1, 2, 2])
        p = purity(pred, true)
        assert 0 < p < 1

    def test_empty(self):
        assert purity(np.array([]), np.array([])) == 0.0


class TestSensitivity:
    def test_perfect(self):
        pred = np.array([0, 0, 1, 1])
        true = np.array([0, 0, 1, 1])
        assert sensitivity(pred, true) == 1.0

    def test_partial(self):
        pred = np.array([0, 0, 0, 1])
        true = np.array([0, 0, 1, 1])
        s = sensitivity(pred, true)
        assert 0 < s <= 1


class TestF1:
    def test_harmonic_mean(self):
        f = f1_score(0.8, 0.8)
        assert abs(f - 0.8) < 1e-10

    def test_zero(self):
        assert f1_score(0, 0) == 0.0


class TestARI:
    def test_perfect(self):
        pred = np.array([0, 0, 1, 1])
        true = np.array([0, 0, 1, 1])
        assert ari(pred, true) == 1.0

    def test_random(self):
        pred = np.array([0, 1, 0, 1])
        true = np.array([0, 0, 1, 1])
        assert ari(pred, true) < 1.0


class TestNMI:
    def test_perfect(self):
        pred = np.array([0, 0, 1, 1])
        true = np.array([0, 0, 1, 1])
        assert nmi(pred, true) == 1.0


class TestComputeAll:
    def test_all_keys(self):
        pred = np.array([0, 0, 1, 1, 2, 2])
        true = np.array([0, 0, 1, 1, 2, 2])
        result = compute_all_metrics(pred, true, 6)
        assert "retention" in result
        assert "purity" in result
        assert "sensitivity" in result
        assert "f1" in result
        assert "ari" in result
        assert "nmi" in result
