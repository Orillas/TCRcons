"""Tests for refinement module."""

import pytest
from tcrconsensus.schema.records import ConsensusCluster, ConsensusEdge
from tcrconsensus.refinement.refiner import refine


def _make_test_data():
    clusters = [
        ConsensusCluster(cluster_id="c1", member_ids=["t1", "t2", "t3"]),
        ConsensusCluster(cluster_id="c2", member_ids=["t4", "t5"]),
    ]
    edges = [
        ConsensusEdge(tcr_id_a="t1", tcr_id_b="t2", final_score=0.9),
        ConsensusEdge(tcr_id_a="t1", tcr_id_b="t3", final_score=0.8),
        ConsensusEdge(tcr_id_a="t2", tcr_id_b="t3", final_score=0.85),
        ConsensusEdge(tcr_id_a="t4", tcr_id_b="t5", final_score=0.7),
    ]
    return clusters, edges


class TestRefine:
    def test_basic(self):
        clusters, edges = _make_test_data()
        result = refine(clusters, edges)
        assert len(result) >= 1
        for c in result:
            assert c.cluster_confidence > 0

    def test_labels(self):
        clusters, edges = _make_test_data()
        result = refine(clusters, edges)
        for c in result:
            total = len(c.core_member_ids) + len(c.peripheral_member_ids)
            assert total <= len(c.member_ids)
