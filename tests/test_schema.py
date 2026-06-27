"""Tests for schema module."""

import pytest
from tcrconsensus.schema.records import (
    TCRRecord, ChainMode, DatasetProfile, RunPlan,
    ClusterAssignment, ConsensusEdge, ConsensusCluster,
    Recommendation, Objective, ConsensusMode, MethodStatus,
)
from tcrconsensus.schema.validation import validate_tcr_record, validate_cdr3_basic


class TestTCRRecord:
    def test_defaults(self):
        r = TCRRecord(tcr_id="t1")
        assert r.chain_mode == ChainMode.BETA_ONLY
        assert r.count == 1
        assert r.metadata == {}

    def test_full_record(self):
        r = TCRRecord(
            tcr_id="t1", chain_mode=ChainMode.PAIRED_AB,
            cdr3_alpha="CAVFSGSNNQPLTF", cdr3_beta="CASSLAPGATNEKLFF",
            v_alpha="TRAV1-1", v_beta="TRBV1",
        )
        assert r.cdr3_alpha.startswith("C")
        assert r.cdr3_beta.startswith("C")


class TestValidation:
    def test_valid_beta(self):
        r = TCRRecord(tcr_id="t1", cdr3_beta="CASSLAPGATNEKLFF")
        errors = validate_tcr_record(r)
        assert len(errors) == 0

    def test_missing_cdr3_beta(self):
        r = TCRRecord(tcr_id="t1")
        errors = validate_tcr_record(r)
        assert any("cdr3_beta" in e for e in errors)

    def test_invalid_cdr3(self):
        r = TCRRecord(tcr_id="t1", cdr3_beta="INVALID")
        errors = validate_tcr_record(r)
        assert len(errors) > 0

    def test_cdr3_basic(self):
        assert validate_cdr3_basic("CASSLAPGATNEKLFF")
        assert not validate_cdr3_basic("")
        assert not validate_cdr3_basic("AB")
        assert not validate_cdr3_basic(None)


class TestEnums:
    def test_chain_mode(self):
        assert ChainMode.BETA_ONLY.value == "beta_only"
        assert ChainMode.PAIRED_AB.value == "paired_ab"

    def test_objective(self):
        assert Objective.BALANCED.value == "balanced"

    def test_consensus_mode(self):
        assert ConsensusMode.CONSERVATIVE.value == "conservative"

    def test_method_status(self):
        assert MethodStatus.SUCCESS.value == "success"


class TestConsensusEdge:
    def test_defaults(self):
        e = ConsensusEdge(tcr_id_a="a", tcr_id_b="b")
        assert e.method_support_count == 0
        assert e.final_score == 0.0


class TestConsensusCluster:
    def test_defaults(self):
        c = ConsensusCluster(cluster_id="c1", member_ids=["a", "b"])
        assert len(c.member_ids) == 2
        assert c.cluster_confidence == 0.0
