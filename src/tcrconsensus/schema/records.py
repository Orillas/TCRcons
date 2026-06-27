"""Canonical data contracts for TCR Consensus Clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ChainMode(str, Enum):
    ALPHA_ONLY = "alpha_only"
    BETA_ONLY = "beta_only"
    PAIRED_AB = "paired_ab"


class RepertoireType(str, Enum):
    BULK = "bulk"
    SINGLE_CELL = "single_cell"
    ANTIGEN_ENRICHED = "antigen_enriched"
    CURATED_DB = "curated_db"


class Objective(str, Enum):
    HIGH_PURITY = "high_purity"
    BALANCED = "balanced"
    HIGH_RECALL = "high_recall"
    NOISE_ROBUST = "noise_robust"
    FAST_SCREENING = "fast_screening"
    PAIRED_CHAIN = "paired_chain_analysis"


class ConsensusMode(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    COVERAGE = "coverage"


class MemberLabel(str, Enum):
    CORE = "core"
    PERIPHERAL = "peripheral"
    LOW_CONFIDENCE = "low_confidence"


class MethodStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TCRRecord:
    tcr_id: str
    chain_mode: ChainMode = ChainMode.BETA_ONLY
    cdr3_alpha: Optional[str] = None
    cdr3_beta: Optional[str] = None
    v_alpha: Optional[str] = None
    j_alpha: Optional[str] = None
    v_beta: Optional[str] = None
    j_beta: Optional[str] = None
    subject_id: Optional[str] = None
    sample_id: Optional[str] = None
    epitope: Optional[str] = None
    hla: Optional[str] = None
    count: int = 1
    frequency: Optional[float] = None
    source_dataset: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    n_tcrs: int = 0
    chain_mode: ChainMode = ChainMode.BETA_ONLY
    vj_completeness: float = 0.0
    cdr3_length_summary: dict[str, float] = field(default_factory=dict)
    unique_ratio: float = 0.0
    clone_expansion_score: float = 0.0
    publicity_score: float = 0.0
    background_noise_score: float = 0.0
    label_availability: bool = False
    repertoire_type: RepertoireType = RepertoireType.BULK
    notes: list[str] = field(default_factory=list)


@dataclass
class RunPlan:
    objective: Objective = Objective.BALANCED
    selected_methods: list[str] = field(default_factory=list)
    consensus_mode: ConsensusMode = ConsensusMode.BALANCED
    method_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    weighting_profile: str = "balanced"
    use_tiered: bool = False                          # Tiered invocation: cheap methods on full data,
                                            # expensive methods only on divergent TCR subset.
                                            # Saves O(n²) compute for tcrdist3/deeptcr.
    refinement_params: dict[str, Any] = field(default_factory=dict)
    reporting_flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterAssignment:
    method: str
    tcr_id: str
    cluster_id: str
    membership_score: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    status: MethodStatus = MethodStatus.SUCCESS


@dataclass
class ConsensusEdge:
    tcr_id_a: str
    tcr_id_b: str
    method_support_count: int = 0
    weighted_support: float = 0.0           # sum of weights of methods that CO-clustered the pair (+)
    repulsion_support: float = 0.0          # Tier-2/4b: sum of weights of HIGH-PURITY methods that
                                            # SEPARATED the pair. Net evidence = weighted_support -
                                            # repulsion_support. A method "separates" (i,j) when it
                                            # assigns both TCRs but to different clusters.
    sequence_support: float = 0.0           # Tier-1/F2: exp(-TCRdist/τ) or BLOSUM62 fallback
    vj_support: float = 0.0                 # Tier-1/F2: shared V/J co-restriction in [0,1]
    noise_penalty: float = 0.0              # Tier-1/F2: -log10(p_null) vs permutation null
    final_score: float = 0.0                # score used for thresholding. Composition:
                                            #   default (Tier-2): weighted_support - repulsion_support
                                            #   Tier-1/F2 fusion: σ(β0 + β_vote·net_vote + β_seq·seq
                                            #                     + β_vj·vj + β_noise·noise)
                                            # where net_vote = weighted_support - repulsion_support.
                                            # The three signal fields above are filled by
                                            # consensus.fusion.enrich_and_fuse (0.0 until then).


@dataclass
class ConsensusCluster:
    cluster_id: str
    member_ids: list[str] = field(default_factory=list)
    core_member_ids: list[str] = field(default_factory=list)
    peripheral_member_ids: list[str] = field(default_factory=list)
    cluster_confidence: float = 0.0
    # Tier-1/F1(b) soft-overlap: affiliation strength of each member to THIS
    # community (BigCLAM F value). A TCR may appear in several clusters'
    # member_ids+membership => overlapping. Empty for hard clustering.
    membership: dict[str, float] = field(default_factory=dict)
    cluster_features: dict[str, Any] = field(default_factory=dict)
    supporting_methods: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    scenario: str = ""
    recommended_mode: ConsensusMode = ConsensusMode.BALANCED
    recommended_methods: list[str] = field(default_factory=list)
    expected_tradeoff: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    justification: str = ""
