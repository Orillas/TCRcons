"""Consensus clustering: co-association, graph, modes, weights."""

from .coassociation import extract_pairwise_support, build_coassociation_matrix
from .graph import build_consensus_graph, connected_components_clustering, community_clustering
from .modes import conservative_consensus, balanced_consensus, coverage_consensus
from .weights import compute_method_weights

__all__ = [
    "extract_pairwise_support", "build_coassociation_matrix",
    "build_consensus_graph", "connected_components_clustering", "community_clustering",
    "conservative_consensus", "balanced_consensus", "coverage_consensus",
    "compute_method_weights",
]
