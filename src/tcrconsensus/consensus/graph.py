"""Consensus graph construction and clustering."""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np

from ..schema.records import ConsensusEdge, ConsensusCluster

logger = logging.getLogger(__name__)


def build_consensus_graph(
    edges: list[ConsensusEdge],
    threshold: float = 0.3,
) -> nx.Graph:
    """Build networkx graph from consensus edges above threshold."""
    G = nx.Graph()
    for edge in edges:
        if edge.final_score >= threshold:
            G.add_edge(
                edge.tcr_id_a,
                edge.tcr_id_b,
                weight=edge.final_score,
                method_support=edge.method_support_count,
            )
    return G


def connected_components_clustering(graph: nx.Graph) -> list[ConsensusCluster]:
    """Extract clusters as connected components (conservative mode)."""
    clusters = []
    for i, component in enumerate(nx.connected_components(graph)):
        members = sorted(component)
        clusters.append(
            ConsensusCluster(
                cluster_id=f"cons_{i:04d}",
                member_ids=members,
                core_member_ids=[],  # filled by refinement
                peripheral_member_ids=[],
                cluster_confidence=0.0,
                supporting_methods=[],
            )
        )
    return clusters


def community_clustering(
    graph: nx.Graph,
    algorithm: str = "leiden",
    resolution: float = 1.0,
) -> list[ConsensusCluster]:
    """Extract clusters via community detection (balanced mode).

    Falls back to Louvain if leidenalg not available.
    """
    if len(graph.edges) == 0:
        return connected_components_clustering(graph)

    try:
        if algorithm == "leiden":
            import leidenalg
            import igraph as ig

            ig_graph = ig.Graph.from_networkx(graph)
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                weights="weight",
            )
            clusters = []
            for i, community in enumerate(partition):
                members = [str(v["_nx_name"]) if "_nx_name" in v.attributes() else (str(v["name"]) if "name" in v.attributes() else str(v.index))
                           for v in ig_graph.vs[community]]
                clusters.append(
                    ConsensusCluster(
                        cluster_id=f"cons_{i:04d}",
                        member_ids=sorted(members),
                    )
                )
            return clusters
    except ImportError:
        logger.warning("leidenalg not available, falling back to Louvain")

    # Louvain via networkx community
    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(graph, resolution=resolution, weight="weight")
        clusters = []
        for i, community in enumerate(communities):
            clusters.append(
                ConsensusCluster(
                    cluster_id=f"cons_{i:04d}",
                    member_ids=sorted(community),
                )
            )
        return clusters

    except ImportError:
        logger.warning("Louvain not available, falling back to connected components")
        return connected_components_clustering(graph)
