"""Cluster refinement — split, merge, filter, label, score confidence.

Tier-2 extensions (5 + 6), all config-gated and backward-compatible:
  * Calibrated confidence: if config["refinement"]["calibrator"] is provided,
    cluster_confidence becomes calibrated P(pure) instead of raw mean edge score.
  * Motif split (config["refinement"]["use_motif_split"]): split a cluster whose
    CDR3s form two distinct motifs (high PWM KL between k-means subgroups).
  * Motif merge (config["refinement"]["use_motif_merge"]): REVIVES the otherwise-
    dead merge step using PWM similarity + shared V/J as the biological criterion
    (the old cross-cluster-edge-score criterion is unreachable because surviving
    edges are intra-cluster by construction).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..schema.records import ConsensusCluster, ConsensusEdge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional Tier-2 imports (lazy, so missing scipy/sklearn doesn't break baseline)
# ---------------------------------------------------------------------------
def _import_motif():
    try:
        from .motif import build_pwm, information_content, pwm_kl, motif_subgroups
        return build_pwm, information_content, pwm_kl, motif_subgroups
    except Exception as e:
        logger.debug(f"motif module unavailable: {e}")
        return None


def _build_lookups(tcr_table: Optional[pd.DataFrame]) -> dict:
    """tcr_id -> {cdr3_beta, v_beta, j_beta} if a table is supplied."""
    if tcr_table is None:
        return {}
    lut: dict[str, dict[str, str]] = {}
    tid_col = "tcr_id" if "tcr_id" in tcr_table.columns else None
    if tid_col is None:
        return {}
    for _, row in tcr_table.iterrows():
        tid = row[tid_col]
        lut[tid] = {
            "cdr3_beta": str(row.get("cdr3_beta", "") or ""),
            "v_beta": str(row.get("v_beta", "") or ""),
            "j_beta": str(row.get("j_beta", "") or ""),
        }
    return lut


# ---------------------------------------------------------------------------
def refine(
    clusters: list[ConsensusCluster],
    edges: list[ConsensusEdge],
    config: dict | None = None,
    tcr_table: Optional[pd.DataFrame] = None,
) -> list[ConsensusCluster]:
    """Refine consensus clusters: split, merge, filter, label."""
    config = config or {}
    refine_cfg = config.get("refinement", {})

    edge_map = _build_edge_map(edges)
    lookups = _build_lookups(tcr_table) if refine_cfg.get("use_motif_features") else {}
    calibrator = refine_cfg.get("calibrator")

    # Step 1: Score cluster confidence (calibrated if calibrator provided)
    for c in clusters:
        c.cluster_confidence = _score_cluster_confidence(c, edge_map, calibrator)

    # Step 2: Edge-score split (original)
    clusters = _split_clusters(clusters, edge_map, refine_cfg)

    # Step 3: Motif split (Tier-2/6) — finds bimodal-motif clusters the edge
    # signal misses.
    if refine_cfg.get("use_motif_split") and lookups:
        clusters = _motif_split(clusters, lookups, refine_cfg)

    # Step 4: Merge — motif merge (Tier-2/6, revived) if enabled, else original
    # cross-cluster-edge merge (which is effectively a no-op given threshold
    # semantics, but preserved for backward compatibility).
    if refine_cfg.get("use_motif_merge") and lookups:
        clusters = _motif_merge(clusters, lookups, refine_cfg)
    else:
        clusters = _merge_clusters(clusters, edge_map, refine_cfg)

    # Step 5: Filter weak members
    clusters = _filter_members(clusters, edge_map, refine_cfg)

    # Step 6: Label core vs peripheral
    clusters = _label_members(clusters, edge_map, refine_cfg)

    # Step 7: Recompute confidence
    for c in clusters:
        c.cluster_confidence = _score_cluster_confidence(c, edge_map, calibrator)

    return clusters


def _build_edge_map(edges: list[ConsensusEdge]) -> dict[tuple[str, str], ConsensusEdge]:
    m = {}
    for e in edges:
        key = tuple(sorted([e.tcr_id_a, e.tcr_id_b]))
        m[key] = e
    return m


def _score_cluster_confidence(
    cluster: ConsensusCluster,
    edge_map: dict[tuple[str, str], ConsensusEdge],
    calibrator=None,
) -> float:
    """Cluster confidence = calibrated(mean edge score) if a Calibrator is
    provided, else the raw mean edge score."""
    members = cluster.member_ids
    if len(members) < 2:
        return 0.0
    scores = []
    method_counts = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            key = tuple(sorted([members[i], members[j]]))
            edge = edge_map.get(key)
            if edge:
                scores.append(edge.final_score)
                method_counts.append(edge.method_support_count)
    if not scores:
        return 0.0
    raw = float(np.mean(scores))
    if calibrator is not None:
        try:
            return float(calibrator.predict(raw))
        except Exception:
            return raw
    return raw


# ---------------------------------------------------------------------------
# Original edge-score split (unchanged)
# ---------------------------------------------------------------------------
def _split_clusters(
    clusters: list[ConsensusCluster],
    edge_map: dict,
    config: dict,
) -> list[ConsensusCluster]:
    split_cfg = config.get("split", {})
    min_score = split_cfg.get("min_consensus_score", 0.2)

    result = []
    for cluster in clusters:
        if cluster.cluster_confidence >= min_score or len(cluster.member_ids) <= 3:
            result.append(cluster)
            continue
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(cluster.member_ids)
        for i, a in enumerate(cluster.member_ids):
            for b in cluster.member_ids[i + 1:]:
                key = tuple(sorted([a, b]))
                edge = edge_map.get(key)
                if edge and edge.final_score >= min_score:
                    G.add_edge(a, b, weight=edge.final_score)
        components = list(nx.connected_components(G))
        if len(components) <= 1:
            result.append(cluster)
            continue
        logger.info(f"Split cluster {cluster.cluster_id} into {len(components)} sub-clusters")
        for idx, comp in enumerate(components):
            result.append(
                ConsensusCluster(
                    cluster_id=f"{cluster.cluster_id}_s{idx}",
                    member_ids=sorted(comp),
                )
            )
    return result


# ---------------------------------------------------------------------------
# Tier-2/6: motif split — bimodal-motif clusters
# ---------------------------------------------------------------------------
def _motif_split(
    clusters: list[ConsensusCluster],
    lookups: dict[str, dict[str, str]],
    config: dict,
) -> list[ConsensusCluster]:
    """Split clusters whose CDR3s form two DISTINCT and each-more-coherent motifs.

    Gates on BOTH:
      (a) PWM KL between the two k-means subgroups >= min_kl  (motifs differ), AND
      (b) the size-weighted sum of subgroup IC EXCEEDS the original cluster's IC
          (the split actually sharpens motif coherence — prevents destructive
          splits of already-coherent clusters, which was the v1 regression).
    """
    mods = _import_motif()
    if mods is None:
        return clusters
    build_pwm, information_content, pwm_kl, motif_subgroups = mods
    ms_cfg = config.get("motif_split", {})
    min_kl = ms_cfg.get("min_kl", 2.5)            # bits — subgroups must differ strongly
    min_size = ms_cfg.get("min_cluster_size", 8)
    min_subgroup = ms_cfg.get("min_subgroup_size", 4)
    min_ic_gain = ms_cfg.get("min_ic_gain_frac", 0.20)  # >=20% IC improvement

    result = []
    for cluster in clusters:
        members = cluster.member_ids
        if len(members) < min_size:
            result.append(cluster)
            continue
        seqs = [lookups.get(m, {}).get("cdr3_beta", "") for m in members]
        seqs = [s for s in seqs if s]
        if len(seqs) < min_size:
            result.append(cluster)
            continue
        groups = motif_subgroups(seqs, k=2)
        if not groups or len(groups) < 2:
            result.append(cluster)
            continue
        if any(len(g) < min_subgroup for g in groups):
            result.append(cluster)
            continue
        sub_seqs = [[seqs[i] for i in g] for g in groups]
        pwms = [build_pwm(s) for s in sub_seqs]
        kl = pwm_kl(pwms[0], pwms[1])
        if kl < min_kl:
            result.append(cluster)
            continue
        # IC improvement gate
        ic_before = information_content(build_pwm(seqs)) * len(seqs)
        ic_after = sum(information_content(p) * len(s) for p, s in zip(pwms, sub_seqs))
        if ic_before <= 0 or (ic_after - ic_before) / ic_before < min_ic_gain:
            result.append(cluster)
            continue
        logger.info(
            f"Motif split {cluster.cluster_id} (KL={kl:.2f} bits, "
            f"IC {ic_before/len(seqs):.2f}->{ic_after/len(seqs):.2f} per seq) "
            f"into {len(groups)} motifs"
        )
        for idx, g in enumerate(groups):
            sub_members = sorted(members[i] for i in g if i < len(members))
            if len(sub_members) >= min_subgroup:
                result.append(ConsensusCluster(
                    cluster_id=f"{cluster.cluster_id}_m{idx}",
                    member_ids=sub_members,
                ))
    return result


# ---------------------------------------------------------------------------
# Original merge (effectively a no-op given threshold semantics; kept for compat)
# ---------------------------------------------------------------------------
def _merge_clusters(
    clusters: list[ConsensusCluster],
    edge_map: dict,
    config: dict,
) -> list[ConsensusCluster]:
    merge_cfg = config.get("merge", {})
    min_cross = merge_cfg.get("min_cross_association", 0.6)
    if len(clusters) <= 1:
        return clusters
    merged = set()
    result = []
    for i in range(len(clusters)):
        if i in merged:
            continue
        current = clusters[i]
        for j in range(i + 1, len(clusters)):
            if j in merged:
                continue
            cross_score = _cross_cluster_score(current, clusters[j], edge_map)
            if cross_score >= min_cross:
                current = ConsensusCluster(
                    cluster_id=current.cluster_id,
                    member_ids=sorted(set(current.member_ids + clusters[j].member_ids)),
                )
                merged.add(j)
                logger.info(f"Merged {clusters[j].cluster_id} into {current.cluster_id}")
        result.append(current)
    return result


def _cross_cluster_score(
    c1: ConsensusCluster, c2: ConsensusCluster, edge_map: dict,
) -> float:
    scores = []
    for a in c1.member_ids:
        for b in c2.member_ids:
            key = tuple(sorted([a, b]))
            edge = edge_map.get(key)
            if edge:
                scores.append(edge.final_score)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Tier-2/6: motif merge — REVIVES the dead merge with a biological criterion
# ---------------------------------------------------------------------------
def _motif_merge(
    clusters: list[ConsensusCluster],
    lookups: dict[str, dict[str, str]],
    config: dict,
) -> list[ConsensusCluster]:
    mods = _import_motif()
    if mods is None:
        return clusters
    build_pwm, _, pwm_kl, _ = mods
    mm_cfg = config.get("motif_merge", {})
    max_kl = mm_cfg.get("max_kl", 1.0)         # bits — motifs similar enough
    min_vj_overlap = mm_cfg.get("min_vj_overlap", 0.0)  # shared V or J fraction

    if len(clusters) <= 1:
        return clusters

    # Precompute each cluster's PWM + V/J sets
    info = []
    for c in clusters:
        seqs = [lookups.get(m, {}).get("cdr3_beta", "") for m in c.member_ids]
        seqs = [s for s in seqs if s]
        vs = {lookups.get(m, {}).get("v_beta", "") for m in c.member_ids}
        js = {lookups.get(m, {}).get("j_beta", "") for m in c.member_ids}
        info.append({
            "pwm": build_pwm(seqs) if len(seqs) >= 2 else None,
            "vs": vs, "js": js, "n": len(seqs),
        })

    merged = set()
    result = []
    for i in range(len(clusters)):
        if i in merged:
            continue
        current = clusters[i]
        cur_info = info[i]
        for j in range(i + 1, len(clusters)):
            if j in merged:
                continue
            oi, oj = info[i], info[j]
            if oi["pwm"] is None or oj["pwm"] is None:
                continue
            kl = pwm_kl(oi["pwm"], oj["pwm"])
            if kl > max_kl:
                continue
            # VJ overlap: fraction of shared V or J genes
            vj_inter = len(oi["vs"] & oj["vs"]) + len(oi["js"] & oj["js"])
            vj_union = max(1, len(oi["vs"] | oj["vs"]) + len(oi["js"] | oj["js"]))
            vj_overlap = vj_inter / vj_union
            if vj_overlap < min_vj_overlap:
                continue
            # Merge j into current
            logger.info(
                f"Motif merge {clusters[j].cluster_id} into {current.cluster_id} "
                f"(KL={kl:.2f}, VJ overlap={vj_overlap:.2f})"
            )
            current = ConsensusCluster(
                cluster_id=current.cluster_id,
                member_ids=sorted(set(current.member_ids + clusters[j].member_ids)),
            )
            merged.add(j)
        result.append(current)
    return result


def _filter_members(
    clusters: list[ConsensusCluster],
    edge_map: dict,
    config: dict,
) -> list[ConsensusCluster]:
    filter_cfg = config.get("filter", {})
    min_conf = filter_cfg.get("min_member_confidence", 0.1)
    result = []
    for cluster in clusters:
        if len(cluster.member_ids) <= 2:
            result.append(cluster)
            continue
        kept = []
        for member in cluster.member_ids:
            scores = []
            for other in cluster.member_ids:
                if other == member:
                    continue
                key = tuple(sorted([member, other]))
                edge = edge_map.get(key)
                if edge:
                    scores.append(edge.final_score)
            avg = float(np.mean(scores)) if scores else 0.0
            if avg >= min_conf:
                kept.append(member)
        if len(kept) >= 2:
            cluster.member_ids = sorted(kept)
        result.append(cluster)
    return result


def _label_members(
    clusters: list[ConsensusCluster],
    edge_map: dict,
    config: dict,
) -> list[ConsensusCluster]:
    conf_cfg = config.get("confidence", {})
    core_thresh = conf_cfg.get("core_threshold", 0.6)
    periph_thresh = conf_cfg.get("peripheral_threshold", 0.3)
    for cluster in clusters:
        core = []
        peripheral = []
        for member in cluster.member_ids:
            score = _member_confidence(member, cluster.member_ids, edge_map)
            if score >= core_thresh:
                core.append(member)
            elif score >= periph_thresh:
                peripheral.append(member)
        cluster.core_member_ids = core
        cluster.peripheral_member_ids = peripheral
    return clusters


def _member_confidence(
    member: str, all_members: list[str], edge_map: dict,
) -> float:
    scores = []
    for other in all_members:
        if other == member:
            continue
        key = tuple(sorted([member, other]))
        edge = edge_map.get(key)
        if edge:
            scores.append(edge.final_score)
    return float(np.mean(scores)) if scores else 0.0
