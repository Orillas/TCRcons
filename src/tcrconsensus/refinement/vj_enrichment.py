"""Tier-2 innovation 6: V/J gene enrichment (hypergeometric).

For a consensus cluster, test whether its V/J gene usage is enriched relative
to the dataset background — a biological coherence signal.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _safe_hypergeom_sf(k, K, n, N) -> float:
    """scipy hypergeom survival function (>= k successes) with safe fallback."""
    try:
        from scipy.stats import hypergeom
        # P(X >= k) = sf(k-1)
        return float(hypergeom.sf(max(k - 1, 0), N, K, n))
    except Exception:
        # crude fallback: assume independence, p ~= (K/N)^n binomial-ish
        if N <= 0 or K <= 0:
            return 1.0
        p = K / N
        from math import comb
        tot = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))
        return float(min(1.0, max(0.0, tot)))


def vj_enrichment(
    cluster_v: list[str],
    cluster_j: list[str],
    background_v: Counter | dict[str, int],
    background_j: Counter | dict[str, int],
) -> dict:
    """Return -log10(p) for the most-enriched V and J in the cluster, plus the
    identity of the enriched genes. Higher = more biologically coherent."""
    n = len(cluster_v)
    N_v = sum(background_v.values()) if background_v else 0
    N_j = sum(background_j.values()) if background_j else 0

    def _enrich(genes, bg, N):
        if not genes or not bg or N == 0:
            return {"gene": None, "neglog10p": 0.0, "frac": 0.0}
        cnt = Counter(genes)
        best_gene, best_p, best_frac = None, 1.0, 0.0
        for g, k in cnt.items():
            K = bg.get(g, 0)
            if K == 0:
                continue
            p = _safe_hypergeom_sf(k, K, n, N)
            if p < best_p:
                best_p, best_gene, best_frac = p, g, k / n
        import math
        return {
            "gene": best_gene,
            "neglog10p": float(-math.log10(best_p)) if best_p > 0 else 99.0,
            "frac": best_frac,
        }

    v_enr = _enrich(cluster_v, background_v, N_v)
    j_enr = _enrich(cluster_j, background_j, N_j)
    return {"v": v_enr, "j": j_enr, "n": n}
