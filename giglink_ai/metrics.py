"""Ranking metrics: Precision@k, Recall@k, nDCG@k and MRR.

All functions take a score matrix ``scores`` (n_gigs, n_seekers) and a graded
relevance matrix ``rel`` of the same shape, and return the mean over gigs.
A seeker counts as *relevant* when rel >= ``threshold`` (default 2, meaning
the seeker truly covers at least half of the required skills).
"""

from __future__ import annotations

import numpy as np


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first (ties: lower index)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :k]


def precision_at_k(scores, rel, k=5, threshold=2):
    top = _topk(scores, k)
    hits = np.take_along_axis(rel, top, axis=1) >= threshold
    return float(hits.mean())


def recall_at_k(scores, rel, k=10, threshold=2):
    top = _topk(scores, k)
    hits = (np.take_along_axis(rel, top, axis=1) >= threshold).sum(axis=1)
    n_rel = (rel >= threshold).sum(axis=1)
    mask = n_rel > 0
    return float((hits[mask] / n_rel[mask]).mean())


def ndcg_at_k(scores, rel, k=10):
    """nDCG@k with gain 2^rel - 1 and discount 1/log2(rank + 1)."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    top = _topk(scores, k)
    gains = 2.0 ** np.take_along_axis(rel, top, axis=1) - 1
    dcg = (gains * discounts).sum(axis=1)
    ideal = 2.0 ** (-np.sort(-rel, axis=1)[:, :k]) - 1
    idcg = (ideal * discounts).sum(axis=1)
    mask = idcg > 0
    return float((dcg[mask] / idcg[mask]).mean())


def mrr(scores, rel, threshold=2):
    """Mean reciprocal rank of the first relevant seeker."""
    order = np.argsort(-scores, axis=1, kind="stable")
    is_rel = np.take_along_axis(rel, order, axis=1) >= threshold
    has = is_rel.any(axis=1)
    first = is_rel.argmax(axis=1)
    return float((1.0 / (first[has] + 1)).mean())


def all_metrics(scores, rel) -> dict[str, float]:
    return {
        "P@5": precision_at_k(scores, rel, 5),
        "R@10": recall_at_k(scores, rel, 10),
        "nDCG@10": ndcg_at_k(scores, rel, 10),
        "MRR": mrr(scores, rel),
    }
