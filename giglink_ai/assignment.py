"""One-to-one assignment of seekers to gigs.

Ranking answers "who is good for this gig?". When many gigs are posted at
once, the platform should also avoid recommending the same top seeker to
every client. We model this as the linear assignment problem

    maximise   sum_{g,s} H[g,s] * x[g,s]
    subject to sum_s x[g,s] <= 1   (each gig gets at most one seeker)
               sum_g x[g,s] <= 1   (each seeker takes at most one gig)
               x[g,s] = 0 if (g,s) is infeasible,   x in {0,1}

and solve it exactly with SciPy's ``linear_sum_assignment`` (a shortest
augmenting path variant of the Hungarian / Jonker-Volgenant method). A greedy
heuristic and an "independent top-1" policy are used for comparison.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# Score given to forbidden pairs. Far below any real score (which lie in
# roughly [0, 1.1]) so the optimiser never picks them when a feasible choice
# exists; any such pair that is still chosen is dropped afterwards.
_FORBIDDEN = -1e6


def optimal_assignment(scores: np.ndarray, feasible: np.ndarray) -> list[tuple[int, int]]:
    H = np.where(feasible, scores, _FORBIDDEN)
    rows, cols = linear_sum_assignment(H, maximize=True)
    return [(g, s) for g, s in zip(rows, cols) if feasible[g, s]]


def greedy_assignment(scores: np.ndarray, feasible: np.ndarray) -> list[tuple[int, int]]:
    """Repeatedly take the highest-scoring remaining feasible pair."""
    H = np.where(feasible, scores, -np.inf)
    order = np.argsort(-H, axis=None, kind="stable")
    used_g, used_s, pairs = set(), set(), []
    n_s = H.shape[1]
    for flat in order:
        g, s = divmod(int(flat), n_s)
        if not np.isfinite(H[g, s]):
            break
        if g in used_g or s in used_s:
            continue
        pairs.append((g, s))
        used_g.add(g)
        used_s.add(s)
    return pairs


def independent_top1(scores: np.ndarray, feasible: np.ndarray) -> list[tuple[int, int]]:
    """Every gig takes its own best seeker; seekers may be double-booked."""
    H = np.where(feasible, scores, -np.inf)
    best = H.argmax(axis=1)
    return [(g, int(s)) for g, s in enumerate(best) if np.isfinite(H[g, s])]


def evaluate_assignment(pairs, scores, rel) -> dict[str, float]:
    """Summarise an assignment using predicted scores and TRUE relevance."""
    n_gigs = rel.shape[0]
    seekers = [s for _, s in pairs]
    double_booked = len(seekers) - len(set(seekers))
    return {
        "predicted_utility": float(sum(scores[g, s] for g, s in pairs)),
        "true_relevance_sum": float(sum(rel[g, s] for g, s in pairs)),
        "good_match_rate": float(sum(rel[g, s] >= 2 for g, s in pairs) / n_gigs),
        "gigs_filled": len(pairs),
        "double_booked": double_booked,
    }
