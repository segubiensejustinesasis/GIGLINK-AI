"""Unit tests for metrics, models and assignment. Run with:  python -m pytest -q"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from giglink_ai.assignment import greedy_assignment, optimal_assignment
from giglink_ai.data_gen import DataConfig, feasibility_matrix, generate, ground_truth_relevance
from giglink_ai.metrics import mrr, ndcg_at_k, precision_at_k
from giglink_ai.models import apply_feasibility, jaccard_scores, tfidf_scores


def test_metrics_perfect_and_worst_ranking():
    rel = np.array([[3, 2, 0, 0, 1]])
    perfect = np.array([[5.0, 4.0, 1.0, 0.0, 3.0]])
    assert ndcg_at_k(perfect, rel, k=5) == 1.0
    assert precision_at_k(perfect, rel, k=2) == 1.0
    assert mrr(perfect, rel) == 1.0
    worst = -perfect
    assert ndcg_at_k(worst, rel, k=5) < 0.6
    assert mrr(worst, rel) == 1 / 4  # first rel>=2 item is at rank 4


def test_ndcg_hand_computed():
    rel = np.array([[0, 3]])
    scores = np.array([[1.0, 0.0]])  # relevant item ranked second
    expected = (7 / np.log2(3)) / (7 / np.log2(2))
    assert np.isclose(ndcg_at_k(scores, rel, k=2), expected)


def test_jaccard_values():
    J = jaccard_scores(pd.Series(["a;b"]), pd.Series(["a;b", "a", "c", ""]))
    assert np.allclose(J, [[1.0, 0.5, 0.0, 0.0]])


def test_tfidf_cosine_bounds():
    S = tfidf_scores(pd.Series(["logo design poster"]), pd.Series(["logo design", "python scripting"]))
    assert S[0, 0] > S[0, 1] and np.all((S >= 0) & (S <= 1 + 1e-9))


def test_generator_is_reproducible():
    a, _ = generate(DataConfig(seed=7, n_gigs=5, n_seekers=5))
    b, _ = generate(DataConfig(seed=7, n_gigs=5, n_seekers=5))
    pd.testing.assert_frame_equal(a, b)


def test_relevance_zero_when_infeasible():
    s, g = generate(DataConfig(seed=1, n_gigs=30, n_seekers=40))
    F = feasibility_matrix(s, g)
    R = ground_truth_relevance(s, g, F)
    assert np.all(R[~F] == 0)


def test_optimal_beats_or_ties_greedy_and_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        H = rng.random((6, 8))
        F = rng.random((6, 8)) < 0.7
        opt = optimal_assignment(H, F)
        gre = greedy_assignment(H, F)
        assert len({s for _, s in opt}) == len(opt)  # one gig per seeker
        assert sum(H[g, s] for g, s in opt) >= sum(H[g, s] for g, s in gre) - 1e-12 or len(opt) > len(gre)
    # Fully feasible case: must equal SciPy's optimum exactly.
    H = rng.random((5, 5))
    r, c = linear_sum_assignment(H, maximize=True)
    opt = optimal_assignment(H, np.ones_like(H, bool))
    assert np.isclose(sum(H[g, s] for g, s in opt), H[r, c].sum())


def test_feasibility_filter_moves_infeasible_to_bottom():
    S = np.array([[0.9, 0.1]])
    F = np.array([[False, True]])
    assert np.argmax(apply_feasibility(S, F)) == 1
