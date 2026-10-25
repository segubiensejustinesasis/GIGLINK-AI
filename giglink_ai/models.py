"""Scoring models that rank seekers for each gig.

Every model returns a score matrix ``S`` of shape (n_gigs, n_seekers) where a
larger ``S[g, s]`` means seeker ``s`` is a better match for gig ``g``.

Models
------
random_scores    : baseline, uniform random scores.
keyword_scores   : count of shared (non-stopword) words, no weighting.
jaccard_scores   : Jaccard similarity of gig tags vs declared seeker tags.
tfidf_scores     : cosine similarity of TF-IDF vectors (vector space model).
lsa_scores       : cosine similarity after truncated SVD (latent semantics).
hybrid_scores    : weighted combination of LSA, Jaccard and rating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .data_gen import split_set


def random_scores(n_gigs: int, n_seekers: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n_gigs, n_seekers))


def keyword_scores(gig_text: pd.Series, seeker_text: pd.Series) -> np.ndarray:
    """Number of distinct words shared by the gig and the seeker bio."""
    vec = CountVectorizer(stop_words="english", binary=True)
    vec.fit(pd.concat([gig_text, seeker_text]))
    G = vec.transform(gig_text)
    S = vec.transform(seeker_text)
    return (G @ S.T).toarray().astype(float)


def jaccard_scores(gig_tags: pd.Series, seeker_tags: pd.Series) -> np.ndarray:
    """J(A, B) = |A ∩ B| / |A ∪ B| between required and declared skill tags.

    Vectorised with binary tag matrices A (gigs x skills) and B (seekers x
    skills): |A ∩ B| = A B^T and |A ∪ B| = |A| + |B| - |A ∩ B|.
    """
    vocab = sorted(set().union(*map(split_set, gig_tags), *map(split_set, seeker_tags)))
    col = {t: i for i, t in enumerate(vocab)}

    def binary(tags: pd.Series) -> np.ndarray:
        M = np.zeros((len(tags), len(vocab)))
        for r, x in enumerate(tags):
            for t in split_set(x):
                M[r, col[t]] = 1.0
        return M

    A, B = binary(gig_tags), binary(seeker_tags)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def fit_tfidf(gig_text: pd.Series, seeker_text: pd.Series):
    """Fit TF-IDF on the joint corpus and return (vectorizer, G, S).

    Weights follow w_{t,d} = (1 + log tf_{t,d}) * (1 + log((1+N)/(1+df_t)))
    and each document vector is L2-normalised, so a dot product is a cosine.
    """
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, norm="l2")
    vec.fit(pd.concat([gig_text, seeker_text]))
    return vec, vec.transform(gig_text), vec.transform(seeker_text)


def tfidf_scores(gig_text: pd.Series, seeker_text: pd.Series) -> np.ndarray:
    _, G, S = fit_tfidf(gig_text, seeker_text)
    return (G @ S.T).toarray()


def lsa_scores(
    gig_text: pd.Series,
    seeker_text: pd.Series,
    k: int = 40,
    n_iter: int = 7,
    seed: int = 0,
) -> np.ndarray:
    """Latent Semantic Analysis similarity.

    The TF-IDF matrix X (documents x terms) is approximated by a rank-k
    truncated SVD X ≈ U_k Σ_k V_k^T. Documents are projected to the k-dim
    latent space (rows of U_k Σ_k), re-normalised, and compared by cosine.
    ``n_iter`` is the number of power iterations of the randomized SVD.
    """
    _, G, S = fit_tfidf(gig_text, seeker_text)
    from scipy.sparse import vstack

    X = vstack([G, S])
    k = min(k, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=seed)
    Z = normalize(svd.fit_transform(X))
    Zg, Zs = Z[: G.shape[0]], Z[G.shape[0] :]
    return Zg @ Zs.T


def rating_feature(ratings: pd.Series, prior: float = 3.5) -> np.ndarray:
    """Map ratings in [1,5] to [0,1]; cold-start seekers get a neutral prior."""
    r = ratings.fillna(prior).to_numpy(dtype=float)
    return (r - 1.0) / 4.0


def hybrid_scores(
    text_sim: np.ndarray,
    tag_sim: np.ndarray,
    rating01: np.ndarray,
    w_text: float = 0.6,
    w_rating: float = 0.05,
) -> np.ndarray:
    """Hybrid score H = w_text*text + (1-w_text)*tag + w_rating*rating.

    The text and tag terms form a convex combination; the rating term is a
    small additive bonus so that reputation breaks ties between equally
    skilled seekers without dominating skill fit.
    """
    return w_text * text_sim + (1.0 - w_text) * tag_sim + w_rating * rating01[None, :]


def apply_feasibility(scores: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Push infeasible pairs to the bottom of every ranking (hard constraint)."""
    out = scores.astype(float).copy()
    out[~feasible] = -np.inf
    return out
