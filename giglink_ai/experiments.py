"""All experiments reported in the paper, on real Upwork job posts and real resumes.

Put the two Kaggle files in data/raw/ first (see README.md), then run from the
repository root::

    python -m giglink_ai.experiments            # full run (a few minutes)
    python -m giglink_ai.experiments --quick    # fewer seeds, for a smoke test

Outputs
-------
results/*.csv|json   every number quoted in the paper
figures/*.pdf|png    every figure in the paper
paper/tables/*.tex   LaTeX tables (via make_tables)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import vstack
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD

from . import plotting
from .assignment import evaluate_assignment, greedy_assignment, independent_top1, optimal_assignment
from .metrics import all_metrics, ndcg_at_k
from .models import (
    fit_tfidf,
    hybrid_scores,
    jaccard_scores,
    keyword_scores,
    lsa_scores,
    random_scores,
    split_set,
    tfidf_scores,
)
from .real_data import RealConfig, _base, data_available, load_gigs, sample

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"

# Hyper-parameter grid searched on the VALIDATION gigs only.
W_TEXT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
LSA_K = 40
SVD_ITERS = 7
N_SCARCE = 300  # seekers in the "scarce" assignment market


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def base_scores(seekers, gigs, seed, k=LSA_K):
    """Every base score matrix (n_gigs x n_seekers) for one sample."""
    return {
        "Random": random_scores(len(gigs), len(seekers), seed=seed + 999),
        "Keyword": keyword_scores(gigs.description, seekers.bio),
        "Jaccard (tags)": jaccard_scores(gigs.required_skills, seekers.declared_skills),
        "TF-IDF": tfidf_scores(gigs.description, seekers.bio),
        "LSA": lsa_scores(gigs.description, seekers.bio, k=k, n_iter=SVD_ITERS, seed=seed),
    }


def tune_hybrid(base, rel, idx):
    """Grid search (text source, w_text) maximising validation nDCG@10."""
    zero = np.zeros(base["TF-IDF"].shape[1])
    best, table = None, []
    for text in ("TF-IDF", "LSA"):
        for wt in W_TEXT_GRID:
            H = hybrid_scores(base[text], base["Jaccard (tags)"], zero, wt, 0.0)
            score = ndcg_at_k(H[idx], rel[idx], 10)
            table.append({"text": text, "w_text": wt, "val_nDCG@10": score})
            if best is None or score > best[0] + 1e-12:
                best = (score, text, float(wt))
    return best, pd.DataFrame(table)


def hybrid(base, text, wt):
    return hybrid_scores(base[text], base["Jaccard (tags)"], np.zeros(base[text].shape[1]), wt, 0.0)


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, half


# ---------------------------------------------------------------------------
# E1  Main comparison + E5 weight grid + E6 assignment, over many random samples
# ---------------------------------------------------------------------------
def run_main(seeds):
    rows, chosen, assign_rows, ablation_rows, per_cat = [], [], [], [], []
    grid_seed0 = None
    for seed in seeds:
        seekers, gigs, rel = sample(RealConfig(seed=seed))
        base = base_scores(seekers, gigs, seed)
        perm = np.random.default_rng(seed).permutation(len(gigs))
        val, test = perm[: len(gigs) // 2], perm[len(gigs) // 2 :]

        (val_score, text, wt), grid = tune_hybrid(base, rel, val)
        if seed == seeds[0]:
            grid_seed0 = grid
        chosen.append({"seed": seed, "text": text, "w_text": wt, "val_nDCG@10": val_score})
        H = hybrid(base, text, wt)
        methods = dict(base, Hybrid=H)

        for name, sc in methods.items():
            rows.append({"seed": seed, "method": name, **all_metrics(sc[test], rel[test])})

        # nDCG@10 of each test gig, to see which kinds of work are easy or hard.
        from .metrics import ndcg_at_k as _n
        for g in test:
            per_cat.append({"seed": seed, "category": gigs.category[g],
                            "Hybrid": _n(H[[g]], rel[[g]], 10), "TF-IDF": _n(base["TF-IDF"][[g]], rel[[g]], 10),
                            "Jaccard (tags)": _n(base["Jaccard (tags)"][[g]], rel[[g]], 10)})

        # Ablations of the hybrid (test gigs).
        other = "LSA" if text == "TF-IDF" else "TF-IDF"
        ab = {
            f"Hybrid (tuned: {text})": H,
            f"Hybrid with {other} text, re-tuned": hybrid(base, other, float(grid[grid.text == other].sort_values("val_nDCG@10").iloc[-1].w_text)),
            "text only ($\\alpha = 1$)": hybrid(base, text, 1.0),
            "tags only ($\\alpha = 0$)": hybrid(base, text, 0.0),
            "equal weights ($\\alpha = 0.5$)": hybrid(base, text, 0.5),
        }
        for name, sc in ab.items():
            key = "Hybrid (tuned)" if name.startswith("Hybrid (tuned") else ("Hybrid, other text model" if name.startswith("Hybrid with") else name)
            ablation_rows.append({"seed": seed, "variant": key, **all_metrics(sc[test], rel[test])})

        # Assignment of ALL sampled gigs, in two market conditions.
        feas = np.ones_like(H, dtype=bool)
        sub = np.sort(np.random.default_rng(seed + 7).choice(len(seekers), N_SCARCE, replace=False))
        for market, cols in ((f"surplus ({len(seekers)} seekers)", np.arange(len(seekers))), (f"scarce ({N_SCARCE} seekers)", sub)):
            Hm, Fm, Rm = H[:, cols], feas[:, cols], rel[:, cols]
            for pol, fn in (("Optimal (Hungarian)", optimal_assignment), ("Greedy", greedy_assignment), ("Independent top-1", independent_top1)):
                t0 = time.perf_counter()
                pairs = fn(Hm, Fm)
                dt = time.perf_counter() - t0
                assign_rows.append({"seed": seed, "market": market, "policy": pol, "seconds": dt, **evaluate_assignment(pairs, Hm, Rm)})
        print(f"  seed {seed}: hybrid text={text} w_text={wt}", flush=True)

    return (pd.DataFrame(rows), pd.DataFrame(chosen), pd.DataFrame(assign_rows),
            pd.DataFrame(ablation_rows), grid_seed0, pd.DataFrame(per_cat))


def summarise(df, by, metrics):
    out = []
    for key, g in df.groupby(by, sort=False):
        row = {by: key, "n_seeds": len(g)}
        for m in metrics:
            mu, sd, half = mean_ci(g[m])
            row[f"{m}_mean"], row[f"{m}_std"], row[f"{m}_ci95"] = mu, sd, half
        out.append(row)
    return pd.DataFrame(out)


def paired_tests(df, a="Hybrid", b="Jaccard (tags)", metric="nDCG@10"):
    x = df[df.method == a].sort_values("seed")[metric].to_numpy()
    y = df[df.method == b].sort_values("seed")[metric].to_numpy()
    t = stats.ttest_rel(x, y)
    w = stats.wilcoxon(x, y)
    return {"a": a, "b": b, "metric": metric, "mean_diff": float((x - y).mean()),
            "t_stat": float(t.statistic), "t_p": float(t.pvalue), "wilcoxon_stat": float(w.statistic),
            "wilcoxon_p": float(w.pvalue), "a_wins": int((x > y).sum()), "n": int(len(x))}


# ---------------------------------------------------------------------------
# E2  LSA dimension sweep   E3  randomized SVD convergence
# ---------------------------------------------------------------------------
def run_lsa_sweep(seeds, ks=(10, 20, 40, 80, 160, 320)):
    rows = []
    for seed in seeds:
        seekers, gigs, rel = sample(RealConfig(seed=seed))
        tf = all_metrics(tfidf_scores(gigs.description, seekers.bio), rel)["nDCG@10"]
        for k in ks:
            sc = lsa_scores(gigs.description, seekers.bio, k=k, n_iter=SVD_ITERS, seed=seed)
            rows.append({"seed": seed, "k": k, "nDCG@10": all_metrics(sc, rel)["nDCG@10"], "tfidf_nDCG@10": tf})
    return pd.DataFrame(rows)


def run_svd_convergence(seed=0, k=LSA_K, iters=range(0, 11)):
    """Relative error of randomized-SVD singular values vs. an exact solver."""
    seekers, gigs, _ = sample(RealConfig(seed=seed))
    _, G, S = fit_tfidf(gigs.description, seekers.bio)
    X = vstack([G, S]).tocsr().astype(float)
    exact = np.sort(svds(X, k=k, tol=0, random_state=seed)[1])[::-1]  # ARPACK, tol=0 -> machine precision
    fro = np.sqrt(X.multiply(X).sum())
    rows = []
    for n_iter in iters:
        for rep in range(5):
            svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=100 + rep).fit(X)
            sv = svd.singular_values_
            rows.append({"n_iter": n_iter, "rep": rep,
                         "max_rel_err": float(np.max(np.abs(sv - exact) / exact)),
                         "mean_rel_err": float(np.mean(np.abs(sv - exact) / exact)),
                         "explained_var": float(svd.explained_variance_ratio_.sum())})
    meta = {"matrix_shape": list(X.shape), "nnz": int(X.nnz), "k": k,
            "exact_low_rank_fro_err": float(np.sqrt(max(fro**2 - (exact**2).sum(), 0)) / fro),
            "sigma_1": float(exact[0]), "sigma_k": float(exact[-1])}
    return pd.DataFrame(rows), meta


# ---------------------------------------------------------------------------
# E4  Robustness: shorter resumes and missing skill tags (perturbing real data)
# ---------------------------------------------------------------------------
def _truncate(bio: pd.Series, n_words: int | None) -> pd.Series:
    return bio if n_words is None else bio.str.split().str[:n_words].str.join(" ")


def _drop_tags(tags: pd.Series, p: float, rng) -> pd.Series:
    return tags.map(lambda x: ";".join(t for t in sorted(split_set(x)) if rng.random() >= p))


def run_robustness(seeds, params):
    rows = []
    levels = {"resume_words": [25, 50, 100, 200, 400, None], "tag_drop_prob": [0.0, 0.2, 0.4, 0.6, 0.8]}
    for field, values in levels.items():
        for v in values:
            for seed in seeds:
                seekers, gigs, rel = sample(RealConfig(seed=seed))
                seekers = seekers.copy()
                if field == "resume_words":
                    seekers["bio"] = _truncate(seekers.bio, v)
                else:
                    seekers["declared_skills"] = _drop_tags(seekers.declared_skills, v, np.random.default_rng(seed + 31))
                base = base_scores(seekers, gigs, seed)
                base["Hybrid"] = hybrid(base, params["text"], params["w_text"])
                for name, sc in base.items():
                    rows.append({"factor": field, "level": "all" if v is None else v, "seed": seed, "method": name,
                                 "nDCG@10": all_metrics(sc, rel)["nDCG@10"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E7  Runtime scaling
# ---------------------------------------------------------------------------
def run_scaling(sizes=(100, 200, 400, 800, 1600, 2400), reps=3):
    rows = []
    for n in sizes:
        seekers, gigs, _ = sample(RealConfig(n_gigs=n, n_seekers=n, seed=0))
        feas = np.ones((n, n), dtype=bool)
        for rep in range(reps):
            t0 = time.perf_counter(); sc = tfidf_scores(gigs.description, seekers.bio); t1 = time.perf_counter()
            jac = jaccard_scores(gigs.required_skills, seekers.declared_skills); t2 = time.perf_counter()
            H = hybrid_scores(sc, jac, np.zeros(n), 0.5, 0.0)
            t3 = time.perf_counter(); optimal_assignment(H, feas); t4 = time.perf_counter()
            greedy_assignment(H, feas); t5 = time.perf_counter()
            rows.append({"n": n, "rep": rep, "TF-IDF scoring": t1 - t0, "Jaccard scoring": t2 - t1,
                         "Hungarian assignment": t4 - t3, "Greedy assignment": t5 - t4})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
ORDER = ["Random", "Keyword", "Jaccard (tags)", "TF-IDF", "LSA", "Hybrid"]


def fig_main(summary):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    s = summary.set_index("method").loc[ORDER]
    for ax, m in zip(axes, ["P@5", "R@100", "nDCG@10", "MRR"]):
        y = np.arange(len(ORDER))
        ax.barh(y, s[f"{m}_mean"], xerr=s[f"{m}_ci95"], color=[plotting.METHOD_COLORS[k] for k in ORDER],
                height=0.65, error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#52514e"})
        for yi, v, h in zip(y, s[f"{m}_mean"], s[f"{m}_ci95"]):
            ax.text(v + h + 0.03, yi, f"{v:.3f}", va="center", fontsize=7, color="#0b0b0b")
        ax.set_title(m); ax.set_xlim(0, 1.15); ax.set_yticks(y, ORDER); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    fig.suptitle("Test-set ranking quality on real Upwork gigs and resumes (mean ± 95% CI over samples)", y=1.03, fontsize=10)
    return fig


def fig_lsa(df):
    import matplotlib.pyplot as plt
    g = df.groupby("k")["nDCG@10"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.errorbar(g.k, g["mean"], yerr=g["std"], marker="D", color=plotting.METHOD_COLORS["LSA"], capsize=2, label="LSA (rank k)")
    ax.axhline(df["tfidf_nDCG@10"].mean(), color=plotting.METHOD_COLORS["TF-IDF"], ls="--", lw=1.5, label="TF-IDF (full rank)")
    ax.set_xscale("log")
    ax.set_xlabel("latent dimension k"); ax.set_ylabel("nDCG@10"); ax.set_title("Effect of LSA rank k"); ax.legend()
    return fig


def fig_svd(df):
    import matplotlib.pyplot as plt
    g = df.groupby("n_iter")[["max_rel_err", "mean_rel_err"]].agg(["mean", "max"])
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.semilogy(g.index, g[("max_rel_err", "mean")], marker="o", color=plotting.PALETTE[0], label="max_i relative error")
    ax.semilogy(g.index, g[("mean_rel_err", "mean")], marker="s", color=plotting.PALETTE[1], label="mean_i relative error")
    ax.axhline(1e-3, color="#8a8984", ls=":", lw=1.2, label="tolerance 1e-3")
    ax.set_xlabel("power iterations q (n_iter)"); ax.set_ylabel(r"$|\hat\sigma_i-\sigma_i|/\sigma_i$")
    ax.set_title(f"Randomized SVD convergence (k = {LSA_K})"); ax.legend()
    return fig


def fig_robustness(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    labels = {"resume_words": "words kept from each resume (log scale)", "tag_drop_prob": "probability of deleting each seeker tag"}
    for ax, f in zip(axes, ["resume_words", "tag_drop_prob"]):
        sub = df[df.factor == f].copy()
        if f == "resume_words":
            full = int(np.ceil(_base()[1].bio.str.split().str.len().max()))
            sub["x"] = sub.level.replace("all", full).astype(float)
        else:
            sub["x"] = sub.level.astype(float)
        for m in ORDER:
            g = sub[sub.method == m].groupby("x")["nDCG@10"].mean()
            ax.plot(g.index, g.values, marker=plotting.MARKERS[m], color=plotting.METHOD_COLORS[m], label=m,
                    lw=2.6 if m == "Hybrid" else 1.6)
        if f == "resume_words":
            ax.set_xscale("log")
        ax.set_xlabel(labels[f]); ax.set_title("Robustness: " + ("shorter resumes" if f == "resume_words" else "missing tags"))
    axes[0].set_ylabel("nDCG@10")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def fig_grid(grid):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3))
    for text, col in (("TF-IDF", plotting.METHOD_COLORS["TF-IDF"]), ("LSA", plotting.METHOD_COLORS["LSA"])):
        g = grid[grid.text == text]
        ax.plot(g.w_text, g["val_nDCG@10"], marker="o", color=col, label=f"Hybrid with {text} text")
    ax.set_xlabel(r"text weight $\alpha$ (0 = tags only, 1 = text only)"); ax.set_ylabel("validation nDCG@10")
    ax.set_title("Choosing the hybrid weight (sample 0)"); ax.legend()
    return fig


def fig_scaling(df):
    import matplotlib.pyplot as plt
    g = df.groupby("n").median(numeric_only=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    cols = {"TF-IDF scoring": plotting.PALETTE[2], "Jaccard scoring": plotting.PALETTE[1],
            "Hungarian assignment": plotting.PALETTE[0], "Greedy assignment": plotting.PALETTE[6]}
    slopes = {}
    for c, col in cols.items():
        ax.loglog(g.index, g[c], marker="o", color=col, label=c)
        slopes[c] = float(np.polyfit(np.log(g.index[2:]), np.log(g[c].iloc[2:]), 1)[0])
    ax.set_xlabel("n gigs = n seekers"); ax.set_ylabel("seconds (median of 3)"); ax.set_title("Computational cost")
    ax.legend()
    return fig, slopes


def fig_mc(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3))
    for m in ["Hybrid", "Jaccard (tags)", "TF-IDF"]:
        x = df[df.method == m].sort_values("seed")["nDCG@10"].to_numpy()
        n = np.arange(1, len(x) + 1)
        mu = np.cumsum(x) / n
        sd = np.array([x[:i].std(ddof=1) if i > 1 else 0 for i in n])
        half = np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(n), np.nan)
        ax.plot(n, mu, color=plotting.METHOD_COLORS[m], label=m)
        ax.fill_between(n, mu - half, mu + half, color=plotting.METHOD_COLORS[m], alpha=0.15, lw=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("number of random samples R"); ax.set_ylabel("running mean nDCG@10")
    ax.set_title("Convergence of the Monte Carlo estimate"); ax.legend()
    return fig


def fig_categories(per_cat, top=15):
    import matplotlib.pyplot as plt
    g = per_cat.groupby("category").agg(n=("Hybrid", "size"), Hybrid=("Hybrid", "mean"), TFIDF=("TF-IDF", "mean"))
    g = g[g.n >= 20].sort_values("n", ascending=False).head(top).sort_values("Hybrid")
    fig, ax = plt.subplots(figsize=(5.6, 4))
    y = np.arange(len(g))
    ax.barh(y, g.Hybrid, color=plotting.METHOD_COLORS["Hybrid"], height=0.6, label="Hybrid")
    ax.scatter(g.TFIDF, y, color=plotting.METHOD_COLORS["TF-IDF"], zorder=3, s=14, label="TF-IDF")
    ax.set_yticks(y, [f"{c} ({n})" for c, n in zip(g.index, g.n)]); ax.set_xlim(0, 1)
    ax.set_xlabel("mean test nDCG@10"); ax.set_title("Ranking quality by Upwork category"); ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return fig


def save(fig, name):
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 samples and smaller sweeps (smoke test)")
    args = ap.parse_args(argv)
    if not data_available():
        sys.exit("Raw data not found. Download the two Kaggle files into data/raw/ (see README.md).")
    plotting.setup()
    RESULTS.mkdir(exist_ok=True)
    n_main = 3 if args.quick else 20
    n_side = 2 if args.quick else 5
    seeds = list(range(n_main))
    side = list(range(n_side))
    t_start = time.perf_counter()

    print(f"[E1] main comparison over {n_main} random samples", flush=True)
    per_seed, chosen, assign, ablation, grid0, per_cat = run_main(seeds)
    per_seed.to_csv(RESULTS / "main_metrics_per_seed.csv", index=False)
    chosen.to_csv(RESULTS / "hybrid_selected_params.csv", index=False)
    grid0.to_csv(RESULTS / "weight_grid_seed0.csv", index=False)
    metrics = ["P@5", "R@100", "nDCG@10", "MRR"]
    summary = summarise(per_seed, "method", metrics)
    summary.to_csv(RESULTS / "main_metrics_summary.csv", index=False)
    ablation_sum = summarise(ablation, "variant", metrics)
    ablation_sum.to_csv(RESULTS / "ablation_summary.csv", index=False)
    assign.to_csv(RESULTS / "assignment_per_seed.csv", index=False)
    assign_sum = assign.groupby(["market", "policy"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    assign_sum.to_csv(RESULTS / "assignment_summary.csv", index=False)
    cat_sum = per_cat.groupby("category").agg(n_test_gigs=("Hybrid", "size"), **{
        f"{m}_nDCG@10": (m, "mean") for m in ("Hybrid", "TF-IDF", "Jaccard (tags)")}).sort_values("n_test_gigs", ascending=False)
    cat_sum.to_csv(RESULTS / "per_category_ndcg.csv")
    tests = [paired_tests(per_seed, "Hybrid", b, m) for b in ("Jaccard (tags)", "TF-IDF", "LSA") for m in ("nDCG@10", "P@5")]
    pd.DataFrame(tests).to_csv(RESULTS / "significance_tests.csv", index=False)

    # Most common tuned configuration -> used for the robustness sweep.
    mode = chosen.groupby(["text", "w_text"]).size().idxmax()
    params = {"text": mode[0], "w_text": float(mode[1])}

    print("[E2] LSA rank sweep", flush=True); lsa = run_lsa_sweep(side); lsa.to_csv(RESULTS / "lsa_rank_sweep.csv", index=False)
    print("[E3] randomized SVD convergence", flush=True); svd_df, svd_meta = run_svd_convergence(); svd_df.to_csv(RESULTS / "svd_convergence.csv", index=False)
    print("[E4] robustness", flush=True); rob = run_robustness(side, params); rob.to_csv(RESULTS / "robustness.csv", index=False)
    print("[E7] runtime scaling", flush=True)
    scaling = run_scaling(sizes=(100, 200, 400, 800) if args.quick else (100, 200, 400, 800, 1600, 2400))
    scaling.to_csv(RESULTS / "runtime_scaling.csv", index=False)

    save(fig_main(summary), "fig_main_metrics")
    save(fig_lsa(lsa), "fig_lsa_rank")
    save(fig_svd(svd_df), "fig_svd_convergence")
    save(fig_robustness(rob), "fig_robustness")
    save(fig_grid(grid0), "fig_weight_grid")
    fs, slopes = fig_scaling(scaling); save(fs, "fig_runtime_scaling")
    save(fig_mc(per_seed), "fig_monte_carlo")
    save(fig_categories(per_cat), "fig_per_category")

    gigs_all, seekers_all, cmap, vocab = _base()
    s0, g0, rel0 = sample(RealConfig(seed=0))
    n_tags = seekers_all.declared_skills.map(lambda x: len(split_set(x)))
    meta = {
        "config": RealConfig().__dict__, "n_main_samples": n_main, "n_side_samples": n_side,
        "lsa_k": LSA_K, "svd_n_iter": SVD_ITERS, "w_text_grid": W_TEXT_GRID.tolist(), "n_scarce": N_SCARCE,
        "selected_params_mode": params,
        "selected_params_counts": chosen.groupby(["text", "w_text"]).size().rename("count").reset_index().to_dict("records"),
        "svd": svd_meta, "runtime_loglog_slopes_n>=400": slopes,
        "data": {
            "gigs_raw": int(len(load_gigs())), "gigs_after_mapping": int(len(gigs_all)), "upwork_categories_mapped": int(len(cmap)),
            "seekers": int(len(seekers_all)), "resume_categories": int(seekers_all.category.nunique()),
            "skill_vocabulary": int(len(vocab)),
            "gig_tags_median": float(gigs_all.required_skills.map(lambda x: len(split_set(x))).median()),
            "seeker_tags_median": float(n_tags.median()),
            "resume_words_median": float(seekers_all.bio.str.split().str.len().median()),
            "sample0_relevance_distribution": (np.bincount(rel0.ravel(), minlength=3) / rel0.size).tolist(),
            "sample0_mean_rel2_per_gig": float((rel0 >= 2).sum(1).mean()),
            "sample0_vocabulary_size": int(len(fit_tfidf(g0.description, s0.bio)[0].vocabulary_)),
        },
        "total_runtime_seconds": time.perf_counter() - t_start,
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    import scipy, sklearn, matplotlib
    meta.update({"scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__})
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    from .make_tables import main as make_tables
    make_tables()
    print(summary.round(3).to_string(index=False))
    print(f"done in {meta['total_runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
