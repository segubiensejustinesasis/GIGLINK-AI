"""All experiments reported in the paper.

Run from the repository root with::

    python -m giglink_ai.experiments            # full run (~1-3 minutes)
    python -m giglink_ai.experiments --quick    # fewer seeds, for a smoke test

Outputs
-------
data/                synthetic dataset for seed 0 (CSV, for inspection)
results/*.csv|json   every number quoted in the paper
figures/*.pdf|png    every figure in the paper
"""

from __future__ import annotations

import argparse
import json
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
from .data_gen import DataConfig, feasibility_matrix, generate, ground_truth_relevance
from .metrics import all_metrics, ndcg_at_k
from .models import (
    apply_feasibility,
    fit_tfidf,
    hybrid_scores,
    jaccard_scores,
    keyword_scores,
    lsa_scores,
    random_scores,
    rating_feature,
    tfidf_scores,
)

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"

# Hyper-parameter grid searched on the VALIDATION gigs only.
W_TEXT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
W_RATING_GRID = [0.0, 0.05, 0.1, 0.2]
LSA_K = 40
SVD_ITERS = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build(cfg: DataConfig):
    """Generate one dataset and compute every base score matrix."""
    seekers, gigs = generate(cfg)
    feas = feasibility_matrix(seekers, gigs)
    rel = ground_truth_relevance(seekers, gigs, feas)
    base = {
        "Random": random_scores(len(gigs), len(seekers), seed=cfg.seed + 999),
        "Keyword": keyword_scores(gigs.description, seekers.bio),
        "Jaccard (tags)": jaccard_scores(gigs.required_skills, seekers.declared_skills),
        "TF-IDF": tfidf_scores(gigs.description, seekers.bio),
        "LSA": lsa_scores(gigs.description, seekers.bio, k=LSA_K, n_iter=SVD_ITERS, seed=cfg.seed),
    }
    return seekers, gigs, feas, rel, base


def tune_hybrid(base, rating01, feas, rel, idx):
    """Grid search (text source, w_text, w_rating) maximising validation nDCG@10."""
    best, table = None, []
    for text in ("TF-IDF", "LSA"):
        for wt in W_TEXT_GRID:
            for wr in W_RATING_GRID:
                H = hybrid_scores(base[text], base["Jaccard (tags)"], rating01, wt, wr)
                score = ndcg_at_k(apply_feasibility(H, feas)[idx], rel[idx], 10)
                table.append({"text": text, "w_text": wt, "w_rating": wr, "val_nDCG@10": score})
                if best is None or score > best[0] + 1e-12:
                    best = (score, text, float(wt), float(wr))
    return best, pd.DataFrame(table)


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, half


# ---------------------------------------------------------------------------
# E1  Main comparison + E5 weight grid + E6 assignment, over many seeds
# ---------------------------------------------------------------------------
def run_main(seeds):
    rows, chosen, assign_rows, ablation_rows = [], [], [], []
    grid_seed0 = None
    for seed in seeds:
        cfg = DataConfig(seed=seed)
        seekers, gigs, feas, rel, base = build(cfg)
        r01 = rating_feature(seekers.rating)
        perm = np.random.default_rng(seed).permutation(len(gigs))
        val, test = perm[: len(gigs) // 2], perm[len(gigs) // 2 :]

        (val_score, text, wt, wr), grid = tune_hybrid(base, r01, feas, rel, val)
        if seed == seeds[0]:
            grid_seed0 = grid
        chosen.append({"seed": seed, "text": text, "w_text": wt, "w_rating": wr, "val_nDCG@10": val_score})
        hybrid = hybrid_scores(base[text], base["Jaccard (tags)"], r01, wt, wr)
        methods = dict(base, Hybrid=hybrid)

        for name, sc in methods.items():
            m = all_metrics(apply_feasibility(sc, feas)[test], rel[test])
            rows.append({"seed": seed, "method": name, **m})

        # Ablations of the tuned hybrid (test gigs).
        ab = {
            "Hybrid (full)": apply_feasibility(hybrid, feas),
            "without feasibility filter": hybrid,
            "without rating term": apply_feasibility(hybrid_scores(base[text], base["Jaccard (tags)"], r01, wt, 0.0), feas),
            "text only (w_text = 1)": apply_feasibility(hybrid_scores(base[text], base["Jaccard (tags)"], r01, 1.0, wr), feas),
            "tags only (w_text = 0)": apply_feasibility(hybrid_scores(base[text], base["Jaccard (tags)"], r01, 0.0, wr), feas),
        }
        for name, sc in ab.items():
            ablation_rows.append({"seed": seed, "variant": name, **all_metrics(sc[test], rel[test])})

        # Assignment on ALL gigs, in two market conditions.
        for market, n_s in (("surplus (400 seekers)", len(seekers)), ("scarce (150 seekers)", 150)):
            H, F, R = hybrid[:, :n_s], feas[:, :n_s], rel[:, :n_s]
            for pol, fn in (("Optimal (Hungarian)", optimal_assignment), ("Greedy", greedy_assignment), ("Independent top-1", independent_top1)):
                t0 = time.perf_counter()
                pairs = fn(H, F)
                dt = time.perf_counter() - t0
                assign_rows.append({"seed": seed, "market": market, "policy": pol, "seconds": dt, **evaluate_assignment(pairs, H, R)})

        if seed == seeds[0]:
            DATA.mkdir(exist_ok=True)
            seekers.to_csv(DATA / "synthetic_seekers_seed0.csv", index=False)
            gigs.to_csv(DATA / "synthetic_gigs_seed0.csv", index=False)
            pd.DataFrame(rel, index=gigs.gig_id, columns=seekers.seeker_id).to_csv(DATA / "ground_truth_relevance_seed0.csv")
        print(f"  seed {seed}: hybrid text={text} w_text={wt} w_rating={wr}")

    return (pd.DataFrame(rows), pd.DataFrame(chosen), pd.DataFrame(assign_rows),
            pd.DataFrame(ablation_rows), grid_seed0)


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
def run_lsa_sweep(seeds, ks=(5, 10, 20, 40, 80, 120, 160)):
    rows = []
    for seed in seeds:
        seekers, gigs = generate(DataConfig(seed=seed))
        feas = feasibility_matrix(seekers, gigs)
        rel = ground_truth_relevance(seekers, gigs, feas)
        tf = all_metrics(apply_feasibility(tfidf_scores(gigs.description, seekers.bio), feas), rel)["nDCG@10"]
        for k in ks:
            sc = lsa_scores(gigs.description, seekers.bio, k=k, n_iter=SVD_ITERS, seed=seed)
            rows.append({"seed": seed, "k": k, "nDCG@10": all_metrics(apply_feasibility(sc, feas), rel)["nDCG@10"], "tfidf_nDCG@10": tf})
    return pd.DataFrame(rows)


def run_svd_convergence(seed=0, k=LSA_K, iters=range(0, 11)):
    """Relative error of randomized-SVD singular values vs. an exact solver."""
    seekers, gigs = generate(DataConfig(seed=seed))
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
# E4  Robustness to vocabulary noise and tag noise
# ---------------------------------------------------------------------------
def run_noise(seeds, params):
    rows = []
    levels = {"synonym_prob": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "tag_drop_prob": [0.0, 0.15, 0.3, 0.45, 0.6]}
    for field, values in levels.items():
        for v in values:
            for seed in seeds:
                cfg = DataConfig(seed=seed, **{field: v})
                seekers, gigs, feas, rel, base = build(cfg)
                r01 = rating_feature(seekers.rating)
                base["Hybrid"] = hybrid_scores(base[params["text"]], base["Jaccard (tags)"], r01, params["w_text"], params["w_rating"])
                for name, sc in base.items():
                    rows.append({"factor": field, "level": v, "seed": seed, "method": name,
                                 "nDCG@10": all_metrics(apply_feasibility(sc, feas), rel)["nDCG@10"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E7  Runtime scaling
# ---------------------------------------------------------------------------
def run_scaling(sizes=(100, 200, 400, 800, 1600, 3200), reps=3):
    rows = []
    for n in sizes:
        seekers, gigs = generate(DataConfig(n_gigs=n, n_seekers=n, seed=0))
        feas = feasibility_matrix(seekers, gigs)
        for rep in range(reps):
            t0 = time.perf_counter(); sc = tfidf_scores(gigs.description, seekers.bio); t1 = time.perf_counter()
            jac = jaccard_scores(gigs.required_skills, seekers.declared_skills); t2 = time.perf_counter()
            H = hybrid_scores(sc, jac, rating_feature(seekers.rating), 0.5, 0.05)
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
    for ax, m in zip(axes, ["P@5", "R@10", "nDCG@10", "MRR"]):
        y = np.arange(len(ORDER))
        ax.barh(y, s[f"{m}_mean"], xerr=s[f"{m}_ci95"], color=[plotting.METHOD_COLORS[k] for k in ORDER],
                height=0.65, error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#52514e"})
        for yi, v, h in zip(y, s[f"{m}_mean"], s[f"{m}_ci95"]):
            ax.text(v + h + 0.03, yi, f"{v:.3f}", va="center", fontsize=7, color="#0b0b0b")
        ax.set_title(m); ax.set_xlim(0, 1.15); ax.set_yticks(y, ORDER); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    fig.suptitle("Test-set ranking quality (mean ± 95% CI over seeds, synthetic data)", y=1.03, fontsize=10)
    return fig


def fig_lsa(df):
    import matplotlib.pyplot as plt
    g = df.groupby("k")["nDCG@10"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.errorbar(g.k, g["mean"], yerr=g["std"], marker="D", color=plotting.METHOD_COLORS["LSA"], capsize=2, label="LSA (rank k)")
    ax.axhline(df["tfidf_nDCG@10"].mean(), color=plotting.METHOD_COLORS["TF-IDF"], ls="--", lw=1.5, label="TF-IDF (full rank)")
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


def fig_noise(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    labels = {"synonym_prob": "synonym probability (vocabulary mismatch)", "tag_drop_prob": "tag drop probability (incomplete profiles)"}
    for ax, f in zip(axes, ["synonym_prob", "tag_drop_prob"]):
        sub = df[df.factor == f]
        for m in ORDER:
            g = sub[sub.method == m].groupby("level")["nDCG@10"].mean()
            ax.plot(g.index, g.values, marker=plotting.MARKERS[m], color=plotting.METHOD_COLORS[m], label=m,
                    lw=2.6 if m == "Hybrid" else 1.6)
        ax.set_xlabel(labels[f]); ax.set_title("Robustness: " + f.replace("_", " "))
    axes[0].set_ylabel("nDCG@10")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def fig_grid(grid, text):
    import matplotlib.pyplot as plt
    sub = grid[grid.text == text].pivot(index="w_rating", columns="w_text", values="val_nDCG@10")
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    im = ax.imshow(sub.values, aspect="auto", cmap="Blues", origin="lower")
    ax.set_xticks(range(len(sub.columns)), [f"{c:.1f}" for c in sub.columns])
    ax.set_yticks(range(len(sub.index)), [f"{r:.2f}" for r in sub.index])
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v > sub.values.min() + 0.6 * (sub.values.max() - sub.values.min()) else "#0b0b0b")
    ax.set_xlabel(r"text weight $\alpha$"); ax.set_ylabel(r"rating weight $\gamma$"); ax.grid(False)
    ax.set_title(f"Validation nDCG@10 (seed 0, text = {text})")
    fig.colorbar(im, ax=ax, fraction=0.04)
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
    ax.set_xlabel("number of random seeds R"); ax.set_ylabel("running mean nDCG@10")
    ax.set_title("Convergence of the Monte Carlo estimate"); ax.legend()
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
    ap.add_argument("--quick", action="store_true", help="3 seeds and smaller sweeps (smoke test)")
    args = ap.parse_args(argv)
    plotting.setup()
    RESULTS.mkdir(exist_ok=True)
    n_main = 3 if args.quick else 20
    n_side = 2 if args.quick else 5
    seeds = list(range(n_main))
    side = list(range(n_side))
    t_start = time.perf_counter()

    print(f"[E1] main comparison over {n_main} seeds")
    per_seed, chosen, assign, ablation, grid0 = run_main(seeds)
    per_seed.to_csv(RESULTS / "main_metrics_per_seed.csv", index=False)
    chosen.to_csv(RESULTS / "hybrid_selected_params.csv", index=False)
    grid0.to_csv(RESULTS / "weight_grid_seed0.csv", index=False)
    metrics = ["P@5", "R@10", "nDCG@10", "MRR"]
    summary = summarise(per_seed, "method", metrics)
    summary.to_csv(RESULTS / "main_metrics_summary.csv", index=False)
    ablation_sum = summarise(ablation, "variant", metrics)
    ablation_sum.to_csv(RESULTS / "ablation_summary.csv", index=False)
    assign.to_csv(RESULTS / "assignment_per_seed.csv", index=False)
    assign_sum = assign.groupby(["market", "policy"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    assign_sum.to_csv(RESULTS / "assignment_summary.csv", index=False)
    tests = [paired_tests(per_seed, "Hybrid", b, m) for b in ("Jaccard (tags)", "TF-IDF") for m in ("nDCG@10", "P@5")]
    pd.DataFrame(tests).to_csv(RESULTS / "significance_tests.csv", index=False)

    # Most common tuned configuration -> used for the noise sweep.
    mode = chosen.groupby(["text", "w_text", "w_rating"]).size().idxmax()
    params = {"text": mode[0], "w_text": float(mode[1]), "w_rating": float(mode[2])}

    print("[E2] LSA rank sweep"); lsa = run_lsa_sweep(side); lsa.to_csv(RESULTS / "lsa_rank_sweep.csv", index=False)
    print("[E3] randomized SVD convergence"); svd_df, svd_meta = run_svd_convergence(); svd_df.to_csv(RESULTS / "svd_convergence.csv", index=False)
    print("[E4] noise robustness"); noise = run_noise(side, params); noise.to_csv(RESULTS / "noise_robustness.csv", index=False)
    print("[E7] runtime scaling")
    scaling = run_scaling(sizes=(100, 200, 400, 800) if args.quick else (100, 200, 400, 800, 1600, 3200))
    scaling.to_csv(RESULTS / "runtime_scaling.csv", index=False)

    save(fig_main(summary), "fig_main_metrics")
    save(fig_lsa(lsa), "fig_lsa_rank")
    save(fig_svd(svd_df), "fig_svd_convergence")
    save(fig_noise(noise), "fig_noise_robustness")
    save(fig_grid(grid0, chosen.iloc[0]["text"]), "fig_weight_grid")
    fs, slopes = fig_scaling(scaling); save(fs, "fig_runtime_scaling")
    save(fig_mc(per_seed), "fig_monte_carlo")

    seekers0, gigs0 = generate(DataConfig(seed=0))
    feas0 = feasibility_matrix(seekers0, gigs0)
    rel0 = ground_truth_relevance(seekers0, gigs0, feas0)
    meta = {
        "config": DataConfig().__dict__, "n_main_seeds": n_main, "n_side_seeds": n_side,
        "lsa_k": LSA_K, "svd_n_iter": SVD_ITERS, "w_text_grid": W_TEXT_GRID.tolist(), "w_rating_grid": W_RATING_GRID,
        "selected_params_mode": params, "selected_params_counts": chosen.groupby(["text", "w_text", "w_rating"]).size().rename("count").reset_index().to_dict("records"),
        "svd": svd_meta, "runtime_loglog_slopes_n>=400": slopes,
        "dataset_seed0": {"feasible_fraction": float(feas0.mean()),
                          "relevance_distribution": (np.bincount(rel0.ravel(), minlength=4) / rel0.size).tolist(),
                          "mean_relevant_per_gig_rel>=2": float((rel0 >= 2).sum(1).mean()),
                          "cold_start_fraction": float(seekers0.rating.isna().mean()),
                          "vocabulary_size": int(len(fit_tfidf(gigs0.description, seekers0.bio)[0].vocabulary_))},
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
