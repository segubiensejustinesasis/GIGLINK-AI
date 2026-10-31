"""Turn the CSV files in results/ into LaTeX tables in paper/tables/.

Generating the tables from the result files (instead of typing numbers by
hand) keeps the paper and the experiments consistent. Run after
``python -m giglink_ai.experiments``::

    python -m giglink_ai.make_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper" / "tables"
METRICS = ["P@5", "R@100", "nDCG@10", "MRR"]


def _fmt(mu, ci, bold=False):
    s = f"{mu:.3f} $\\pm$ {ci:.3f}"
    return f"\\textbf{{{mu:.3f}}} $\\pm$ {ci:.3f}" if bold else s


def _metric_table(df, key, order, caption, label):
    df = df.set_index(key).loc[order]
    best = {m: df[f"{m}_mean"].max() for m in METRICS}
    lines = [
        "\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(METRICS) + "}", "\\toprule",
        "Method & " + " & ".join(m.replace("@", "@") for m in METRICS) + " \\\\", "\\midrule",
    ]
    for name, row in df.iterrows():
        name = name.replace("w_text", "$\\alpha$")
        cells = [_fmt(row[f"{m}_mean"], row[f"{m}_ci95"], abs(row[f"{m}_mean"] - best[m]) < 1e-12) for m in METRICS]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_sum = pd.read_csv(RES / "main_metrics_summary.csv")
    meta = json.loads((RES / "run_metadata.json").read_text())
    order = ["Random", "Keyword", "Jaccard (tags)", "TF-IDF", "LSA", "Hybrid"]
    (OUT / "main_metrics.tex").write_text(_metric_table(
        main_sum, "method", order,
        "Test-set ranking quality on real Upwork job posts and real resumes (mean $\\pm$ half-width of the 95\\% confidence interval over "
        f"{int(main_sum.n_seeds.iloc[0])} random samples; 200 test gigs $\\times$ {meta['data']['seekers']:,} resumes per sample). "
        "Best value per column in bold.", "tab:main"))

    ab = pd.read_csv(RES / "ablation_summary.csv")
    (OUT / "ablation.tex").write_text(_metric_table(
        ab, "variant", list(ab.variant),
        f"Ablation of the hybrid model (test gigs, {int(ab.n_seeds.iloc[0])} samples, mean $\\pm$ 95\\% CI). "
        "$\\alpha$ is the weight on text similarity; $1-\\alpha$ goes to tag overlap.", "tab:ablation"))

    asg = pd.read_csv(RES / "assignment_summary.csv")
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             f"\\caption{{Simultaneous assignment of all {meta['config']['n_gigs']} sampled gigs (mean over {int(main_sum.n_seeds.iloc[0])} samples). "
             "\\emph{True relevance} sums the category-based relevance of the chosen pairs; \\emph{good match rate} is the share of the gigs "
             "that receive a resume from the primary matching category ($\\mathrm{rel}=2$).}",
             "\\label{tab:assign}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{llrrrrr}", "\\toprule",
             "Market & Policy & Pred.\\ utility & True relevance & Good match rate & Gigs filled & Double-booked \\\\", "\\midrule"]
    for i, r in asg.iterrows():
        market = r.market if i % 3 == 0 else ""
        lines.append(f"{market} & {r.policy} & {r.predicted_utility:.2f} & {r.true_relevance_sum:.2f} & "
                     f"{r.good_match_rate:.3f} & {r.gigs_filled:.1f} & {r.double_booked:.1f} \\\\")
        if i == 2:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (OUT / "assignment.tex").write_text("\n".join(lines) + "\n")

    sig = pd.read_csv(RES / "significance_tests.csv")
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             f"\\caption{{Paired comparisons over the same {int(main_sum.n_seeds.iloc[0])} random samples (Hybrid minus baseline).}}", "\\label{tab:sig}",
             "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{llrrrr}", "\\toprule",
             "Baseline & Metric & Mean diff. & Paired $t$ ($p$) & Wilcoxon $p$ & Hybrid wins \\\\", "\\midrule"]
    for _, r in sig.iterrows():
        lines.append(f"{r.b} & {r.metric} & {r.mean_diff:+.3f} & {r.t_stat:.1f} ({r.t_p:.1e}) & {r.wilcoxon_p:.1e} & {r.a_wins}/{r.n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    (OUT / "significance.tex").write_text("\n".join(lines) + "\n")

    sc = pd.read_csv(RES / "runtime_scaling.csv").groupby("n").median(numeric_only=True)
    slopes = meta["runtime_loglog_slopes_n>=400"]
    cols = ["TF-IDF scoring", "Jaccard scoring", "Hungarian assignment", "Greedy assignment"]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Wall-clock time in seconds (median of 3 runs) for $n$ gigs and $n$ seekers. The last row is the slope of a "
             "least-squares line fitted to $\\log t$ versus $\\log n$ for $n \\ge 400$ (empirical complexity exponent).}",
             "\\label{tab:runtime}", "\\begin{tabular}{rrrrr}", "\\toprule",
             "$n$ & TF-IDF scoring & Jaccard scoring & Hungarian & Greedy \\\\", "\\midrule"]
    for n, r in sc.iterrows():
        lines.append(f"{n} & " + " & ".join(f"{r[c]:.4f}" for c in cols) + " \\\\")
    lines += ["\\midrule", "slope & " + " & ".join(f"{slopes[c]:.2f}" for c in cols) + " \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "runtime.tex").write_text("\n".join(lines) + "\n")

    svd = pd.read_csv(RES / "svd_convergence.csv").groupby("n_iter").mean(numeric_only=True)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             f"\\caption{{Randomized SVD versus exact ARPACK singular values ($k={meta['svd']['k']}$, "
             f"matrix {meta['svd']['matrix_shape'][0]}$\\times${meta['svd']['matrix_shape'][1]}, mean of 5 random starts).}}",
             "\\label{tab:svd}", "\\begin{tabular}{rccc}", "\\toprule",
             "$q$ & max rel.\\ error & mean rel.\\ error & explained variance \\\\", "\\midrule"]
    for q, r in svd.iterrows():
        if q in (0, 1, 2, 4, 6, 7, 8, 10):
            lines.append(f"{q} & {r.max_rel_err:.2e} & {r.mean_rel_err:.2e} & {r.explained_var:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "svd.tex").write_text("\n".join(lines) + "\n")
    d = meta["data"]
    macros = {
        "DataGigsRaw": f"{d['gigs_raw']:,}", "DataGigsMapped": f"{d['gigs_after_mapping']:,}",
        "DataMappedCats": str(d["upwork_categories_mapped"]), "DataSeekers": f"{d['seekers']:,}",
        "DataVocab": f"{d['skill_vocabulary']:,}", "DataGigTags": f"{d['gig_tags_median']:.0f}",
        "DataSeekerTags": f"{d['seeker_tags_median']:.0f}", "DataResumeWords": f"{d['resume_words_median']:,.0f}",
        "DataRelTwo": f"{d['sample0_mean_rel2_per_gig']:.0f}",
        "DataRelTwoPct": f"{100 * d['sample0_mean_rel2_per_gig'] / d['seekers']:.1f}\\%",
        "DataRuntimeMin": f"{meta['total_runtime_seconds'] / 60:.0f}",
    }
    (OUT / "data_macros.tex").write_text(
        "% Generated by giglink_ai/make_tables.py from results/run_metadata.json\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(f"wrote tables to {OUT}")


if __name__ == "__main__":
    main()
