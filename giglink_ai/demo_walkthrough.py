"""GigLink AI matching demo on REAL Kaggle data (for the video presentation).

Walks through what the GigLink AI does, step by step:

  1. Load the real data: Upwork job posts (gigs) and LiveCareer resumes (seekers).
  2. Give every resume skill tags (Upwork skill phrases found in the resume text).
  3. A client posts a gig (default: milk tea shop pubmat poster + logo).
  4. Rank all resumes with the hybrid score  H = 0.2 * LSA text + 0.8 * Jaccard tags.
  5. Seeker view: the top resume sees the real Upwork gigs that fit it best.

Run from the repository root (needs data/raw/upwork-jobs.csv and data/raw/Resume.csv)::

    python -m giglink_ai.demo_walkthrough
    python -m giglink_ai.demo_walkthrough --text "Need someone to encode 500 receipts in Excel" \
        --tags "data entry;microsoft excel"

The category column is NOT used by the AI; it is shown only to check the answer
against our category table (data/category_map.csv).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from giglink_ai.demo import DEFAULT_W_TEXT, recommend
from giglink_ai.models import hybrid_scores, jaccard_scores, lsa_scores
from giglink_ai.real_data import ROOT, _base, load_category_map

SUMMARY = ROOT / "results" / "main_metrics_summary.csv"  # written by giglink_ai.experiments

LINE = "-" * 78


def step(n: int, title: str) -> None:
    print(f"\n[{n}/5] {title}")
    print(LINE)


def pause(sec: float) -> None:
    sys.stdout.flush()
    time.sleep(sec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default="Need a pubmat poster and logo for our milk tea shop opening this Saturday")
    ap.add_argument("--tags", default="graphic design;logo design;poster design;adobe photoshop")
    ap.add_argument("--category", default="Graphic Design",
                    help="Upwork category of the gig (only used to check the answer)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--pause", type=float, default=0.6, help="seconds between steps (for screen recording)")
    a = ap.parse_args(argv)

    print("=" * 78)
    print("  GigLink AI Matching Demo  |  REAL DATA (Kaggle)")
    print("=" * 78)

    step(1, "Loading real data")
    t0 = time.perf_counter()
    gigs, seekers, _, vocab = _base()
    print(f"  Gigs    : {len(gigs):,} real Upwork job posts (Dec 2023 - Feb 2024)")
    print(f"            (posts in categories covered by data/category_map.csv)")
    print(f"  Seekers : {len(seekers):,} real anonymized resumes ({seekers.category.nunique()} job categories)")
    print(f"  Loaded in {time.perf_counter() - t0:.1f} s")
    pause(a.pause)

    step(2, "Giving each resume skill tags")
    n_tags = seekers.declared_skills.str.count(";").add(1).where(seekers.declared_skills.ne(""), 0)
    print(f"  Skill vocabulary : {len(vocab):,} Upwork skill tags")
    print(f"  Tags per resume  : median {int(n_tags.median())}")
    pause(a.pause)

    step(3, "Client posts a gig")
    print(f"  Gig  : {a.text}")
    print(f"  Tags : {a.tags.replace(';', ', ')}")
    pause(a.pause)

    step(4, f"AI ranks all {len(seekers):,} resumes")
    print(f"  Text match : TF-IDF -> LSA (k = 40) -> cosine similarity")
    print(f"  Tag match  : Jaccard = shared tags / all tags")
    print(f"  Score      : H = {DEFAULT_W_TEXT:.1f} x text + {1 - DEFAULT_W_TEXT:.1f} x tags")
    t0 = time.perf_counter()
    gig = {"gig_id": "DEMO", "description": a.text, "required_skills": a.tags}
    top = recommend(gig, seekers, a.top)
    print(f"  Ranked in {time.perf_counter() - t0:.2f} s\n")

    cmap = load_category_map()
    primary, related = cmap.get(a.category, ("", set()))
    top["check"] = np.where(top.category == primary, "expected",
                            np.where(top.category.isin(related), "related", "-"))
    table = top.drop(columns="resume_start").copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    print(table.to_string(index=False))
    b = top.iloc[0]
    print(f"\n  #1 score: {DEFAULT_W_TEXT:.1f} x {b.text_sim:.3f} + {1 - DEFAULT_W_TEXT:.1f} x {b.tag_jaccard:.3f}"
          f" = {b.score:.3f}")
    ok = int((top.check != "-").sum())
    print(f"  Check   : {ok} of {len(top)} fit a '{a.category}' gig (expected: {primary})")
    pause(a.pause)

    step(5, f"Seeker view: resume {b.seeker_id} ({b.category}) sees matching gigs")
    me = seekers[seekers.seeker_id == b.seeker_id]
    t0 = time.perf_counter()
    text = lsa_scores(me.bio, gigs.description, k=40)
    tags = jaccard_scores(me.declared_skills, gigs.required_skills)
    H = hybrid_scores(text, tags, np.zeros(len(gigs)), DEFAULT_W_TEXT, 0.0)[0]
    best = np.argsort(-H)[:3]
    print(f"  Searched {len(gigs):,} gigs in {time.perf_counter() - t0:.1f} s\n")
    print(f"  {'score':>5}  {'upwork_category':<18}  gig_title")
    for i in best:
        print(f"  {H[i]:5.3f}  {gigs.category.iloc[i]:<18}  {gigs.title.iloc[i][:48]}")

    print("\n" + "=" * 78)
    if SUMMARY.exists():
        m = pd.read_csv(SUMMARY, index_col=0)["nDCG@10_mean"]
        print(f"  Paper result (20 random test sets): "
              f"hybrid nDCG@10 = {m['Hybrid']:.3f}")
        print(f"  vs LSA {m['LSA']:.3f}, TF-IDF {m['TF-IDF']:.3f}, tags only {m['Jaccard (tags)']:.3f}, "
              f"random {m['Random']:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
