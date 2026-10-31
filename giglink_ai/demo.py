"""Command-line demo: type a gig and see which real resumes GigLink would suggest.

Example::

    python -m giglink_ai.demo --text "Need a logo and brand kit for a small coffee shop" \
        --tags "logo design;graphic design;adobe illustrator"

The seeker pool is the 2,483 real resumes in data/raw/Resume.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .models import hybrid_scores, jaccard_scores, lsa_scores
from .real_data import _base

# Most frequently selected configuration in results/hybrid_selected_params.csv
DEFAULT_W_TEXT = 0.2


def recommend(gig: dict, seekers: pd.DataFrame, top: int = 5, w_text=DEFAULT_W_TEXT) -> pd.DataFrame:
    gigs = pd.DataFrame([gig])
    text = lsa_scores(gigs.description, seekers.bio, k=40)
    tags = jaccard_scores(gigs.required_skills.str.lower(), seekers.declared_skills)
    H = hybrid_scores(text, tags, np.zeros(len(seekers)), w_text, 0.0)[0]
    best = np.argsort(-H)[:top]
    out = seekers.iloc[best][["seeker_id", "category"]].copy()
    out.insert(1, "score", H[best].round(3))
    out.insert(2, "text_sim", text[0, best].round(3))
    out.insert(3, "tag_jaccard", tags[0, best].round(3))
    out["resume_start"] = seekers.bio.iloc[best].str.slice(0, 70).to_numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default="Need a logo and brand kit for a small coffee shop opening next month")
    ap.add_argument("--tags", default="logo design;graphic design;adobe illustrator;brand identity",
                    help="';'-separated Upwork-style skill tags")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    _, seekers, _, _ = _base()
    gig = {"gig_id": "DEMO", "description": a.text, "required_skills": a.tags}
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(f"Gig: {a.text}\nTags: {a.tags}\n")
    print(recommend(gig, seekers, a.top).to_string(index=False))


if __name__ == "__main__":
    main()
