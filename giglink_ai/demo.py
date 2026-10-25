"""Interactive demo: type a gig and see the top seekers GigLink would suggest.

Example (run from the repository root, the folder that contains giglink_ai/)::

    python -m giglink_ai.demo --text "Need a logo and pubmat for our org event" --tags "logo_design,poster_design" --days "sat,sun" --hours 6 --budget 1800

Tags and days may be separated by commas or semicolons; quote them so the
shell does not treat ';' as the end of the command.

The seeker pool is the synthetic seed-0 dataset (no real users).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_gen import DAYS, SKILL_NAMES, DataConfig, feasibility_matrix, generate
    from .models import apply_feasibility, hybrid_scores, jaccard_scores, lsa_scores, rating_feature
except ImportError:
    # Run directly as a file (python giglink_ai/demo.py): make the package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from giglink_ai.data_gen import DAYS, SKILL_NAMES, DataConfig, feasibility_matrix, generate
    from giglink_ai.models import apply_feasibility, hybrid_scores, jaccard_scores, lsa_scores, rating_feature

# Most frequently selected configuration in results/hybrid_selected_params.csv
DEFAULT_W_TEXT, DEFAULT_W_RATING = 0.3, 0.0


def parse_list(value: str, allowed: list[str], what: str) -> str:
    """Accept 'a,b', 'a;b' or 'a b'; check each item; return the ';'-joined form the models expect."""
    items = [x.strip().lower() for x in re.split(r"[;,\s]+", value) if x.strip()]
    bad = [x for x in items if x not in allowed]
    if not items or bad:
        raise argparse.ArgumentTypeError(f"invalid {what}: {', '.join(bad) or '(empty)'}; choose from: {', '.join(allowed)}")
    return ";".join(items)


def recommend(gig: dict, seekers: pd.DataFrame, top: int = 5, w_text=DEFAULT_W_TEXT, w_rating=DEFAULT_W_RATING) -> pd.DataFrame:
    gigs = pd.DataFrame([gig])
    text = lsa_scores(gigs.description, seekers.bio, k=40)
    tags = jaccard_scores(gigs.required_skills, seekers.declared_skills)
    H = hybrid_scores(text, tags, rating_feature(seekers.rating), w_text, w_rating)
    H = apply_feasibility(H, feasibility_matrix(seekers, gigs))[0]
    best = np.argsort(-H)[:top]
    best = best[np.isfinite(H[best])]  # drop seekers who fail the day/budget constraints
    out = seekers.iloc[best][["seeker_id", "bio", "declared_skills", "hourly_rate_php", "rating"]].copy()
    out.insert(1, "score", H[best].round(3))
    out.insert(2, "text_sim", text[0, best].round(3))
    out.insert(3, "tag_jaccard", tags[0, best].round(3))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default="Small business needs a logo and flyer layout for a weekend bazaar")
    ap.add_argument("--tags", default="logo_design,poster_design",
                    type=lambda v: parse_list(v, SKILL_NAMES, "tag"),
                    help=f"comma-separated, from: {', '.join(SKILL_NAMES)}")
    ap.add_argument("--days", default="sat,sun", type=lambda v: parse_list(v, DAYS, "day"),
                    help=f"comma-separated, from: {', '.join(DAYS)}")
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--budget", type=int, default=1800)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    seekers, _ = generate(DataConfig(seed=0))
    gig = {"gig_id": "DEMO", "description": a.text, "required_skills": a.tags,
           "work_days": a.days, "est_hours": a.hours, "budget_php": a.budget}
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(f"Gig: {a.text}\nTags: {a.tags} | days: {a.days} | {a.hours} h | budget PHP {a.budget}\n")
    result = recommend(gig, seekers, a.top)
    if result.empty:
        print("No seeker shares a working day and fits the budget; try more days or a bigger --budget.")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
