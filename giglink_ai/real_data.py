"""Real-data loader: Upwork job posts (gigs) and LiveCareer resumes (seekers).

Two public Kaggle datasets are used (download links in README.md):

* ``data/raw/upwork-jobs.csv``  real Upwork job posts collected from the
  Upwork RSS feed (Dec 2023 - Feb 2024). The client's Upwork category and the
  Upwork skill tags are embedded in the ``description`` field and parsed here.
* ``data/raw/Resume.csv``  2,484 real resumes scraped from LiveCareer, each
  filed under one of 24 job categories (e.g. DESIGNER, ACCOUNTANT).

Ground truth
------------
There is no public record of who was actually hired for an Upwork post. We
therefore judge relevance through a category compatibility table written by
the authors, ``data/category_map.csv``: for every Upwork category it lists the
resume category that does that work (``primary``) and the categories that do
closely related work (``related``). Graded relevance is

    rel(g, s) = 2  if category(s) is the primary category for gig g,
                1  if category(s) is a related category,
                0  otherwise.

The table only uses the two category labels, which the models never see, so
the models are graded against an external judgement and not their own inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GIGS_CSV = RAW / "upwork-jobs.csv"
RESUMES_CSV = RAW / "Resume.csv"
CATEGORY_MAP = ROOT / "data" / "category_map.csv"

_PAY_MARKERS = r"(?:Hourly Range:|Budget:|Posted On:)"
_FIELD_END = r"(?:Skills:|Location Requirement:|Country:|click to apply)"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _job_text(desc: str) -> str:
    return re.split(_PAY_MARKERS, desc, maxsplit=1)[0].replace("\xa0", " ").strip()


def _category(desc: str) -> str:
    m = re.search(r"Category:\s*(.*?)" + _FIELD_END, desc)
    return m.group(1).strip() if m else ""


def _skills(desc: str) -> str:
    m = re.search(r"Skills:(.*?)" + _FIELD_END, desc, re.S)
    if not m:
        return ""
    tags = [t.strip().lower() for t in m.group(1).split(",")]
    return ";".join(dict.fromkeys(t for t in tags if t))


def load_gigs(path: Path = GIGS_CSV) -> pd.DataFrame:
    """Parse the raw Upwork file into one row per job post."""
    d = pd.read_csv(path)
    gigs = pd.DataFrame({
        "gig_id": d["link"].str.extract(r"(?:~|%7E)(\w+)")[0],
        "title": d["title"].str.strip(),
        "category": d["description"].map(_category),
        "required_skills": d["description"].map(_skills),
        "is_hourly": d["is_hourly"].astype(str).eq("True"),
        "hourly_low": d["hourly_low"],
        "hourly_high": d["hourly_high"],
        "budget": d["budget"],
        "country": d["country"].fillna(""),
        "body": d["description"].map(_job_text),
    })
    gigs["description"] = gigs["title"] + ". " + gigs["body"]
    return gigs.drop_duplicates("gig_id").reset_index(drop=True)


def load_resumes(path: Path = RESUMES_CSV) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["ID", "Resume_str", "Category"])
    bio = d["Resume_str"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    out = pd.DataFrame({"seeker_id": d["ID"].astype(str), "category": d["Category"], "bio": bio})
    return out[out.bio.str.len() >= 200].reset_index(drop=True)  # drops a handful of near-empty files


def load_category_map(path: Path = CATEGORY_MAP) -> dict[str, tuple[str, set[str]]]:
    m = pd.read_csv(path).fillna("")
    return {r.upwork_category: (r.primary_resume_category, {c for c in r.related_resume_categories.split(";") if c})
            for r in m.itertuples()}


# ---------------------------------------------------------------------------
# Seeker skill tags: Upwork skill phrases that occur in the resume text
# ---------------------------------------------------------------------------
def skill_vocabulary(gigs: pd.DataFrame, min_gigs: int = 5) -> set[str]:
    """Upwork skill tags (at most four words) used by at least ``min_gigs`` job posts."""
    counts = gigs.required_skills.str.split(";").explode().value_counts()
    return {t for t, c in counts.items() if c >= min_gigs and len(t) >= 2 and len(t.split()) <= 4}


def _ngrams(text: str, n_max: int) -> set[str]:
    toks = re.findall(r"[a-z0-9+#./&-]+", text.lower())
    toks = [t.strip(".-/") or t for t in toks]
    grams = set()
    for n in range(1, n_max + 1):
        grams.update(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return grams


def extract_tags(texts: pd.Series, vocab: set[str]) -> pd.Series:
    """Tag each resume with every vocabulary phrase it contains (exact n-gram match)."""
    n_max = max(len(v.split()) for v in vocab)
    return texts.map(lambda t: ";".join(sorted(_ngrams(t, n_max) & vocab)))


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
@dataclass
class RealConfig:
    n_gigs: int = 400          # gigs sampled per seed (half validation, half test)
    n_seekers: int | None = None  # None = every resume
    min_tags: int = 1          # gigs need at least this many Upwork skill tags
    seed: int = 0


_CACHE: dict = {}


def _base():
    """Parse the raw files once per process (about 10 s)."""
    if "base" not in _CACHE:
        cmap = load_category_map()
        gigs = load_gigs()
        vocab = skill_vocabulary(gigs)
        gigs = gigs[gigs.category.isin(cmap)].reset_index(drop=True)
        seekers = load_resumes()
        seekers["declared_skills"] = extract_tags(seekers.bio, vocab)
        _CACHE["base"] = (gigs, seekers, cmap, vocab)
    return _CACHE["base"]


def relevance_matrix(gigs: pd.DataFrame, seekers: pd.DataFrame, cmap) -> np.ndarray:
    rel = np.zeros((len(gigs), len(seekers)), dtype=int)
    scat = seekers.category.to_numpy()
    for i, cat in enumerate(gigs.category):
        primary, related = cmap[cat]
        rel[i, np.isin(scat, list(related))] = 1
        rel[i, scat == primary] = 2
    return rel


def sample(cfg: RealConfig):
    """Draw one random evaluation set: (seekers, gigs, relevance)."""
    gigs, seekers, cmap, _ = _base()
    rng = np.random.default_rng(cfg.seed)
    pool = gigs[gigs.required_skills.str.count(";") + 1 >= cfg.min_tags]
    g = pool.iloc[rng.choice(len(pool), size=cfg.n_gigs, replace=False)].reset_index(drop=True)
    s = seekers
    if cfg.n_seekers is not None:
        s = seekers.iloc[np.sort(rng.choice(len(seekers), size=cfg.n_seekers, replace=False))].reset_index(drop=True)
    return s, g, relevance_matrix(g, s, cmap)


def data_available() -> bool:
    return GIGS_CSV.exists() and RESUMES_CSV.exists()
