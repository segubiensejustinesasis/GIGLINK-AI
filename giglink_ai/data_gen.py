"""Synthetic GigLink data generator.

GigLink has not been deployed yet, so there is no real interaction log to
learn from. This module builds a *clearly labelled synthetic* dataset that
mimics the gig categories described in Chapter 1 of the GigLink proposal
(tutoring, graphic design, writing, programming, photography, social media,
online selling and administrative support).

The key design idea is that every seeker has a set of **hidden true skills**.
The matching models never see these. They only see what a real platform would
see:

* free-text bios and gig descriptions, where each skill is written using one
  of several synonyms ("logo design", "brand identity", ...), and
* self-declared skill tags, which are noisy (some true skills are left out
  and some untrue skills are over-claimed).

Ground-truth relevance is computed from the hidden skills, so the evaluation
measures how well a model recovers the truth from noisy observations instead
of grading a model against its own inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Skill taxonomy: canonical skill -> (category, list of surface forms)
# The first surface form is the "canonical" wording; the others are synonyms.
# ---------------------------------------------------------------------------
SKILLS: dict[str, tuple[str, list[str]]] = {
    "math_tutoring": ("tutoring", ["math tutoring", "algebra lessons", "calculus coaching", "mathematics review sessions"]),
    "science_tutoring": ("tutoring", ["science tutoring", "physics lessons", "chemistry review", "biology coaching"]),
    "english_tutoring": ("tutoring", ["english tutoring", "grammar lessons", "esl conversation practice", "reading comprehension coaching"]),
    "logo_design": ("design", ["logo design", "brand identity", "brand mark creation", "visual identity kit"]),
    "poster_design": ("design", ["poster design", "flyer layout", "pubmat creation", "event banner graphics"]),
    "ui_design": ("design", ["ui design", "figma mockups", "app wireframes", "user interface prototyping"]),
    "copywriting": ("writing", ["copywriting", "product descriptions", "marketing copy", "ad caption writing"]),
    "proofreading": ("writing", ["proofreading", "grammar checking", "document editing", "manuscript polishing"]),
    "translation": ("writing", ["filipino english translation", "tagalog translation", "document localization", "subtitle translation"]),
    "python_programming": ("programming", ["python programming", "python scripting", "automation scripts", "pandas data cleaning"]),
    "web_development": ("programming", ["web development", "html css javascript", "website building", "landing page coding"]),
    "spreadsheet_work": ("programming", ["excel spreadsheets", "google sheets formulas", "spreadsheet automation", "pivot table reports"]),
    "event_photography": ("photography", ["event photography", "event coverage shoot", "party photos", "occasion photoshoot"]),
    "product_photography": ("photography", ["product photography", "flat lay shots", "item photoshoot", "catalog photos"]),
    "video_editing": ("media", ["video editing", "capcut editing", "premiere pro cuts", "reels and tiktok editing"]),
    "social_media_mgmt": ("media", ["social media management", "facebook page handling", "instagram content scheduling", "community moderation"]),
    "online_selling": ("commerce", ["online selling", "shopee store management", "lazada listings", "live selling hosting"]),
    "data_entry": ("admin", ["data entry", "encoding records", "database encoding", "form digitization"]),
    "virtual_assistance": ("admin", ["virtual assistance", "email management", "calendar scheduling", "online admin support"]),
    "transcription": ("admin", ["transcription", "audio to text typing", "meeting minutes writing", "interview transcribing"]),
}
SKILL_NAMES = list(SKILLS)
CATEGORIES = sorted({cat for cat, _ in SKILLS.values()})
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_SEEKER_OPENERS = [
    "BulSU student offering", "Third year student available for", "Fresh graduate with experience in",
    "Working student who enjoys", "Part-time freelancer skilled in", "Young professional offering",
]
_GIG_OPENERS = [
    "Looking for a student who can do", "Need help with", "Hiring a freelancer for",
    "Small business needs", "Seeking someone experienced in", "Short task requiring",
]
_FILLER = [
    "reliable and responsive", "can meet deadlines", "friendly communication",
    "flexible schedule", "detail oriented", "willing to learn", "quick turnaround",
]


@dataclass
class DataConfig:
    """Parameters that control the synthetic dataset.

    Attributes
    ----------
    n_gigs, n_seekers : number of gig postings and seeker profiles.
    synonym_prob : probability that a skill is written with a non-canonical
        synonym instead of its canonical name (vocabulary mismatch level).
    tag_drop_prob : probability that a seeker forgets to declare a true skill.
    tag_spurious_prob : probability that a seeker over-claims one extra skill.
    cold_start_prob : probability that a seeker has no rating yet.
    """

    n_gigs: int = 200
    n_seekers: int = 400
    synonym_prob: float = 0.6
    tag_drop_prob: float = 0.25
    tag_spurious_prob: float = 0.3
    cold_start_prob: float = 0.3
    seed: int = 0


def _skill_phrase(rng: np.random.Generator, skill: str, synonym_prob: float) -> str:
    forms = SKILLS[skill][1]
    if rng.random() < synonym_prob:
        return forms[rng.integers(1, len(forms))]
    return forms[0]


def _pick_skills(rng: np.random.Generator, k: int) -> list[str]:
    """Pick k skills, mostly from one or two related categories (realistic)."""
    main_cat = CATEGORIES[rng.integers(len(CATEGORIES))]
    in_cat = [s for s in SKILL_NAMES if SKILLS[s][0] == main_cat]
    chosen: list[str] = []
    while len(chosen) < k:
        pool = in_cat if (rng.random() < 0.65 and set(in_cat) - set(chosen)) else SKILL_NAMES
        s = pool[rng.integers(len(pool))]
        if s not in chosen:
            chosen.append(s)
    return chosen


def _text_for(rng: np.random.Generator, skills: list[str], openers: list[str], synonym_prob: float) -> str:
    parts = [openers[rng.integers(len(openers))]]
    phrases = []
    for s in skills:
        phrases.append(_skill_phrase(rng, s, synonym_prob))
        # Sometimes people describe the same skill twice in different words,
        # and often they also mention the broad category.
        if rng.random() < 0.3:
            phrases.append(_skill_phrase(rng, s, 1.0))
        if rng.random() < 0.4:
            phrases.append(SKILLS[s][0])
    parts.append(", ".join(phrases))
    parts.append(_FILLER[rng.integers(len(_FILLER))])
    return " ".join(parts)


def generate(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate seekers and gigs. Returns (seekers_df, gigs_df)."""
    rng = np.random.default_rng(cfg.seed)

    seekers = []
    for i in range(cfg.n_seekers):
        true_skills = _pick_skills(rng, int(rng.integers(2, 5)))
        declared = [s for s in true_skills if rng.random() > cfg.tag_drop_prob]
        if rng.random() < cfg.tag_spurious_prob:
            extra = SKILL_NAMES[rng.integers(len(SKILL_NAMES))]
            if extra not in declared:
                declared.append(extra)
        n_days = int(rng.integers(2, 6))
        days = sorted(rng.choice(7, size=n_days, replace=False).tolist())
        quality = float(np.clip(rng.normal(3.9, 0.6), 1.0, 5.0))
        has_rating = rng.random() >= cfg.cold_start_prob
        seekers.append(
            {
                "seeker_id": f"S{i:04d}",
                "bio": _text_for(rng, true_skills, _SEEKER_OPENERS, cfg.synonym_prob),
                "declared_skills": ";".join(declared),
                "true_skills": ";".join(true_skills),  # HIDDEN from models
                "available_days": ";".join(DAYS[d] for d in days),
                "hourly_rate_php": int(rng.integers(8, 41) * 10),
                "rating": round(float(np.clip(quality + rng.normal(0, 0.3), 1, 5)), 2) if has_rating else np.nan,
            }
        )

    gigs = []
    for j in range(cfg.n_gigs):
        req = _pick_skills(rng, int(rng.integers(1, 4)))
        n_days = int(rng.integers(1, 4))
        days = sorted(rng.choice(7, size=n_days, replace=False).tolist())
        hours = int(rng.integers(2, 11))
        gigs.append(
            {
                "gig_id": f"G{j:04d}",
                "description": _text_for(rng, req, _GIG_OPENERS, cfg.synonym_prob),
                "required_skills": ";".join(req),  # clients pick tags from a fixed list
                "work_days": ";".join(DAYS[d] for d in days),
                "est_hours": hours,
                "budget_php": int(hours * rng.integers(10, 46) * 10),
            }
        )
    return pd.DataFrame(seekers), pd.DataFrame(gigs)


def split_set(s: str) -> set[str]:
    """Turn a ';'-joined string into a set (empty string -> empty set)."""
    return set(s.split(";")) if isinstance(s, str) and s else set()


def feasibility_matrix(seekers: pd.DataFrame, gigs: pd.DataFrame) -> np.ndarray:
    """Hard constraints observable on the platform.

    A seeker s is feasible for gig g when (i) they share at least one working
    day and (ii) the seeker's rate times the estimated hours fits the budget.
    Returns a boolean matrix of shape (n_gigs, n_seekers).
    """
    s_days = [split_set(x) for x in seekers["available_days"]]
    g_days = [split_set(x) for x in gigs["work_days"]]
    day_ok = np.array([[len(gd & sd) > 0 for sd in s_days] for gd in g_days])
    cost = np.outer(gigs["est_hours"].to_numpy(), seekers["hourly_rate_php"].to_numpy())
    budget_ok = cost <= gigs["budget_php"].to_numpy()[:, None]
    return day_ok & budget_ok


def ground_truth_relevance(seekers: pd.DataFrame, gigs: pd.DataFrame, feasible: np.ndarray) -> np.ndarray:
    """Graded relevance in {0,1,2,3} computed from HIDDEN true skills.

    coverage = |true skills of seeker  ∩  required skills| / |required skills|
    rel = 3 if coverage = 1, 2 if coverage >= 0.5, 1 if coverage > 0, else 0,
    and rel = 0 whenever the pair violates a hard constraint.
    """
    t_sk = [split_set(x) for x in seekers["true_skills"]]
    req = [split_set(x) for x in gigs["required_skills"]]
    rel = np.zeros((len(gigs), len(seekers)), dtype=int)
    for g, r in enumerate(req):
        for s, t in enumerate(t_sk):
            cov = len(r & t) / len(r)
            rel[g, s] = 3 if cov == 1 else 2 if cov >= 0.5 else 1 if cov > 0 else 0
    return rel * feasible
