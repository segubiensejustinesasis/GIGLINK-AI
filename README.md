# GigLink: AI-Powered Skill-Based Gig Matching

Collaborative mini-research project for **MCS 306 – Introduction to Artificial Intelligence** and
**TWM 301 – Technical Writing in Mathematics**, Bulacan State University (BSM CS 3B – G2).

> **Skill-Based Gig Matching for GigLink Using TF-IDF, Latent Semantic Analysis, and Optimal Assignment**

GigLink is a planned task-based gig marketplace for students and young professionals. This repository
contains the AI component for its *provider matching* module: given a gig post, rank the seekers who are most
likely to have the right skills and can actually take the job, and match many gigs to seekers at once without
double-booking anyone.

| | |
|---|---|
| Paper (PDF) | [`paper/main.pdf`](paper/main.pdf) — LaTeX source in [`paper/main.tex`](paper/main.tex) |
| Research proposal / approach evaluation | [`docs/research_proposal.md`](docs/research_proposal.md) |
| Reference check log | [`docs/reference_verification.md`](docs/reference_verification.md) |
| Team Git workflow | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## ⚠️ About the data

GigLink is **not deployed yet, so there is no real user data**. All experiments use a **synthetic, seeded
dataset** produced by [`giglink_ai/data_gen.py`](giglink_ai/data_gen.py). Each synthetic seeker has *hidden*
true skills; the models only see noisy bios (with synonyms) and noisy self-declared tags, and are scored against
the hidden truth. The results describe behaviour under these stated assumptions, not the behaviour of real users.

## Method in one paragraph

Gig descriptions and seeker bios become **TF-IDF** vectors (vector space model). **Latent semantic analysis**
(randomized truncated SVD, rank *k* = 40, *q* = 7 power iterations) maps them into a concept space where we measure
cosine similarity. A **hybrid score** `H = α·text + (1−α)·Jaccard(tags) + γ·rating` is tuned on validation gigs,
and **hard constraints** (shared working day, rate × hours ≤ budget) remove seekers who cannot take the gig. For
batch matching, the **linear assignment problem** is solved exactly with SciPy's Hungarian-type solver and compared
with a greedy heuristic.

## Headline results (synthetic data, 20 seeds, test gigs)

| Method | P@5 | nDCG@10 | MRR |
|---|---|---|---|
| Random | 0.169 | 0.184 | 0.347 |
| Keyword overlap | 0.686 | 0.664 | 0.883 |
| Jaccard (tags) | 0.872 | 0.872 | 0.962 |
| TF-IDF | 0.761 | 0.739 | 0.906 |
| LSA (k = 40) | 0.705 | 0.690 | 0.876 |
| **Hybrid (tuned)** | **0.895** | **0.903** | **0.976** |

The hybrid beat tag matching on nDCG@10 in 20 of 20 seeds. Exact assignment gave 3.8% (surplus) to 9.5% (scarce)
more true relevance than greedy assignment and never double-booked a seeker. Full tables: [`results/`](results/).

## Repository structure

```
giglink_ai/
  data_gen.py      synthetic seekers & gigs, feasibility constraints, ground truth
  models.py        Random, Keyword, Jaccard, TF-IDF, LSA, Hybrid scoring
  metrics.py       Precision@k, Recall@k, nDCG@k, MRR
  assignment.py    optimal (Hungarian) vs greedy vs independent assignment
  experiments.py   runs every experiment, writes results/ and figures/
  make_tables.py   turns results/*.csv into paper/tables/*.tex
  demo.py          command-line demo: type a gig, get top seekers
  plotting.py      shared figure style
tests/             unit tests (pytest)
data/              synthetic dataset for seed 0 (CSV, for inspection)
results/           every number in the paper (CSV / JSON)
figures/           every figure in the paper (PDF + PNG)
paper/             LaTeX source, APA 7 reference list, generated tables, compiled PDF
docs/              proposal, video script, reference verification log
```

## How to run

Requires Python 3.10+ (tested on 3.13).

```bash
git clone https://github.com/segubiensejustinesasis/GigLink-An-AI-Powered-Gig-Marketplace.git
cd GigLink-An-AI-Powered-Gig-Marketplace
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                  # 8 unit tests
python -m giglink_ai.experiments     # all experiments (~1–2 min); --quick for a smoke test
python -m giglink_ai.make_tables     # regenerate LaTeX tables from results/
python -m giglink_ai.demo            # try the matcher
python -m giglink_ai.demo --text "Need someone for algebra lessons this weekend" \
    --tags math_tutoring --days "sat;sun" --hours 4 --budget 1200
```

`python -m giglink_ai.experiments` overwrites `data/`, `results/` and `figures/`. Because every random draw is
seeded, the metric values are identical on every run; only the runtime measurements change with the machine.

### Compile the paper

The paper uses the BulSU mini-research template (title page with logos, roman-numbered front matter). Add the
logos as `paper/assets/bulsu.png` and `paper/assets/college_of_science.png`; without them a placeholder box is drawn.
References are a hand-formatted APA 7 list in `paper/apa_bibliography.tex` (cited with natbib), so only
pdfLaTeX is needed, no biber. `paper/references.bib` keeps the same sources in BibTeX form.

```bash
cd paper
latexmk -pdf main.tex      # or run pdflatex main.tex two or three times
```

**Overleaf / single file:** `python paper/build_standalone.py` writes `paper/giglink_paper_standalone.tex`, which
has the tables and the reference list built in and compiles in seconds (fits Overleaf's free plan). Upload it together with a `figures/` folder containing `fig_*.pdf`
(and optionally an `assets/` folder with the logos), set the compiler to pdfLaTeX, and compile.

## Experiments and settings

| Experiment | What it shows | Output |
|---|---|---|
| E1 main comparison | ranking quality of 6 methods, 20 seeds, weights tuned on validation split | `main_metrics_*.csv`, `significance_tests.csv`, `fig_main_metrics` |
| Ablation | effect of constraints, rating, text, tags | `ablation_summary.csv` |
| E2 LSA rank sweep | nDCG@10 vs *k* | `lsa_rank_sweep.csv`, `fig_lsa_rank` |
| E3 SVD convergence | singular-value error vs power iterations *q* | `svd_convergence.csv`, `fig_svd_convergence` |
| E4 robustness | vocabulary noise and tag noise | `noise_robustness.csv`, `fig_noise_robustness` |
| E5 weight grid | validation surface over α, γ | `weight_grid_seed0.csv`, `fig_weight_grid` |
| E6 assignment | Hungarian vs greedy vs independent top-1 | `assignment_*.csv` |
| E7 runtime | scaling with problem size | `runtime_scaling.csv`, `fig_runtime_scaling` |
| Monte Carlo | stability of estimates over seeds | `fig_monte_carlo` |

Key parameters (see `results/run_metadata.json`): 200 gigs × 400 seekers per dataset; LSA rank 40; 7 power
iterations; oversampling 10; α ∈ {0, 0.1, …, 1}; γ ∈ {0, 0.05, 0.1, 0.2}; exact-SVD reference with ARPACK `tol=0`.

## What is implemented vs proposed

| Component | Status |
|---|---|
| Data generator, models, metrics, assignment, experiments, demo | Implemented, unit-tested |
| Integration into the GigLink web app (API endpoint + UI) | Proposed, not implemented |
| Evaluation on real GigLink data, user study, fairness audit | Requires data / not done |

## Authors

Evangelista, Kristiana T. · Jacinto, Thealyne G. · Perez, Jammele E. · Santos, Tifany B. · Segubiense, Justine S.

