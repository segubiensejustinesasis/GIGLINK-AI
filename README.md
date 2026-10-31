# GigLink: AI-Powered Skill-Based Gig Matching

Collaborative mini-research project for **MCS 306 – Introduction to Artificial Intelligence** and
**TWM 301 – Technical Writing in Mathematics**, Bulacan State University (BSM CS 3B – G2).

> **Skill-Based Gig Matching for GigLink Using TF-IDF, Latent Semantic Analysis, and Optimal Assignment**

GigLink is a planned task-based gig marketplace for students and young professionals. This repository
contains the AI component for its *provider matching* module: given a gig post, rank the seekers who are most
likely to have the right skills, and match many gigs to seekers at once without double-booking anyone.

| | |
|---|---|
| Paper (PDF) | [`paper/main.pdf`](paper/main.pdf) — LaTeX source in [`paper/main.tex`](paper/main.tex) |
| Google Colab notebook | [`notebooks/GigLink_Colab.ipynb`](notebooks/GigLink_Colab.ipynb) |
| Video presentation | [Watch on Google Drive](PASTE_GOOGLE_DRIVE_LINK_HERE) (too large for GitHub) |
| Reference check log | [`docs/reference_verification.md`](docs/reference_verification.md) |
| Team Git workflow | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## About the data (real, not synthetic)

GigLink is not deployed yet, so it has no users of its own. The method is evaluated on **two real, public
Kaggle datasets**. No synthetic or generated data is used anywhere.

| Role in GigLink | Dataset | What we use |
|---|---|---|
| **Gigs** | [Upwork Job Postings Dataset 2024 (50K Records)](https://www.kaggle.com/datasets/asaniczka/upwork-job-postings-dataset-2024-50k-records) by asaniczka | 52,977 real Upwork job posts (Dec 2023 – Feb 2024): title, description, the client's Upwork category and skill tags |
| **Seekers** | [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by Snehaan Bhawal | 2,483 real LiveCareer resumes in 24 job categories (personal details already anonymised) |

**Ground truth.** Nobody publishes who was hired for which Upwork post, so relevance is judged with a
hand-written table, [`data/category_map.csv`](data/category_map.csv): for 94 Upwork categories it names the
resume category that does that work (relevance 2) and closely related categories (relevance 1). The models never
see these category labels.

**The raw data is not in this repository.** `data/raw/` is ignored by Git. To run the experiments, download the
two files from Kaggle and put them in `data/raw/` (see [`data/raw/README.md`](data/raw/README.md)).

## Method in one paragraph

Gig posts and resumes become **TF-IDF** vectors (vector space model). **Latent semantic analysis**
(randomized truncated SVD, rank *k* = 40, *q* = 7 power iterations) maps them into a concept space where we measure
cosine similarity. Each resume is tagged with the Upwork skill phrases it contains, and a **hybrid score**
`H = α·text + (1−α)·Jaccard(tags)` is tuned on validation gigs. For batch matching, the **linear assignment
problem** is solved exactly with SciPy's Hungarian-type solver and compared with a greedy heuristic. Schedule and
budget constraints are implemented but could not be evaluated, because the resumes have no availability or rates.

## Headline results (real data, 20 random samples, 200 test gigs × 2,483 resumes each)

| Method | P@5 | R@100 | nDCG@10 | MRR |
|---|---|---|---|---|
| Random | 0.044 | 0.041 | 0.068 | 0.142 |
| Keyword overlap | 0.236 | 0.129 | 0.278 | 0.438 |
| Jaccard (tags) | 0.277 | 0.124 | 0.324 | 0.462 |
| TF-IDF | 0.314 | 0.176 | 0.362 | 0.484 |
| LSA (k = 40) | 0.348 | 0.236 | 0.406 | 0.541 |
| **Hybrid (tuned)** | **0.389** | **0.237** | **0.438** | **0.566** |

The hybrid beat every baseline on nDCG@10 in 20 of 20 samples. It works best for design and marketing gigs
(nDCG@10 ≈ 0.75) and worst for data-analytics and support gigs (below 0.2). Exact assignment collected 0.9%
(all resumes) to 6.1% (300 resumes) more relevance than greedy assignment and never double-booked a seeker.
Full tables: [`results/`](results/).

## Repository structure

```
giglink_ai/
  real_data.py     loads the Upwork posts and resumes, tags resumes, builds the ground truth
  models.py        Random, Keyword, Jaccard, TF-IDF, LSA, Hybrid scoring
  metrics.py       Precision@k, Recall@k, nDCG@k, MRR
  assignment.py    optimal (Hungarian) vs greedy vs independent assignment
  experiments.py   runs every experiment, writes results/ and figures/
  make_tables.py   turns results/*.csv into paper/tables/*.tex
  demo.py          command-line demo: type a gig, get top resumes
  demo_walkthrough.py  step-by-step demo shown in the video
  plotting.py      shared figure style
data/
  category_map.csv Upwork category -> resume category table (the ground truth)
  raw/             put the Kaggle files here (ignored by Git)
  preview/         first 100 cleaned Upwork posts (descriptions cut to 300 characters), for a quick look
scripts/
  make_preview.py  builds the preview CSV from data/raw/upwork-jobs.csv
notebooks/         Google Colab notebook
tests/             unit tests (pytest)
results/           every number in the paper (CSV / JSON)
figures/           every figure in the paper (PDF + PNG)
paper/             LaTeX source, APA 7 reference list, generated tables, compiled PDF
docs/              reference verification log
```

## How to run

Requires Python 3.10+ (tested on 3.13).

```bash
git clone https://github.com/segubiensejustinesasis/GigLink-An-AI-Powered-Gig-Marketplace.git
cd GigLink-An-AI-Powered-Gig-Marketplace
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put upwork-jobs.csv and Resume.csv in data/raw/ first
python -m pytest -q                  # 9 unit tests (no data needed)
python -m giglink_ai.experiments     # all experiments (~15-20 min); --quick for a smoke test
python -m giglink_ai.make_tables     # regenerate LaTeX tables from results/
python -m giglink_ai.demo            # try the matcher
python -m giglink_ai.demo_walkthrough   # 5-step demo shown in the video
python -m giglink_ai.demo --text "Need someone to edit TikTok videos for our shop" \
    --tags "video editing;adobe premiere pro;tiktok"
```

`python -m giglink_ai.experiments` overwrites `results/`, `figures/` and `paper/tables/`. Because every random
draw is seeded, the metric values are identical on every run; only the runtime measurements change with the machine.

**Google Colab:** open [`notebooks/GigLink_Colab.ipynb`](notebooks/GigLink_Colab.ipynb) in Colab, upload a ZIP of
this repository and the two Kaggle files when asked, and choose *Runtime → Run all*.

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
| E1 main comparison | ranking quality of 6 methods, 20 random samples, weight tuned on a validation split | `main_metrics_*.csv`, `significance_tests.csv`, `fig_main_metrics` |
| Per category | which kinds of work are easy or hard | `per_category_ndcg.csv`, `fig_per_category` |
| Ablation | text model, text only, tags only, fixed weight | `ablation_summary.csv` |
| E2 LSA rank sweep | nDCG@10 vs *k* | `lsa_rank_sweep.csv`, `fig_lsa_rank` |
| E3 SVD convergence | singular-value error vs power iterations *q* | `svd_convergence.csv`, `fig_svd_convergence` |
| E4 robustness | shorter resumes, missing seeker tags | `robustness.csv`, `fig_robustness` |
| E5 weight curve | validation nDCG@10 over α | `weight_grid_seed0.csv`, `fig_weight_grid` |
| E6 assignment | Hungarian vs greedy vs independent top-1 | `assignment_*.csv` |
| E7 runtime | scaling with problem size | `runtime_scaling.csv`, `fig_runtime_scaling` |
| Monte Carlo | stability of estimates over samples | `fig_monte_carlo` |

Key parameters (see `results/run_metadata.json`): 400 gigs per sample (200 validation, 200 test), all 2,483
resumes; LSA rank 40; 7 power iterations; oversampling 10; α ∈ {0, 0.1, …, 1}; exact-SVD reference with ARPACK `tol=0`.

## What is implemented vs proposed

| Component | Status |
|---|---|
| Real-data loader, models, metrics, assignment, experiments, demo, Colab notebook | Implemented, unit-tested |
| Schedule and budget constraints | Implemented; not evaluated (no data) |
| Integration into the GigLink web app (API endpoint + UI) | Proposed, not implemented |
| Evaluation on GigLink data, user study, fairness audit | Requires data / not done |

## Data credits

* asaniczka. (2024). *Upwork job postings dataset 2024 (50K records)* [Data set]. Kaggle. https://www.kaggle.com/datasets/asaniczka/upwork-job-postings-dataset-2024-50k-records
* Bhawal, S. (2021). *Resume dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset


## Authors

Evangelista, Kristiana T. · Jacinto, Thealyne G. · Perez, Jammele E. · Santos, Tifany B. · Segubiense, Justine S.
