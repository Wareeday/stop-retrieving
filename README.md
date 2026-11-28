<div align="center">

# Learning When to Stop Retrieving

**In multi-hop retrieval, when is the evidence enough to stop, and when is a question hopeless enough to give up?**

</div>

A small, fully reproducible study of the **stopping decision in iterative multi-hop retrieval**. After each retrieval hop, a classifier over 20 signals computed from the retrieval itself decides whether the evidence collected so far is sufficient. A second classifier, added after the first results had been seen, lets the policy **give up early** on questions that will never become sufficient. Everything runs on a laptop CPU with BM25; no language model is needed for the main results.

Cost is the average number of hops (retrieval calls). Quality is *full-support recall*: the share of questions whose retrieved evidence contains **all** gold supporting passages at the moment the policy stops.

## Key findings

- **HotpotQA:** gradient boosting beats fixed-depth retrieval at the same cost by **2.4 to 6.0 percentage points** (95% bootstrap intervals exclude zero; best at tau = 0.7: **+6.0 pp, 95% CI [+4.4, +7.6]**). Mean recall over budgets: GBM **64.4%** vs fixed **61.2%**. The oracle reaches 74.5% at 1.59 hops, so most of the possible saving is not captured.
- **2WikiMultihopQA:** in-domain, the learned policy mostly does **not** beat fixed depth (45.0% vs 45.6%); with the give-up rule it is slightly ahead (46.5%).
- **MuSiQue:** no gain. Only 18.2% of questions are ever sufficient with this retriever, so **the retriever, not the stopper, is the bottleneck**.
- **Transfer:** a model trained on HotpotQA only transfers to 2WikiMultihopQA (+1.2 to +3.1 pp) but **fails on MuSiQue** (down to -1.9 pp).
- **What matters:** evidence-*structure* features (titles named in the question, passages that mention each other) matter most; question-only features are nearly useless (AUC 0.545).
- **LLM judge (single model):** `openai/gpt-oss-120b` via Groq, first 100 HotpotQA questions: 61.0% recall at 2.10 hops vs 62.8% for fixed depth at the same cost (-1.8 pp, 95% CI [-8.7, +5.6], so no difference can be established with n = 100).

## Results

All numbers below come from `results/metrics.json`, `results/results_tables.md` and `results/llm_judge_groq_openai_gpt-oss-120b_xa8da54.json`.

### Datasets and retrieval ceilings

| Dataset | Questions | Corpus passages | Gold passages per question (questions) | Recall (%) after hop 1 → 4 | Oracle (recall at avg. hops) |
|:---|---:|---:|:---|:---|:---|
| HotpotQA | 1,000 | 9,811 | 2 (1000) | 36.4 → 60.3 → 67.9 → 74.5 | 74.5% at 1.59 |
| 2WikiMultihopQA | 1,000 | 6,119 | 2 (765), 4 (235) | 26.5 → 46.6 → 50.4 → 53.1 | 53.1% at 1.36 |
| MuSiQue | 1,000 | 11,656 | 2 (518), 3 (316), 4 (166) | 6.2 → 13.6 → 16.7 → 18.2 | 18.2% at 1.18 |

### Fixed hops vs heuristics vs learned policies vs oracle

Recall (%) at a given average number of hops (5-fold cross-validation, first split). Best non-oracle value per column in **bold**.

**HotpotQA**

| Policy | 1.5 hops | 2.0 | 2.5 | 3.0 | Mean 1-4 |
|:---|---:|---:|---:|---:|---:|
| Fixed hops | 48.4 | 60.3 | 64.1 | 67.9 | 61.2 |
| Heuristic: coverage | 46.2 | 54.8 | 61.5 | 66.1 | 57.8 |
| Heuristic: top-score margin | 44.2 | 51.5 | 57.5 | 63.1 | 56.8 |
| Learned: logistic regression | 48.4 | 60.1 | 68.0 | 72.0 | 63.1 |
| Learned: gradient boosting (GBM) | **50.7** | **62.7** | **70.3** | **72.4** | **64.4** |
| Learned: GBM + give-up (rho=0.1) | **50.7** | **62.7** | **70.3** | **72.4** | **64.4** |
| Oracle | 74.5% at 1.59 hops |  |  |  |  |

**2WikiMultihopQA**

| Policy | 1.5 hops | 2.0 | 2.5 | 3.0 | Mean 1-4 |
|:---|---:|---:|---:|---:|---:|
| Fixed hops | 36.6 | **46.6** | 48.5 | 50.4 | 45.6 |
| Heuristic: coverage | 33.8 | 39.0 | 42.5 | 45.6 | 40.7 |
| Heuristic: top-score margin | 33.2 | 37.4 | 41.5 | 45.5 | 41.2 |
| Learned: logistic regression | 33.3 | 38.9 | 43.9 | 48.6 | 42.8 |
| Learned: gradient boosting (GBM) | 36.2 | 42.6 | 47.2 | 51.0 | 45.0 |
| Learned: GBM + give-up (rho=0.1) | **40.1** | 44.8 | **49.5** | **51.6** | **46.5** |
| Oracle | 53.1% at 1.36 hops |  |  |  |  |

**MuSiQue**

| Policy | 1.5 hops | 2.0 | 2.5 | 3.0 | Mean 1-4 |
|:---|---:|---:|---:|---:|---:|
| Fixed hops | 9.9 | 13.6 | 15.2 | **16.7** | **14.2** |
| Heuristic: coverage | 8.1 | 10.0 | 11.9 | 13.9 | 11.8 |
| Heuristic: top-score margin | 8.2 | 10.2 | 12.2 | 14.2 | 12.2 |
| Learned: logistic regression | 8.3 | 10.4 | 12.5 | 14.6 | 12.5 |
| Learned: gradient boosting (GBM) | 8.3 | 10.5 | 12.6 | 14.7 | 12.5 |
| Learned: GBM + give-up (rho=0.1) | **11.1** | **14.0** | **15.6** | 15.6 | 13.8 |
| Oracle | 18.2% at 1.18 hops |  |  |  |  |

### HotpotQA: learned policy vs fixed hops at the same cost

| tau | Avg. hops | GBM recall % | Fixed hops, same cost % | Difference (pp) | 95% CI |
|---:|---:|---:|---:|---:|:---|
| 0.3 | 1.92 | 61.1 | 58.5 | +2.6 | [+0.7, +4.8] |
| 0.5 | 2.20 | 66.0 | 61.8 | +4.2 | [+2.3, +6.0] |
| 0.7 | 2.59 | 70.8 | 64.8 | +6.0 | [+4.4, +7.6] |
| 0.9 | 3.54 | 73.9 | 71.5 | +2.4 | [+1.6, +3.1] |

### In-domain training vs transfer from HotpotQA

| Dataset | Fixed hops | GBM trained in-domain (CV) | GBM trained on HotpotQA only |
|:---|---:|---:|---:|
| 2WikiMultihopQA | 45.6 | 45.0 | 47.2 |
| MuSiQue | 14.2 | 12.5 | 13.2 |

Difference to fixed hops at the same cost, in percentage points with 95% bootstrap intervals (GBM):

| Test dataset | Training | tau=0.3 | tau=0.5 | tau=0.7 | tau=0.9 |
|:---|:---|:---|:---|:---|:---|
| 2WikiMultihopQA | in-domain (CV) | -2.9 [-4.4, -1.6] | -0.3 [-1.4, +0.8] | +0.8 [0.0, +1.5] | +0.3 [+0.2, +0.4] |
| 2WikiMultihopQA | HotpotQA only | +3.1 [+1.6, +4.8] | +2.2 [+0.3, +3.8] | +2.3 [+1.0, +3.6] | +1.2 [+0.1, +2.3] |
| MuSiQue | in-domain (CV) | -0.4 [-1.0, +0.2] | 0.0 [-0.2, +0.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| MuSiQue | HotpotQA only | -1.9 [-3.0, -0.9] | -1.9 [-2.9, -0.9] | -0.9 [-1.7, -0.1] | -0.1 [-0.5, +0.2] |

### Feature ablation (HotpotQA, gradient boosting, mean over 5 CV seeds)

| Features used | Count | AUC | Mean recall 1-4 hops (%) |
|:---|---:|---:|---:|
| All features | 20 | 0.881 | 64.3 |
| Without score | 14 | 0.872 | 64.1 |
| Without stability | 18 | 0.881 | 64.2 |
| Without coverage | 16 | 0.841 | 63.8 |
| Without structure | 17 | 0.794 | 62.9 |
| Without question | 16 | 0.866 | 64.1 |
| Without hop | 19 | 0.881 | 64.3 |
| Only score | 6 | 0.705 | 62.3 |
| Only stability | 2 | 0.676 | 62.3 |
| Only coverage | 4 | 0.740 | 62.5 |
| Only structure | 3 | 0.805 | 63.2 |
| Only question | 4 | 0.545 | 56.3 |
| Only hop | 1 | 0.649 | 61.3 |

### Give-up rule (rho = 0.1) vs no give-up

| Dataset | tau | GBM hops | GBM recall % | GBM vs fixed (pp) [95% CI] | +give-up hops | +give-up recall % | +give-up vs fixed (pp) [95% CI] |
|:---|---:|---:|---:|:---|---:|---:|:---|
| HotpotQA | 0.3 | 1.92 | 61.1 | +2.6 [+0.7, +4.8] | 1.92 | 61.1 | +2.6 [+0.7, +4.8] |
|  | 0.5 | 2.20 | 66.0 | +4.2 [+2.3, +6.0] | 2.20 | 66.0 | +4.2 [+2.3, +6.0] |
|  | 0.7 | 2.59 | 70.8 | +6.0 [+4.4, +7.6] | 2.59 | 70.8 | +6.0 [+4.4, +7.6] |
|  | 0.9 | 3.54 | 73.9 | +2.4 [+1.6, +3.1] | 3.54 | 73.9 | +2.4 [+1.6, +3.1] |
| 2WikiMultihopQA | 0.3 | 2.30 | 44.8 | -2.9 [-4.4, -1.6] | 2.02 | 44.4 | -2.3 [-3.6, -0.6] |
|  | 0.5 | 2.76 | 49.2 | -0.3 [-1.4, +0.8] | 2.48 | 48.8 | +0.4 [-0.8, +1.5] |
|  | 0.7 | 3.26 | 51.9 | +0.8 [0.0, +1.5] | 2.98 | 51.5 | +1.2 [+0.2, +2.0] |
|  | 0.9 | 3.90 | 53.1 | +0.3 [+0.2, +0.4] | 3.62 | 52.7 | +0.6 [+0.1, +1.1] |
| MuSiQue | 0.3 | 3.65 | 17.3 | -0.4 [-1.0, +0.2] | 2.20 | 14.8 | +0.6 [-0.4, +1.6] |
|  | 0.5 | 3.95 | 18.1 | 0.0 [-0.2, +0.1] | 2.48 | 15.5 | +0.4 [-0.5, +1.4] |
|  | 0.7 | 4.00 | 18.2 | 0.0 [0.0, 0.0] | 2.53 | 15.6 | +0.4 [-0.5, +1.2] |
|  | 0.9 | 4.00 | 18.2 | 0.0 [0.0, 0.0] | 2.53 | 15.6 | +0.4 [-0.6, +1.2] |

Sensitivity to rho. Only rho = 0.1 was fixed in advance; the others were examined afterwards:

| Dataset (mean recall 1-4 hops, %) | Fixed hops | rho=0.05 | rho=0.1 | rho=0.2 | rho=0.3 |
|:---|---:|---:|---:|---:|---:|
| HotpotQA | 61.2 | 64.4 | 64.4 | 64.4 | 64.2 |
| 2WikiMultihopQA | 45.6 | 45.6 | 46.5 | 46.9 | 46.9 |
| MuSiQue | 14.2 | 14.0 | 13.8 | 13.0 | 9.6 |

### Where the hops are wasted

An oracle stops at hop 1 on a question that never becomes sufficient. The learned policy keeps asking for more:

| Dataset | Gold passages | Questions | Sufficient by hop 4 (%) | Avg. hops spent (learned, tau=0.5) |
|:---|---:|---:|---:|---:|
| HotpotQA | 2 | 1,000 | 74.5 | 2.20 |
| *HotpotQA: never sufficient* |  | 255 |  | 3.09 |
| 2WikiMultihopQA | 2 | 765 | 68.4 | 2.44 |
| 2WikiMultihopQA | 4 | 235 | 3.4 | 3.81 |
| *2WikiMultihopQA: never sufficient* |  | 469 |  | 3.57 |
| MuSiQue | 2 | 518 | 32.8 | 3.91 |
| MuSiQue | 3 | 316 | 3.5 | 4.00 |
| MuSiQue | 4 | 166 | 0.6 | 4.00 |
| *MuSiQue: never sufficient* |  | 818 |  | 3.96 |

### LLM-judge baseline (HotpotQA only, one model, 100 questions)

| Policy (first 100 HotpotQA questions) | Avg. hops | Recall % |
|:---|---:|---:|
| LLM judge (groq: openai/gpt-oss-120b) | 2.10 | 61.0 |
| Fixed hops, same cost | 2.10 | 62.8 |
| Learned GBM, same cost | 2.10 | 65.3 |
| Oracle | 1.55 | 74.0 |

Judge versus fixed hops at the same cost: **-1.8 pp, 95% CI [-8.7, +5.6]**.

**How the run was done and what "120 requests" means.** Only one judge model was run: `openai/gpt-oss-120b` on Groq (temperature 0, reasoning effort low, 512 output tokens, at least 2 s between requests, paced to 8,000 tokens per minute). For safety and cost it was run in **three successive runs (patches)** that together cover the 100 questions, re-using an on-disk **cache** keyed by a hash of the full prompt and all settings, so an answer that already exists is never requested again. The counters in the result file describe **only the last run**: it sent **120 requests** and read **90 answers from the cache**. This does **not** mean that 120 questions were judged. The judge asks about a question once per hop until its first YES, so one question needs one to four requests; in total 210 judge decisions were needed for the 100 questions (the sum of the stopping hops, i.e. 2.10 x 100).

| Item | Value |
|:---|---:|
| Questions judged | 100 |
| Judge decisions in total (= sum of stopping hops = 2.10 x 100) | 210 |
| Requests sent in the run that produced the result file (**last run only**) | 120 |
| Answers read from the cache in that run (sent in earlier runs) | 90 |
| Requests needed by questions 1-40 / by questions 41-100 | 90 / 120 |
| Prompts declined by the provider / replies that were neither YES nor NO | 0 / 0 |
| Estimated prompt tokens in the last run | 96,491 |

Judge decisions against the gold-derived sufficiency labels (derived from `stop_hop_per_question` and `trajectories_hotpotqa.csv`; see `scripts/judge_error_analysis.py`):

| Quantity (first 100 HotpotQA questions) | Value |
|:---|---:|
| Judge stopped at hop 1 / 2 / 3 / 4 (questions) | 49 / 18 / 7 / 26 |
| Accepted at hop 1 and truly sufficient | 34 of 49 (69.4%) |
| Truly sufficient at hop 1 and accepted at hop 1 | 34 of 35 |
| Stops before hop 4 where the evidence was not yet sufficient | 20 of 74 |
| Never-sufficient questions: average hops spent | 3.35 (26 questions; 19 ran to hop 4) |

### Figures

| HotpotQA frontier | All datasets (in-domain vs transfer, with give-up) | Feature ablation |
|---|---|---|
| ![HotpotQA](results/fig_hotpotqa_frontier.png) | ![All datasets](results/fig_all_datasets.png) | ![Ablation](results/fig_ablation.png) |

### Which policy is best, and why

On HotpotQA the **gradient-boosting policy** is clearly best: it combines evidence-structure signals (is the chain closed?) and coverage signals (how much of the question is explained?) that no single heuristic captures, and a non-linear model uses them better than logistic regression. Elsewhere no learned policy beats fixed depth reliably. On MuSiQue the reason is the retriever: most questions never become sufficient, so a stopper can save cost but cannot add evidence.

## How it works

1. **Retriever.** BM25 (k1 = 1.5, b = 0.75) over title + text. Hop 1 searches with the question; each later hop searches with the question (x3) plus the best new passage of the previous hop and adds the 3 best passages not seen before. At most 4 hops.
2. **Label.** The evidence is *sufficient* after hop t if all gold supporting passages are among the passages retrieved so far.
3. **Features (20, gold-free).** score (6), stability (2), coverage (4), structure (3), question (4), hop (1). A unit test checks that changing the gold labels does not change any feature.
4. **Policies.** Fixed hops; two single-signal heuristics; logistic regression; gradient boosting; oracle. A policy stops at the first hop whose score reaches a threshold tau.
5. **Give-up rule.** A second classifier predicts whether a question will ever become sufficient; the policy also stops when that probability is below rho (0.1).
6. **Evaluation.** 5-fold cross-validation split by question; policies are compared through the upper convex hull of their (cost, recall) points; paired bootstrap (1,000 resamples) against fixed depth at equal cost.

## Quick start

Python 3.10 or newer (developed and tested on 3.12).

```bash
git clone https://github.com/Wareeday/stop-retrieving.git
cd stop-retrieving
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m stopret download     # 6 files, about 43 MB, pinned commit, SHA-256 verified
python -m stopret run          # trajectories, CV, transfer, ablation, figures (a few minutes on one CPU core)
python -m pytest               # 127 tests (5 need the `requests` library and are skipped without it)
python -m stopret report       # redraw figures and tables from results/metrics.json
```

### Optional: LLM-judge baseline (free Groq tier)

```bash
export GROQ_API_KEY=...        # never commit your key
python -m stopret check --backend groq --model openai/gpt-oss-120b       # key, connection and model id; costs no tokens
python -m stopret judge --dry-run --n 100 --tokens-per-minute 8000      # estimate tokens and requests, no API call
python -m stopret judge --backend groq --model openai/gpt-oss-120b --n 100 \
    --rpm 30 --tokens-per-minute 8000 --max-tokens 512 \
    --extra-body '{"reasoning_effort": "low"}'
python -m stopret report
```

The client sends a browser-like User-Agent (some APIs sit behind Cloudflare, which rejects Python's default), retries 429/5xx answers, **stops immediately** on a 401/403 with `STOP: API Key invalid or blocked by Cloudflare (1010)`, never records a failed request as answered, and caches every answer so a rerun resumes where it stopped. Other OpenAI-compatible servers work with `--backend openai-compatible --host URL --api-key-env NAME`.

## Repository layout

```
stopret/          the package: data, bm25, trajectory, features, models, policies, evaluation,
                  experiments, plots, report, llm_judge, llm_http, cli
scripts/          hopeless_questions.py, judge_error_analysis.py
tests/            127 tests (BM25 vs rank_bm25, pipeline invariants, data integrity, HTTP layer,
                  judge client against mock servers, Cloudflare simulation)
results/          metrics.json, results_tables.md, figures, per-hop feature tables (CSV), judge result
examples/         requests_guard.py (the same HTTP rules for code that uses `requests`)
report/           Learning When to Stop Retrieving.pdf
requirements*.txt, pytest.ini, Makefile, LICENSE
```

## Reproducibility

The data are fetched from one pinned commit of the HippoRAG repository and checked against SHA-256 hashes. Cross-validation, bootstrap and model training are deterministic, so a repeated run reproduces the stored metrics. Retrieval trajectories and features are identical across library versions tested (numpy 1.26 to 2.5, scikit-learn 1.4 to 1.9); logistic-regression results are identical, and gradient-boosting results move by at most 0.3 percentage points in the differences to the fixed baseline.

## Honest notes and limitations

- **The give-up rule was added after the first results had been seen.** It was motivated by their error analysis. Only rho = 0.1 was fixed in advance; the other values of rho were examined afterwards, and the best value differs by dataset.
- **Thresholds were discussed after the curves had been seen.** The four thresholds used for the bootstrap (0.3, 0.5, 0.7, 0.9) were fixed in the code in advance, but which results to discuss was decided after inspecting the curves.
- **The LLM judge is a single model** (`openai/gpt-oss-120b`), run only on HotpotQA and only on 100 questions, in three patches with caching. Other models, prompts and datasets were not tested. The interval for its difference to fixed depth is wide.
- **BM25 only.** No dense or hybrid retriever; absolute recall values reflect a lexical retriever over small pooled corpora (6k to 12k passages).
- **Retriever settings were chosen on HotpotQA recall alone**, on the same questions that are later evaluated.
- **Sufficiency is a retrieval proxy:** all gold passages were retrieved, not that a reader answers correctly.
- **Mean-recall columns:** the heuristic sweeps lack the point at which a policy never stops (adding it raises the coverage heuristic's HotpotQA mean from 57.8 to 59.2, no ranking changes), and the learned frontier is the best of 101 thresholds on the same out-of-fold predictions, so it is mildly optimistic; the bootstrap rows at four pre-specified thresholds are less affected.
- Hyperparameters were not tuned and no correction for multiple comparisons was applied.

## Technical report

A full report (related work, method, setup, all tables, error analysis, limitations) is in report.

## Licences and data

Code: MIT (see `LICENSE`). Data are downloaded, not stored here, and keep their own licences: HotpotQA (CC BY-SA 4.0), MuSiQue (CC BY 4.0), the 2WikiMultihopQA repository (Apache-2.0; check the dataset terms), and the HippoRAG repository that hosts the subsets (MIT). The judge model `openai/gpt-oss-120b` is released under Apache-2.0; hosted APIs have their own terms of service.

## Citation

```bibtex
@misc{masood2026stopretrieving,
  author = {Masood, Warda},
  title  = {Learning When to Stop Retrieving},
  year   = {2026},
  note   = {Technical report and code},
  url    = {https://github.com/Wareeday/stop-retrieving}
}
```
## Sources

- HotpotQA: Yang et al., 2018. 2WikiMultihopQA: Ho et al., 2020. MuSiQue: Trivedi et al., 2022.
- The 1,000-question subsets and pooled corpora come from the HippoRAG repository
  (Gutierrez et al., 2024), pinned to commit `d5c8329`. The datasets keep their original licences
  (see the table above). The raw files are not included in this repository.
- BM25 follows the Okapi formulation as implemented in the `rank_bm25` package.
- Provider limits in the LLM-judge section: Groq's rate-limit documentation, Mistral's help centre,
  Cerebras documentation and third-party trackers, read in autumn 2026. They change; check your
  own console.
