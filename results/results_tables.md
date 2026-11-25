## Retrieval recall by hop (share of questions with ALL gold passages retrieved)

| Dataset | questions | hop 1 | hop 2 | hop 3 | hop 4 |
|---|---:|---:|---:|---:|---:|
| hotpotqa | 1000 | 36.4 | 60.3 | 67.9 | 74.5 |
| 2wikimultihopqa | 1000 | 26.5 | 46.6 | 50.4 | 53.1 |
| musique | 1000 | 6.2 | 13.6 | 16.7 | 18.2 |

## HotpotQA: recall (%) at a given average number of hops

| Policy | 1.5 hops | 2.0 hops | 2.5 hops | 3.0 hops | mean over 1-4 hops |
|---|---:|---:|---:|---:|---:|
| Fixed number of hops | 48.4 | 60.3 | 64.1 | 67.9 | 61.2 |
| Heuristic: question coverage | 46.2 | 54.8 | 61.5 | 66.1 | 57.8 |
| Heuristic: top-score margin | 44.2 | 51.5 | 57.5 | 63.1 | 56.8 |
| Learned: logistic regression | 48.4 | 60.1 | 68.0 | 72.0 | 63.1 |
| Learned: gradient boosting | 50.7 | 62.7 | 70.3 | 72.4 | 64.4 |
| Learned: gradient boosting + give-up rule | 50.7 | 62.7 | 70.3 | 72.4 | 64.4 |

Oracle (stops at the first sufficient hop): 74.5% recall at 1.59 average hops.

## hotpotqa: fixed thresholds, paired bootstrap against the fixed-hop baseline

| policy | tau | avg hops | recall % | fixed-hop recall % at same cost | difference (pp) | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| gbm | 0.3 | 1.92 | 61.1 | 58.5 | +2.6 | [+0.7, +4.8] |
| gbm | 0.5 | 2.20 | 66.0 | 61.8 | +4.2 | [+2.3, +6.0] |
| gbm | 0.7 | 2.59 | 70.8 | 64.8 | +6.0 | [+4.4, +7.6] |
| gbm | 0.9 | 3.54 | 73.9 | 71.5 | +2.4 | [+1.6, +3.1] |
| gbm + give-up (rho=0.1) | 0.3 | 1.92 | 61.1 | 58.5 | +2.6 | [+0.7, +4.8] |
| gbm + give-up (rho=0.1) | 0.5 | 2.20 | 66.0 | 61.8 | +4.2 | [+2.3, +6.0] |
| gbm + give-up (rho=0.1) | 0.7 | 2.59 | 70.8 | 64.8 | +6.0 | [+4.4, +7.6] |
| gbm + give-up (rho=0.1) | 0.9 | 3.54 | 73.9 | 71.5 | +2.4 | [+1.6, +3.1] |

## 2wikimultihopqa: fixed thresholds, paired bootstrap against the fixed-hop baseline

| policy | tau | avg hops | recall % | fixed-hop recall % at same cost | difference (pp) | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| gbm | 0.3 | 2.30 | 44.8 | 47.7 | -2.9 | [-4.4, -1.6] |
| gbm | 0.5 | 2.76 | 49.2 | 49.5 | -0.3 | [-1.4, +0.8] |
| gbm | 0.7 | 3.26 | 51.9 | 51.1 | +0.8 | [+0.0, +1.5] |
| gbm | 0.9 | 3.90 | 53.1 | 52.8 | +0.3 | [+0.2, +0.4] |
| gbm + give-up (rho=0.1) | 0.3 | 2.02 | 44.4 | 46.7 | -2.3 | [-3.6, -0.6] |
| gbm + give-up (rho=0.1) | 0.5 | 2.48 | 48.8 | 48.4 | +0.4 | [-0.8, +1.5] |
| gbm + give-up (rho=0.1) | 0.7 | 2.98 | 51.5 | 50.3 | +1.2 | [+0.2, +2.0] |
| gbm + give-up (rho=0.1) | 0.9 | 3.62 | 52.7 | 52.1 | +0.6 | [+0.1, +1.1] |

## musique: fixed thresholds, paired bootstrap against the fixed-hop baseline

| policy | tau | avg hops | recall % | fixed-hop recall % at same cost | difference (pp) | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| gbm | 0.3 | 3.65 | 17.3 | 17.7 | -0.4 | [-1.0, +0.2] |
| gbm | 0.5 | 3.95 | 18.1 | 18.1 | -0.0 | [-0.2, +0.1] |
| gbm | 0.7 | 4.00 | 18.2 | 18.2 | +0.0 | [+0.0, +0.0] |
| gbm | 0.9 | 4.00 | 18.2 | 18.2 | +0.0 | [+0.0, +0.0] |
| gbm + give-up (rho=0.1) | 0.3 | 2.20 | 14.8 | 14.2 | +0.6 | [-0.4, +1.6] |
| gbm + give-up (rho=0.1) | 0.5 | 2.48 | 15.5 | 15.1 | +0.4 | [-0.5, +1.4] |
| gbm + give-up (rho=0.1) | 0.7 | 2.53 | 15.6 | 15.2 | +0.4 | [-0.5, +1.2] |
| gbm + give-up (rho=0.1) | 0.9 | 2.53 | 15.6 | 15.2 | +0.4 | [-0.6, +1.2] |

## Trained on HotpotQA only, tested on the other datasets: paired bootstrap against the fixed-hop baseline (gradient boosting)

| test dataset | tau | avg hops | recall % | fixed-hop recall % at same cost | difference (pp) | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| 2wikimultihopqa | 0.3 | 1.73 | 44.4 | 41.3 | +3.1 | [+1.6, +4.8] |
| 2wikimultihopqa | 0.5 | 1.93 | 47.3 | 45.1 | +2.2 | [+0.3, +3.8] |
| 2wikimultihopqa | 0.7 | 2.16 | 49.5 | 47.2 | +2.3 | [+1.0, +3.6] |
| 2wikimultihopqa | 0.9 | 2.97 | 51.5 | 50.3 | +1.2 | [+0.1, +2.3] |
| musique | 0.3 | 2.80 | 14.2 | 16.1 | -1.9 | [-3.0, -0.9] |
| musique | 0.5 | 3.12 | 15.0 | 16.9 | -1.9 | [-2.9, -0.9] |
| musique | 0.7 | 3.38 | 16.4 | 17.3 | -0.9 | [-1.7, -0.1] |
| musique | 0.9 | 3.89 | 17.9 | 18.0 | -0.1 | [-0.5, +0.2] |

## Give-up rule: sensitivity to rho (recall % at a given average number of hops)

| Dataset | rho | 1.5 hops | 2.0 hops | 2.5 hops | 3.0 hops | mean over 1-4 hops |
|---|---:|---:|---:|---:|---:|---:|
| hotpotqa | fixed hops | 48.4 | 60.3 | 64.1 | 67.9 | 61.2 |
| hotpotqa | 0.05 | 50.7 | 62.7 | 70.3 | 72.4 | 64.4 |
| hotpotqa | 0.1 | 50.7 | 62.7 | 70.3 | 72.4 | 64.4 |
| hotpotqa | 0.2 | 50.9 | 62.8 | 70.3 | 72.4 | 64.4 |
| hotpotqa | 0.3 | 51.1 | 62.9 | 70.0 | 71.8 | 64.2 |
| 2wikimultihopqa | fixed hops | 36.6 | 46.6 | 48.5 | 50.4 | 45.6 |
| 2wikimultihopqa | 0.05 | 38.0 | 43.5 | 48.1 | 51.3 | 45.6 |
| 2wikimultihopqa | 0.1 | 40.1 | 44.8 | 49.5 | 51.6 | 46.5 |
| 2wikimultihopqa | 0.2 | 41.2 | 46.4 | 50.0 | 51.2 | 46.9 |
| 2wikimultihopqa | 0.3 | 42.1 | 47.8 | 49.7 | 50.7 | 46.9 |
| musique | fixed hops | 9.9 | 13.6 | 15.2 | 16.7 | 14.2 |
| musique | 0.05 | 9.5 | 12.8 | 15.4 | 17.1 | 14.0 |
| musique | 0.1 | 11.1 | 14.0 | 15.6 | 15.6 | 13.8 |
| musique | 0.2 | 12.7 | 13.6 | 13.6 | 13.6 | 13.0 |
| musique | 0.3 | 9.7 | 9.7 | 9.7 | 9.7 | 9.6 |

## Classifier quality (ROC-AUC of 'evidence is sufficient', 5-fold CV)

| Dataset | model | AUC (mean ± std over 5 CV seeds) | AUC hop 1 | hop 2 | hop 3 | hop 4 |
|---|---|---|---:|---:|---:|---:|
| hotpotqa | logreg | 0.821 ± 0.002 | 0.858 | 0.799 | 0.770 | 0.740 |
| hotpotqa | gbm | 0.881 ± 0.002 | 0.921 | 0.872 | 0.847 | 0.806 |
| 2wikimultihopqa | logreg | 0.842 ± 0.001 | 0.881 | 0.818 | 0.825 | 0.814 |
| 2wikimultihopqa | gbm | 0.891 ± 0.002 | 0.931 | 0.886 | 0.868 | 0.856 |
| musique | logreg | 0.757 ± 0.004 | 0.684 | 0.750 | 0.750 | 0.727 |
| musique | gbm | 0.785 ± 0.003 | 0.793 | 0.764 | 0.761 | 0.761 |

## 2wikimultihopqa: in-domain vs. trained on HotpotQA only (gradient boosting)

| Policy | 1.5 hops | 2.0 hops | 2.5 hops | 3.0 hops | mean over 1-4 hops |
|---|---:|---:|---:|---:|---:|
| Fixed number of hops | 36.6 | 46.6 | 48.5 | 50.4 | 45.6 |
| Learned, trained on this dataset (CV) | 36.2 | 42.6 | 47.2 | 51.0 | 45.0 |
| Learned + give-up, trained on this dataset (CV) | 40.1 | 44.8 | 49.5 | 51.6 | 46.5 |
| Learned, trained on HotpotQA only | 39.8 | 48.6 | 50.5 | 51.7 | 47.2 |
| Learned + give-up, trained on HotpotQA only | 39.8 | 48.6 | 50.5 | 51.7 | 47.2 |

Oracle: 53.1% recall at 1.36 average hops.

## musique: in-domain vs. trained on HotpotQA only (gradient boosting)

| Policy | 1.5 hops | 2.0 hops | 2.5 hops | 3.0 hops | mean over 1-4 hops |
|---|---:|---:|---:|---:|---:|
| Fixed number of hops | 9.9 | 13.6 | 15.2 | 16.7 | 14.2 |
| Learned, trained on this dataset (CV) | 8.3 | 10.5 | 12.6 | 14.7 | 12.5 |
| Learned + give-up, trained on this dataset (CV) | 11.1 | 14.0 | 15.6 | 15.6 | 13.8 |
| Learned, trained on HotpotQA only | 9.4 | 11.6 | 13.4 | 15.2 | 13.2 |
| Learned + give-up, trained on HotpotQA only | 9.4 | 11.6 | 13.4 | 15.2 | 13.2 |

Oracle: 18.2% recall at 1.18 average hops.

## Feature-group ablation (HotpotQA, gradient boosting)

| Features used | # features | AUC | mean recall over 1-4 hops (%) |
|---|---:|---:|---:|
| all features | 20 | 0.881 | 64.3 ± 0.1 |
| without score | 14 | 0.872 | 64.1 ± 0.1 |
| without stability | 18 | 0.881 | 64.2 ± 0.1 |
| without coverage | 16 | 0.841 | 63.8 ± 0.1 |
| without structure | 17 | 0.794 | 62.9 ± 0.1 |
| without question | 16 | 0.866 | 64.1 ± 0.0 |
| without hop | 19 | 0.881 | 64.3 ± 0.1 |
| only score | 6 | 0.705 | 62.3 ± 0.1 |
| only stability | 2 | 0.676 | 62.3 ± 0.1 |
| only coverage | 4 | 0.740 | 62.5 ± 0.1 |
| only structure | 3 | 0.805 | 63.2 ± 0.1 |
| only question | 4 | 0.545 | 56.3 ± 0.2 |
| only hop | 1 | 0.649 | 61.3 ± 0.1 |

## LLM-judge stopping baseline (HotpotQA, same questions for every row)

| Judge | questions | avg hops | recall % | fixed hops, same cost | difference vs fixed (pp, 95% CI) | learned (gbm), same cost | oracle | requests sent |
|---|---:|---:|---:|---:|---|---:|---|---:|
| gemini: gemini-3.5-flash | 5 | 3.40 | 40.0 | 40.0 | +0.0 [+0.0, +8.0] | 40.0 | 40.0 at 1.60 | 17 |

'Requests sent' counts only the last run; answers already cached from an earlier run are not sent again.
