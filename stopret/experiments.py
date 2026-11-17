"""All experiments reported in the README. Run with `python -m stopret run`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import DATASETS, load
from .evaluation import (default_taus, fixed_points, frontier_auc, outcome,
                         paired_bootstrap_vs_fixed, recall_at_budget, sweep)
from .features import FEATURE_GROUPS, FEATURE_NAMES, build_table
from .llm_judge import OllamaJudge
from .models import MODEL_KINDS, cross_val_predict, fit_predict
from .policies import oracle_stops, stop_hops, stop_hops_giveup, to_matrix
from .trajectory import IterativeRetriever, RetrievalConfig

BUDGETS = (1.5, 2.0, 2.5, 3.0)
BOOTSTRAP_TAUS = (0.3, 0.5, 0.7, 0.9)
MAIN_RHO = 0.1                       # give-up threshold used in the headline numbers
GIVEUP_RHOS = (0.05, 0.1, 0.2, 0.3)  # shown in the sensitivity table
CV_SEEDS = (0, 1, 2, 3, 4)
HEURISTICS = {"coverage": "cov", "top_score": "z1"}


def load_or_build_table(name: str, cfg: RetrievalConfig, data_dir: str, out_dir: Path,
                        rebuild: bool = False) -> pd.DataFrame:
    path = out_dir / f"trajectories_{name}.csv"
    if path.exists() and not rebuild:
        return pd.read_csv(path, dtype={"qid": str})
    print(f"[{name}] building retrieval trajectories ...", flush=True)
    table = build_table(IterativeRetriever(load(name, data_dir), cfg))
    table.to_csv(path, index=False)
    return table


def _curve_summary(points: list[tuple[float, float]], max_hops: int) -> dict:
    return {
        "frontier_auc": frontier_auc(points, max_hops),
        "recall_at_budget": {str(b): recall_at_budget(points, b) for b in BUDGETS},
    }


def _heuristic_points(table: pd.DataFrame, suff: np.ndarray, max_hops: int) -> dict:
    out = {}
    for name, column in HEURISTICS.items():
        scores = to_matrix(table, column, max_hops)
        taus = np.linspace(scores.min(), scores.max(), 201)
        out[name] = sweep(scores, suff, taus)
    return out


def evaluate_probabilities(proba: np.ndarray, table: pd.DataFrame, suff: np.ndarray,
                           max_hops: int, with_bootstrap: bool = True) -> dict:
    labels = table["sufficient"].to_numpy()
    scores = proba.reshape(-1, max_hops)
    points = sweep(scores, suff, default_taus())
    result = {
        "auc": float(roc_auc_score(labels, proba)),
        "auc_by_hop": [float(roc_auc_score(labels[table["hop"] == h], proba[table["hop"] == h]))
                       for h in range(1, max_hops + 1)],
        **_curve_summary(points, max_hops),
        "points": points,
    }
    if with_bootstrap:
        result["vs_fixed_at_same_cost"] = [
            {"tau": tau, **paired_bootstrap_vs_fixed(lambda idx, t=tau: stop_hops(scores[idx], t), suff)}
            for tau in BOOTSTRAP_TAUS
        ]
    return result


def evaluate_giveup(p_now: np.ndarray, p_ever: np.ndarray, table: pd.DataFrame,
                    suff: np.ndarray, max_hops: int) -> dict:
    """Stopping with a give-up rule. p_now: P(sufficient now), p_ever: P(sufficient by the last
    hop). One curve per give-up threshold rho; bootstrap only for the headline rho."""
    now = p_now.reshape(-1, max_hops)
    ever = p_ever.reshape(-1, max_hops)
    labels = table["eventually"].to_numpy()
    result = {
        "auc_ever": float(roc_auc_score(labels, p_ever)),
        "auc_ever_by_hop": [float(roc_auc_score(labels[table["hop"] == h], p_ever[table["hop"] == h]))
                            for h in range(1, max_hops + 1)],
        "by_rho": {},
    }
    for rho in GIVEUP_RHOS:
        points = [outcome(suff, stop_hops_giveup(now, ever, tau, rho)) for tau in default_taus()]
        result["by_rho"][str(rho)] = {"points": points, **_curve_summary(points, max_hops)}
    result["main_rho"] = MAIN_RHO
    result["vs_fixed_at_same_cost"] = [
        {"tau": tau, "rho": MAIN_RHO, **paired_bootstrap_vs_fixed(
            lambda idx, t=tau: stop_hops_giveup(now[idx], ever[idx], t, MAIN_RHO), suff)}
        for tau in BOOTSTRAP_TAUS
    ]
    return result


def run_dataset(name: str, table: pd.DataFrame, max_hops: int) -> dict:
    suff = to_matrix(table, "sufficient", max_hops).astype(bool)
    fixed = fixed_points(suff)
    oracle_cost, oracle_recall = outcome(suff, oracle_stops(suff))
    result = {
        "n_questions": int(len(suff)),
        "gold_per_question": {
            int(k): int(v) for k, v in table.groupby(table.index // max_hops)["n_gold"].first()
                                            .value_counts().sort_index().items()
        },
        "recall_by_hop": [r for _c, r in fixed],
        "fixed": {"points": fixed, **_curve_summary(fixed, max_hops)},
        "oracle": {"cost": oracle_cost, "recall": oracle_recall},
        "heuristics": {},
        "in_domain_cv": {},
    }
    for hname, pts in _heuristic_points(table, suff, max_hops).items():
        result["heuristics"][hname] = {"points": pts, **_curve_summary(pts, max_hops)}

    for kind in MODEL_KINDS:
        proba = cross_val_predict(table, max_hops, kind, seed=0)
        result["in_domain_cv"][kind] = evaluate_probabilities(proba, table, suff, max_hops)
        seeds = []
        for seed in CV_SEEDS:
            p = proba if seed == 0 else cross_val_predict(table, max_hops, kind, seed=seed)
            pts = sweep(p.reshape(-1, max_hops), suff, default_taus())
            seeds.append((float(roc_auc_score(table["sufficient"], p)), frontier_auc(pts, max_hops)))
        seeds = np.array(seeds)
        result["in_domain_cv"][kind]["across_cv_seeds"] = {
            "auc_mean": float(seeds[:, 0].mean()), "auc_std": float(seeds[:, 0].std()),
            "frontier_auc_mean": float(seeds[:, 1].mean()),
            "frontier_auc_std": float(seeds[:, 1].std()),
        }
        if kind == "gbm":
            p_ever = cross_val_predict(table, max_hops, kind, seed=0, target="eventually")
            result["in_domain_cv"][kind]["giveup"] = evaluate_giveup(proba, p_ever, table, suff,
                                                                     max_hops)
    return result


def run_transfer(train: pd.DataFrame, test: pd.DataFrame, max_hops: int) -> dict:
    suff = to_matrix(test, "sufficient", max_hops).astype(bool)
    result = {}
    for kind in MODEL_KINDS:
        p_now = fit_predict(train, test, kind)
        result[kind] = evaluate_probabilities(p_now, test, suff, max_hops)
        if kind == "gbm":
            p_ever = fit_predict(train, test, kind, target="eventually")
            result[kind]["giveup"] = evaluate_giveup(p_now, p_ever, test, suff, max_hops)
    return result


def run_ablation(table: pd.DataFrame, max_hops: int, kind: str = "gbm") -> dict:
    suff = to_matrix(table, "sufficient", max_hops).astype(bool)
    results = {}
    variants = {"all features": FEATURE_NAMES}
    for group, cols in FEATURE_GROUPS.items():
        variants[f"without {group}"] = [f for f in FEATURE_NAMES if f not in cols]
    for group, cols in FEATURE_GROUPS.items():
        variants[f"only {group}"] = list(cols) if group != "hop" else ["hop"]
    for label, feats in variants.items():
        aucs, faucs = [], []
        for seed in CV_SEEDS:
            p = cross_val_predict(table, max_hops, kind, features=feats, seed=seed)
            pts = sweep(p.reshape(-1, max_hops), suff, default_taus())
            aucs.append(roc_auc_score(table["sufficient"], p))
            faucs.append(frontier_auc(pts, max_hops))
        results[label] = {"n_features": len(feats), "auc": float(np.mean(aucs)),
                          "frontier_auc": float(np.mean(faucs)),
                          "frontier_auc_std": float(np.std(faucs))}
    return results


def run_all(data_dir: str = "data/raw", out_dir: str = "results", rebuild: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = RetrievalConfig()
    tables = {n: load_or_build_table(n, cfg, data_dir, out, rebuild) for n in DATASETS}

    metrics = {"config": {"retrieval": cfg.__dict__, "features": FEATURE_NAMES,
                          "cv_seeds": list(CV_SEEDS), "budgets": list(BUDGETS)},
               "datasets": {}, "transfer_from_hotpotqa": {}, "ablation_hotpotqa_gbm": {}}
    for name, table in tables.items():
        print(f"[{name}] in-domain cross-validation ...", flush=True)
        metrics["datasets"][name] = run_dataset(name, table, cfg.max_hops)
    for name in DATASETS:
        if name != "hotpotqa":
            print(f"[hotpotqa -> {name}] zero-shot transfer ...", flush=True)
            metrics["transfer_from_hotpotqa"][name] = run_transfer(
                tables["hotpotqa"], tables[name], cfg.max_hops)
    print("[hotpotqa] feature-group ablation ...", flush=True)
    metrics["ablation_hotpotqa_gbm"] = run_ablation(tables["hotpotqa"], cfg.max_hops)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    return metrics


def run_llm_judge(model: str, n_questions: int = 200, host: str = "http://localhost:11434",
                  data_dir: str = "data/raw", out_dir: str = "results") -> dict:
    """LLM-judge stopping on the first `n_questions` HotpotQA questions, compared with the fixed
    and learned policies on exactly the same questions. Needs a running Ollama server."""
    out = Path(out_dir)
    cfg = RetrievalConfig()
    ds = load("hotpotqa", data_dir)
    retriever = IterativeRetriever(ds, cfg)
    table = load_or_build_table("hotpotqa", cfg, data_dir, out)
    suff = to_matrix(table, "sufficient", cfg.max_hops).astype(bool)[:n_questions]

    safe_model = model.replace(":", "_").replace("/", "_")
    judge = OllamaJudge(model, host, cache_path=out / f"llm_judge_cache_{safe_model}.json")
    verdicts = np.zeros((len(suff), cfg.max_hops))
    for qi, ex in enumerate(ds.examples[: len(suff)]):
        for hop in retriever.retrieve(ex):
            passages = [(ds.titles[i], ds.texts[i]) for i in hop.seen_ids]
            verdicts[qi, hop.hop - 1] = judge.is_sufficient(ex.question, passages,
                                                            cache_key=f"{ex.qid}|{hop.hop}")
        if (qi + 1) % 20 == 0:
            print(f"  judged {qi + 1}/{len(suff)} questions", flush=True)

    cost, recall = outcome(suff, stop_hops(verdicts, 0.5))
    learned = cross_val_predict(table, cfg.max_hops, "gbm").reshape(-1, cfg.max_hops)[: len(suff)]
    learned_points = sweep(learned, suff, default_taus())
    result = {
        "model": model, "n_questions": len(suff),
        "llm_judge": {"cost": cost, "recall": recall},
        "fixed_hops_recall_at_same_cost": recall_at_budget(fixed_points(suff), cost),
        "learned_gbm_recall_at_same_cost": recall_at_budget(learned_points, cost),
        "oracle": dict(zip(("cost", "recall"), outcome(suff, oracle_stops(suff)))),
    }
    (out / f"llm_judge_{safe_model}.json").write_text(json.dumps(result, indent=1))
    return result
