"""Turn metrics.json into the markdown tables used in the README."""
from __future__ import annotations

from pathlib import Path

POLICY_ROWS = [
    ("fixed", "Fixed number of hops", lambda r: r["fixed"]),
    ("coverage", "Heuristic: question coverage", lambda r: r["heuristics"]["coverage"]),
    ("top_score", "Heuristic: top-score margin", lambda r: r["heuristics"]["top_score"]),
    ("logreg", "Learned: logistic regression", lambda r: r["in_domain_cv"]["logreg"]),
    ("gbm", "Learned: gradient boosting", lambda r: r["in_domain_cv"]["gbm"]),
    ("gbm_giveup", "Learned: gradient boosting + give-up rule",
     lambda r: _giveup_main(r["in_domain_cv"]["gbm"]["giveup"])),
]


def _giveup_main(giveup: dict) -> dict:
    return giveup["by_rho"][str(giveup["main_rho"])]


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def _budget_table(rows: list[tuple[str, dict]], budgets: list[float]) -> str:
    head = "| Policy | " + " | ".join(f"{b} hops" for b in budgets) + " | mean over 1-4 hops |"
    sep = "|---|" + "---:|" * (len(budgets) + 1)
    lines = [head, sep]
    for label, summary in rows:
        cells = [_pct(summary["recall_at_budget"][str(b)]) for b in budgets]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {_pct(summary['frontier_auc'])} |")
    return "\n".join(lines)


def write_tables(metrics: dict, path: Path) -> None:
    budgets = metrics["config"]["budgets"]
    out: list[str] = []

    out.append("## Retrieval recall by hop (share of questions with ALL gold passages retrieved)\n")
    out.append("| Dataset | questions | hop 1 | hop 2 | hop 3 | hop 4 |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, res in metrics["datasets"].items():
        cells = " | ".join(_pct(r) for r in res["recall_by_hop"])
        out.append(f"| {name} | {res['n_questions']} | {cells} |")

    hot = metrics["datasets"]["hotpotqa"]
    out.append("\n## HotpotQA: recall (%) at a given average number of hops\n")
    rows = [(label, getter(hot)) for _k, label, getter in POLICY_ROWS]
    out.append(_budget_table(rows, budgets))
    out.append(f"\nOracle (stops at the first sufficient hop): {_pct(hot['oracle']['recall'])}% "
               f"recall at {hot['oracle']['cost']:.2f} average hops.")

    for dname, res in metrics["datasets"].items():
        gbm = res["in_domain_cv"]["gbm"]
        out.append(f"\n## {dname}: fixed thresholds, paired bootstrap against the fixed-hop baseline\n")
        out.append("| policy | tau | avg hops | recall % | fixed-hop recall % at same cost | "
                   "difference (pp) | 95% CI |")
        out.append("|---|---:|---:|---:|---:|---:|---|")
        for label, rows in (("gbm", gbm["vs_fixed_at_same_cost"]),
                            (f"gbm + give-up (rho={gbm['giveup']['main_rho']})",
                             gbm["giveup"]["vs_fixed_at_same_cost"])):
            for b in rows:
                out.append(f"| {label} | {b['tau']} | {b['cost']:.2f} | {_pct(b['recall'])} | "
                           f"{_pct(b['fixed_recall_same_cost'])} | {100 * b['diff']:+.1f} | "
                           f"[{100 * b['diff_lo']:+.1f}, {100 * b['diff_hi']:+.1f}] |")

    out.append("\n## Trained on HotpotQA only, tested on the other datasets: paired bootstrap "
               "against the fixed-hop baseline (gradient boosting)\n")
    out.append("| test dataset | tau | avg hops | recall % | fixed-hop recall % at same cost | "
               "difference (pp) | 95% CI |")
    out.append("|---|---:|---:|---:|---:|---:|---|")
    for dname, tr in metrics["transfer_from_hotpotqa"].items():
        for b in tr["gbm"]["vs_fixed_at_same_cost"]:
            out.append(f"| {dname} | {b['tau']} | {b['cost']:.2f} | {_pct(b['recall'])} | "
                       f"{_pct(b['fixed_recall_same_cost'])} | {100 * b['diff']:+.1f} | "
                       f"[{100 * b['diff_lo']:+.1f}, {100 * b['diff_hi']:+.1f}] |")

    out.append("\n## Give-up rule: sensitivity to rho (recall % at a given average number of hops)\n")
    out.append("| Dataset | rho | " + " | ".join(f"{b} hops" for b in budgets) + " | mean over 1-4 hops |")
    out.append("|---|---:|" + "---:|" * (len(budgets) + 1))
    for dname, res in metrics["datasets"].items():
        fx = res["fixed"]
        cells = [_pct(fx["recall_at_budget"][str(b)]) for b in budgets]
        out.append(f"| {dname} | fixed hops | " + " | ".join(cells) + f" | {_pct(fx['frontier_auc'])} |")
        for rho, v in res["in_domain_cv"]["gbm"]["giveup"]["by_rho"].items():
            cells = [_pct(v["recall_at_budget"][str(b)]) for b in budgets]
            out.append(f"| {dname} | {rho} | " + " | ".join(cells) + f" | {_pct(v['frontier_auc'])} |")

    out.append("\n## Classifier quality (ROC-AUC of 'evidence is sufficient', 5-fold CV)\n")
    out.append("| Dataset | model | AUC (mean ± std over 5 CV seeds) | AUC hop 1 | hop 2 | hop 3 | hop 4 |")
    out.append("|---|---|---|---:|---:|---:|---:|")
    for name, res in metrics["datasets"].items():
        for kind, ev in res["in_domain_cv"].items():
            s = ev["across_cv_seeds"]
            by_hop = " | ".join(f"{a:.3f}" for a in ev["auc_by_hop"])
            out.append(f"| {name} | {kind} | {s['auc_mean']:.3f} ± {s['auc_std']:.3f} | {by_hop} |")

    for name, res in metrics["datasets"].items():
        if name == "hotpotqa":
            continue
        out.append(f"\n## {name}: in-domain vs. trained on HotpotQA only (gradient boosting)\n")
        tr = metrics["transfer_from_hotpotqa"][name]["gbm"]
        rows = [("Fixed number of hops", res["fixed"]),
                ("Learned, trained on this dataset (CV)", res["in_domain_cv"]["gbm"]),
                ("Learned + give-up, trained on this dataset (CV)",
                 _giveup_main(res["in_domain_cv"]["gbm"]["giveup"])),
                ("Learned, trained on HotpotQA only", tr),
                ("Learned + give-up, trained on HotpotQA only", _giveup_main(tr["giveup"]))]
        out.append(_budget_table(rows, budgets))
        out.append(f"\nOracle: {_pct(res['oracle']['recall'])}% recall at {res['oracle']['cost']:.2f} "
                   "average hops.")

    out.append("\n## Feature-group ablation (HotpotQA, gradient boosting)\n")
    out.append("| Features used | # features | AUC | mean recall over 1-4 hops (%) |")
    out.append("|---|---:|---:|---:|")
    for label, v in metrics["ablation_hotpotqa_gbm"].items():
        out.append(f"| {label} | {v['n_features']} | {v['auc']:.3f} | "
                   f"{_pct(v['frontier_auc'])} ± {100 * v['frontier_auc_std']:.1f} |")
    Path(path).write_text("\n".join(out) + "\n")
