"""Where does a "sufficient now?" classifier waste hops? Prints, per dataset and per number of
gold passages, how often questions become sufficient and how many hops the plain learned policy
(tau = 0.5, no give-up rule) spends on them.

Run from the repo root after `python -m stopret run`:
    python scripts/hopeless_questions.py | tee results/hopeless_questions.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopret.models import cross_val_predict  # noqa: E402
from stopret.policies import stop_hops, to_matrix  # noqa: E402

T, TAU = 4, 0.5
for name in ("hotpotqa", "2wikimultihopqa", "musique"):
    table = pd.read_csv(f"results/trajectories_{name}.csv", dtype={"qid": str})
    suff = to_matrix(table, "sufficient", T).astype(bool)
    n_gold = to_matrix(table, "n_gold", T)[:, 0]
    stops = stop_hops(cross_val_predict(table, T, "gbm").reshape(-1, T), TAU)
    print(f"== {name}")
    print("gold passages | questions | sufficient by hop 4 | avg hops spent (learned, tau=0.5)")
    for g in sorted(set(n_gold)):
        m = n_gold == g
        print(f"{int(g):13d} | {m.sum():9d} | {100 * suff[m, -1].mean():18.1f}% | {stops[m].mean():.2f}")
    never = ~suff[:, -1]
    print(f"never sufficient: {never.sum()} questions, learned policy spends {stops[never].mean():.2f} "
          f"hops on them on average (an oracle would stop at hop 1)\n")
