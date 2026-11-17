"""Command line entry point: python -m stopret {download,run,report,judge}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import download
from .experiments import run_all, run_llm_judge
from .plots import plot_ablation, plot_all_datasets, plot_hotpotqa
from .report import write_tables


def main() -> None:
    parser = argparse.ArgumentParser(prog="stopret", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="fetch the three datasets (about 43 MB)")
    p_dl.add_argument("--data-dir", default="data/raw")

    p_run = sub.add_parser("run", help="build trajectories, run all experiments, write results/")
    p_run.add_argument("--data-dir", default="data/raw")
    p_run.add_argument("--out-dir", default="results")
    p_run.add_argument("--rebuild", action="store_true",
                       help="recompute the retrieval trajectories instead of reusing the CSVs")

    p_j = sub.add_parser("judge", help="optional: LLM-judge stopping baseline through Ollama")
    p_j.add_argument("--model", required=True, help="Ollama model name, e.g. llama3.2:3b")
    p_j.add_argument("--n", type=int, default=200, help="number of HotpotQA questions to judge")
    p_j.add_argument("--host", default="http://localhost:11434")
    p_j.add_argument("--data-dir", default="data/raw")
    p_j.add_argument("--out-dir", default="results")

    p_rep = sub.add_parser("report", help="redraw figures and tables from results/metrics.json")
    p_rep.add_argument("--out-dir", default="results")

    args = parser.parse_args()
    if args.command == "download":
        download(args.data_dir)
    elif args.command == "run":
        metrics = run_all(args.data_dir, args.out_dir, args.rebuild)
        _write_outputs(metrics, Path(args.out_dir))
    elif args.command == "judge":
        print(json.dumps(run_llm_judge(args.model, args.n, args.host, args.data_dir, args.out_dir),
                         indent=1))
    elif args.command == "report":
        out = Path(args.out_dir)
        _write_outputs(json.loads((out / "metrics.json").read_text()), out)


def _write_outputs(metrics: dict, out: Path) -> None:
    plot_hotpotqa(metrics["datasets"]["hotpotqa"], out / "fig_hotpotqa_frontier.png")
    plot_all_datasets(metrics, out / "fig_all_datasets.png")
    plot_ablation(metrics["ablation_hotpotqa_gbm"], out / "fig_ablation.png")
    write_tables(metrics, out / "results_tables.md")
    print(f"done, see {out}/")
