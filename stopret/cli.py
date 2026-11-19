"""Command line entry point: python -m stopret {download,run,report,judge}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import download
from .experiments import run_all, run_llm_judge
from .llm_judge import BACKENDS, THINKING_CHOICES
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

    p_j = sub.add_parser("judge", help="optional: LLM-judge stopping baseline (Ollama, Groq or Gemini)")
    p_j.add_argument("--backend", choices=BACKENDS, default="ollama")
    p_j.add_argument("--model", required=True,
                     help="e.g. llama3.2:3b for ollama, or a model id from the Groq / Google AI Studio console")
    p_j.add_argument("--n", type=int, default=100, help="number of HotpotQA questions to judge")
    p_j.add_argument("--host", default=None,
                     help="server URL (default: localhost:11434 for ollama, the provider's API otherwise)")
    p_j.add_argument("--min-interval", type=float, default=None,
                     help="minimum seconds between requests "
                          "(default: ollama 0, groq 2, gemini 6; check your own rate limits)")
    p_j.add_argument("--thinking", choices=THINKING_CHOICES, default=None,
                     help="gemini only: thinking setting (omit, off, minimal, low, medium, high)")
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
        result = run_llm_judge(args.backend, args.model, args.n, args.host, args.data_dir,
                               args.out_dir, args.min_interval, args.thinking)
        print(json.dumps(result, indent=1))
        print("run `python -m stopret report` to add it to results/results_tables.md")
    elif args.command == "report":
        out = Path(args.out_dir)
        _write_outputs(json.loads((out / "metrics.json").read_text()), out)


def _write_outputs(metrics: dict, out: Path) -> None:
    plot_hotpotqa(metrics["datasets"]["hotpotqa"], out / "fig_hotpotqa_frontier.png")
    plot_all_datasets(metrics, out / "fig_all_datasets.png")
    plot_ablation(metrics["ablation_hotpotqa_gbm"], out / "fig_ablation.png")
    write_tables(metrics, out / "results_tables.md")
    print(f"done, see {out}/")

