"""Command line entry point: python -m stopret {download,run,report,judge}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import download, verify
from .experiments import estimate_judge_cost, run_all, run_llm_judge
from .llm_judge import BACKENDS, PROVIDERS, THINKING_CHOICES, RateLimitExhausted
from .plots import plot_ablation, plot_all_datasets, plot_hotpotqa
from .report import write_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stopret", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="fetch the three datasets (about 43 MB) and verify them")
    p_dl.add_argument("--data-dir", default="data/raw")
    p_dl.add_argument("--force", action="store_true", help="download again even if files are valid")

    p_run = sub.add_parser("run", help="build trajectories, run all experiments, write results/")
    p_run.add_argument("--data-dir", default="data/raw")
    p_run.add_argument("--out-dir", default="results")
    p_run.add_argument("--rebuild", action="store_true",
                       help="recompute the retrieval trajectories instead of reusing the CSVs")

    p_rep = sub.add_parser("report", help="redraw figures and tables from results/metrics.json")
    p_rep.add_argument("--out-dir", default="results")

    p_j = sub.add_parser(
        "judge", help="optional: LLM-judge stopping baseline (free hosted open models or Ollama)")
    p_j.add_argument("--backend", choices=BACKENDS,
                     help="ollama, a free hosted provider (" + ", ".join(PROVIDERS) + "), "
                          "openai-compatible (any other server, needs --host), or gemini")
    p_j.add_argument("--model", help="model id exactly as shown in the provider's console")
    p_j.add_argument("--n", type=int, default=100, help="number of HotpotQA questions to judge")
    p_j.add_argument("--host", default=None,
                     help="server URL (needed for openai-compatible; otherwise the provider's default)")
    p_j.add_argument("--api-key-env", default=None,
                     help="name of the environment variable holding the key (openai-compatible)")
    p_j.add_argument("--rpm", type=int, default=None,
                     help="your requests-per-minute limit; sets the spacing between requests")
    p_j.add_argument("--min-interval", type=float, default=None,
                     help="seconds between requests (overrides --rpm; default: ollama 0, hosted 2, gemini 6)")
    p_j.add_argument("--tokens-per-minute", type=int, default=None,
                     help="your tokens-per-minute limit; requests are paced to stay under it")
    p_j.add_argument("--max-tokens", type=int, default=None,
                     help="output token limit (default 16; Gemini 1024; reasoning models need more)")
    p_j.add_argument("--extra-body", default=None,
                     help="""JSON merged into the request, e.g. '{"reasoning_effort": "low"}'""")
    p_j.add_argument("--thinking", choices=THINKING_CHOICES, default=None,
                     help="gemini only: thinking setting")
    p_j.add_argument("--dry-run", action="store_true",
                     help="no API calls: estimate tokens and requests, and how long your limits allow")
    p_j.add_argument("--tpd", type=int, default=None, help="dry run: your tokens-per-day limit")
    p_j.add_argument("--rpd", type=int, default=None, help="dry run: your requests-per-day limit")
    p_j.add_argument("--data-dir", default="data/raw")
    p_j.add_argument("--out-dir", default="results")
    return parser


def _judge(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.dry_run:
        estimate = estimate_judge_cost(
            args.n, args.data_dir, args.out_dir, tpm=args.tokens_per_minute, tpd=args.tpd,
            rpm=args.rpm, rpd=args.rpd)
        print(json.dumps(estimate, indent=1))
        print("Worst case (every hop sent). Real usage is lower because the judge stops at its "
              "first YES. Token counts are estimates (4 characters per token).")
        return
    if not args.backend or not args.model:
        parser.error("judge needs --backend and --model (or --dry-run)")
    extra_body = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except ValueError as err:
            parser.error(f"--extra-body is not valid JSON: {err}")
        if not isinstance(extra_body, dict):
            parser.error("--extra-body must be a JSON object")
    min_interval = args.min_interval
    if min_interval is None and args.rpm:
        min_interval = 60.0 / args.rpm
    try:
        result = run_llm_judge(
            args.backend, args.model, n_questions=args.n, host=args.host, data_dir=args.data_dir,
            out_dir=args.out_dir, min_interval=min_interval,
            tokens_per_minute=args.tokens_per_minute, max_tokens=args.max_tokens,
            extra_body=extra_body, api_key_env=args.api_key_env, thinking=args.thinking)
    except RateLimitExhausted as err:
        raise SystemExit(f"stopped: {err}") from err
    except (RuntimeError, ValueError, FileNotFoundError) as err:
        raise SystemExit(f"error: {err}") from err
    shown = {k: v for k, v in result.items() if k != "stop_hop_per_question"}
    print(json.dumps(shown, indent=1))
    print("run `python -m stopret report` to add it to results/results_tables.md")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.data_dir, force=args.force)
        bad = verify(args.data_dir)
        if bad:
            raise SystemExit(f"verification failed for: {', '.join(bad)}")
        print("all dataset files present and verified (SHA-256)")
    elif args.command == "run":
        metrics = run_all(args.data_dir, args.out_dir, args.rebuild)
        _write_outputs(metrics, Path(args.out_dir))
    elif args.command == "judge":
        _judge(args, parser)
    elif args.command == "report":
        out = Path(args.out_dir)
        _write_outputs(json.loads((out / "metrics.json").read_text()), out)


def _write_outputs(metrics: dict, out: Path) -> None:
    plot_hotpotqa(metrics["datasets"]["hotpotqa"], out / "fig_hotpotqa_frontier.png")
    plot_all_datasets(metrics, out / "fig_all_datasets.png")
    plot_ablation(metrics["ablation_hotpotqa_gbm"], out / "fig_ablation.png")
    write_tables(metrics, out / "results_tables.md")
    print(f"done, see {out}/")
