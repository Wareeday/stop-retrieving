"""Command line entry point: python -m stopret {download,run,report,judge,check}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import download, verify
from .experiments import estimate_judge_cost, run_all, run_llm_judge
from .llm_http import ApiError, AuthBlocked, RateLimitExhausted
from .llm_judge import BACKENDS, PROVIDERS, THINKING_CHOICES, check_backend
from .plots import plot_ablation, plot_all_datasets, plot_hotpotqa
from .report import write_tables


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def _add_provider_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=BACKENDS,
                   help="ollama, a free hosted provider (" + ", ".join(PROVIDERS) + "), "
                        "openai-compatible (any other server, needs --host), or gemini")
    p.add_argument("--model", help="model id exactly as shown in the provider's console")
    p.add_argument("--host", default=None,
                   help="server URL (needed for openai-compatible; otherwise the provider's default)")
    p.add_argument("--api-key-env", default=None,
                   help="name of the environment variable holding the key (openai-compatible)")
    p.add_argument("--user-agent", default=None,
                   help="User-Agent header (default: a browser-like one, which avoids Cloudflare "
                        "error 1010; the STOPRET_USER_AGENT variable works too)")
    p.add_argument("--rpm", type=_positive_int, default=None,
                   help="your requests-per-minute limit; sets the spacing between requests")
    p.add_argument("--min-interval", type=_non_negative_float, default=None,
                   help="seconds between requests (overrides --rpm; default: ollama 0, hosted 2, gemini 6)")
    p.add_argument("--tokens-per-minute", type=_positive_int, default=None,
                   help="your tokens-per-minute limit; requests are paced to stay under it")
    p.add_argument("--max-tokens", type=_positive_int, default=None,
                   help="output token limit (default 16; Gemini 1024; reasoning models need more)")
    p.add_argument("--extra-body", default=None,
                   help="""JSON merged into the request, e.g. '{"reasoning_effort": "low"}'""")
    p.add_argument("--thinking", choices=THINKING_CHOICES, default=None,
                   help="gemini only: thinking setting")


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
    _add_provider_options(p_j)
    p_j.add_argument("--n", type=_positive_int, default=100,
                     help="number of HotpotQA questions to judge")
    p_j.add_argument("--dry-run", action="store_true",
                     help="no API calls: estimate tokens and requests, and how long your limits allow")
    p_j.add_argument("--tpd", type=_positive_int, default=None, help="dry run: your tokens-per-day limit")
    p_j.add_argument("--rpd", type=_positive_int, default=None, help="dry run: your requests-per-day limit")
    p_j.add_argument("--data-dir", default="data/raw")
    p_j.add_argument("--out-dir", default="results")

    p_c = sub.add_parser(
        "check", help="test your key and connection with one request that costs no tokens "
                      "(lists the model ids your key can use)")
    _add_provider_options(p_c)
    p_c.add_argument("--ping", action="store_true",
                     help="also ask one tiny real question (needs --model; uses about 100 tokens)")
    return parser


def _parse_extra_body(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict | None:
    if not args.extra_body:
        return None
    try:
        extra_body = json.loads(args.extra_body)
    except ValueError as err:
        parser.error(f"--extra-body is not valid JSON: {err}")
    if not isinstance(extra_body, dict):
        parser.error("--extra-body must be a JSON object")
    return extra_body


def _min_interval(args: argparse.Namespace) -> float | None:
    if args.min_interval is not None:
        return args.min_interval
    return 60.0 / args.rpm if args.rpm else None


def _provider_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    return {"min_interval": _min_interval(args), "tokens_per_minute": args.tokens_per_minute,
            "max_tokens": args.max_tokens, "extra_body": _parse_extra_body(args, parser),
            "api_key_env": args.api_key_env, "thinking": args.thinking,
            "user_agent": args.user_agent}


def _fail(err: Exception) -> SystemExit:
    """One place that turns every expected failure into a clean message and a non-zero exit."""
    if isinstance(err, AuthBlocked):
        return SystemExit(str(err))                    # already starts with "STOP: ..."
    if isinstance(err, RateLimitExhausted):
        return SystemExit(f"stopped: {err}")
    return SystemExit(f"error: {err}")


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
    options = _provider_options(args, parser)
    try:
        result = run_llm_judge(args.backend, args.model, n_questions=args.n, host=args.host,
                               data_dir=args.data_dir, out_dir=args.out_dir, **options)
    except KeyboardInterrupt:
        raise SystemExit("interrupted: answers received so far are saved; run the same command "
                         "to continue where it stopped") from None
    except (ApiError, RuntimeError, ValueError, OSError) as err:
        raise _fail(err) from err
    shown = {k: v for k, v in result.items() if k != "stop_hop_per_question"}
    print(json.dumps(shown, indent=1))
    print("run `python -m stopret report` to add it to results/results_tables.md")


def _check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.backend:
        parser.error("check needs --backend")
    options = {k: v for k, v in _provider_options(args, parser).items() if v is not None}
    try:
        result = check_backend(args.backend, args.model, args.host, ping=args.ping, **options)
    except KeyboardInterrupt:
        raise SystemExit("interrupted") from None
    except (ApiError, RuntimeError, ValueError, OSError) as err:
        raise _fail(err) from err
    print(f"OK: key accepted and {args.backend} reachable; {result['models_available']} model ids "
          "visible to this key.")
    if args.model:
        if result["model_found"]:
            print(f"OK: model id {args.model!r} exists.")
        else:
            print(f"WARNING: model id {args.model!r} was not found.")
            if result["close_matches"]:
                print("  did you mean: " + ", ".join(result["close_matches"]))
    print("model ids (first 40):")
    for name in result["sample_models"]:
        print("  " + name)
    if "ping" in result:
        ping = result["ping"]
        print(f"ping: the model answered {'YES' if ping['answered_yes'] else 'NO/other'} in "
              f"{ping['seconds']} s (the correct answer is YES)")


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
    elif args.command == "check":
        _check(args, parser)
    elif args.command == "report":
        out = Path(args.out_dir)
        _write_outputs(json.loads((out / "metrics.json").read_text()), out)


def _write_outputs(metrics: dict, out: Path) -> None:
    plot_hotpotqa(metrics["datasets"]["hotpotqa"], out / "fig_hotpotqa_frontier.png")
    plot_all_datasets(metrics, out / "fig_all_datasets.png")
    plot_ablation(metrics["ablation_hotpotqa_gbm"], out / "fig_ablation.png")
    write_tables(metrics, out / "results_tables.md")
    print(f"done, see {out}/")
