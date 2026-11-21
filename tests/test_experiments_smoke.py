"""End-to-end smoke tests of the LLM-judge runner and CLI against mock servers (no real model).
They need the datasets: run `python -m stopret download` first."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest

from stopret.cli import main
from stopret.experiments import estimate_judge_cost, run_llm_judge
from stopret.report import write_tables

DATA = Path("data/raw")
pytestmark = pytest.mark.skipif(not (DATA / "hotpotqa.json").exists(),
                                reason="run `python -m stopret download` first")
KEYS = {"groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY", "mistral": "MISTRAL_API_KEY",
        "openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY",
        "openai-compatible": "MY_TEST_KEY"}


def mock_server(answer: str):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(self.path)
            if answer == "BLOCKED":
                body = {"promptFeedback": {"blockReason": "SAFETY"}}
            elif ":generateContent" in self.path:
                body = {"candidates": [{"content": {"parts": [{"text": answer}]},
                                        "finishReason": "STOP"}]}
            elif "api/generate" in self.path:
                body = {"response": answer}
            else:
                body = {"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    for key in KEYS.values():
        monkeypatch.setenv(key, "test-key")
    table = Path("results/trajectories_hotpotqa.csv")
    if not table.exists():
        pytest.skip("run `python -m stopret run` first (builds results/trajectories_hotpotqa.csv)")
    (tmp_path / table.name).write_bytes(table.read_bytes())
    return tmp_path


def run(workdir, backend, answer, **options):
    server, requests, url = mock_server(answer)
    options.setdefault("min_interval", 0)
    if backend == "openai-compatible":
        options["api_key_env"] = "MY_TEST_KEY"
    result = run_llm_judge(backend, "mock-model", n_questions=20, host=url, data_dir=str(DATA),
                           out_dir=str(workdir), **options)
    server.shutdown()
    return result, requests


@pytest.mark.parametrize("backend", ["ollama", "groq", "cerebras", "mistral", "openrouter",
                                     "openai-compatible", "gemini"])
def test_always_yes_stops_after_one_hop_and_sends_one_request_per_question(workdir, backend):
    result, requests = run(workdir, backend, "YES")
    assert result["llm_judge"]["cost"] == 1.0
    assert len(requests) == result["requests_sent_this_run"] == 20
    assert result["stop_hop_per_question"] == [1] * 20
    assert (workdir / f"llm_judge_{backend}_mock-model.json").exists()
    assert set(result["vs_fixed_at_same_cost"]) >= {"diff", "diff_lo", "diff_hi"}


def test_always_no_runs_all_four_hops_and_second_run_uses_the_cache(workdir):
    server, requests, url = mock_server("NO")          # one server, so the cache key stays equal
    kwargs = dict(n_questions=20, host=url, data_dir=str(DATA), out_dir=str(workdir), min_interval=0)
    result = run_llm_judge("groq", "mock-model", **kwargs)
    assert result["llm_judge"]["cost"] == 4.0 and len(requests) == 20 * 4
    again = run_llm_judge("groq", "mock-model", **kwargs)    # same settings -> all from cache
    assert len(requests) == 80 and again["answers_from_cache_this_run"] == 80
    assert again["requests_sent_this_run"] == 0 and again["llm_judge"] == result["llm_judge"]
    server.shutdown()


@pytest.mark.skipif(not Path("results/metrics.json").exists(),
                    reason="run `python -m stopret run` first (writes results/metrics.json)")
def test_report_picks_up_judge_result_files(workdir):
    run(workdir, "groq", "YES")
    metrics = json.loads(Path("results/metrics.json").read_text())
    write_tables(metrics, workdir / "results_tables.md")
    text = (workdir / "results_tables.md").read_text()
    assert "LLM-judge stopping baseline" in text and "groq: mock-model" in text


def test_refused_prompts_count_as_no_and_are_reported(workdir):
    result, requests = run(workdir, "gemini", "BLOCKED")
    assert result["llm_judge"]["cost"] == 4.0
    assert result["refused_prompts_this_run"] == len(requests) == 20 * 4


def test_options_that_do_not_apply_are_rejected(workdir):
    with pytest.raises(ValueError, match="thinking"):
        run_llm_judge("groq", "m", n_questions=1, data_dir=str(DATA), out_dir=str(workdir),
                      thinking="off")


def test_extra_body_changes_the_result_file_name(workdir):
    result, _ = run(workdir, "groq", "YES", extra_body={"reasoning_effort": "low"})
    assert result["settings"]["extra_body"] == {"reasoning_effort": "low"}
    assert len(list(workdir.glob("llm_judge_groq_mock-model_x*.json"))) == 1


def test_trajectory_table_that_does_not_match_the_dataset_is_detected(workdir):
    table = (workdir / "trajectories_hotpotqa.csv")
    frame = pd.read_csv(table, dtype={"qid": str})
    frame["qid"] = "wrong"
    frame.to_csv(table, index=False)
    with pytest.raises(ValueError, match="does not match the dataset"):
        run_llm_judge("ollama", "m", n_questions=5, data_dir=str(DATA), out_dir=str(workdir))


def test_dry_run_estimates_without_any_api_call(workdir):
    estimate = estimate_judge_cost(10, str(DATA), str(workdir), tpm=8000, tpd=200_000,
                                   rpm=30, rpd=1000)
    assert estimate["questions"] == 10 and estimate["worst_case_requests"] == 40
    per_hop = estimate["mean_prompt_tokens_per_request_by_hop"]
    assert per_hop == sorted(per_hop) and per_hop[0] > 100       # prompts grow with every hop
    assert estimate["worst_case_prompt_tokens"] == pytest.approx(sum(per_hop) * 10, rel=0.01)
    assert estimate["days_if_limited_by_tokens_per_day"] == pytest.approx(
        estimate["worst_case_prompt_tokens"] / 200_000, abs=0.01)
    assert estimate["minutes_if_limited_by_requests_per_minute"] == pytest.approx(40 / 30, abs=0.1)


def test_cli_judge_end_to_end_and_input_validation(workdir, capsys):
    server, requests, url = mock_server("YES")
    common = ["judge", "--backend", "openai-compatible", "--host", url, "--api-key-env", "MY_TEST_KEY",
              "--model", "m", "--n", "10", "--data-dir", str(DATA), "--out-dir", str(workdir),
              "--rpm", "6000"]
    main(common + ["--extra-body", '{"reasoning_effort": "low"}'])
    shown = json.loads(capsys.readouterr().out.split("\nrun `python")[0])
    assert shown["llm_judge"]["cost"] == 1.0 and len(requests) == 10
    assert "stop_hop_per_question" not in shown
    with pytest.raises(SystemExit):
        main(common + ["--extra-body", "{broken"])
    with pytest.raises(SystemExit):
        main(common + ["--extra-body", "[1, 2]"])
    with pytest.raises(SystemExit):
        main(["judge", "--data-dir", str(DATA), "--out-dir", str(workdir)])     # no backend/model
    server.shutdown()
    main(["judge", "--dry-run", "--n", "5", "--data-dir", str(DATA), "--out-dir", str(workdir),
          "--tpd", "200000"])
    assert "worst_case_requests" in capsys.readouterr().out


def test_cli_stops_cleanly_when_the_daily_limit_is_reached_and_on_config_errors(workdir, monkeypatch):
    server, requests, url = mock_server("YES")
    server.shutdown()                      # replace with a server that always says "try in an hour"

    class Limited(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(429)
            self.send_header("Retry-After", "3600")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    limited = HTTPServer(("127.0.0.1", 0), Limited)
    threading.Thread(target=limited.serve_forever, daemon=True).start()
    base = ["judge", "--backend", "groq", "--model", "m", "--n", "5", "--data-dir", str(DATA),
            "--out-dir", str(workdir), "--min-interval", "0"]
    with pytest.raises(SystemExit, match="stopped: .*resume"):
        main(base + ["--host", f"http://127.0.0.1:{limited.server_port}"])
    limited.shutdown()
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(SystemExit, match="error: set the GROQ_API_KEY"):
        main(base)
    with pytest.raises(SystemExit, match="error: option"):
        main(base[:1] + ["--backend", "ollama", "--model", "m", "--thinking", "off",
                         "--data-dir", str(DATA), "--out-dir", str(workdir)])
