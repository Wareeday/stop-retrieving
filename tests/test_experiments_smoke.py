"""End-to-end smoke test of the LLM-judge runner against mock servers (no real model)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from stopret.experiments import run_llm_judge

DATA = Path("data/raw")
pytestmark = pytest.mark.skipif(not (DATA / "hotpotqa.json").exists(),
                                reason="run `python -m stopret download` first")


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
                body = {"choices": [{"message": {"content": answer}}]}
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


def run(tmp_path, backend, answer, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    (tmp_path / "trajectories_hotpotqa.csv").write_bytes(
        Path("results/trajectories_hotpotqa.csv").read_bytes())
    server, requests, url = mock_server(answer)
    result = run_llm_judge(backend, "mock-model", n_questions=20, host=url, data_dir=str(DATA),
                           out_dir=str(tmp_path), min_interval=0)
    server.shutdown()
    return result, requests


@pytest.mark.parametrize("backend", ["ollama", "groq", "gemini"])
def test_always_yes_stops_after_one_hop_and_sends_one_request_per_question(tmp_path, monkeypatch, backend):
    result, requests = run(tmp_path, backend, "YES", monkeypatch)
    assert result["llm_judge"]["cost"] == 1.0
    assert len(requests) == result["requests_sent_this_run"] == 20
    assert (tmp_path / f"llm_judge_{backend}_mock-model.json").exists()


def test_always_no_runs_all_four_hops(tmp_path, monkeypatch):
    result, requests = run(tmp_path, "groq", "NO", monkeypatch)
    assert result["llm_judge"]["cost"] == 4.0
    assert len(requests) == 20 * 4
    # the result file is picked up by the report
    from stopret.report import write_tables
    import json as _json
    metrics = _json.loads(Path("results/metrics.json").read_text())
    write_tables(metrics, tmp_path / "results_tables.md")
    assert "LLM-judge stopping baseline" in (tmp_path / "results_tables.md").read_text()


def test_refused_prompts_count_as_no_and_are_reported(tmp_path, monkeypatch):
    result, requests = run(tmp_path, "gemini", "BLOCKED", monkeypatch)
    assert result["llm_judge"]["cost"] == 4.0                     # every hop answered "no"
    assert result["refused_prompts_this_run"] == len(requests) == 20 * 4


def test_thinking_setting_only_for_gemini(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="only applies to the gemini backend"):
        run_llm_judge("groq", "m", n_questions=1, data_dir=str(DATA), out_dir=str(tmp_path),
                      thinking="off")
