"""End-to-end smoke test of the LLM-judge runner against a mock Ollama server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from stopret.experiments import run_llm_judge

DATA = Path("data/raw")


@pytest.mark.skipif(not (DATA / "hotpotqa.json").exists(), reason="run `python -m stopret download` first")
def test_llm_judge_runner_with_always_yes_mock(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"response": "YES"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (tmp_path / "trajectories_hotpotqa.csv").write_bytes(Path("results/trajectories_hotpotqa.csv").read_bytes())
    result = run_llm_judge("mock", n_questions=20, host=f"http://127.0.0.1:{server.server_port}",
                           data_dir=str(DATA), out_dir=str(tmp_path))
    server.shutdown()
    assert result["llm_judge"]["cost"] == 1.0  # always YES -> always stops after hop 1
    assert 0.0 <= result["llm_judge"]["recall"] <= 1.0
    assert (tmp_path / "llm_judge_mock.json").exists()
