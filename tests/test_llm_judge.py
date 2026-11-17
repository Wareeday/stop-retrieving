import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from stopret.llm_judge import OllamaJudge


def _serve(reply: str):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            body = json.dumps({"response": reply}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_yes_and_no_are_parsed_and_prompt_contains_evidence():
    for reply, expected in [(" Yes.", True), ("NO", False), ("I think so", False)]:
        server, seen = _serve(reply)
        judge = OllamaJudge("fake-model", f"http://127.0.0.1:{server.server_port}")
        assert judge.is_sufficient("Who wrote X?", [("X", "X was written by Y.")]) is expected
        assert "Who wrote X?" in seen[0]["prompt"] and "X was written by Y." in seen[0]["prompt"]
        assert seen[0]["model"] == "fake-model" and seen[0]["stream"] is False
        server.shutdown()


def test_cache_avoids_second_request(tmp_path):
    server, seen = _serve("YES")
    judge = OllamaJudge("m", f"http://127.0.0.1:{server.server_port}", cache_path=tmp_path / "c.json")
    assert judge.is_sufficient("q", [("t", "b")], cache_key="a|1")
    assert judge.is_sufficient("q", [("t", "b")], cache_key="a|1")
    assert len(seen) == 1
    # a fresh judge picks the cache up from disk
    again = OllamaJudge("m", f"http://127.0.0.1:{server.server_port}", cache_path=tmp_path / "c.json")
    assert again.is_sufficient("q", [("t", "b")], cache_key="a|1")
    assert len(seen) == 1
    server.shutdown()
