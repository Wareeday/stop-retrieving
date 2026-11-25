import pytest

from stopret.data import Dataset, Example


@pytest.fixture
def tiny_dataset() -> Dataset:
    """A hand-made two-hop corpus: the answer to each question needs two passages."""
    titles = ["Alice Smith", "Brightwater", "Bob Jones", "Cedar Falls", "Dana Lee", "Eastport"]
    texts = [
        "Alice Smith is a painter who was born in Brightwater.",
        "Brightwater is a small town with a population of 3,000 people.",
        "Bob Jones is a sculptor who lives in Cedar Falls.",
        "Cedar Falls is a city on the Redstone river.",
        "Dana Lee is a poet from Eastport.",
        "Eastport is a harbour town known for its lighthouse.",
    ]
    examples = [
        Example("q1", "What is the population of the town where Alice Smith was born?", "3,000", [0, 1]),
        Example("q2", "Which river runs through the city where Bob Jones lives?", "Redstone", [2, 3]),
    ]
    return Dataset("tiny", titles, texts, examples)


# ---------------------------------------------------------------- a scriptable HTTP server

import json as _json  # noqa: E402
import threading as _threading  # noqa: E402
from http.server import BaseHTTPRequestHandler as _Handler  # noqa: E402
from http.server import HTTPServer as _Server  # noqa: E402


@pytest.fixture
def scripted_server():
    """make(handler) -> (server, log, url). handler(index, request) returns (status, headers,
    body); body may be bytes, str or a dict (sent as JSON). `request` holds method, path,
    lower-cased headers and the decoded JSON payload. The log lists every request received."""
    servers = []

    def make(handler):
        log = []

        class Handler(_Handler):
            def _serve(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                payload = _json.loads(self.rfile.read(length)) if length else None
                request = {"method": method, "path": self.path, "payload": payload,
                           "headers": {k.lower(): v for k, v in self.headers.items()}}
                log.append(request)
                status, headers, body = handler(len(log) - 1, request)
                if isinstance(body, dict):
                    data = _json.dumps(body).encode()
                elif isinstance(body, bytes):
                    data = body
                else:
                    data = str(body).encode()
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self._serve("POST")

            def do_GET(self):
                self._serve("GET")

            def log_message(self, *args):
                pass

        server = _Server(("127.0.0.1", 0), Handler)
        _threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, log, f"http://127.0.0.1:{server.server_port}"

    yield make
    for server in servers:
        server.shutdown()
