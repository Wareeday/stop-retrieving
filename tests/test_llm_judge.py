"""LLM-judge plumbing, tested against a mock server that speaks both the Ollama and the
OpenAI-style (Groq) protocol. No real model is involved."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from stopret.llm_judge import (GeminiJudge, GroqJudge, JudgeRefused, LLMJudge, OllamaJudge,
                               RateLimitExhausted, make_judge)


def serve(script):
    """script: list of (status, headers, reply_text); the last entry repeats forever."""
    log = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            log.append({"path": self.path, "headers": dict(self.headers), "payload": payload})
            status, headers, text = script[min(len(log), len(script)) - 1]
            if isinstance(text, dict):           # a ready-made reply body (errors, blocks, ...)
                body = text
            elif "api/generate" in self.path:    # ollama
                body = {"response": text}
            elif ":generateContent" in self.path:  # gemini
                body = {"candidates": [{"content": {"parts": [{"text": text}]},
                                        "finishReason": "STOP"}]}
            else:                                # groq / openai-style
                body = {"choices": [{"message": {"content": text}}]}
            data = json.dumps(body).encode()
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, log, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture
def no_sleep():
    slept = []
    return slept, slept.append


def test_ollama_yes_no_parsing_and_request_body():
    for reply, expected in [(" Yes.", True), ("NO", False), ("I think so", False)]:
        server, log, url = serve([(200, {}, reply)])
        judge = OllamaJudge("fake-model", url)
        assert judge.is_sufficient("Who wrote X?", [("X", "X was written by Y.")]) is expected
        body = log[0]["payload"]
        assert log[0]["path"] == "/api/generate"
        assert "Who wrote X?" in body["prompt"] and "X was written by Y." in body["prompt"]
        assert body["model"] == "fake-model" and body["stream"] is False
        assert body["options"]["temperature"] == 0
        server.shutdown()


def test_groq_request_format_and_auth_header():
    server, log, url = serve([(200, {}, "YES")])
    judge = GroqJudge("some-model", base_url=url, api_key="test-key", min_interval=0)
    assert judge.is_sufficient("Who wrote X?", [("X", "X was written by Y.")]) is True
    request = log[0]
    assert request["path"] == "/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    body = request["payload"]
    assert body["model"] == "some-model" and body["temperature"] == 0 and body["max_tokens"] == 8
    assert body["messages"][0]["role"] == "user"
    assert "Who wrote X?" in body["messages"][0]["content"]
    server.shutdown()


def test_groq_needs_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqJudge("m")
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert GroqJudge("m").api_key == "from-env"


def test_retries_429_with_retry_after_then_succeeds(no_sleep):
    slept, sleep = no_sleep
    server, log, url = serve([(429, {"Retry-After": "7"}, ""), (200, {}, "YES")])
    judge = GroqJudge("m", base_url=url, api_key="k", min_interval=0, sleep=sleep)
    assert judge.is_sufficient("q", [("t", "b")]) is True
    assert len(log) == 2 and slept == [7.0]
    server.shutdown()


def test_server_errors_use_exponential_backoff(no_sleep):
    slept, sleep = no_sleep
    server, log, url = serve([(503, {}, ""), (503, {}, ""), (200, {}, "NO")])
    judge = OllamaJudge("m", url, sleep=sleep)
    assert judge.is_sufficient("q", [("t", "b")]) is False
    assert len(log) == 3 and slept == [1.0, 2.0]
    server.shutdown()


def test_long_rate_limit_wait_raises_instead_of_sleeping(no_sleep):
    slept, sleep = no_sleep
    server, log, url = serve([(429, {"Retry-After": "3600"}, "")])
    judge = GroqJudge("m", base_url=url, api_key="k", min_interval=0, sleep=sleep)
    with pytest.raises(RateLimitExhausted, match="resume"):
        judge.is_sufficient("q", [("t", "b")])
    assert slept == [] and len(log) == 1
    server.shutdown()


def test_client_errors_are_not_retried(no_sleep):
    slept, sleep = no_sleep
    server, log, url = serve([(401, {}, "")])
    judge = GroqJudge("m", base_url=url, api_key="bad", min_interval=0, sleep=sleep)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        judge.is_sufficient("q", [("t", "b")])
    assert len(log) == 1 and slept == []
    server.shutdown()


def test_min_interval_spaces_requests(no_sleep):
    slept, sleep = no_sleep
    server, log, url = serve([(200, {}, "NO")])
    judge = GroqJudge("m", base_url=url, api_key="k", min_interval=5.0, sleep=sleep)
    judge.is_sufficient("q1", [("t", "b")])
    judge.is_sufficient("q2", [("t", "b")])
    assert len(slept) == 1 and 4.0 < slept[0] <= 5.0  # nothing before the first request
    server.shutdown()


def test_parse_strips_think_blocks_and_rejects_empty_answers():
    assert LLMJudge.parse("<think>hmm, maybe\nyes</think>\nYES") is True
    assert LLMJudge.parse("<think>x</think> No") is False
    with pytest.raises(ValueError, match="cut off"):
        LLMJudge.parse("<think>still thinking about it")
    with pytest.raises(ValueError, match="empty answer"):
        LLMJudge.parse("<think>done</think>")
    with pytest.raises(ValueError, match="empty answer"):
        LLMJudge.parse("   ")


def test_cache_avoids_second_request_and_survives_restart(tmp_path):
    server, log, url = serve([(200, {}, "YES")])
    judge = OllamaJudge("m", url, cache_path=tmp_path / "c.json")
    assert judge.is_sufficient("q", [("t", "b")], cache_key="a|1")
    assert judge.is_sufficient("q", [("t", "b")], cache_key="a|1")
    assert len(log) == 1 and judge.n_calls == 1 and judge.prompt_chars > 0
    again = OllamaJudge("m", url, cache_path=tmp_path / "c.json")
    assert again.is_sufficient("q", [("t", "b")], cache_key="a|1")
    assert len(log) == 1 and again.n_calls == 0
    server.shutdown()


def test_make_judge_dispatch(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(make_judge("ollama", "m"), OllamaJudge)
    assert isinstance(make_judge("groq", "m"), GroqJudge)
    assert isinstance(make_judge("gemini", "m"), GeminiJudge)
    with pytest.raises(ValueError):
        make_judge("openai", "m")


# ---------------------------------------------------------------- Gemini

def gemini(url, **kwargs):
    kwargs.setdefault("min_interval", 0)
    return GeminiJudge("gemini-test", base_url=url, api_key="test-key", **kwargs)


def test_gemini_request_format_and_header():
    server, log, url = serve([(200, {}, "YES")])
    assert gemini(url).is_sufficient("Who wrote X?", [("X", "X was written by Y.")]) is True
    request = log[0]
    assert request["path"] == "/models/gemini-test:generateContent"
    headers = {k.lower(): v for k, v in request["headers"].items()}   # header names are case-insensitive
    assert headers["x-goog-api-key"] == "test-key"
    assert "key=" not in request["path"]            # the key never goes into the URL
    body = request["payload"]
    assert "Who wrote X?" in body["contents"][0]["parts"][0]["text"]
    assert body["contents"][0]["role"] == "user"
    assert body["generationConfig"] == {"temperature": 0, "maxOutputTokens": 1024}  # no thinkingConfig
    server.shutdown()


def test_gemini_thinking_settings_and_model_prefix():
    for thinking, expected in [("off", {"thinkingBudget": 0}), ("minimal", {"thinkingLevel": "minimal"}),
                               ("high", {"thinkingLevel": "high"})]:
        server, log, url = serve([(200, {}, "NO")])
        judge = GeminiJudge("models/some-model", base_url=url, api_key="k", min_interval=0,
                            thinking=thinking)
        judge.is_sufficient("q", [("t", "b")])
        assert log[0]["path"] == "/models/some-model:generateContent"   # 'models/' not doubled
        assert log[0]["payload"]["generationConfig"]["thinkingConfig"] == expected
        server.shutdown()
    with pytest.raises(ValueError, match="thinking must be one of"):
        GeminiJudge("m", api_key="k", thinking="extreme")


def test_gemini_api_key_from_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiJudge("m")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert GeminiJudge("m").api_key == "google-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")   # takes precedence
    assert GeminiJudge("m").api_key == "gemini-key"


def test_gemini_ignores_thought_parts():
    reply = {"candidates": [{"content": {"parts": [{"text": "let me think...", "thought": True},
                                                    {"text": "YES"}]}, "finishReason": "STOP"}]}
    server, log, url = serve([(200, {}, reply)])
    assert gemini(url).is_sufficient("q", [("t", "b")]) is True
    server.shutdown()


def test_gemini_429_uses_retry_delay_from_body(no_sleep):
    slept, sleep = no_sleep
    error = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}]}}
    server, log, url = serve([(429, {}, error), (200, {}, "NO")])
    assert gemini(url, sleep=sleep).is_sufficient("q", [("t", "b")]) is False
    assert len(log) == 2 and slept == [34.0]
    server.shutdown()


def test_gemini_daily_quota_style_delay_stops_the_run(no_sleep):
    slept, sleep = no_sleep
    error = {"error": {"code": 429, "details": [{"retryDelay": "40000s"}]}}
    server, log, url = serve([(429, {}, error)])
    with pytest.raises(RateLimitExhausted, match="resume"):
        gemini(url, sleep=sleep).is_sufficient("q", [("t", "b")])
    assert slept == [] and len(log) == 1
    server.shutdown()


def test_persistent_429_without_hints_ends_as_rate_limit_exhausted(no_sleep):
    slept, sleep = no_sleep
    server, log, url = serve([(429, {}, {})])
    with pytest.raises(RateLimitExhausted, match="still rate limited"):
        gemini(url, sleep=sleep).is_sufficient("q", [("t", "b")])
    assert len(log) == 9 and slept[:3] == [1.0, 2.0, 4.0]    # 1 request + 8 retries
    server.shutdown()


def test_gemini_safety_block_is_a_refusal_not_a_crash():
    server, log, url = serve([(200, {}, {"promptFeedback": {"blockReason": "SAFETY"}})])
    judge = gemini(url)
    with pytest.raises(JudgeRefused, match="SAFETY"):
        judge.is_sufficient("q", [("t", "b")])
    assert judge.n_refused == 1 and judge.n_calls == 1
    server.shutdown()
    server, log, url = serve([(200, {}, {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]})])
    with pytest.raises(JudgeRefused):
        gemini(url).is_sufficient("q", [("t", "b")])
    server.shutdown()


def test_gemini_empty_answer_from_token_limit_is_an_error_with_advice():
    server, log, url = serve([(200, {}, {"candidates": [{"content": {"parts": []},
                                                          "finishReason": "MAX_TOKENS"}]})])
    with pytest.raises(ValueError, match="MAX_TOKENS.*thinking"):
        gemini(url).is_sufficient("q", [("t", "b")])
    server.shutdown()
