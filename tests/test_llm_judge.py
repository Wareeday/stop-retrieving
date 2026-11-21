"""LLM-judge plumbing, tested against a mock server that speaks the Ollama, OpenAI-style
(Groq, Cerebras, Mistral, OpenRouter, ...) and Gemini protocols. No real model is involved."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from stopret.llm_judge import (BACKENDS, PROVIDERS, GeminiJudge, JudgeRefused, LLMJudge,
                               OllamaJudge, OpenAICompatibleJudge, RateLimiter,
                               RateLimitExhausted, build_prompt, estimate_tokens, make_judge)

PASSAGES = [("X", "X was written by Y.")]


def serve(script):
    """script: list of (status, headers, reply); the last entry repeats forever. A reply that is a
    dict is sent as the raw body, a string is wrapped in the protocol the path belongs to."""
    log = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            log.append({"path": self.path, "headers": dict(self.headers), "payload": payload})
            status, headers, reply = script[min(len(log), len(script)) - 1]
            if isinstance(reply, dict):
                body = reply
            elif "api/generate" in self.path:        # ollama
                body = {"response": reply}
            elif ":generateContent" in self.path:     # gemini
                body = {"candidates": [{"content": {"parts": [{"text": reply}]},
                                        "finishReason": "STOP"}]}
            else:                                     # openai-style
                body = {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]}
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


class FakeClock:
    """A clock whose sleep() moves time forward, so pacing can be tested instantly."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


def compat(url, clock=None, **kwargs):
    kwargs.setdefault("min_interval", 0)
    if clock is not None:
        kwargs.update(clock=clock, sleep=clock.sleep)
    return OpenAICompatibleJudge("test-model", base_url=url, api_key="test-key", **kwargs)


# ------------------------------------------------------------------ prompt and parsing

def test_build_prompt_contains_question_and_numbered_passages():
    prompt = build_prompt("Who wrote X?", PASSAGES + [("Z", "Z is a place.")])
    assert "Who wrote X?" in prompt and "[1] X: X was written by Y." in prompt
    assert "[2] Z: Z is a place." in prompt and prompt.rstrip().endswith("YES or NO.")
    assert estimate_tokens("a" * 40) == 10


def test_parse_strips_think_blocks_and_rejects_bad_answers():
    assert LLMJudge.parse(" Yes.") is True and LLMJudge.parse("NO") is False
    assert LLMJudge.parse("I think so") is False
    assert LLMJudge.parse("<think>hmm, maybe\nyes</think>\nYES") is True
    with pytest.raises(ValueError, match="cut off"):
        LLMJudge.parse("<think>still thinking about it")
    with pytest.raises(ValueError, match="empty answer"):
        LLMJudge.parse("<think>done</think>")
    with pytest.raises(ValueError, match="empty answer"):
        LLMJudge.parse("   ")


# ------------------------------------------------------------------ ollama

def test_ollama_request_body_and_answers():
    for reply, expected in [(" Yes.", True), ("NO", False)]:
        server, log, url = serve([(200, {}, reply)])
        judge = OllamaJudge("fake-model", url)
        assert judge.is_sufficient("Who wrote X?", PASSAGES) is expected
        body = log[0]["payload"]
        assert log[0]["path"] == "/api/generate" and body["model"] == "fake-model"
        assert "Who wrote X?" in body["prompt"] and body["stream"] is False
        assert body["options"] == {"temperature": 0, "num_predict": 16}
        server.shutdown()


# ------------------------------------------------------------------ openai-compatible providers

def test_provider_presets_have_the_documented_endpoints():
    assert PROVIDERS["groq"].base_url == "https://api.groq.com/openai/v1"
    assert PROVIDERS["cerebras"].base_url == "https://api.cerebras.ai/v1"
    assert PROVIDERS["mistral"].base_url == "https://api.mistral.ai/v1"
    assert PROVIDERS["openrouter"].base_url == "https://openrouter.ai/api/v1"
    assert {p.key_env for p in PROVIDERS.values()} == {
        "GROQ_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"}
    assert set(BACKENDS) == {"ollama", *PROVIDERS, "openai-compatible", "gemini"}


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_every_preset_sends_an_openai_style_request(provider, monkeypatch):
    monkeypatch.setenv(PROVIDERS[provider].key_env, "secret-key")
    server, log, url = serve([(200, {}, "YES")])
    judge = make_judge(provider, "some-model", host=url, min_interval=0)
    assert judge.backend == provider
    assert judge.is_sufficient("Who wrote X?", PASSAGES) is True
    request = log[0]
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert request["path"] == "/chat/completions" and headers["authorization"] == "Bearer secret-key"
    body = request["payload"]
    assert body["model"] == "some-model" and body["temperature"] == 0 and body["max_tokens"] == 16
    assert body["messages"] == [{"role": "user", "content": build_prompt("Who wrote X?", PASSAGES)}]
    server.shutdown()


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_missing_key_names_the_environment_variable(provider, monkeypatch):
    monkeypatch.delenv(PROVIDERS[provider].key_env, raising=False)
    with pytest.raises(RuntimeError, match=PROVIDERS[provider].key_env):
        make_judge(provider, "m")


def test_generic_openai_compatible_backend(monkeypatch):
    monkeypatch.setenv("MY_SERVER_KEY", "abc")
    server, log, url = serve([(200, {}, "NO")])
    judge = make_judge("openai-compatible", "m", host=url + "/v1/", api_key_env="MY_SERVER_KEY",
                       min_interval=0)
    assert judge.is_sufficient("q", PASSAGES) is False
    assert log[0]["path"] == "/v1/chat/completions"
    server.shutdown()
    with pytest.raises(ValueError, match="base URL"):
        make_judge("openai-compatible", "m", api_key_env="MY_SERVER_KEY")
    with pytest.raises(RuntimeError, match="--api-key-env"):
        make_judge("openai-compatible", "m", host=url)


def test_extra_body_is_merged_and_cannot_override_the_prompt():
    server, log, url = serve([(200, {}, "YES")])
    judge = compat(url, extra_body={"reasoning_effort": "low"}, max_tokens=256)
    judge.is_sufficient("q", PASSAGES)
    body = log[0]["payload"]
    assert body["reasoning_effort"] == "low" and body["max_tokens"] == 256
    server.shutdown()
    with pytest.raises(ValueError, match="messages"):
        compat(url, extra_body={"messages": []})


def test_reasoning_only_reply_explains_how_to_fix_it():
    reply = {"choices": [{"message": {"content": "", "reasoning": "let me think"},
                          "finish_reason": "length"}]}
    server, log, url = serve([(200, {}, reply)])
    with pytest.raises(ValueError) as info:
        compat(url).is_sufficient("q", PASSAGES)
    message = str(info.value)
    assert "only reasoning" in message and "--max-tokens" in message
    assert "reasoning_effort" in message and "OpenRouter" in message
    server.shutdown()


def test_content_filter_is_a_refusal():
    reply = {"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}
    server, log, url = serve([(200, {}, reply)])
    judge = compat(url)
    with pytest.raises(JudgeRefused):
        judge.is_sufficient("q", PASSAGES)
    assert judge.n_refused == 1
    server.shutdown()


# ------------------------------------------------------------------ retries and rate limits

def test_retries_429_with_retry_after_then_succeeds(clock):
    server, log, url = serve([(429, {"Retry-After": "7"}, {}), (200, {}, "YES")])
    assert compat(url, clock).is_sufficient("q", PASSAGES) is True
    assert len(log) == 2 and clock.slept == [7.0]
    server.shutdown()


def test_server_errors_use_exponential_backoff(clock):
    server, log, url = serve([(503, {}, {}), (503, {}, {}), (200, {}, "NO")])
    assert OllamaJudge("m", url, clock=clock, sleep=clock.sleep).is_sufficient("q", PASSAGES) is False
    assert len(log) == 3 and clock.slept == [1.0, 2.0]
    server.shutdown()


def test_long_rate_limit_wait_raises_instead_of_sleeping(clock):
    server, log, url = serve([(429, {"Retry-After": "3600"}, {})])
    with pytest.raises(RateLimitExhausted, match="resume"):
        compat(url, clock).is_sufficient("q", PASSAGES)
    assert clock.slept == [] and len(log) == 1
    server.shutdown()


def test_persistent_429_without_hints_ends_as_rate_limit_exhausted(clock):
    server, log, url = serve([(429, {}, {})])
    with pytest.raises(RateLimitExhausted, match="still rate limited"):
        compat(url, clock).is_sufficient("q", PASSAGES)
    assert len(log) == 9 and clock.slept[:3] == [1.0, 2.0, 4.0]    # 1 request + 8 retries
    server.shutdown()


def test_client_errors_are_not_retried(clock):
    server, log, url = serve([(401, {}, {})])
    with pytest.raises(RuntimeError, match="HTTP 401"):
        compat(url, clock).is_sufficient("q", PASSAGES)
    assert len(log) == 1 and clock.slept == []
    server.shutdown()


# ------------------------------------------------------------------ pacing

def test_limiter_spaces_requests(clock):
    limiter = RateLimiter(min_interval=5.0, clock=clock, sleep=clock.sleep)
    limiter.wait(10)
    limiter.wait(10)
    limiter.wait(10)
    assert clock.slept == [5.0, 5.0]       # nothing before the first request


def test_limiter_keeps_tokens_per_minute_under_the_limit(clock):
    limiter = RateLimiter(tokens_per_minute=1000, clock=clock, sleep=clock.sleep)
    start = clock.now
    limiter.wait(600)                       # 600 in the window
    limiter.wait(300)                       # 900 in the window: still fits, no waiting
    assert clock.slept == []
    limiter.wait(300)                       # 1,200 would exceed 1,000: wait until the first
    assert clock.now - start == pytest.approx(60.0)   # entry leaves the 60-second window
    limiter.wait(600)                       # only the 300 just sent remains in the window: 900 fits
    assert clock.now - start == pytest.approx(60.0)   # so no further waiting


def test_limiter_never_blocks_on_a_single_oversized_request(clock):
    limiter = RateLimiter(tokens_per_minute=100, clock=clock, sleep=clock.sleep)
    limiter.wait(5000)                      # can never fit: send it and let the server decide
    assert clock.slept == []


def test_judge_paces_by_estimated_tokens(clock):
    server, log, url = serve([(200, {}, "NO")])
    big = [("T", "word " * 400)]            # ~2,000 characters -> ~500 tokens (+20% +16 output)
    judge = compat(url, clock, tokens_per_minute=1000)
    judge.is_sufficient("q1", big)
    judge.is_sufficient("q2", big)          # fits in the window together? 2 * ~630 > 1000 -> waits
    assert len(log) == 2 and sum(clock.slept) == pytest.approx(60.0)
    server.shutdown()


# ------------------------------------------------------------------ cache

def test_cache_is_keyed_by_prompt_and_survives_restart(tmp_path):
    server, log, url = serve([(200, {}, "YES")])
    path = tmp_path / "c.json"
    judge = compat(url, cache_path=path)
    assert judge.is_sufficient("q", PASSAGES) and judge.is_sufficient("q", PASSAGES)
    assert len(log) == 1 and judge.n_calls == 1 and judge.n_cache_hits == 1
    judge.is_sufficient("q", PASSAGES + [("Z", "more evidence")])     # different prompt -> new request
    assert len(log) == 2
    again = compat(url, cache_path=path)
    assert again.is_sufficient("q", PASSAGES) and again.n_calls == 0 and len(log) == 2
    assert not list(tmp_path.glob("*.tmp"))                           # atomic write left no temp file
    server.shutdown()


def test_cache_does_not_leak_between_different_settings(tmp_path):
    server, log, url = serve([(200, {}, "YES")])
    path = tmp_path / "c.json"
    compat(url, cache_path=path).is_sufficient("q", PASSAGES)
    other = compat(url, cache_path=path, extra_body={"reasoning_effort": "low"})
    other.is_sufficient("q", PASSAGES)
    assert len(log) == 2                  # same prompt, different settings -> asked again
    server.shutdown()


def test_corrupt_cache_is_set_aside_not_fatal(tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text("{not json")
    judge = compat("http://127.0.0.1:1", cache_path=path)
    assert judge.cache == {} and (tmp_path / "c.corrupt").exists()
    assert "unreadable" in capsys.readouterr().out


# ------------------------------------------------------------------ factory

def test_make_judge_rejects_options_that_do_not_apply(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    with pytest.raises(ValueError, match="thinking"):
        make_judge("ollama", "m", thinking="off")
    with pytest.raises(ValueError, match="extra_body"):
        make_judge("gemini", "m", extra_body={"a": 1})
    with pytest.raises(ValueError, match="thinking"):
        make_judge("groq", "m", thinking="off")
    with pytest.raises(ValueError, match="unknown backend"):
        make_judge("nonexistent", "m")
    assert isinstance(make_judge("ollama", "m"), OllamaJudge)
    assert isinstance(make_judge("groq", "m"), OpenAICompatibleJudge)
    assert isinstance(make_judge("gemini", "m"), GeminiJudge)


# ------------------------------------------------------------------ gemini (optional, proprietary)

def gemini(url, clock=None, **kwargs):
    kwargs.setdefault("min_interval", 0)
    if clock is not None:
        kwargs.update(clock=clock, sleep=clock.sleep)
    return GeminiJudge("gemini-test", base_url=url, api_key="test-key", **kwargs)


def test_gemini_request_format_and_header():
    server, log, url = serve([(200, {}, "YES")])
    assert gemini(url).is_sufficient("Who wrote X?", PASSAGES) is True
    request = log[0]
    assert request["path"] == "/models/gemini-test:generateContent"
    headers = {k.lower(): v for k, v in request["headers"].items()}   # header names are case-insensitive
    assert headers["x-goog-api-key"] == "test-key" and "key=" not in request["path"]
    body = request["payload"]
    assert "Who wrote X?" in body["contents"][0]["parts"][0]["text"]
    assert body["generationConfig"] == {"temperature": 0, "maxOutputTokens": 1024}
    server.shutdown()


def test_gemini_thinking_settings_and_model_prefix():
    for thinking, expected in [("off", {"thinkingBudget": 0}), ("minimal", {"thinkingLevel": "minimal"}),
                               ("high", {"thinkingLevel": "high"})]:
        server, log, url = serve([(200, {}, "NO")])
        GeminiJudge("models/some-model", base_url=url, api_key="k", min_interval=0,
                    thinking=thinking).is_sufficient("q", PASSAGES)
        assert log[0]["path"] == "/models/some-model:generateContent"
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
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert GeminiJudge("m").api_key == "gemini-key"


def test_gemini_ignores_thought_parts():
    reply = {"candidates": [{"content": {"parts": [{"text": "let me think...", "thought": True},
                                                    {"text": "YES"}]}, "finishReason": "STOP"}]}
    server, log, url = serve([(200, {}, reply)])
    assert gemini(url).is_sufficient("q", PASSAGES) is True
    server.shutdown()


def test_gemini_429_uses_retry_delay_from_body(clock):
    error = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}]}}
    server, log, url = serve([(429, {}, error), (200, {}, "NO")])
    assert gemini(url, clock).is_sufficient("q", PASSAGES) is False
    assert len(log) == 2 and clock.slept == [34.0]
    server.shutdown()


def test_gemini_daily_quota_style_delay_stops_the_run(clock):
    error = {"error": {"code": 429, "details": [{"retryDelay": "40000s"}]}}
    server, log, url = serve([(429, {}, error)])
    with pytest.raises(RateLimitExhausted, match="resume"):
        gemini(url, clock).is_sufficient("q", PASSAGES)
    assert clock.slept == [] and len(log) == 1
    server.shutdown()


def test_gemini_safety_block_is_a_refusal_not_a_crash():
    server, log, url = serve([(200, {}, {"promptFeedback": {"blockReason": "SAFETY"}})])
    judge = gemini(url)
    with pytest.raises(JudgeRefused, match="SAFETY"):
        judge.is_sufficient("q", PASSAGES)
    assert judge.n_refused == 1 and judge.n_calls == 1
    server.shutdown()
    server, log, url = serve([(200, {}, {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]})])
    with pytest.raises(JudgeRefused):
        gemini(url).is_sufficient("q", PASSAGES)
    server.shutdown()


def test_gemini_empty_answer_from_token_limit_is_an_error_with_advice():
    reply = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
    server, log, url = serve([(200, {}, reply)])
    with pytest.raises(ValueError, match="MAX_TOKENS.*thinking"):
        gemini(url).is_sufficient("q", PASSAGES)
    server.shutdown()
