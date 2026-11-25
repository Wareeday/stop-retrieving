"""The HTTP layer: browser User-Agent, Cloudflare 1010, 401/403 stop, retries, network faults,
secrets never leaking, and "a failed request is never counted as done"."""
import email.utils
import http.client
import ssl
import time
import urllib.error
import urllib.request

import pytest

from stopret import llm_http
from stopret.llm_http import (
    DEFAULT_USER_AGENT,
    STOP_HEADLINE,
    USER_AGENT_ENV,
    ApiError,
    AuthBlocked,
    RateLimitExhausted,
    clean_api_key,
    get_json,
    post_json,
    resolve_user_agent,
)
from stopret.llm_judge import (
    PROVIDERS,
    GeminiJudge,
    JudgeRefused,
    LLMJudge,
    OllamaJudge,
    OpenAICompatibleJudge,
    check_backend,
    make_judge,
)

PASSAGES = [("X", "X was written by Y.")]
GOOD = {"choices": [{"message": {"content": "NO"}, "finish_reason": "stop"}]}
CHROME = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"


class Clock:
    def __init__(self):
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)


@pytest.fixture
def clock():
    return Clock()


def cloudflare_like(index, request):
    """What api.groq.com does to Python's default signature, and to a normal browser one."""
    if "python" in request["headers"].get("user-agent", "").lower():
        return 403, {"Content-Type": "text/plain"}, "error code: 1010"
    return 200, {}, GOOD


# ------------------------------------------------------------------ the User-Agent

def test_default_user_agent_is_the_requested_browser_string():
    assert DEFAULT_USER_AGENT == CHROME
    assert resolve_user_agent() == CHROME


def test_user_agent_override_order_and_validation(monkeypatch):
    monkeypatch.delenv(USER_AGENT_ENV, raising=False)
    monkeypatch.setenv(USER_AGENT_ENV, "from-env/1.0")
    assert resolve_user_agent() == "from-env/1.0"
    assert resolve_user_agent("explicit/2.0") == "explicit/2.0"       # argument beats the variable
    for bad in ("with\nnewline", "ünïcode"):
        with pytest.raises(ValueError, match="printable ASCII"):
            resolve_user_agent(bad)


def test_every_backend_sends_the_browser_user_agent_on_posts_and_gets(scripted_server, monkeypatch):
    for provider in PROVIDERS.values():
        monkeypatch.setenv(provider.key_env, "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("MY_KEY", "k")
    monkeypatch.delenv(USER_AGENT_ENV, raising=False)
    reply = {"choices": [{"message": {"content": "YES"}}], "response": "YES", "data": [], "models": [],
             "candidates": [{"content": {"parts": [{"text": "YES"}]}}]}
    server, log, url = scripted_server(lambda i, r: (200, {}, reply))
    for backend in ("ollama", *PROVIDERS, "openai-compatible", "gemini"):
        options = {"api_key_env": "MY_KEY"} if backend == "openai-compatible" else {}
        judge = make_judge(backend, "m", url, min_interval=0, **options)
        judge.is_sufficient("q", PASSAGES)
        judge.list_models()
    assert len(log) == 14
    assert {r["headers"]["user-agent"] for r in log} == {CHROME}
    assert {r["headers"]["accept"] for r in log} == {"application/json"}
    assert "python" not in " ".join(r["headers"]["user-agent"] for r in log).lower()


def test_user_agent_can_be_changed_per_judge(scripted_server):
    server, log, url = scripted_server(lambda i, r: (200, {}, GOOD))
    OpenAICompatibleJudge("m", base_url=url, api_key="k", min_interval=0,
                          user_agent="my-tool/9").is_sufficient("q", PASSAGES)
    assert log[0]["headers"]["user-agent"] == "my-tool/9"


# ------------------------------------------------------------------ Cloudflare 1010 and 401/403

def test_python_default_signature_is_blocked_but_the_fixed_client_gets_through(scripted_server, clock):
    """Reproduces the reported failure: the server answers 403 / error code 1010 to Python's
    default User-Agent. The old client sent one; the new client does not."""
    server, log, url = scripted_server(cloudflare_like)
    old_style = urllib.request.Request(url + "/chat/completions", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(old_style)
    assert info.value.code == 403 and b"1010" in info.value.read()
    judge = OpenAICompatibleJudge("m", base_url=url, api_key="k", min_interval=0)
    assert judge.is_sufficient("q", PASSAGES) is False            # NO, but a real answer
    with pytest.raises(AuthBlocked):                              # and a Python UA still fails loudly
        OpenAICompatibleJudge("m", base_url=url, api_key="k", min_interval=0,
                              user_agent="python-requests/2.31").is_sufficient("q", PASSAGES)


@pytest.mark.parametrize(("status", "body", "expected"), [
    (403, "error code: 1010", "browser signature"),
    (403, "<html>Error 1010: the owner has banned your access based on your browser's signature</html>",
     "browser signature"),
    (403, "error code: 1015", "rate limiting your IP"),
    (403, "error code: 1020", "firewall rule"),
    (403, "error code: 1006", "banned your IP"),
    (403, '{"error": {"message": "forbidden"}}', "refused access"),
    (401, '{"error": {"message": "Invalid API Key"}}', "key was rejected"),
])
def test_401_and_403_stop_at_once_with_the_stop_message_and_a_specific_hint(
        scripted_server, clock, status, body, expected):
    server, log, url = scripted_server(lambda i, r: (status, {}, body))
    with pytest.raises(AuthBlocked) as info:
        post_json(url + "/chat/completions", {"x": 1}, sleep=clock.sleep)
    message = str(info.value)
    assert message.startswith(STOP_HEADLINE)
    assert STOP_HEADLINE == "STOP: API Key invalid or blocked by Cloudflare (1010)"
    assert f"HTTP {status}" in message and expected in message and "asked again" in message
    assert info.value.status == status
    assert len(log) == 1 and clock.slept == []                     # no retry, no waiting


def test_the_api_key_never_appears_in_error_messages(scripted_server):
    secret = "gsk_TOPSECRET1234567890"
    server, log, url = scripted_server(
        lambda i, r: (401, {}, '{"error": "Incorrect API key provided: ' + secret + '"}'))
    judge = OpenAICompatibleJudge("m", base_url=url, api_key=secret, min_interval=0)
    with pytest.raises(AuthBlocked) as info:
        judge.is_sufficient("q", PASSAGES)
    assert secret not in str(info.value) and "***" in str(info.value)


def test_a_failed_request_is_not_cached_so_the_question_is_asked_again(scripted_server, tmp_path):
    mode = {"fail": True}
    server, log, url = scripted_server(
        lambda i, r: (403, {}, "error code: 1010") if mode["fail"] else (200, {}, GOOD))
    cache = tmp_path / "c.json"
    judge = OpenAICompatibleJudge("m", base_url=url, api_key="k", min_interval=0, cache_path=cache)
    with pytest.raises(AuthBlocked):
        judge.is_sufficient("q", PASSAGES)
    assert judge.cache == {} and not cache.exists()
    mode["fail"] = False
    again = OpenAICompatibleJudge("m", base_url=url, api_key="k", min_interval=0, cache_path=cache)
    assert again.is_sufficient("q", PASSAGES) is False
    assert len(log) == 2 and again.n_calls == 1                    # the failed question was retried


# ------------------------------------------------------------------ retries

def test_429_408_and_5xx_are_retried_with_backoff(scripted_server, clock):
    script = [(429, {}, "{}"), (408, {}, "{}"), (500, {}, "{}"), (502, {}, "{}"), (200, {}, GOOD)]
    server, log, url = scripted_server(lambda i, r: script[i])
    assert post_json(url, {}, sleep=clock.sleep) == GOOD
    assert len(log) == 5 and clock.slept == [1.0, 2.0, 4.0, 8.0]


def test_retry_after_as_http_date_and_garbage(scripted_server, clock):
    future = email.utils.formatdate(time.time() + 30, usegmt=True)
    script = [(429, {"Retry-After": future}, "{}"), (429, {"Retry-After": "soon"}, "{}"),
              (200, {}, GOOD)]
    server, log, url = scripted_server(lambda i, r: script[i])
    post_json(url, {}, sleep=clock.sleep)
    assert 25 <= clock.slept[0] <= 31 and clock.slept[1] == 2.0     # date parsed; garbage -> backoff


@pytest.mark.parametrize(("status", "hint"), [
    (400, "model id and any options"), (404, "stopret check"), (413, "tokens-per-minute"),
    (422, "not accepted")])
def test_other_client_errors_are_not_retried_and_explain_themselves(
        scripted_server, clock, status, hint):
    server, log, url = scripted_server(lambda i, r: (status, {}, '{"error": "nope"}'))
    with pytest.raises(ApiError) as info:
        post_json(url, {}, sleep=clock.sleep)
    assert not isinstance(info.value, AuthBlocked)
    assert f"HTTP {status}" in str(info.value) and hint in str(info.value)
    assert len(log) == 1 and clock.slept == []


# ------------------------------------------------------------------ network faults (no server)

def flaky(monkeypatch, errors, final=None):
    """Make urllib.request.urlopen raise `errors` in order, then return `final`."""
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    def fake(request, timeout=None):
        calls.append(request)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return Response(final)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return calls


def test_timeouts_resets_dropped_connections_and_dns_errors_are_retried(monkeypatch, clock):
    calls = flaky(monkeypatch, [TimeoutError("read timed out"), ConnectionResetError(),
                                http.client.RemoteDisconnected("closed"),
                                http.client.IncompleteRead(b"par"),
                                urllib.error.URLError("temporary failure in name resolution"),
                                ssl.SSLError("EOF occurred in violation of protocol")],
                  final=b'{"ok": true}')
    assert post_json("http://x/y", {}, sleep=clock.sleep) == {"ok": True}
    assert len(calls) == 7 and clock.slept == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def test_connection_problems_that_never_go_away_end_in_a_clear_error(monkeypatch, clock):
    flaky(monkeypatch, [TimeoutError("read timed out")] * 20)
    with pytest.raises(ApiError, match="connection problem.*8 retries.*TimeoutError"):
        post_json("http://x/y", {}, sleep=clock.sleep)
    flaky(monkeypatch, [urllib.error.URLError("no route to host")] * 20)
    with pytest.raises(ApiError, match="could not reach http://x/y.*no route to host"):
        post_json("http://x/y", {}, sleep=clock.sleep)


def test_certificate_problems_are_not_retried(monkeypatch, clock):
    flaky(monkeypatch, [urllib.error.URLError(ssl.SSLCertVerificationError("self-signed"))] * 3)
    with pytest.raises(ApiError, match="certificate.*Retrying will not help"):
        post_json("https://x/y", {}, sleep=clock.sleep)
    assert clock.slept == []


@pytest.mark.parametrize("body", [b"<html>Just a moment...</html>", b"", b"[1, 2]"])
def test_answers_that_are_not_a_json_object_are_reported_not_crashed_on(monkeypatch, clock, body):
    flaky(monkeypatch, [], final=body)
    with pytest.raises(ApiError, match="not with JSON|not an object"):
        post_json("http://x/y", {}, sleep=clock.sleep)


def test_http_200_with_an_error_object_is_an_error(monkeypatch, clock):
    flaky(monkeypatch, [], final=b'{"error": {"message": "overloaded"}}')
    with pytest.raises(ApiError, match="reported an error.*overloaded"):
        post_json("http://x/y", {}, sleep=clock.sleep)


def test_the_query_string_is_never_shown(monkeypatch, clock):
    flaky(monkeypatch, [], final=b"nope")
    with pytest.raises(ApiError) as info:
        get_json("https://x.example/v1/models?key=SECRET", sleep=clock.sleep)
    assert "SECRET" not in str(info.value) and "x.example/v1/models" in str(info.value)


# ------------------------------------------------------------------ API keys

@pytest.mark.parametrize(("raw", "expected"), [
    ("abc123", "abc123"), ("  abc123\n", "abc123"), ('"abc123"', "abc123"), ("'abc123'\n", "abc123"),
    (' " abc123 " ', "abc123")])
def test_clean_api_key_removes_copy_paste_debris(raw, expected):
    assert clean_api_key(raw, "MY_KEY") == expected


@pytest.mark.parametrize(("raw", "message"), [
    (None, "set the MY_KEY environment variable"), ("", "set but empty"), ('""', "set but empty"),
    ("abc def", "position 4"), ("abc\u2019def", "position 4"), ("ab\ncd", "position 3")])
def test_clean_api_key_rejects_what_cannot_be_a_key_without_echoing_it(raw, message):
    with pytest.raises(RuntimeError, match=message) as info:
        clean_api_key(raw, "MY_KEY")
    assert "def" not in str(info.value) or raw is None


def test_a_pasted_key_with_a_newline_still_works(scripted_server, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  'gsk_abc'\n")
    server, log, url = scripted_server(lambda i, r: (200, {}, GOOD))
    make_judge("groq", "m", url, min_interval=0).is_sufficient("q", PASSAGES)
    assert log[0]["headers"]["authorization"] == "Bearer gsk_abc"
    monkeypatch.setenv("GEMINI_API_KEY", "key with space")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY contains a space"):
        GeminiJudge("m")


# ------------------------------------------------------------------ answers that are not YES/NO

def test_unparsed_replies_count_as_no_are_reported_and_not_cached(scripted_server, tmp_path):
    replies = iter(["Maybe", "**YES**", "Hard to say", "no"])
    server, log, url = scripted_server(
        lambda i, r: (200, {}, {"choices": [{"message": {"content": next(replies)}}]}))
    judge = OpenAICompatibleJudge("m", base_url=url, api_key="k", min_interval=0,
                                  cache_path=tmp_path / "c.json")
    answers = [judge.is_sufficient(f"q{i}", PASSAGES) for i in range(4)]
    assert answers == [False, True, False, False]
    assert judge.n_unparsed == 2 and judge.n_parsed == 2 and len(judge.cache) == 2


def test_a_model_that_never_says_yes_or_no_is_stopped_early(scripted_server):
    server, log, url = scripted_server(
        lambda i, r: (200, {}, {"choices": [{"message": {"content": "I would need more context."}}]}))
    judge = OpenAICompatibleJudge("m", base_url=url, api_key="k", min_interval=0)
    with pytest.raises(ValueError, match="did not answer YES or NO in its first 5 replies"):
        for i in range(10):
            judge.is_sufficient(f"q{i}", PASSAGES)
    assert len(log) == 5


# ------------------------------------------------------------------ `check`

def test_check_lists_models_finds_the_id_and_suggests_close_ones(scripted_server, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    models = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "llama-3.1-8b-instant"}]}
    server, log, url = scripted_server(lambda i, r: (200, {}, models))
    found = check_backend("groq", "llama-3.1-8b-instant", url)
    assert found["model_found"] and found["models_available"] == 2
    assert log[0]["method"] == "GET" and log[0]["path"] == "/models"
    missing = check_backend("groq", "gpt-oss-120b", url)
    assert missing["model_found"] is False and "openai/gpt-oss-120b" in missing["close_matches"]


def test_check_ping_asks_one_question_and_other_backends_list_models_too(scripted_server, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def handler(i, r):
        if r["method"] == "GET":
            return 200, {}, {"models": [{"name": "models/gemini-x"}, {"name": "gemma-y"}]}
        return 200, {}, {"candidates": [{"content": {"parts": [{"text": "YES"}]}}]}

    server, log, url = scripted_server(handler)
    result = check_backend("gemini", "gemini-x", url, ping=True)
    assert result["model_found"] and result["ping"]["answered_yes"] is True
    assert [r["method"] for r in log] == ["GET", "POST"]
    ollama = check_backend("ollama", "m", url)
    assert log[-1]["path"] == "/api/tags" and ollama["models_available"] == 2
    with pytest.raises(ValueError, match="--ping needs --model"):
        check_backend("ollama", None, url, ping=True)


def test_check_reports_a_blocked_key_with_the_stop_message(scripted_server, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    server, log, url = scripted_server(lambda i, r: (403, {}, "error code: 1010"))
    with pytest.raises(AuthBlocked, match="STOP: API Key invalid or blocked by Cloudflare"):
        check_backend("groq", None, url)


# ------------------------------------------------------------------ the module is self-contained

def test_llm_http_imports_nothing_from_the_rest_of_the_package():
    with open(llm_http.__file__, encoding="utf-8") as f:
        source = f.read()
    assert "stopret" not in source.replace("STOPRET_USER_AGENT", "").replace("stopret check", "")
    assert "import requests" not in source and "from requests" not in source


def test_exception_classes_keep_their_relationships():
    assert issubclass(AuthBlocked, ApiError) and issubclass(RateLimitExhausted, ApiError)
    assert issubclass(ApiError, RuntimeError) and not issubclass(JudgeRefused, ApiError)
    assert LLMJudge.backend == "base" and OllamaJudge.backend == "ollama"
