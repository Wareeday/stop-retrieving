"""The requests-based patch in examples/ (skipped when the requests library is not installed)."""
import sys
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from requests_guard import BROWSER_UA, ApiStop, post_json_guarded  # noqa: E402

OK = {"choices": [{"message": {"content": "YES"}}]}


def test_sends_browser_agent_and_returns_json(scripted_server):
    server, log, url = scripted_server(lambda i, r: (200, {}, OK))
    assert post_json_guarded(url, {"a": 1}, " key123\n") == OK
    assert log[0]["headers"]["user-agent"] == BROWSER_UA
    assert log[0]["headers"]["authorization"] == "Bearer key123"
    assert "python-requests" not in log[0]["headers"]["user-agent"]


def test_cloudflare_1010_for_python_requests_agent_is_the_reported_failure(scripted_server):
    def cloudflare(i, r):
        return (403, {}, "error code: 1010") if "python" in r["headers"]["user-agent"].lower() else (200, {}, OK)

    server, log, url = scripted_server(cloudflare)
    with pytest.raises(ApiStop):                       # what happens with requests' default agent
        post_json_guarded(url, {}, "k", extra_headers={"User-Agent": "python-requests/2.31"})
    assert post_json_guarded(url, {}, "k") == OK       # and with the fix


def test_401_and_403_stop_immediately_without_retry_and_without_leaking_the_key(scripted_server):
    for status in (401, 403):
        sleeps = []
        server, log, url = scripted_server(
            lambda i, r, s=status: (s, {}, '{"error": "bad key SECRETKEY99"}'))
        with pytest.raises(ApiStop) as info:
            post_json_guarded(url, {}, "SECRETKEY99", sleep=sleeps.append)
        assert str(info.value).startswith("STOP: API Key invalid or blocked by Cloudflare (1010)")
        assert f"HTTP {status}" in str(info.value) and "SECRETKEY99" not in str(info.value)
        assert len(log) == 1 and sleeps == []


def test_429_and_5xx_are_retried(scripted_server):
    script = [(429, {"Retry-After": "3"}, "{}"), (503, {}, "{}"), (200, {}, OK)]
    sleeps = []
    server, log, url = scripted_server(lambda i, r: script[i])
    assert post_json_guarded(url, {}, "k", sleep=sleeps.append) == OK
    assert sleeps == [3.0, 2]


def test_a_failed_question_is_not_recorded_so_a_rerun_asks_it_again(scripted_server):
    """The pattern from the module docstring."""
    mode = {"blocked": True}
    server, log, url = scripted_server(
        lambda i, r: (403, {}, "error code: 1010") if mode["blocked"] and i == 2 else (200, {}, OK))
    results = {}

    def run(questions):
        for question in questions:
            if question in results:
                continue
            try:
                reply = post_json_guarded(url, {"q": question}, "k")
            except ApiStop:
                return False
            results[question] = reply["choices"][0]["message"]["content"]
        return True

    assert run(range(5)) is False and sorted(results) == [0, 1]       # stopped at question 2
    assert len(log) == 3
    mode["blocked"] = False
    assert run(range(5)) is True and sorted(results) == [0, 1, 2, 3, 4]
    assert len(log) == 3 + 3                                          # 2 was asked again, 0-1 not
