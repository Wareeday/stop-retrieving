"""Optional baseline: ask an LLM whether the retrieved evidence is sufficient.

The policy is the obvious one: after each hop, show the question and everything retrieved so
far to the model, and stop at the first "YES". Three backends are available:

    ollama   a model running on your own machine (http://localhost:11434)
    groq     open-weight models hosted by Groq, through its OpenAI-compatible API
             (needs the environment variable GROQ_API_KEY)
    gemini   Google's Gemini models through the Gemini API (generateContent)
             (needs the environment variable GEMINI_API_KEY, or GOOGLE_API_KEY)

NOTE: I built this project in a sandbox with no Ollama, no GPU and no access to the Groq or
Gemini APIs, so the plumbing (request format, answer parsing, retries, caching, rate-limit
handling) is covered by tests against mock servers (tests/test_llm_judge.py), but no number for
this baseline has been produced yet. Run `python -m stopret judge ...` to get one.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = """You are checking retrieved evidence for a multi-hop question.

Question: {question}

Retrieved passages:
{passages}

Do these passages together contain all the information needed to answer the question?
Answer with exactly one word: YES or NO."""

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
BACKENDS = ("ollama", "groq", "gemini")
THINKING_CHOICES = ("omit", "off", "minimal", "low", "medium", "high")


class RateLimitExhausted(RuntimeError):
    """The server keeps refusing for rate-limit reasons, or asked us to wait longer than we are
    willing to (for example a daily quota)."""


class JudgeRefused(ValueError):
    """The provider declined to answer this particular prompt (e.g. a safety block)."""


def _retry_delay_from_body(raw: bytes) -> float | None:
    """Google-style 429 bodies carry the wait in error.details[].retryDelay, e.g. '34s'."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    for detail in (error or {}).get("details") or []:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        match = re.fullmatch(r"([0-9.]+)s", delay or "")
        if match:
            return float(match.group(1))
    return None


def _post_json(url: str, payload: dict, headers: dict, timeout: float, max_retries: int = 8,
               max_wait: float = 120.0, sleep=time.sleep) -> dict:
    """POST JSON and return the decoded reply. Retries 429 and 5xx responses and connection
    errors, waiting as long as the server says (Retry-After header, or retryDelay in the body)
    and otherwise backing off exponentially. If the server asks for a wait longer than
    `max_wait` seconds, or keeps answering 429 after all retries, raise RateLimitExhausted
    instead of sleeping for hours."""
    body = json.dumps(payload).encode()
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            raw = err.read()
            retryable = err.code == 429 or err.code >= 500
            if not retryable or attempt == max_retries:
                if err.code == 429:
                    raise RateLimitExhausted(
                        f"still rate limited after {attempt} retries. Progress is cached; run "
                        "the same command again later and it will resume.") from err
                raise RuntimeError(
                    f"HTTP {err.code} from {url}: {raw.decode(errors='replace')[:300]}") from err
            retry_after = err.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else _retry_delay_from_body(raw)
            if wait is None:
                wait = min(2.0 ** attempt, 60.0)
            if wait > max_wait:
                raise RateLimitExhausted(
                    f"server asked to wait {wait:.0f}s (limit reached). Progress is cached; "
                    "run the same command again later and it will resume.") from err
        except urllib.error.URLError as err:
            if attempt == max_retries:
                raise RuntimeError(f"could not reach {url}: {err.reason}") from err
            wait = min(2.0 ** attempt, 60.0)
        sleep(wait)
    raise AssertionError("unreachable")


class LLMJudge:
    """Shared logic: prompt, answer parsing, caching, call counting, minimum request spacing."""

    backend = "base"

    def __init__(self, model: str, cache_path: str | Path | None = None,
                 min_interval: float = 0.0, timeout: float = 300.0, max_tokens: int = 8,
                 sleep=time.sleep):
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens
        self.min_interval, self._sleep = min_interval, sleep
        self._last_request = float("-inf")
        self.n_calls = 0              # requests actually sent (cache hits are not counted)
        self.n_refused = 0            # prompts the provider declined to answer
        self.prompt_chars = 0         # characters sent, for a rough token estimate
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, bool] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

    def _complete(self, prompt: str) -> str:
        raise NotImplementedError

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    @staticmethod
    def parse(answer: str) -> bool:
        """YES/NO from the model's reply. Reasoning models may emit <think>...</think> first."""
        cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        if "<think>" in cleaned:  # cut off inside the reasoning, e.g. by max_tokens
            raise ValueError(
                "the answer was cut off inside a <think> block; use a non-reasoning (instruct) "
                "model, or raise max_tokens")
        if not cleaned:
            raise ValueError(
                "the model returned an empty answer; use a non-reasoning (instruct) model, "
                "or raise max_tokens")
        return cleaned.upper().startswith("YES")

    def is_sufficient(self, question: str, passages: list[tuple[str, str]],
                      cache_key: str | None = None) -> bool:
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]
        text = "\n".join(f"[{i}] {title}: {body}" for i, (title, body) in enumerate(passages, 1))
        prompt = PROMPT.format(question=question, passages=text)
        self._throttle()
        try:
            verdict = self.parse(self._complete(prompt))
        except JudgeRefused:
            self.n_refused += 1
            self.n_calls += 1
            self.prompt_chars += len(prompt)
            raise
        self.n_calls += 1
        self.prompt_chars += len(prompt)
        if cache_key is not None:
            self.cache[cache_key] = verdict
            if self.cache_path:
                self.cache_path.write_text(json.dumps(self.cache))
        return verdict


class OllamaJudge(LLMJudge):
    backend = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.host = host.rstrip("/")

    def _complete(self, prompt: str) -> str:
        reply = _post_json(
            f"{self.host}/api/generate",
            {"model": self.model, "prompt": prompt, "stream": False,
             "options": {"temperature": 0, "num_predict": self.max_tokens}},
            {"Content-Type": "application/json"}, self.timeout, sleep=self._sleep)
        return reply["response"]


class GroqJudge(LLMJudge):
    """Groq's chat-completions endpoint (OpenAI-compatible). The key is read from the
    GROQ_API_KEY environment variable and is never written to disk or to the results."""

    backend = "groq"

    def __init__(self, model: str, base_url: str = GROQ_BASE_URL, api_key: str | None = None,
                 min_interval: float = 2.0, **kwargs):
        super().__init__(model, min_interval=min_interval, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("set the GROQ_API_KEY environment variable "
                               "(create a key in the Groq console)")

    def _complete(self, prompt: str) -> str:
        reply = _post_json(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": [{"role": "user", "content": prompt}],
             "temperature": 0, "max_tokens": self.max_tokens},
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            self.timeout, sleep=self._sleep)
        return reply["choices"][0]["message"]["content"] or ""


class GeminiJudge(LLMJudge):
    """Gemini API, native generateContent endpoint. The key is read from GEMINI_API_KEY (or
    GOOGLE_API_KEY), sent in the x-goog-api-key header, and never written to disk or results.

    Newer Gemini models "think" before answering by default, and thinking tokens count towards
    maxOutputTokens. That is why max_tokens defaults to 1024 here instead of 8, and why
    `thinking` is configurable:

        omit     send no thinking setting (works with every model; the default)
        off      thinkingBudget = 0 (Gemini 2.5-style models that allow switching thinking off)
        minimal | low | medium | high
                 thinkingLevel (Gemini 3-style models)

    Which of these a given model accepts depends on the model family. If the API answers
    HTTP 400 about the thinking setting, try another value or `omit`."""

    backend = "gemini"

    def __init__(self, model: str, base_url: str = GEMINI_BASE_URL, api_key: str | None = None,
                 min_interval: float = 6.0, thinking: str = "omit", max_tokens: int = 1024,
                 **kwargs):
        super().__init__(model, min_interval=min_interval, max_tokens=max_tokens, **kwargs)
        if thinking not in THINKING_CHOICES:
            raise ValueError(f"thinking must be one of {THINKING_CHOICES}, got {thinking!r}")
        self.thinking = thinking
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("set the GEMINI_API_KEY environment variable "
                               "(create a key in Google AI Studio)")

    def _generation_config(self) -> dict:
        config: dict = {"temperature": 0, "maxOutputTokens": self.max_tokens}
        if self.thinking == "off":
            config["thinkingConfig"] = {"thinkingBudget": 0}
        elif self.thinking != "omit":
            config["thinkingConfig"] = {"thinkingLevel": self.thinking}
        return config

    def _complete(self, prompt: str) -> str:
        model = self.model.removeprefix("models/")
        reply = _post_json(
            f"{self.base_url}/models/{model}:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
             "generationConfig": self._generation_config()},
            {"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            self.timeout, sleep=self._sleep)
        return self._extract_text(reply)

    @staticmethod
    def _extract_text(reply: dict) -> str:
        candidates = reply.get("candidates") or []
        if not candidates:
            reason = (reply.get("promptFeedback") or {}).get("blockReason", "unknown")
            raise JudgeRefused(f"Gemini returned no answer (blockReason: {reason})")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if text.strip():
            return text
        finish = candidate.get("finishReason", "unknown")
        if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}:
            raise JudgeRefused(f"Gemini declined to answer (finishReason: {finish})")
        raise ValueError(
            f"the model returned an empty answer (finishReason: {finish}); if this is MAX_TOKENS, "
            "thinking probably used up the output budget: try --thinking off (or minimal) "
            "or raise max_tokens")


def make_judge(backend: str, model: str, host: str | None = None, **kwargs) -> LLMJudge:
    if backend == "ollama":
        return OllamaJudge(model, host or "http://localhost:11434", **kwargs)
    if backend == "groq":
        return GroqJudge(model, base_url=host or GROQ_BASE_URL, **kwargs)
    if backend == "gemini":
        return GeminiJudge(model, base_url=host or GEMINI_BASE_URL, **kwargs)
    raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
