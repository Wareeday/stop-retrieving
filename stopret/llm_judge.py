"""Optional baseline: ask an LLM whether the retrieved evidence is sufficient.

The policy is the obvious one: after each hop, show the question and everything retrieved so
far to the model, and stop at the first "YES".

Backends
--------
ollama             a model on your own machine (http://localhost:11434). Free, but slow on a CPU.
groq, cerebras,    hosted open-weight models behind an OpenAI-compatible API with a free tier.
mistral,           The API key comes from an environment variable (GROQ_API_KEY, CEREBRAS_API_KEY,
openrouter         MISTRAL_API_KEY, OPENROUTER_API_KEY).
openai-compatible  any other OpenAI-compatible server: pass --host (base URL) and --api-key-env.
gemini             Google's Gemini API. These are proprietary models, so this backend is not part
                   of a fully open-source setup. It is kept as an optional extra.

Free tiers are limited by tokens per minute and tokens per day as much as by requests, so the
client paces itself by tokens too (`tokens_per_minute`), retries 429/5xx answers using the wait
the server states, and caches every answer so an interrupted run resumes where it stopped.

NOTE: I built this project in a sandbox with no access to any of these APIs, so the plumbing
(request format, answer parsing, retries, pacing, caching) is covered by tests against mock
servers (tests/test_llm_judge.py). No number for this baseline has been produced yet.
Run `python -m stopret judge ...` to get one.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROMPT = """You are checking retrieved evidence for a multi-hop question.

Question: {question}

Retrieved passages:
{passages}

Do these passages together contain all the information needed to answer the question?
Answer with exactly one word: YES or NO."""

CHARS_PER_TOKEN = 4          # rough English average, used for estimates only
PACING_SAFETY = 1.2          # pace as if prompts were 20% longer than the estimate


@dataclass(frozen=True)
class Provider:
    base_url: str
    key_env: str
    note: str


PROVIDERS: dict[str, Provider] = {
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY",
                     "very fast, but small free token budgets per minute and per day"),
    "cerebras": Provider("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
                         "very fast, larger free daily token budget (check the current terms)"),
    "mistral": Provider("https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                        "large free token budget, 1 request per second; use a model whose weights "
                        "are open (check the model card)"),
    "openrouter": Provider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                           "many ':free' open models, but small daily request limits"),
}
BACKENDS = ("ollama", *PROVIDERS, "openai-compatible", "gemini")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
THINKING_CHOICES = ("omit", "off", "minimal", "low", "medium", "high")


class RateLimitExhausted(RuntimeError):
    """The server keeps refusing for rate-limit reasons, or asked us to wait longer than we are
    willing to (for example a daily quota)."""


class JudgeRefused(ValueError):
    """The provider declined to answer this particular prompt (e.g. a safety block)."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def build_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    text = "\n".join(f"[{i}] {title}: {body}" for i, (title, body) in enumerate(passages, 1))
    return PROMPT.format(question=question, passages=text)


# ---------------------------------------------------------------- HTTP with retries

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


# ---------------------------------------------------------------- pacing

class RateLimiter:
    """Keeps requests at least `min_interval` seconds apart and, if `tokens_per_minute` is set,
    keeps the (estimated) tokens sent in any rolling 60 seconds under that limit."""

    def __init__(self, min_interval: float = 0.0, tokens_per_minute: int | None = None,
                 clock=time.monotonic, sleep=time.sleep):
        self.min_interval, self.tpm = min_interval, tokens_per_minute
        self._clock, self._sleep = clock, sleep
        self._last_start = float("-inf")
        self._events: list[tuple[float, int]] = []

    def wait(self, tokens: int) -> None:
        while True:
            now = self._clock()
            self._events = [(t, n) for t, n in self._events if now - t < 60.0]
            delay = max(0.0, self.min_interval - (now - self._last_start))
            # A single request bigger than the whole budget could never fit, so only wait for
            # the window to clear when something is actually in it.
            if (self.tpm is not None and self._events
                    and sum(n for _, n in self._events) + tokens > self.tpm):
                delay = max(delay, self._events[0][0] + 60.0 - now)
            if delay <= 1e-9:
                break
            self._sleep(delay)
        self._last_start = self._clock()
        self._events.append((self._last_start, tokens))


# ---------------------------------------------------------------- judges

class LLMJudge:
    """Shared logic: prompt, answer parsing, pacing, call counting, and a disk cache keyed by a
    hash of everything that affects the answer (backend, model, settings, full prompt)."""

    backend = "base"

    def __init__(self, model: str, cache_path: str | Path | None = None,
                 min_interval: float = 0.0, tokens_per_minute: int | None = None,
                 timeout: float = 300.0, max_tokens: int = 16, clock=time.monotonic,
                 sleep=time.sleep):
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens
        self._sleep = sleep
        self.limiter = RateLimiter(min_interval, tokens_per_minute, clock, sleep)
        self.n_calls = 0              # requests actually sent (cache hits are not counted)
        self.n_cache_hits = 0
        self.n_refused = 0            # prompts the provider declined to answer
        self.prompt_chars = 0         # characters sent, for a rough token estimate
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, bool] = self._load_cache()

    # --- to be provided by subclasses
    def _complete(self, prompt: str) -> str:
        raise NotImplementedError

    def settings(self) -> dict:
        """Everything besides the prompt that can change the answer (part of the cache key)."""
        return {"backend": self.backend, "model": self.model, "max_tokens": self.max_tokens}

    # --- cache
    def _load_cache(self) -> dict[str, bool]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text())
        except ValueError:
            broken = self.cache_path.with_suffix(".corrupt")
            self.cache_path.replace(broken)
            print(f"warning: cache {self.cache_path} was unreadable, moved to {broken}")
            return {}

    def _save_cache(self) -> None:
        if self.cache_path:
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            os.replace(tmp, self.cache_path)       # atomic: Ctrl-C can't leave half a file

    def cache_key(self, prompt: str) -> str:
        blob = json.dumps(self.settings(), sort_keys=True) + "\n" + prompt
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    # --- answer parsing
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

    def is_sufficient(self, question: str, passages: list[tuple[str, str]]) -> bool:
        prompt = build_prompt(question, passages)
        key = self.cache_key(prompt)
        if key in self.cache:
            self.n_cache_hits += 1
            return self.cache[key]
        self.limiter.wait(int(estimate_tokens(prompt) * PACING_SAFETY) + self.max_tokens)
        self.n_calls += 1
        self.prompt_chars += len(prompt)
        try:
            verdict = self.parse(self._complete(prompt))
        except JudgeRefused:
            self.n_refused += 1
            raise
        self.cache[key] = verdict
        self._save_cache()
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


class OpenAICompatibleJudge(LLMJudge):
    """Any server that speaks POST {base_url}/chat/completions (Groq, Cerebras, Mistral,
    OpenRouter, vLLM, ...). The key is read from an environment variable and is never written to
    disk or to the results.

    `extra_body` is merged into the request for provider-specific options, for example
    {"reasoning_effort": "low"} for gpt-oss models on Groq or Cerebras, or
    {"reasoning": {"effort": "low"}} on OpenRouter."""

    def __init__(self, model: str, provider: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, api_key_env: str | None = None,
                 extra_body: dict | None = None, min_interval: float = 2.0, **kwargs):
        super().__init__(model, min_interval=min_interval, **kwargs)
        preset = PROVIDERS.get(provider) if provider else None
        self.backend = provider or "openai-compatible"
        base_url = base_url or (preset.base_url if preset else None)
        if not base_url:
            raise ValueError("an OpenAI-compatible backend needs a base URL (--host)")
        self.base_url = base_url.rstrip("/")
        key_env = api_key_env or (preset.key_env if preset else None)
        self.api_key = api_key or (os.environ.get(key_env) if key_env else None)
        if not self.api_key:
            raise RuntimeError(
                f"set the {key_env or 'API key'} environment variable"
                + ("" if key_env else " and pass its name with --api-key-env"))
        self.extra_body = dict(extra_body or {})
        for reserved in ("model", "messages"):
            if reserved in self.extra_body:
                raise ValueError(f"extra_body may not set {reserved!r}")

    def settings(self) -> dict:
        return {**super().settings(), "base_url": self.base_url, "extra_body": self.extra_body}

    def _complete(self, prompt: str) -> str:
        reply = _post_json(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": [{"role": "user", "content": prompt}],
             "temperature": 0, "max_tokens": self.max_tokens, **self.extra_body},
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            self.timeout, sleep=self._sleep)
        choice = (reply.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish = choice.get("finish_reason")
        if content.strip():
            return content
        if finish == "content_filter":
            raise JudgeRefused("the provider's content filter declined to answer")
        only_reasoning = bool(message.get("reasoning") or message.get("reasoning_content"))
        raise ValueError(
            f"the model returned an empty answer (finish_reason: {finish}"
            + ("; it produced only reasoning" if only_reasoning else "") + "). Reasoning models "
            "spend the output budget on thinking: raise --max-tokens (for example 512) and lower "
            "the effort with --extra-body, e.g. '{\"reasoning_effort\": \"low\"}' (Groq, Cerebras) "
            "or '{\"reasoning\": {\"effort\": \"low\"}}' (OpenRouter). Or use a non-reasoning model.")


class GeminiJudge(LLMJudge):
    """Gemini API, native generateContent endpoint (proprietary models; optional extra). The key is
    read from GEMINI_API_KEY (or GOOGLE_API_KEY), sent in the x-goog-api-key header, and never
    written to disk or results.

    Newer Gemini models "think" before answering by default, and thinking tokens count towards
    maxOutputTokens. That is why max_tokens defaults to 1024 here, and why `thinking` is
    configurable:

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

    def settings(self) -> dict:
        return {**super().settings(), "base_url": self.base_url, "thinking": self.thinking}

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


# ---------------------------------------------------------------- factory

_COMMON_OPTIONS = {"cache_path", "min_interval", "tokens_per_minute", "max_tokens", "timeout"}


def make_judge(backend: str, model: str, host: str | None = None, **options) -> LLMJudge:
    """Build a judge. Unsupported options for a backend raise ValueError instead of being
    silently ignored."""
    def check(allowed: set[str]) -> None:
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"option(s) {sorted(unknown)} do not apply to the {backend!r} backend")

    if backend == "ollama":
        check(_COMMON_OPTIONS)
        return OllamaJudge(model, host or "http://localhost:11434", **options)
    if backend == "gemini":
        check(_COMMON_OPTIONS | {"thinking"})
        return GeminiJudge(model, base_url=host or GEMINI_BASE_URL, **options)
    if backend in PROVIDERS or backend == "openai-compatible":
        check(_COMMON_OPTIONS | {"extra_body", "api_key_env"})
        return OpenAICompatibleJudge(model, provider=backend if backend in PROVIDERS else None,
                                     base_url=host, **options)
    raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
