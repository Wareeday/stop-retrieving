"""Optional baseline: ask a local LLM (through Ollama) whether the evidence is sufficient.

The policy is the obvious one: after each hop, show the question and everything retrieved so
far to the model, and stop at the first "YES".

NOTE: the HTTP plumbing and the answer parsing are covered by tests against a mock server
(tests/test_llm_judge.py). I built this project in a sandbox without Ollama or a GPU, so the
numbers for this baseline have NOT been produced yet; run `python -m stopret judge` locally.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

PROMPT = """You are checking retrieved evidence for a multi-hop question.

Question: {question}

Retrieved passages:
{passages}

Do these passages together contain all the information needed to answer the question?
Answer with exactly one word: YES or NO."""


class OllamaJudge:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 300.0,
                 cache_path: str | Path | None = None):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, bool] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

    def _generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0, "num_predict": 4}}).encode()
        request = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())["response"]

    def is_sufficient(self, question: str, passages: list[tuple[str, str]],
                      cache_key: str | None = None) -> bool:
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]
        text = "\n".join(f"[{i}] {title}: {body}" for i, (title, body) in enumerate(passages, 1))
        answer = self._generate(PROMPT.format(question=question, passages=text))
        verdict = answer.strip().upper().startswith("YES")
        if cache_key is not None:
            self.cache[cache_key] = verdict
            if self.cache_path:
                self.cache_path.write_text(json.dumps(self.cache))
        return verdict
