"""Iterative (multi-hop) retrieval without an LLM.

Hop 1 retrieves with the question alone. Every later hop rewrites the query as

    question (repeated `question_weight` times) + text of the best passage found in the previous hop

and retrieves `k` passages that have not been retrieved before. So after hop t the
evidence pool holds up to k * t passages, and every extra hop costs one more retrieval
call (and k more passages that a reader would have to process). The stopping policies in
`policies.py` decide, after each hop, whether to continue.

The query rewriting is deliberately simple. The point of this project is the stopping
decision, not the retriever, and a retriever that needs no model download keeps the whole
pipeline runnable on a laptop CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25, tokenize
from .data import Dataset, Example


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 3                 # new passages added per hop
    max_hops: int = 4
    question_weight: int = 3   # how many times the question tokens are repeated in the query


@dataclass
class HopResult:
    hop: int                    # 1-indexed
    query_tokens: list[str]
    new_ids: list[int]          # passages added at this hop, best first
    new_scores: np.ndarray      # their BM25 scores
    seen_ids: list[int]         # everything retrieved up to and including this hop
    max_possible: float         # upper bound of a BM25 score for this query
    top100_mean: float          # score statistics of the 100 best passages for this query
    top100_std: float
    rerank_overlap: float       # share of the unconstrained top-k that was already retrieved


class IterativeRetriever:
    def __init__(self, ds: Dataset, cfg: RetrievalConfig = RetrievalConfig()):
        self.ds = ds
        self.cfg = cfg
        self.doc_tokens = [tokenize(f"{t} {x}") for t, x in zip(ds.titles, ds.texts)]
        self.bm25 = BM25(self.doc_tokens)

    def retrieve(self, example: Example) -> list[HopResult]:
        cfg = self.cfg
        question_tokens = tokenize(example.question)
        seen: list[int] = []
        seen_set: set[int] = set()
        hops: list[HopResult] = []

        for hop in range(1, cfg.max_hops + 1):
            query = question_tokens * cfg.question_weight
            if hops:
                query = query + self.doc_tokens[hops[-1].new_ids[0]]

            scores = self.bm25.scores(query)
            order = np.argsort(-scores, kind="stable")
            unconstrained_top = order[: cfg.k]
            new_ids = [int(i) for i in order if int(i) not in seen_set][: cfg.k]

            top100 = scores[order[:100]]
            rerank_overlap = (
                float(np.mean([int(i) in seen_set for i in unconstrained_top])) if hops else 0.0
            )
            seen.extend(new_ids)
            seen_set.update(new_ids)
            hops.append(HopResult(
                hop=hop,
                query_tokens=query,
                new_ids=new_ids,
                new_scores=scores[new_ids],
                seen_ids=list(seen),
                max_possible=self.bm25.max_possible(query),
                top100_mean=float(top100.mean()),
                top100_std=float(top100.std()),
                rerank_overlap=rerank_overlap,
            ))
        return hops
