"""Features the stopping classifier sees after each hop.

Nothing in here looks at the gold passages. Gold is only used to build the labels in
`build_table`, and that column is never fed to the model.

Feature groups (also used for the ablation in the experiments):

score      how confident does the retriever look? (top score, gaps between the new passages,
           change since the previous hop, z-score against the 100 best passages)
stability  is the retrieval settling down? (how many of this hop's top results were already
           retrieved before, how similar the retrieved passages are to each other)
coverage   how much of the question is explained by the evidence so far? (idf-weighted share of
           question terms found in the evidence, and how much the last hop added)
structure  does the evidence look like a closed chain? (titles of retrieved passages that are
           named in the question, passages that mention the title of another retrieved passage)
question   cheap cues from the question text (length, yes/no form, comparison words, names)
hop        the hop index itself
"""
from __future__ import annotations

import re
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd

from .bm25 import tokenize
from .data import Example
from .trajectory import HopResult, IterativeRetriever

FEATURE_GROUPS: dict[str, list[str]] = {
    "score": ["s1n", "gap12n", "gap1kn", "mean_newn", "s1n_delta", "z1"],
    "stability": ["rerank_overlap", "pair_overlap"],
    "coverage": ["cov", "cov_gain", "new_cov", "n_uncovered"],
    "structure": ["title_hits", "title_hit_frac", "links"],
    "question": ["q_len", "is_yesno", "cmp_cue", "n_cap"],
    "hop": ["hop"],
}
FEATURE_NAMES: list[str] = [f for group in FEATURE_GROUPS.values() for f in group]

_YESNO_STARTS = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have",
                 "had", "will", "would", "should"}
_CMP_WORDS = {"both", "same", "older", "younger", "more", "less", "first", "earlier", "later",
              "larger", "smaller", "longer", "shorter", "or"}


def normalize_title(title: str) -> str:
    """'Chris Evans (actor)' -> 'chris evans'"""
    return re.sub(r"\s*\(.*?\)\s*$", "", title).strip().lower()


def _contains_phrase(haystack: str, phrase: str) -> bool:
    if len(phrase) < 3:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", haystack) is not None


class FeatureExtractor:
    def __init__(self, retriever: IterativeRetriever):
        self.retriever = retriever
        ds = retriever.ds
        self.norm_titles = [normalize_title(t) for t in ds.titles]
        self.lower_texts = [x.lower() for x in ds.texts]
        self.token_sets = [set(toks) for toks in retriever.doc_tokens]
        self._title_in_text = lru_cache(maxsize=None)(self._title_in_text_uncached)

    def _title_in_text_uncached(self, title_id: int, text_id: int) -> bool:
        return _contains_phrase(self.lower_texts[text_id], self.norm_titles[title_id])

    def question_features(self, question: str) -> dict[str, float]:
        words = question.split()
        lowered = [w.strip("?,.!").lower() for w in words]
        return {
            "q_len": float(len(words)),
            "is_yesno": float(bool(lowered) and lowered[0] in _YESNO_STARTS),
            "cmp_cue": float(any(w in _CMP_WORDS for w in lowered)),
            "n_cap": float(sum(1 for w in words[1:] if w[:1].isupper())),
        }

    def _coverage(self, terms: list[str], pool: list[int]) -> tuple[float, int]:
        """idf-weighted share of the question terms that occur in the pooled passages."""
        if not terms:
            return 0.0, 0
        pooled: set[str] = set()
        for i in pool:
            pooled |= self.token_sets[i]
        bm25 = self.retriever.bm25
        total = sum(bm25.idf_of(t) for t in terms)
        covered = sum(bm25.idf_of(t) for t in terms if t in pooled)
        n_uncovered = sum(1 for t in terms if t not in pooled)
        return (covered / total if total > 0 else 0.0), n_uncovered

    def hop_features(self, example: Example, hops: list[HopResult]) -> list[dict[str, float]]:
        q_lower = example.question.lower()
        terms = sorted(set(tokenize(example.question)))
        static = self.question_features(example.question)
        rows: list[dict[str, float]] = []
        prev_cov, prev_s1n = None, None

        for h in hops:
            norm = h.max_possible if h.max_possible > 0 else 1.0
            s = h.new_scores
            s1n = float(s[0]) / norm
            cov, n_uncovered = self._coverage(terms, h.seen_ids)
            new_cov, _ = self._coverage(terms, h.new_ids)

            pairs = list(combinations(h.seen_ids, 2))
            pair_overlap = float(np.mean([
                len(self.token_sets[a] & self.token_sets[b])
                / max(1, len(self.token_sets[a] | self.token_sets[b]))
                for a, b in pairs
            ])) if pairs else 0.0

            title_hits = sum(_contains_phrase(q_lower, self.norm_titles[i]) for i in h.seen_ids)
            links = sum(
                self._title_in_text(b, a)
                for a in h.seen_ids for b in h.seen_ids if a != b
            )

            rows.append({
                "hop": float(h.hop),
                "s1n": s1n,
                "gap12n": float(s[0] - s[1]) / norm,
                "gap1kn": float(s[0] - s[-1]) / norm,
                "mean_newn": float(s.mean()) / norm,
                "s1n_delta": 0.0 if prev_s1n is None else s1n - prev_s1n,
                "z1": float((s[0] - h.top100_mean) / (h.top100_std + 1e-9)),
                "rerank_overlap": h.rerank_overlap,
                "pair_overlap": pair_overlap,
                "cov": cov,
                "cov_gain": 0.0 if prev_cov is None else cov - prev_cov,
                "new_cov": new_cov,
                "n_uncovered": float(n_uncovered),
                "title_hits": float(title_hits),
                "title_hit_frac": title_hits / len(h.seen_ids),
                "links": float(links),
                **static,
            })
            prev_cov, prev_s1n = cov, s1n
        return rows


def build_table(retriever: IterativeRetriever) -> pd.DataFrame:
    """One row per (question, hop): features plus the gold-derived labels."""
    extractor = FeatureExtractor(retriever)
    records = []
    for ex in retriever.ds.examples:
        hops = retriever.retrieve(ex)
        gold = set(ex.gold_ids)
        for h, feats in zip(hops, extractor.hop_features(ex, hops)):
            found = len(gold & set(h.seen_ids))
            records.append({
                "qid": ex.qid,
                "n_gold": len(gold),
                "gold_found": found / len(gold),
                "sufficient": int(found == len(gold)),
                **feats,
            })
    table = pd.DataFrame.from_records(records)
    # 'eventually' = does the question become sufficient by the last hop? It is the same for
    # every hop of a question and only ever used as a training target, never as a feature.
    max_hops = retriever.cfg.max_hops
    final = table["sufficient"].to_numpy().reshape(-1, max_hops)[:, -1]
    table["eventually"] = np.repeat(final, max_hops)
    columns = ["qid", "n_gold", "gold_found", "sufficient", "eventually"] + FEATURE_NAMES
    return table[columns]
