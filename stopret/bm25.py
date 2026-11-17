"""A small sparse BM25 (Okapi) retriever.

I wrote this instead of using rank_bm25 directly because the iterative retrieval runs
tens of thousands of long queries, and a scipy sparse matrix-vector product is a lot
faster than rank_bm25's python loop. The scoring formula is the same as
rank_bm25.BM25Okapi (including its handling of negative idf values), and
tests/test_bm25.py checks the two against each other.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop English stop words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25):
        self.k1, self.b = k1, b
        self.n_docs = len(docs_tokens)

        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        doc_len = np.zeros(self.n_docs)
        for i, tokens in enumerate(docs_tokens):
            doc_len[i] = len(tokens)
            for term, count in Counter(tokens).items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(count)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n_docs, len(self.vocab)),
                               dtype=np.float64)
        self.avgdl = doc_len.mean()

        # idf, with rank_bm25's floor for terms that appear in more than half the docs
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log((self.n_docs - df + 0.5) / (df + 0.5))
        floor = epsilon * idf.mean()
        idf[idf < 0] = floor
        self.idf = idf

        # Pre-compute the document side of the score once.
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * doc_len[tf.row] / self.avgdl)
        weights = self.idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm)
        self._weights = sparse.csr_matrix((weights, (tf.row, tf.col)), shape=tf.shape)

    def idf_of(self, term: str) -> float:
        j = self.vocab.get(term)
        return float(self.idf[j]) if j is not None else 0.0

    def query_vector(self, query_tokens: list[str]) -> np.ndarray:
        """Term counts for the query; repeated tokens count repeatedly, as in rank_bm25."""
        vec = np.zeros(len(self.vocab))
        for term, count in Counter(query_tokens).items():
            j = self.vocab.get(term)
            if j is not None:
                vec[j] = count
        return vec

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        return self._weights @ self.query_vector(query_tokens)

    def max_possible(self, query_tokens: list[str]) -> float:
        """Upper bound of a document's score for this query (used to normalise scores)."""
        vec = self.query_vector(query_tokens)
        return float((vec * self.idf).sum() * (self.k1 + 1))
