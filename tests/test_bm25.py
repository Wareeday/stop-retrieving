import numpy as np
from rank_bm25 import BM25Okapi

from stopret.bm25 import BM25, tokenize

DOCS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Paris is the capital and most populous city of France.",
    "Gustave Eiffel was a French civil engineer who designed the tower.",
    "The Statue of Liberty was designed by Frederic Bartholdi in France.",
    "Mount Everest is the highest mountain above sea level.",
    "The tower was completed in 1889 for the World's Fair held in Paris.",
]


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The Tower of Paris!") == ["tower", "paris"]


def test_scores_match_rank_bm25():
    toks = [tokenize(d) for d in DOCS]
    ours = BM25(toks)
    ref = BM25Okapi(toks)  # same defaults: k1=1.5, b=0.75, epsilon=0.25
    for query in ["who designed the Eiffel tower", "Paris France tower tower", "everest"]:
        q = tokenize(query)
        np.testing.assert_allclose(ours.scores(q), ref.get_scores(q), rtol=1e-9, atol=1e-9)


def test_unknown_terms_are_ignored():
    bm25 = BM25([tokenize(d) for d in DOCS])
    assert np.all(bm25.scores(["zzzunknown"]) == 0)
    assert bm25.max_possible(["zzzunknown"]) == 0
