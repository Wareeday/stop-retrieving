import copy

import numpy as np
import pytest

from stopret.evaluation import (fixed_points, outcome, recall_at_budget, upper_hull)
from stopret.features import FEATURE_NAMES, FeatureExtractor, build_table, normalize_title
from stopret.policies import oracle_stops, stop_hops, stop_hops_giveup, to_matrix
from stopret.trajectory import IterativeRetriever, RetrievalConfig


def test_stop_hops_first_crossing_and_last_hop_fallback():
    scores = np.array([[0.1, 0.6, 0.9], [0.1, 0.2, 0.3]])
    assert stop_hops(scores, 0.5).tolist() == [2, 3]


def test_giveup_stops_early_on_hopeless_questions():
    now = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    ever = np.array([[0.01, 0.01, 0.01], [0.9, 0.9, 0.9]])
    assert stop_hops_giveup(now, ever, tau=0.5, rho=0.1).tolist() == [1, 3]


def test_oracle_stops_at_first_sufficient_hop_or_one():
    suff = np.array([[False, True, True], [False, False, False], [True, True, True]])
    assert oracle_stops(suff).tolist() == [2, 1, 1]


def test_fixed_points_and_outcome():
    suff = np.array([[False, True, True], [True, True, True], [False, False, True]])
    pts = fixed_points(suff)
    assert pts == [(1.0, 1 / 3), (2.0, 2 / 3), (3.0, 1.0)]
    assert outcome(suff, np.array([2, 1, 3])) == (2.0, 1.0)


def test_upper_hull_and_interpolation():
    pts = [(1.0, 0.2), (2.0, 0.3), (3.0, 0.8), (4.0, 0.8)]
    hull = upper_hull(pts)
    assert hull[0] == (1.0, 0.2) and (2.0, 0.3) not in hull  # below the line from (1,.2) to (3,.8)
    assert recall_at_budget(pts, 2.0) == pytest.approx(0.5)
    assert recall_at_budget(pts, 10.0) == pytest.approx(0.8)
    assert np.isnan(recall_at_budget(pts, 0.5))
    assert recall_at_budget([[1, 0.2], [3, 0.8]], 2.0) == pytest.approx(0.5)  # JSON-style input


def test_retriever_adds_k_new_passages_per_hop_without_repeats(tiny_dataset):
    cfg = RetrievalConfig(k=2, max_hops=3)
    hops = IterativeRetriever(tiny_dataset, cfg).retrieve(tiny_dataset.examples[0])
    assert [len(h.new_ids) for h in hops] == [2, 2, 2]
    all_ids = [i for h in hops for i in h.new_ids]
    assert len(all_ids) == len(set(all_ids)) == 6
    assert hops[-1].seen_ids == all_ids
    assert set(hops[0].new_ids) == {0, 1}  # both gold passages are the obvious top hits here


def test_table_labels_and_shape(tiny_dataset):
    cfg = RetrievalConfig(k=2, max_hops=3)
    table = build_table(IterativeRetriever(tiny_dataset, cfg))
    assert len(table) == 2 * 3
    assert table[FEATURE_NAMES].notna().all().all()
    suff = to_matrix(table, "sufficient", 3)
    assert (np.diff(suff, axis=1) >= 0).all()  # once sufficient, always sufficient
    assert (to_matrix(table, "eventually", 3) == suff[:, [-1]]).all()


def test_features_never_depend_on_gold(tiny_dataset):
    cfg = RetrievalConfig(k=2, max_hops=3)
    retriever = IterativeRetriever(tiny_dataset, cfg)
    extractor = FeatureExtractor(retriever)
    ex = tiny_dataset.examples[0]
    hops = retriever.retrieve(ex)
    changed = copy.deepcopy(ex)
    changed.gold_ids = [4, 5]
    assert extractor.hop_features(ex, hops) == extractor.hop_features(changed, hops)


def test_normalize_title_strips_disambiguation():
    assert normalize_title("Chris Evans (actor)") == "chris evans"
