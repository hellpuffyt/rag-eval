"""Hand-computed correctness tests for retrieval metrics."""

from __future__ import annotations

import math

import pytest

from rag_eval.retrieval_metrics import (
    dcg_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# precision@k
# ---------------------------------------------------------------------------


def test_precision_at_k_basic() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    gold = ["b", "d", "z"]
    # top-3: a, b, c -> 1 relevant (b) / 3
    assert precision_at_k(retrieved, gold, 3) == pytest.approx(1 / 3)


def test_precision_at_k_all_relevant() -> None:
    retrieved = ["a", "b", "c"]
    gold = ["a", "b", "c"]
    assert precision_at_k(retrieved, gold, 3) == pytest.approx(1.0)


def test_precision_at_k_no_relevant() -> None:
    retrieved = ["a", "b", "c"]
    gold = ["x", "y"]
    assert precision_at_k(retrieved, gold, 3) == pytest.approx(0.0)


def test_precision_at_k_no_gold_at_all() -> None:
    retrieved = ["a", "b", "c"]
    assert precision_at_k(retrieved, [], 3) == pytest.approx(0.0)


def test_precision_at_k_larger_than_result_set() -> None:
    # only 2 retrieved, k=5 -> denominator is still 5 (fixed-k convention)
    retrieved = ["a", "b"]
    gold = ["a", "b"]
    assert precision_at_k(retrieved, gold, 5) == pytest.approx(2 / 5)


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k([], ["a"], 3) == pytest.approx(0.0)


def test_precision_at_k_invalid_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], 0)
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], -1)


def test_precision_at_k_ties_use_list_order() -> None:
    # precision@k depends only on rank order, so identical scores upstream
    # (already broken into an order before reaching this function) just use
    # whatever order the list provides.
    retrieved = ["b", "a", "c"]
    gold = ["a"]
    assert precision_at_k(retrieved, gold, 2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------


def test_recall_at_k_basic() -> None:
    retrieved = ["a", "b", "c", "d"]
    gold = ["b", "d", "z"]
    # top-2: a, b -> 1 of 3 gold found
    assert recall_at_k(retrieved, gold, 2) == pytest.approx(1 / 3)


def test_recall_at_k_all_found() -> None:
    retrieved = ["a", "b", "c"]
    gold = ["a", "b"]
    assert recall_at_k(retrieved, gold, 3) == pytest.approx(1.0)


def test_recall_at_k_no_relevant_documents_is_vacuous_one() -> None:
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, [], 3) == pytest.approx(1.0)


def test_recall_at_k_k_larger_than_result_set() -> None:
    retrieved = ["a"]
    gold = ["a", "b"]
    assert recall_at_k(retrieved, gold, 10) == pytest.approx(0.5)


def test_recall_at_k_empty_retrieved_nonzero_gold() -> None:
    assert recall_at_k([], ["a", "b"], 5) == pytest.approx(0.0)


def test_recall_at_k_invalid_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], 0)


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


def test_mrr_first_result_relevant() -> None:
    assert mrr(["a", "b", "c"], ["a"]) == pytest.approx(1.0)


def test_mrr_third_result_relevant() -> None:
    assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_mrr_no_relevant_found() -> None:
    assert mrr(["x", "y", "z"], ["a"]) == pytest.approx(0.0)


def test_mrr_no_gold_at_all() -> None:
    assert mrr(["x", "y", "z"], []) == pytest.approx(0.0)


def test_mrr_restricted_by_k() -> None:
    # relevant doc is at rank 3, but we only search top-2
    assert mrr(["x", "y", "a"], ["a"], k=2) == pytest.approx(0.0)


def test_mrr_uses_first_relevant_when_multiple() -> None:
    assert mrr(["x", "a", "b"], ["a", "b"]) == pytest.approx(1 / 2)


def test_mrr_empty_retrieved() -> None:
    assert mrr([], ["a"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# DCG / nDCG
# ---------------------------------------------------------------------------


def test_dcg_at_k_hand_computed() -> None:
    # relevance [1, 0, 1] at k=3
    # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5
    relevance = [1.0, 0.0, 1.0]
    expected = 1.0 / math.log2(2) + 0.0 / math.log2(3) + 1.0 / math.log2(4)
    assert dcg_at_k(relevance, 3) == pytest.approx(expected)
    assert dcg_at_k(relevance, 3) == pytest.approx(1.5)


def test_dcg_at_k_truncates_to_k() -> None:
    relevance = [1.0, 1.0, 1.0, 1.0]
    dcg_full = dcg_at_k(relevance, 4)
    dcg_2 = dcg_at_k(relevance, 2)
    assert dcg_2 < dcg_full
    expected_2 = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert dcg_2 == pytest.approx(expected_2)


def test_dcg_at_k_invalid_k() -> None:
    with pytest.raises(ValueError):
        dcg_at_k([1.0], 0)


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    retrieved = ["a", "b", "c"]
    gold = ["a", "b"]
    assert ndcg_at_k(retrieved, gold, 3) == pytest.approx(1.0)


def test_ndcg_at_k_hand_computed_imperfect_ranking() -> None:
    # gold = {a, c}; retrieved order b, a, c -> relevance [0, 1, 1]
    # DCG = 0/log2(2) + 1/log2(3) + 1/log2(4)
    # ideal ranking for 2 relevant docs at k=3: [1,1,0]
    # IDCG = 1/log2(2) + 1/log2(3) + 0/log2(4)
    retrieved = ["b", "a", "c"]
    gold = ["a", "c"]
    dcg = 0.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 0.0 / math.log2(4)
    assert ndcg_at_k(retrieved, gold, 3) == pytest.approx(dcg / idcg)


def test_ndcg_at_k_no_relevant_documents_is_vacuous_one() -> None:
    assert ndcg_at_k(["a", "b"], [], 2) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_found_is_zero() -> None:
    assert ndcg_at_k(["a", "b", "c"], ["z"], 3) == pytest.approx(0.0)


def test_ndcg_at_k_gold_count_exceeds_k() -> None:
    # 3 gold docs but k=1: ideal DCG only accounts for 1 slot
    retrieved = ["a", "x", "y"]
    gold = ["a", "b", "c"]
    # DCG@1 = 1/log2(2) = 1.0 ; IDCG@1 = 1/log2(2) = 1.0 -> ndcg = 1.0
    assert ndcg_at_k(retrieved, gold, 1) == pytest.approx(1.0)


def test_ndcg_at_k_k_larger_than_result_set() -> None:
    retrieved = ["a"]
    gold = ["a", "b"]
    # DCG@5 = 1/log2(2); IDCG@5 (2 relevant, k=5) = 1/log2(2) + 1/log2(3)
    dcg = 1.0 / math.log2(2)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(retrieved, gold, 5) == pytest.approx(dcg / idcg)


def test_ndcg_at_k_invalid_k() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], ["a"], 0)


def test_ndcg_at_k_ties_in_relevance_order_matters() -> None:
    # both relevant docs retrieved, order affects nDCG when not all in top-k... here full recall
    # but placing them last vs first changes DCG even though set is identical
    gold = ["a", "b"]
    ndcg_best = ndcg_at_k(["a", "b", "c"], gold, 3)
    ndcg_worst = ndcg_at_k(["c", "b", "a"], gold, 3)
    assert ndcg_best == pytest.approx(1.0)
    assert ndcg_worst < ndcg_best


# ---------------------------------------------------------------------------
# hit rate
# ---------------------------------------------------------------------------


def test_hit_rate_at_k_hit() -> None:
    assert hit_rate_at_k(["a", "b", "c"], ["c", "z"], 3) == pytest.approx(1.0)


def test_hit_rate_at_k_miss() -> None:
    assert hit_rate_at_k(["a", "b", "c"], ["z"], 3) == pytest.approx(0.0)


def test_hit_rate_at_k_hit_outside_cutoff_is_miss() -> None:
    assert hit_rate_at_k(["a", "b", "c", "z"], ["z"], 2) == pytest.approx(0.0)


def test_hit_rate_at_k_no_gold_is_vacuous_one() -> None:
    assert hit_rate_at_k(["a", "b"], [], 2) == pytest.approx(1.0)


def test_hit_rate_at_k_empty_retrieved() -> None:
    assert hit_rate_at_k([], ["a"], 3) == pytest.approx(0.0)


def test_hit_rate_at_k_invalid_k() -> None:
    with pytest.raises(ValueError):
        hit_rate_at_k(["a"], ["a"], 0)


def test_all_metrics_agree_on_full_result_edge_case() -> None:
    # a single retrieved doc, single gold doc, they match, k=1
    retrieved = ["only"]
    gold = ["only"]
    assert precision_at_k(retrieved, gold, 1) == pytest.approx(1.0)
    assert recall_at_k(retrieved, gold, 1) == pytest.approx(1.0)
    assert mrr(retrieved, gold) == pytest.approx(1.0)
    assert ndcg_at_k(retrieved, gold, 1) == pytest.approx(1.0)
    assert hit_rate_at_k(retrieved, gold, 1) == pytest.approx(1.0)
