"""Metric values checked against arithmetic done by hand (shown in comments),
plus regressions found in the production-readiness audit."""

from __future__ import annotations

import math

import pytest

from rag_eval.attribution import classify_case
from rag_eval.groundedness import evaluate_groundedness
from rag_eval.retrieval_metrics import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag_eval.text_utils import tokenize
from rag_eval.types import Case

RETRIEVED = ["x", "a", "y", "b", "z"]
GOLD = ["a", "b", "c"]


def test_precision_recall_hit_by_hand() -> None:
    # top-4 = x a y b -> 2 relevant; precision 2/4, recall 2/3, hit 1
    assert precision_at_k(RETRIEVED, GOLD, 4) == pytest.approx(0.5)
    assert recall_at_k(RETRIEVED, GOLD, 4) == pytest.approx(2 / 3)
    assert hit_rate_at_k(RETRIEVED, GOLD, 1) == 0.0
    assert hit_rate_at_k(RETRIEVED, GOLD, 2) == 1.0
    # k beyond the list: denominator stays k (TREC) -> 2/10
    assert precision_at_k(RETRIEVED, GOLD, 10) == pytest.approx(0.2)


def test_mrr_by_hand() -> None:
    assert mrr(RETRIEVED, GOLD) == pytest.approx(1 / 2)  # first hit at rank 2
    assert mrr(RETRIEVED, GOLD, k=1) == 0.0


def test_ndcg_by_hand() -> None:
    # DCG@4  = 1/log2(3) + 1/log2(5)            = 0.630930 + 0.430677 = 1.061606
    # IDCG@4 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.630930 + 0.5   = 2.130930
    # nDCG@4 = 1.061606 / 2.130930              = 0.498189
    assert ndcg_at_k(RETRIEVED, GOLD, 4) == pytest.approx(0.498189, abs=1e-6)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3) + 0.5)
    assert ndcg_at_k(RETRIEVED, GOLD, 4) == pytest.approx(expected)


def test_duplicate_retrieved_ids_count_once() -> None:
    # Before: recall@3 = 2.0 and nDCG@3 = 1.63 for ["c1", "c1", "c2"].
    # Deduplicated ranking: c1 (rank 1), c2 (rank 2).
    dup = ["c1", "c1", "c2"]
    assert recall_at_k(dup, ["c1"], 3) == 1.0
    assert precision_at_k(dup, ["c1"], 3) == pytest.approx(1 / 3)
    assert ndcg_at_k(dup, ["c1"], 3) == 1.0
    # A duplicate ahead of the gold chunk must not push it down the ranking.
    assert mrr(["x", "x", "g"], ["g"]) == pytest.approx(1 / 2)
    for k in (1, 2, 3, 5):
        for gold in (["c1"], ["c2"], ["c1", "c2"], ["c3"]):
            assert 0.0 <= recall_at_k(dup, gold, k) <= 1.0
            assert 0.0 <= ndcg_at_k(dup, gold, k) <= 1.0


def test_attribution_ignores_duplicates_when_ranking() -> None:
    case = Case(
        id="d", question="q", retrieved_chunk_ids=["x", "x", "x", "g"],
        gold_chunk_ids=["g"], answer="alpha beta gamma delta",
        chunk_texts={"g": "alpha beta gamma delta", "x": "unrelated words here"},
    )
    # Unique ranking: x (1), g (2) -> within k=2, answer fully grounded.
    result = classify_case(case, k=2)
    assert result.label is None
    assert result.best_gold_rank == 2


def test_groundedness_by_hand() -> None:
    # answer tokens: the(0) tower(1) was(2) finished(3) in(4) 1889(5) by(6) engineers(7)
    # context has "the tower was" (3) and "in 1889" (2 < min 3, not counted).
    result = evaluate_groundedness(
        "The tower was finished in 1889 by engineers.",
        "The tower was completed in 1889.",
    )
    assert result.score == pytest.approx(3 / 8)
    # 3 of 6 context tokens used
    assert result.context_utilization == pytest.approx(3 / 6)
    assert result.unsupported_spans == ["finished in 1889 by engineers"]


def test_non_ascii_answer_is_not_vacuously_grounded() -> None:
    # Before: the ASCII-only tokenizer produced no tokens, so a fabricated
    # answer in Chinese scored 1.0 ("nothing to be unsupported").
    assert tokenize("Müller sagte: Ärger") == ["müller", "sagte", "ärger"]
    assert evaluate_groundedness("答案完全是编造的", "上下文说了别的事情").score == 0.0
    supported = evaluate_groundedness(
        "Die Straße heißt Hauptstraße in München", "Die Straße heißt Hauptstraße."
    )
    assert supported.score == pytest.approx(4 / 6)
