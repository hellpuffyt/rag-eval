"""Failure-attribution classifier tests: all three failure directions plus success."""

from __future__ import annotations

import pytest

from rag_eval.attribution import classify_case
from rag_eval.types import Case


def make_case(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: list[str],
    answer: str,
    chunk_texts: dict[str, str] | None = None,
) -> Case:
    return Case(
        id="c1",
        question="q?",
        retrieved_chunk_ids=retrieved_chunk_ids,
        gold_chunk_ids=gold_chunk_ids,
        answer=answer,
        chunk_texts=chunk_texts or {},
    )


def test_retrieval_miss_gold_never_retrieved() -> None:
    case = make_case(
        retrieved_chunk_ids=["a", "b", "c"],
        gold_chunk_ids=["z"],
        answer="whatever the model said",
        chunk_texts={"a": "text a", "b": "text b", "c": "text c"},
    )
    result = classify_case(case, k=3)
    assert result.label == "retrieval_miss"
    assert result.best_gold_rank is None
    assert result.groundedness_score is None


def test_retrieval_rank_gold_found_but_below_cutoff() -> None:
    case = make_case(
        retrieved_chunk_ids=["a", "b", "c", "gold"],
        gold_chunk_ids=["gold"],
        answer="some answer text",
        chunk_texts={"a": "a", "b": "b", "c": "c", "gold": "the real answer content"},
    )
    result = classify_case(case, k=2)
    assert result.label == "retrieval_rank"
    assert result.best_gold_rank == 4
    assert result.groundedness_score is None


def test_generation_failure_gold_retrieved_but_answer_ungrounded() -> None:
    case = make_case(
        retrieved_chunk_ids=["gold", "b", "c"],
        gold_chunk_ids=["gold"],
        answer="completely unrelated fabricated nonsense about dragons and spaceships",
        chunk_texts={
            "gold": "the quarterly revenue increased due to strong enterprise demand",
            "b": "irrelevant chunk about weather patterns",
            "c": "irrelevant chunk about cooking recipes",
        },
    )
    result = classify_case(case, k=3, groundedness_threshold=0.5)
    assert result.label == "generation"
    assert result.best_gold_rank == 1
    assert result.groundedness_score is not None
    assert result.groundedness_score < 0.5


def test_success_gold_retrieved_and_answer_grounded() -> None:
    case = make_case(
        retrieved_chunk_ids=["gold", "b"],
        gold_chunk_ids=["gold"],
        answer="the quarterly revenue increased due to strong enterprise demand",
        chunk_texts={
            "gold": "the quarterly revenue increased due to strong enterprise demand",
            "b": "irrelevant chunk about weather patterns",
        },
    )
    result = classify_case(case, k=2, groundedness_threshold=0.5)
    assert result.label is None
    assert result.best_gold_rank == 1
    assert result.groundedness_score == pytest.approx(1.0)


def test_no_gold_chunks_is_always_success() -> None:
    case = make_case(retrieved_chunk_ids=["a", "b"], gold_chunk_ids=[], answer="anything")
    result = classify_case(case, k=2)
    assert result.label is None
    assert result.best_gold_rank is None
    assert result.groundedness_score is None


def test_gold_at_exact_cutoff_boundary_counts_as_retrieved() -> None:
    case = make_case(
        retrieved_chunk_ids=["a", "b", "gold"],
        gold_chunk_ids=["gold"],
        answer="matching content here",
        chunk_texts={"a": "a", "b": "b", "gold": "matching content here"},
    )
    result = classify_case(case, k=3)
    assert result.best_gold_rank == 3
    assert result.label is None


def test_gold_one_past_cutoff_boundary_is_retrieval_rank() -> None:
    case = make_case(
        retrieved_chunk_ids=["a", "b", "gold"],
        gold_chunk_ids=["gold"],
        answer="matching content here",
        chunk_texts={"a": "a", "b": "b", "gold": "matching content here"},
    )
    result = classify_case(case, k=2)
    assert result.label == "retrieval_rank"


def test_multiple_gold_ids_uses_best_rank() -> None:
    case = make_case(
        retrieved_chunk_ids=["gold2", "x", "gold1"],
        gold_chunk_ids=["gold1", "gold2"],
        answer="text matching gold2 content",
        chunk_texts={"gold2": "text matching gold2 content", "x": "x", "gold1": "gold1 content"},
    )
    result = classify_case(case, k=3)
    assert result.best_gold_rank == 1


def test_invalid_k_raises() -> None:
    case = make_case(["a"], ["a"], "a")
    with pytest.raises(ValueError):
        classify_case(case, k=0)


def test_groundedness_threshold_boundary_is_inclusive_success() -> None:
    case = make_case(
        retrieved_chunk_ids=["gold"],
        gold_chunk_ids=["gold"],
        answer="exact match text",
        chunk_texts={"gold": "exact match text"},
    )
    # groundedness will be 1.0, well above any reasonable threshold
    result = classify_case(case, k=1, groundedness_threshold=1.0)
    assert result.label is None
