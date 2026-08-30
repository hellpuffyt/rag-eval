"""Tests for the Case dataclass."""

from __future__ import annotations

import pytest

from rag_eval.types import Case


def test_case_context_text_defaults_to_all_retrieved() -> None:
    case = Case(
        id="c1",
        question="q",
        retrieved_chunk_ids=["a", "b"],
        gold_chunk_ids=["a"],
        answer="ans",
        chunk_texts={"a": "text a", "b": "text b"},
    )
    assert case.context_text() == "text a\ntext b"


def test_case_context_text_with_explicit_ids() -> None:
    case = Case(
        id="c1",
        question="q",
        retrieved_chunk_ids=["a", "b", "c"],
        gold_chunk_ids=["a"],
        answer="ans",
        chunk_texts={"a": "text a", "b": "text b", "c": "text c"},
    )
    assert case.context_text(["c", "a"]) == "text c\ntext a"


def test_case_context_text_skips_unknown_chunk_ids() -> None:
    case = Case(
        id="c1",
        question="q",
        retrieved_chunk_ids=["a", "missing"],
        gold_chunk_ids=[],
        answer="ans",
        chunk_texts={"a": "text a"},
    )
    assert case.context_text() == "text a"


def test_case_mismatched_scores_length_raises() -> None:
    with pytest.raises(ValueError):
        Case(
            id="c1",
            question="q",
            retrieved_chunk_ids=["a", "b"],
            gold_chunk_ids=[],
            answer="ans",
            retrieved_scores=[0.5],
        )


def test_case_matching_scores_length_ok() -> None:
    case = Case(
        id="c1",
        question="q",
        retrieved_chunk_ids=["a", "b"],
        gold_chunk_ids=[],
        answer="ans",
        retrieved_scores=[0.9, 0.1],
    )
    assert case.retrieved_scores == [0.9, 0.1]
