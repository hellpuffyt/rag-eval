"""Tests for aggregate reporting."""

from __future__ import annotations

import pytest

from rag_eval.report import evaluate_case, evaluate_dataset, format_case_table, format_table
from rag_eval.types import Case


def make_case(
    id_: str, retrieved: list[str], gold: list[str], answer: str, texts: dict[str, str]
) -> Case:
    return Case(
        id=id_,
        question="q?",
        retrieved_chunk_ids=retrieved,
        gold_chunk_ids=gold,
        answer=answer,
        chunk_texts=texts,
    )


def test_evaluate_case_computes_all_fields() -> None:
    case = make_case(
        "c1",
        retrieved=["gold", "b"],
        gold=["gold"],
        answer="the sky is blue",
        texts={"gold": "the sky is blue", "b": "unrelated"},
    )
    result = evaluate_case(case, k=2)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)
    assert result.mrr == pytest.approx(1.0)
    assert result.groundedness == pytest.approx(1.0)
    assert result.attribution.label is None


def test_evaluate_dataset_aggregates_means() -> None:
    cases = [
        make_case(
            "c1",
            ["a"],
            ["a"],
            "the sky is blue today",
            {"a": "the sky is blue today"},
        ),
        make_case(
            "c2",
            ["z"],
            ["a"],
            "totally unrelated content",
            {"z": "totally unrelated content here", "a": "the sky is blue today"},
        ),
    ]
    report = evaluate_dataset(cases, k=1)
    assert report.num_cases == 2
    assert report.mean_precision == pytest.approx((1.0 + 0.0) / 2)
    assert report.num_retrieval_miss == 1
    assert report.num_success == 1


def test_evaluate_dataset_empty_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_dataset([], k=1)


def test_evaluate_dataset_failure_attribution_counts_sum_to_total() -> None:
    cases = [
        make_case("c1", ["a"], ["a"], "matches a", {"a": "matches a"}),
        make_case("c2", ["z"], ["a"], "nope", {"z": "z content", "a": "matches a"}),
        make_case(
            "c3",
            ["x", "a"],
            ["a"],
            "nothing grounded at all here",
            {"x": "x", "a": "matches a totally different topic content xyz"},
        ),
    ]
    report = evaluate_dataset(cases, k=1, groundedness_threshold=0.9)
    total = (
        report.num_success
        + report.num_retrieval_miss
        + report.num_retrieval_rank
        + report.num_generation_failure
    )
    assert total == report.num_cases


def test_to_dict_includes_cases_by_default() -> None:
    cases = [make_case("c1", ["a"], ["a"], "matches a", {"a": "matches a"})]
    report = evaluate_dataset(cases, k=1)
    data = report.to_dict()
    assert "cases" in data
    assert data["cases"][0]["id"] == "c1"


def test_to_dict_can_exclude_cases() -> None:
    cases = [make_case("c1", ["a"], ["a"], "matches a", {"a": "matches a"})]
    report = evaluate_dataset(cases, k=1)
    data = report.to_dict(include_cases=False)
    assert "cases" not in data


def test_to_dict_has_expected_top_level_keys() -> None:
    cases = [make_case("c1", ["a"], ["a"], "matches a", {"a": "matches a"})]
    report = evaluate_dataset(cases, k=1)
    data = report.to_dict()
    assert set(
        ["k", "num_cases", "retrieval", "generation", "failure_attribution", "cases"]
    ) <= set(data.keys())


def test_format_table_contains_key_metrics() -> None:
    cases = [make_case("c1", ["a"], ["a"], "matches a", {"a": "matches a"})]
    report = evaluate_dataset(cases, k=1)
    text = format_table(report)
    assert "precision@1" in text
    assert "recall@1" in text
    assert "ndcg@1" in text
    assert "groundedness" in text
    assert "retrieval_miss" in text


def test_format_case_table_lists_each_case() -> None:
    cases = [
        make_case(
            "c1",
            ["a"],
            ["a"],
            "the sky is blue today",
            {"a": "the sky is blue today"},
        ),
        make_case(
            "c2",
            ["b"],
            ["a"],
            "no match here at all",
            {"b": "b content", "a": "the sky is blue today"},
        ),
    ]
    report = evaluate_dataset(cases, k=1)
    text = format_case_table(report)
    assert "c1" in text
    assert "c2" in text
    assert "success" in text
    assert "retrieval_miss" in text
