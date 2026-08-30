"""Tests for JSONL dataset loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.io import DatasetError, iter_cases, load_dataset


def write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_valid_dataset(tmp_path: Path) -> None:
    line = (
        '{"id": "q1", "question": "What?", "retrieved_chunk_ids": ["a", "b"], '
        '"gold_chunk_ids": ["a"], "answer": "It is a.", '
        '"chunk_texts": {"a": "text a", "b": "text b"}}'
    )
    path = write_jsonl(tmp_path / "data.jsonl", [line])
    cases = load_dataset(path)
    assert len(cases) == 1
    assert cases[0].id == "q1"
    assert cases[0].retrieved_chunk_ids == ["a", "b"]
    assert cases[0].chunk_texts["a"] == "text a"


def test_load_dataset_skips_blank_lines(tmp_path: Path) -> None:
    line = (
        '{"id": "q1", "question": "What?", "retrieved_chunk_ids": [], '
        '"gold_chunk_ids": [], "answer": "x"}'
    )
    path = write_jsonl(tmp_path / "data.jsonl", ["", line, "   ", ""])
    cases = load_dataset(path)
    assert len(cases) == 1


def test_load_dataset_multiple_lines(tmp_path: Path) -> None:
    lines = [
        f'{{"id": "q{i}", "question": "Q{i}?", "retrieved_chunk_ids": [], '
        f'"gold_chunk_ids": [], "answer": "a{i}"}}'
        for i in range(5)
    ]
    path = write_jsonl(tmp_path / "data.jsonl", lines)
    cases = load_dataset(path)
    assert [c.id for c in cases] == [f"q{i}" for i in range(5)]


def test_load_dataset_missing_required_field(tmp_path: Path) -> None:
    line = '{"id": "q1", "question": "What?", "gold_chunk_ids": [], "answer": "x"}'
    path = write_jsonl(tmp_path / "data.jsonl", [line])
    with pytest.raises(DatasetError, match="missing required field"):
        load_dataset(path)


def test_load_dataset_invalid_json(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "data.jsonl", ["{not valid json"])
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_dataset(path)


def test_load_dataset_not_an_object(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "data.jsonl", ["[1, 2, 3]"])
    with pytest.raises(DatasetError, match="expected a JSON object"):
        load_dataset(path)


def test_load_dataset_empty_file_raises(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "data.jsonl", [])
    with pytest.raises(DatasetError, match="no cases"):
        load_dataset(path)


def test_load_dataset_duplicate_ids_raises(tmp_path: Path) -> None:
    line = (
        '{{"id": "dup", "question": "Q?", "retrieved_chunk_ids": [], '
        '"gold_chunk_ids": [], "answer": "a"}}'
    )
    path = write_jsonl(tmp_path / "data.jsonl", [line.format(), line.format()])
    with pytest.raises(DatasetError, match="duplicate case id"):
        load_dataset(path)


def test_load_dataset_with_scores(tmp_path: Path) -> None:
    line = (
        '{"id": "q1", "question": "What?", "retrieved_chunk_ids": ["a", "b"], '
        '"retrieved_scores": [0.9, 0.4], "gold_chunk_ids": ["a"], "answer": "x"}'
    )
    path = write_jsonl(tmp_path / "data.jsonl", [line])
    cases = load_dataset(path)
    assert cases[0].retrieved_scores == [0.9, 0.4]


def test_load_dataset_mismatched_scores_length_raises(tmp_path: Path) -> None:
    line = (
        '{"id": "q1", "question": "What?", "retrieved_chunk_ids": ["a", "b"], '
        '"retrieved_scores": [0.9], "gold_chunk_ids": ["a"], "answer": "x"}'
    )
    path = write_jsonl(tmp_path / "data.jsonl", [line])
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_iter_cases_is_lazy_generator(tmp_path: Path) -> None:
    line = (
        '{"id": "q1", "question": "What?", "retrieved_chunk_ids": [], '
        '"gold_chunk_ids": [], "answer": "x"}'
    )
    path = write_jsonl(tmp_path / "data.jsonl", [line])
    gen = iter_cases(path)
    first = next(gen)
    assert first.id == "q1"
    with pytest.raises(StopIteration):
        next(gen)


def test_load_dataset_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_dataset(tmp_path / "does_not_exist.jsonl")
