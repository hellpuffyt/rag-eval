"""Tests for the rag-eval CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.cli import main

GOOD_LINE = (
    '{{"id": "{id}", "question": "Q?", "retrieved_chunk_ids": ["a"], '
    '"gold_chunk_ids": ["a"], "answer": "text a", "chunk_texts": {{"a": "text a"}}}}'
)


@pytest.fixture()
def dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    path.write_text(GOOD_LINE.format(id="q1") + "\n", encoding="utf-8")
    return path


def test_cli_table_output(capsys: pytest.CaptureFixture[str], dataset_path: Path) -> None:
    code = main([str(dataset_path), "-k", "1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "rag-eval report" in out


def test_cli_json_output(capsys: pytest.CaptureFixture[str], dataset_path: Path) -> None:
    code = main([str(dataset_path), "-k", "1", "--format", "json"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["num_cases"] == 1


def test_cli_per_case_table(capsys: pytest.CaptureFixture[str], dataset_path: Path) -> None:
    code = main([str(dataset_path), "-k", "1", "--per-case"])
    assert code == 0
    out = capsys.readouterr().out
    assert "q1" in out


def test_cli_missing_dataset_returns_error_code(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = main([str(tmp_path / "nope.jsonl")])
    assert code == 2
    err = capsys.readouterr().err
    assert "error" in err


def test_cli_malformed_dataset_returns_error_code(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    code = main([str(path)])
    assert code == 2


def test_cli_gate_pass(capsys: pytest.CaptureFixture[str], dataset_path: Path) -> None:
    code = main([str(dataset_path), "-k", "1", "--fail-under-precision", "0.5"])
    assert code == 0


def test_cli_gate_fail(capsys: pytest.CaptureFixture[str], dataset_path: Path) -> None:
    code = main([str(dataset_path), "-k", "1", "--fail-under-groundedness", "1.1"])
    assert code == 1
    err = capsys.readouterr().err
    assert "GATE FAILED" in err


def test_cli_writes_to_output_file(tmp_path: Path, dataset_path: Path) -> None:
    out_path = tmp_path / "out.json"
    code = main([str(dataset_path), "-k", "1", "--format", "json", "--output", str(out_path)])
    assert code == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["num_cases"] == 1


def test_cli_multiple_gate_failures_all_reported(
    capsys: pytest.CaptureFixture[str], dataset_path: Path
) -> None:
    code = main(
        [
            str(dataset_path),
            "-k",
            "1",
            "--fail-under-groundedness",
            "1.1",
            "--fail-under-precision",
            "1.1",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert err.count("GATE FAILED") == 2


def test_cli_custom_min_match_len(capsys: pytest.CaptureFixture[str], dataset_path: Path) -> None:
    code = main([str(dataset_path), "-k", "1", "--min-match-len", "1", "--format", "json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["generation"]["mean_groundedness"] == pytest.approx(1.0)
