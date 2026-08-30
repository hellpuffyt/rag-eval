"""JSONL dataset loading for rag-eval."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from rag_eval.types import Case


class DatasetError(ValueError):
    """Raised when a dataset file is malformed."""


_REQUIRED_FIELDS = ("id", "question", "retrieved_chunk_ids", "gold_chunk_ids", "answer")


def _parse_line(line: str, line_no: int) -> Case:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"line {line_no}: invalid JSON ({exc})") from exc

    if not isinstance(raw, dict):
        raise DatasetError(f"line {line_no}: expected a JSON object, got {type(raw).__name__}")

    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise DatasetError(f"line {line_no}: missing required field(s): {', '.join(missing)}")

    try:
        return Case(
            id=str(raw["id"]),
            question=str(raw["question"]),
            retrieved_chunk_ids=[str(c) for c in raw["retrieved_chunk_ids"]],
            gold_chunk_ids=[str(c) for c in raw["gold_chunk_ids"]],
            answer=str(raw["answer"]),
            chunk_texts={str(k): str(v) for k, v in raw.get("chunk_texts", {}).items()},
            retrieved_scores=(
                [float(s) for s in raw["retrieved_scores"]]
                if raw.get("retrieved_scores") is not None
                else None
            ),
        )
    except ValueError as exc:
        raise DatasetError(f"line {line_no}: {exc}") from exc


def iter_cases(path: str | Path) -> Iterator[Case]:
    """Stream cases from a JSONL dataset file, one Case per non-blank line."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            yield _parse_line(line, line_no)


def load_dataset(path: str | Path) -> list[Case]:
    """Load an entire JSONL dataset into memory as a list of Case objects."""
    cases = list(iter_cases(path))
    if not cases:
        raise DatasetError(f"dataset {path!s} contains no cases")
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise DatasetError(f"duplicate case id(s): {sorted(dupes)}")
    return cases
