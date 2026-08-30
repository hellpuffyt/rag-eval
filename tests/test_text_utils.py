"""Tests for tokenization helpers."""

from __future__ import annotations

from rag_eval.text_utils import find_token_spans, tokenize


def test_tokenize_lowercases() -> None:
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_splits_on_punctuation() -> None:
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_tokenize_keeps_apostrophes() -> None:
    assert tokenize("don't stop") == ["don't", "stop"]


def test_tokenize_handles_numbers() -> None:
    assert tokenize("in 1889 exactly") == ["in", "1889", "exactly"]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_find_token_spans_matches_tokenize_length() -> None:
    text = "The quick, brown fox!"
    assert len(find_token_spans(text)) == len(tokenize(text))


def test_find_token_spans_are_correct_offsets() -> None:
    text = "hi there"
    spans = find_token_spans(text)
    assert text[spans[0][0] : spans[0][1]] == "hi"
    assert text[spans[1][0] : spans[1][1]] == "there"
