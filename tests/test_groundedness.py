"""Tests for lexical groundedness / unsupported-span detection."""

from __future__ import annotations

import pytest

from rag_eval.groundedness import evaluate_groundedness, ngram_precision
from rag_eval.text_utils import tokenize


def test_fully_supported_answer_scores_one() -> None:
    context = "The Eiffel Tower is located in Paris and was completed in 1889."
    answer = "The Eiffel Tower is located in Paris."
    result = evaluate_groundedness(answer, context, min_match_len=3)
    assert result.score == pytest.approx(1.0)
    assert result.unsupported_spans == []


def test_fully_unsupported_answer_scores_zero() -> None:
    context = "Bananas are a good source of potassium."
    answer = "Quantum entanglement violates local realism."
    result = evaluate_groundedness(answer, context, min_match_len=3)
    assert result.score == pytest.approx(0.0)
    assert len(result.unsupported_spans) == 1


def test_partially_supported_answer_reports_unsupported_span() -> None:
    context = "The report was published in March 2023 by the research team."
    answer = "The report was published in March 2023 by aliens from Mars."
    result = evaluate_groundedness(answer, context, min_match_len=3)
    assert 0.0 < result.score < 1.0
    assert any("aliens" in span.lower() for span in result.unsupported_spans)


def test_empty_answer_is_vacuously_grounded() -> None:
    result = evaluate_groundedness("", "some context here", min_match_len=3)
    assert result.score == pytest.approx(1.0)
    assert result.unsupported_spans == []
    assert result.answer_token_count == 0


def test_empty_context_zero_groundedness_for_nonempty_answer() -> None:
    result = evaluate_groundedness("some claim about the world", "", min_match_len=3)
    assert result.score == pytest.approx(0.0)
    assert result.context_utilization == pytest.approx(0.0)
    assert result.unsupported_spans == ["some claim about the world"]


def test_empty_answer_and_empty_context() -> None:
    result = evaluate_groundedness("", "", min_match_len=3)
    assert result.score == pytest.approx(1.0)
    assert result.context_utilization == pytest.approx(0.0)


def test_context_utilization_full_when_answer_covers_all_context() -> None:
    context = "cats sleep a lot during the day"
    answer = "cats sleep a lot during the day, apparently"
    result = evaluate_groundedness(answer, context, min_match_len=3)
    assert result.context_utilization == pytest.approx(1.0)


def test_context_utilization_low_when_answer_ignores_most_context() -> None:
    context = (
        "The quarterly revenue grew by twelve percent driven by strong demand "
        "in the enterprise segment and improved retention across all regions."
    )
    answer = "Revenue grew."
    result = evaluate_groundedness(answer, context, min_match_len=2)
    assert result.context_utilization < 0.3


def test_min_match_len_filters_short_coincidental_overlaps() -> None:
    context = "a completely unrelated sentence about gardening tools"
    answer = "the cat sat on a mat"
    # "a" alone is a 1-token overlap; with min_match_len=2 it should not count
    result = evaluate_groundedness(answer, context, min_match_len=2)
    assert result.score == pytest.approx(0.0)


def test_min_match_len_one_allows_single_token_matches() -> None:
    context = "gardening tools are useful"
    answer = "gardening is fun"
    result = evaluate_groundedness(answer, context, min_match_len=1)
    assert result.score > 0.0


def test_invalid_min_match_len_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_groundedness("a", "b", min_match_len=0)


def test_unsupported_spans_preserve_original_casing_and_punctuation() -> None:
    context = "Paris is the capital of France."
    answer = "Paris is the capital of France, home to the Eiffel Tower!"
    result = evaluate_groundedness(answer, context, min_match_len=3)
    assert any("Eiffel Tower" in span for span in result.unsupported_spans)


def test_multiple_unsupported_spans_are_all_reported() -> None:
    context = "The cat sat on the mat. The dog ran in the park."
    answer = "Elephants danced. The cat sat on the mat. Robots exploded."
    result = evaluate_groundedness(answer, context, min_match_len=3)
    assert len(result.unsupported_spans) >= 2


def test_repeated_context_phrase_can_support_multiple_answer_occurrences() -> None:
    context = "quarterly earnings beat expectations"
    answer = "quarterly earnings beat expectations, and quarterly earnings beat expectations again"
    result = evaluate_groundedness(answer, context, min_match_len=3)
    # both occurrences of the 4-token phrase are matched (8 of 10 answer tokens);
    # only the connective "and" and the trailing "again" remain unsupported.
    assert result.score == pytest.approx(0.8)
    assert result.unsupported_spans == ["and", "again"]


def test_matches_list_reports_correct_token_offsets() -> None:
    context = "the quick brown fox jumps over the lazy dog"
    answer = "the quick brown fox is fast"
    result = evaluate_groundedness(answer, context, min_match_len=3)
    assert len(result.matches) >= 1
    m = result.matches[0]
    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)
    assert (
        answer_tokens[m.answer_start : m.answer_end]
        == context_tokens[m.context_start : m.context_end]
    )


def test_ngram_precision_full_overlap() -> None:
    assert ngram_precision("hello world", "hello world foo", n=1) == pytest.approx(1.0)


def test_ngram_precision_partial_overlap() -> None:
    score = ngram_precision("hello world foo", "hello world bar", n=1)
    assert score == pytest.approx(2 / 3)


def test_ngram_precision_bigrams() -> None:
    # answer bigrams: (hello, world), (world, foo) -> only first present
    score = ngram_precision("hello world foo", "hello world bar", n=2)
    assert score == pytest.approx(0.5)


def test_ngram_precision_answer_shorter_than_n_is_vacuous_one() -> None:
    assert ngram_precision("hi", "anything else entirely", n=3) == pytest.approx(1.0)


def test_ngram_precision_invalid_n() -> None:
    with pytest.raises(ValueError):
        ngram_precision("a", "b", n=0)


def test_case_insensitive_matching() -> None:
    context = "The Capital Of France Is Paris"
    answer = "the capital of france is paris"
    result = evaluate_groundedness(answer, context, min_match_len=3)
    assert result.score == pytest.approx(1.0)
