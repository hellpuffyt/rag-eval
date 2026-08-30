"""Lexical groundedness: how much of an answer is supported by its context.

**This is a lexical proxy, not entailment.** It measures word-overlap and
longest-common-substring alignment between the generated answer and the
retrieved context. It cannot tell that "the meeting was moved to Tuesday" is
contradicted by context saying "the meeting stays on Monday" -- both share
almost every word. It also cannot verify factual correctness, only textual
support. Use it to catch answers that are largely *unsupported*
(fabricated/hallucinated relative to the retrieved text) or that ignore the
context entirely, not as a substitute for human or entailment-model review.

Method
------
We greedily align the answer against the context using repeated
longest-common-substring (LCS, contiguous token match) extraction:

1. Tokenize both answer and context (see :mod:`rag_eval.text_utils`).
2. Find the longest contiguous run of tokens shared between any still-uncovered
   region of the answer and the full context.
3. If that run is at least ``min_match_len`` tokens, mark it as "supported" on
   both sides (answer tokens covered, context tokens used) and repeat on the
   remaining uncovered answer regions.
4. Stop when no remaining region has a common substring of at least
   ``min_match_len`` tokens with the context.

``groundedness score`` = fraction of answer tokens covered by some match.
``context utilization`` = fraction of context tokens used by some match.
``unsupported spans`` = the original-text substrings of the answer's
uncovered token runs (each run reported as one span, in reading order).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.text_utils import find_token_spans, tokenize

DEFAULT_MIN_MATCH_LEN = 3


@dataclass(frozen=True)
class Match:
    """A contiguous token match between the answer and the context."""

    length: int
    answer_start: int
    answer_end: int
    context_start: int
    context_end: int


@dataclass(frozen=True)
class GroundednessResult:
    """Result of aligning an answer against its retrieved context."""

    score: float
    context_utilization: float
    unsupported_spans: list[str]
    matches: list[Match]
    answer_token_count: int
    context_token_count: int


def _lcs_substring(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """Longest common (contiguous) substring of token lists a and b.

    Returns (length, a_start, b_start). length is 0 if there is no overlap.
    """
    if not a or not b:
        return 0, 0, 0
    prev = [0] * (len(b) + 1)
    best_len = 0
    best_a_end = 0
    best_b_end = 0
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_a_end = i
                    best_b_end = j
        prev = curr
    return best_len, best_a_end - best_len, best_b_end - best_len


def _greedy_align(
    answer_tokens: list[str], context_tokens: list[str], min_match_len: int
) -> tuple[list[Match], list[bool], list[bool]]:
    answer_covered = [False] * len(answer_tokens)
    context_covered = [False] * len(context_tokens)
    matches: list[Match] = []
    segments: list[tuple[int, int]] = [(0, len(answer_tokens))]

    while segments:
        best_len = 0
        best_seg_idx = -1
        best_a_start = 0
        best_b_start = 0
        for idx, (s, e) in enumerate(segments):
            length, a_off, b_start = _lcs_substring(answer_tokens[s:e], context_tokens)
            if length > best_len:
                best_len = length
                best_seg_idx = idx
                best_a_start = s + a_off
                best_b_start = b_start
        if best_len < min_match_len or best_seg_idx < 0:
            break

        a_start, a_end = best_a_start, best_a_start + best_len
        b_start, b_end = best_b_start, best_b_start + best_len
        for i in range(a_start, a_end):
            answer_covered[i] = True
        for j in range(b_start, b_end):
            context_covered[j] = True
        matches.append(
            Match(
                length=best_len,
                answer_start=a_start,
                answer_end=a_end,
                context_start=b_start,
                context_end=b_end,
            )
        )

        s, e = segments.pop(best_seg_idx)
        if s < a_start:
            segments.append((s, a_start))
        if a_end < e:
            segments.append((a_end, e))

    return matches, answer_covered, context_covered


def _spans_from_coverage(text: str, covered: list[bool]) -> list[str]:
    if not covered:
        return []
    token_spans = find_token_spans(text)
    spans: list[str] = []
    run_start: int | None = None
    for idx, is_covered in enumerate(covered):
        if not is_covered:
            if run_start is None:
                run_start = idx
        else:
            if run_start is not None:
                char_start = token_spans[run_start][0]
                char_end = token_spans[idx - 1][1]
                spans.append(text[char_start:char_end])
                run_start = None
    if run_start is not None:
        char_start = token_spans[run_start][0]
        char_end = token_spans[-1][1]
        spans.append(text[char_start:char_end])
    return spans


def evaluate_groundedness(
    answer: str, context: str, min_match_len: int = DEFAULT_MIN_MATCH_LEN
) -> GroundednessResult:
    """Score how much of ``answer`` is lexically supported by ``context``.

    An empty answer is vacuously fully "grounded" (score 1.0, no unsupported
    spans -- there is no content to be unsupported). An empty context yields a
    groundedness score of 0.0 for any non-empty answer (nothing to support it)
    and context_utilization of 0.0 in all cases (nothing to utilize).
    """
    if min_match_len <= 0:
        raise ValueError("min_match_len must be a positive integer")

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)

    if not answer_tokens:
        return GroundednessResult(
            score=1.0,
            context_utilization=0.0,
            unsupported_spans=[],
            matches=[],
            answer_token_count=0,
            context_token_count=len(context_tokens),
        )

    if not context_tokens:
        return GroundednessResult(
            score=0.0,
            context_utilization=0.0,
            unsupported_spans=_spans_from_coverage(answer, [False] * len(answer_tokens)),
            matches=[],
            answer_token_count=len(answer_tokens),
            context_token_count=0,
        )

    matches, answer_covered, context_covered = _greedy_align(
        answer_tokens, context_tokens, min_match_len
    )
    score = sum(answer_covered) / len(answer_covered)
    utilization = sum(context_covered) / len(context_covered)
    unsupported = _spans_from_coverage(answer, answer_covered)

    return GroundednessResult(
        score=score,
        context_utilization=utilization,
        unsupported_spans=unsupported,
        matches=matches,
        answer_token_count=len(answer_tokens),
        context_token_count=len(context_tokens),
    )


def ngram_precision(answer: str, context: str, n: int = 1) -> float:
    """Fraction of the answer's unique n-grams that also occur in the context.

    A coarser, order-independent companion metric to :func:`evaluate_groundedness`.
    Returns 1.0 if the answer has fewer than n tokens (nothing to check).
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)
    answer_ngrams = {tuple(answer_tokens[i : i + n]) for i in range(len(answer_tokens) - n + 1)}
    if not answer_ngrams:
        return 1.0
    context_ngrams = {tuple(context_tokens[i : i + n]) for i in range(len(context_tokens) - n + 1)}
    hits = sum(1 for ng in answer_ngrams if ng in context_ngrams)
    return hits / len(answer_ngrams)
