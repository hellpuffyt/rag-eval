"""Failure attribution: is a bad case a retrieval problem or a generation problem?

For a case to be scored as a "success" it must both (a) retrieve at least one
gold-relevant chunk within the top-k cutoff, and (b) the generated answer must
be adequately grounded in that retrieved context (see
:mod:`rag_eval.groundedness`). Whenever a case is not a full success, exactly
one of the following three labels explains why:

- ``retrieval_miss``: none of the gold-relevant chunk ids appear anywhere in
  the retrieved list at all. The retriever never found the right chunk.
- ``retrieval_rank``: a gold-relevant chunk *was* retrieved, but not within
  the top-k cutoff -- it was found, but ranked too low to be used.
- ``generation``: a gold-relevant chunk was retrieved within the top-k cutoff,
  but the generated answer is not adequately grounded in the retrieved
  context (groundedness score below ``groundedness_threshold``). Retrieval did
  its job; generation ignored or contradicted it.

A case with no gold-relevant chunks at all is always a success (there was
nothing to retrieve), matching the vacuous-truth convention used throughout
:mod:`rag_eval.retrieval_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag_eval.groundedness import DEFAULT_MIN_MATCH_LEN, evaluate_groundedness
from rag_eval.types import Case

FailureLabel = Literal["retrieval_miss", "retrieval_rank", "generation"]

DEFAULT_GROUNDEDNESS_THRESHOLD = 0.5


@dataclass(frozen=True)
class AttributionResult:
    """Outcome of classifying a single case."""

    label: FailureLabel | None
    """None means the case succeeded; otherwise one of the FailureLabel values."""
    best_gold_rank: int | None
    """1-indexed rank of the best-ranked retrieved gold chunk, or None if none
    was retrieved at all."""
    groundedness_score: float | None
    """Groundedness score computed against the top-k context, or None if it
    was not computed (case already failed at the retrieval stage)."""


def classify_case(
    case: Case,
    k: int,
    groundedness_threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
    min_match_len: int = DEFAULT_MIN_MATCH_LEN,
) -> AttributionResult:
    """Classify why (if at all) a single case failed.

    Args:
        case: The evaluation case.
        k: The retrieval cutoff used to judge whether a gold chunk was
            "usably" ranked (must match the k used for the retrieval metrics
            being reported alongside this attribution).
        groundedness_threshold: Minimum groundedness score (see
            :func:`rag_eval.groundedness.evaluate_groundedness`) for the
            answer to be considered adequately grounded, given retrieval
            succeeded.
        min_match_len: Minimum contiguous token match length used by the
            groundedness aligner.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    gold = set(case.gold_chunk_ids)
    if not gold:
        return AttributionResult(label=None, best_gold_rank=None, groundedness_score=None)

    best_rank: int | None = None
    for rank, rid in enumerate(case.retrieved_chunk_ids, start=1):
        if rid in gold:
            best_rank = rank
            break

    if best_rank is None:
        return AttributionResult(
            label="retrieval_miss", best_gold_rank=None, groundedness_score=None
        )

    if best_rank > k:
        return AttributionResult(
            label="retrieval_rank", best_gold_rank=best_rank, groundedness_score=None
        )

    top_k_ids = case.retrieved_chunk_ids[:k]
    context = case.context_text(top_k_ids)
    result = evaluate_groundedness(case.answer, context, min_match_len=min_match_len)

    if result.score < groundedness_threshold:
        return AttributionResult(
            label="generation", best_gold_rank=best_rank, groundedness_score=result.score
        )

    return AttributionResult(label=None, best_gold_rank=best_rank, groundedness_score=result.score)
