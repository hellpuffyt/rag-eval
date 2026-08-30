"""Retrieval-quality metrics: precision@k, recall@k, MRR, nDCG@k, hit rate.

All metrics take a ranked list of retrieved chunk ids (best result first) and a
set of gold-relevant chunk ids. Relevance is treated as binary (a chunk is
relevant iff it is in ``gold_ids``); scores are not required for any of these
metrics because ranking is defined entirely by list order, matching how a
retriever's ``top_k`` output is normally represented.

Edge-case conventions (documented explicitly since they are judgment calls):

- If ``gold_ids`` is empty (no relevant documents exist for the question),
  recall@k, nDCG@k, and hit_rate are all vacuously ``1.0`` -- there is nothing
  to find, so "finding everything relevant" is trivially satisfied.
  precision@k is unaffected by this convention: it is still computed as
  "relevant retrieved / k" and will be ``0.0`` in this situation, since none of
  the (irrelevant, because nothing is relevant) retrieved items count.
- precision@k uses a fixed denominator of ``k`` (the requested cutoff), not
  ``min(k, len(retrieved))``. This is the conventional TREC-style definition:
  if a retriever returns fewer than ``k`` results, the missing slots count
  against it as non-relevant. If you want precision over "what was actually
  returned", pass ``k=len(retrieved_chunk_ids)``.
- MRR is computed over the full retrieved list by default (``k=None``), matching
  the standard definition of mean reciprocal rank. Pass ``k`` to restrict the
  search to the top-k results only.
"""

from __future__ import annotations

import math


def _relevant_flags(retrieved_ids: list[str], gold_ids: set[str]) -> list[bool]:
    return [rid in gold_ids for rid in retrieved_ids]


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str] | set[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are relevant.

    Denominator is always ``k`` (TREC-style): missing results count as
    non-relevant. Raises ValueError if k <= 0.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    gold = set(gold_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in gold)
    return hits / k


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str] | set[str], k: int) -> float:
    """Fraction of relevant ids found within the top-k retrieved results.

    Returns 1.0 (vacuously) if there are no gold-relevant ids at all.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    gold = set(gold_ids)
    if not gold:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in gold)
    return hits / len(gold)


def mrr(retrieved_ids: list[str], gold_ids: list[str] | set[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant retrieved id (0.0 if none found).

    If ``k`` is given, only the top-k retrieved ids are searched.
    """
    gold = set(gold_ids)
    candidates = retrieved_ids if k is None else retrieved_ids[:k]
    for rank, rid in enumerate(candidates, start=1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance: list[float], k: int) -> float:
    """Discounted cumulative gain of a relevance-graded list, at cutoff k.

    Uses the standard log2(rank + 1) discount: DCG = sum(rel_i / log2(i + 1))
    for i = 1..k (1-indexed rank).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    total = 0.0
    for i, rel in enumerate(relevance[:k], start=1):
        total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str] | set[str], k: int) -> float:
    """Normalized DCG@k with binary relevance and ideal-DCG normalization.

    IDCG is the DCG of the best possible ranking: all relevant items first
    (up to k). If there are no relevant ids, nDCG@k is vacuously 1.0.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    gold = set(gold_ids)
    if not gold:
        return 1.0
    relevance = [1.0 if rid in gold else 0.0 for rid in retrieved_ids]
    dcg = dcg_at_k(relevance, k)
    ideal_relevance = [1.0] * min(len(gold), k) + [0.0] * max(0, k - len(gold))
    idcg = dcg_at_k(ideal_relevance, k)
    if idcg == 0.0:
        return 1.0
    return dcg / idcg


def hit_rate_at_k(retrieved_ids: list[str], gold_ids: list[str] | set[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k retrieved results, else 0.0.

    Returns 1.0 (vacuously) if there are no gold-relevant ids at all.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    gold = set(gold_ids)
    if not gold:
        return 1.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in gold for rid in top_k) else 0.0
