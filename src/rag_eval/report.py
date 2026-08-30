"""Aggregate reporting: per-case detail plus dataset-level summary statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_eval.attribution import (
    DEFAULT_GROUNDEDNESS_THRESHOLD,
    AttributionResult,
    classify_case,
)
from rag_eval.groundedness import DEFAULT_MIN_MATCH_LEN, evaluate_groundedness
from rag_eval.retrieval_metrics import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag_eval.types import Case


@dataclass(frozen=True)
class CaseReport:
    """Per-case computed metrics."""

    id: str
    precision: float
    recall: float
    mrr: float
    ndcg: float
    hit_rate: float
    groundedness: float
    context_utilization: float
    unsupported_spans: list[str]
    attribution: AttributionResult


@dataclass(frozen=True)
class AggregateReport:
    """Dataset-level summary: mean metrics plus failure-attribution counts."""

    k: int
    num_cases: int
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    mean_ndcg: float
    mean_hit_rate: float
    mean_groundedness: float
    mean_context_utilization: float
    num_success: int
    num_retrieval_miss: int
    num_retrieval_rank: int
    num_generation_failure: int
    cases: list[CaseReport] = field(default_factory=list)

    def to_dict(self, include_cases: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "k": self.k,
            "num_cases": self.num_cases,
            "retrieval": {
                "mean_precision_at_k": round(self.mean_precision, 6),
                "mean_recall_at_k": round(self.mean_recall, 6),
                "mean_mrr": round(self.mean_mrr, 6),
                "mean_ndcg_at_k": round(self.mean_ndcg, 6),
                "mean_hit_rate_at_k": round(self.mean_hit_rate, 6),
            },
            "generation": {
                "mean_groundedness": round(self.mean_groundedness, 6),
                "mean_context_utilization": round(self.mean_context_utilization, 6),
            },
            "failure_attribution": {
                "success": self.num_success,
                "retrieval_miss": self.num_retrieval_miss,
                "retrieval_rank": self.num_retrieval_rank,
                "generation": self.num_generation_failure,
            },
        }
        if include_cases:
            data["cases"] = [
                {
                    "id": c.id,
                    "precision_at_k": round(c.precision, 6),
                    "recall_at_k": round(c.recall, 6),
                    "mrr": round(c.mrr, 6),
                    "ndcg_at_k": round(c.ndcg, 6),
                    "hit_rate_at_k": round(c.hit_rate, 6),
                    "groundedness": round(c.groundedness, 6),
                    "context_utilization": round(c.context_utilization, 6),
                    "unsupported_spans": c.unsupported_spans,
                    "failure_label": c.attribution.label,
                    "best_gold_rank": c.attribution.best_gold_rank,
                }
                for c in self.cases
            ]
        return data


def evaluate_case(
    case: Case,
    k: int,
    groundedness_threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
    min_match_len: int = DEFAULT_MIN_MATCH_LEN,
) -> CaseReport:
    """Compute all metrics for a single case."""
    context = case.context_text(case.retrieved_chunk_ids[:k])
    grounding = evaluate_groundedness(case.answer, context, min_match_len=min_match_len)
    attribution = classify_case(
        case, k=k, groundedness_threshold=groundedness_threshold, min_match_len=min_match_len
    )
    return CaseReport(
        id=case.id,
        precision=precision_at_k(case.retrieved_chunk_ids, case.gold_chunk_ids, k),
        recall=recall_at_k(case.retrieved_chunk_ids, case.gold_chunk_ids, k),
        mrr=mrr(case.retrieved_chunk_ids, case.gold_chunk_ids, k=k),
        ndcg=ndcg_at_k(case.retrieved_chunk_ids, case.gold_chunk_ids, k),
        hit_rate=hit_rate_at_k(case.retrieved_chunk_ids, case.gold_chunk_ids, k),
        groundedness=grounding.score,
        context_utilization=grounding.context_utilization,
        unsupported_spans=grounding.unsupported_spans,
        attribution=attribution,
    )


def evaluate_dataset(
    cases: list[Case],
    k: int,
    groundedness_threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
    min_match_len: int = DEFAULT_MIN_MATCH_LEN,
) -> AggregateReport:
    """Compute per-case metrics and dataset-level aggregates."""
    if not cases:
        raise ValueError("cannot evaluate an empty case list")

    case_reports = [
        evaluate_case(
            c, k=k, groundedness_threshold=groundedness_threshold, min_match_len=min_match_len
        )
        for c in cases
    ]
    n = len(case_reports)

    num_success = sum(1 for c in case_reports if c.attribution.label is None)
    num_retrieval_miss = sum(1 for c in case_reports if c.attribution.label == "retrieval_miss")
    num_retrieval_rank = sum(1 for c in case_reports if c.attribution.label == "retrieval_rank")
    num_generation_failure = sum(1 for c in case_reports if c.attribution.label == "generation")

    return AggregateReport(
        k=k,
        num_cases=n,
        mean_precision=sum(c.precision for c in case_reports) / n,
        mean_recall=sum(c.recall for c in case_reports) / n,
        mean_mrr=sum(c.mrr for c in case_reports) / n,
        mean_ndcg=sum(c.ndcg for c in case_reports) / n,
        mean_hit_rate=sum(c.hit_rate for c in case_reports) / n,
        mean_groundedness=sum(c.groundedness for c in case_reports) / n,
        mean_context_utilization=sum(c.context_utilization for c in case_reports) / n,
        num_success=num_success,
        num_retrieval_miss=num_retrieval_miss,
        num_retrieval_rank=num_retrieval_rank,
        num_generation_failure=num_generation_failure,
        cases=case_reports,
    )


def format_table(report: AggregateReport) -> str:
    """Render a human-readable summary table."""
    lines = []
    lines.append(f"rag-eval report  (k={report.k}, cases={report.num_cases})")
    lines.append("")
    lines.append("Retrieval metrics")
    lines.append(f"  precision@{report.k}         {report.mean_precision:.4f}")
    lines.append(f"  recall@{report.k}            {report.mean_recall:.4f}")
    lines.append(f"  mrr                    {report.mean_mrr:.4f}")
    lines.append(f"  ndcg@{report.k}              {report.mean_ndcg:.4f}")
    lines.append(f"  hit_rate@{report.k}          {report.mean_hit_rate:.4f}")
    lines.append("")
    lines.append("Generation metrics")
    lines.append(f"  groundedness           {report.mean_groundedness:.4f}")
    lines.append(f"  context_utilization    {report.mean_context_utilization:.4f}")
    lines.append("")
    lines.append("Failure attribution")
    lines.append(f"  success                {report.num_success}")
    lines.append(f"  retrieval_miss         {report.num_retrieval_miss}")
    lines.append(f"  retrieval_rank         {report.num_retrieval_rank}")
    lines.append(f"  generation             {report.num_generation_failure}")
    return "\n".join(lines)


def format_case_table(report: AggregateReport) -> str:
    """Render a per-case human-readable table."""
    header = (
        f"{'id':<12}{'prec':>8}{'rec':>8}{'mrr':>8}{'ndcg':>8}"
        f"{'hit':>6}{'ground':>9}{'util':>8}  label"
    )
    lines = [header, "-" * len(header)]
    for c in report.cases:
        label = c.attribution.label or "success"
        lines.append(
            f"{c.id:<12}{c.precision:>8.3f}{c.recall:>8.3f}{c.mrr:>8.3f}{c.ndcg:>8.3f}"
            f"{c.hit_rate:>6.2f}{c.groundedness:>9.3f}{c.context_utilization:>8.3f}  {label}"
        )
    return "\n".join(lines)
