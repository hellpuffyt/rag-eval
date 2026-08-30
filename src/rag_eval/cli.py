"""Command-line interface for rag-eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rag_eval.attribution import DEFAULT_GROUNDEDNESS_THRESHOLD
from rag_eval.groundedness import DEFAULT_MIN_MATCH_LEN
from rag_eval.io import DatasetError, load_dataset
from rag_eval.report import evaluate_dataset, format_case_table, format_table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Offline evaluation for retrieval-augmented generation pipelines.",
    )
    parser.add_argument(
        "dataset", type=str, help="Path to a JSONL dataset file (see README for schema)."
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Retrieval cutoff for precision/recall/nDCG/hit-rate (default: 5).",
    )
    parser.add_argument(
        "--groundedness-threshold",
        type=float,
        default=DEFAULT_GROUNDEDNESS_THRESHOLD,
        help=(
            "Minimum groundedness score to count as generation success "
            f"(default: {DEFAULT_GROUNDEDNESS_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--min-match-len",
        type=int,
        default=DEFAULT_MIN_MATCH_LEN,
        help=(
            "Minimum contiguous token match length for groundedness alignment "
            f"(default: {DEFAULT_MIN_MATCH_LEN})."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--per-case",
        action="store_true",
        help="Include a per-case table (table format) or per-case detail (json).",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Write output to a file instead of stdout."
    )

    gates = parser.add_argument_group("CI gates (exit code 1 if any threshold is not met)")
    gates.add_argument("--fail-under-precision", type=float, default=None)
    gates.add_argument("--fail-under-recall", type=float, default=None)
    gates.add_argument("--fail-under-mrr", type=float, default=None)
    gates.add_argument("--fail-under-ndcg", type=float, default=None)
    gates.add_argument("--fail-under-hit-rate", type=float, default=None)
    gates.add_argument("--fail-under-groundedness", type=float, default=None)
    gates.add_argument("--fail-under-context-utilization", type=float, default=None)

    return parser


def _check_gates(report_dict: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures = []
    retrieval = report_dict["retrieval"]
    generation = report_dict["generation"]
    checks = [
        (
            "--fail-under-precision",
            args.fail_under_precision,
            retrieval["mean_precision_at_k"],
            "precision@k",
        ),
        ("--fail-under-recall", args.fail_under_recall, retrieval["mean_recall_at_k"], "recall@k"),
        ("--fail-under-mrr", args.fail_under_mrr, retrieval["mean_mrr"], "mrr"),
        ("--fail-under-ndcg", args.fail_under_ndcg, retrieval["mean_ndcg_at_k"], "ndcg@k"),
        (
            "--fail-under-hit-rate",
            args.fail_under_hit_rate,
            retrieval["mean_hit_rate_at_k"],
            "hit_rate@k",
        ),
        (
            "--fail-under-groundedness",
            args.fail_under_groundedness,
            generation["mean_groundedness"],
            "groundedness",
        ),
        (
            "--fail-under-context-utilization",
            args.fail_under_context_utilization,
            generation["mean_context_utilization"],
            "context_utilization",
        ),
    ]
    for flag, threshold, actual, name in checks:
        if threshold is not None and actual < threshold:
            failures.append(f"{name} = {actual:.4f} is below threshold {threshold:.4f} ({flag})")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cases = load_dataset(args.dataset)
    except (DatasetError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        report = evaluate_dataset(
            cases,
            k=args.k,
            groundedness_threshold=args.groundedness_threshold,
            min_match_len=args.min_match_len,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report_dict = report.to_dict(include_cases=args.per_case)

    if args.format == "json":
        output = json.dumps(report_dict, indent=2)
    else:
        output = format_table(report)
        if args.per_case:
            output += "\n\n" + format_case_table(report)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    gate_failures = _check_gates(report.to_dict(include_cases=False), args)
    if gate_failures:
        for failure in gate_failures:
            print(f"GATE FAILED: {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
