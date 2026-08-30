# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-30

### Added

- Initial release.
- Retrieval metrics: precision@k, recall@k, MRR, nDCG@k, hit rate, with
  documented edge-case conventions (empty gold sets, k larger than the result
  set, ties).
- Lexical groundedness scoring via greedy longest-common-substring alignment,
  with per-case unsupported-span extraction.
- Context utilization metric (fraction of retrieved context actually used by
  the answer).
- Three-way failure attribution: `retrieval_miss`, `retrieval_rank`,
  `generation`.
- `rag-eval` CLI with table/JSON output, per-case detail, and CI threshold
  gates (`--fail-under-*` flags).
- JSONL dataset format and loader with validation.
