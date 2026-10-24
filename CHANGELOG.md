# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Duplicate ids in `retrieved_chunk_ids` were counted repeatedly: recall@k
  and nDCG@k could exceed 1.0 (e.g. 2.0 and 1.63), and duplicates pushed gold
  chunks down the attribution ranking. Duplicates are now ignored after the
  first occurrence everywhere.
- The tokenizer only recognised ASCII letters: a non-English answer lost its
  words, and an answer entirely in e.g. Chinese scored as vacuously 100%
  grounded. Tokenization is now Unicode-aware.
- README installation commands were Windows-only and did not warn that the
  PyPI name `rag-eval` belongs to an unrelated project.

### Added

- Hand-computed metric fixtures; `SECURITY.md`, `.editorconfig`,
  `docs/PRODUCTION_READINESS.md`; CI token limited to `contents: read`.

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
