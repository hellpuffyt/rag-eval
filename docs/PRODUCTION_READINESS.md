# Production readiness — rag-eval

Audit date: 2026-10-22. Tested code commit: `ec6b334` (branch
`audit/production-readiness`); later commits change documentation and
repository metadata only.

## Scope

Offline, standard-library-only evaluator for RAG datasets (JSONL):
precision@k, recall@k, MRR, nDCG@k, hit rate, a lexical groundedness score
(greedy longest-common-substring alignment), context utilization, and a
retrieval-miss / retrieval-rank / generation failure attribution, as a
library and CLI with JSON output and `--fail-under-*` gates. No model calls,
no network. Scores are computed only from the dataset; nothing is invented.

## Platforms and prerequisites

Python ≥ 3.10, no runtime dependencies. Verified on macOS 27 (Apple
Silicon), Python 3.12 (full + clean clone), 3.10 and 3.13 (suite). Not on
PyPI; the PyPI name `rag-eval` belongs to an unrelated project.
Linux/Windows CI legs not run for this audit.

## Clean-clone verification

```bash
git clone -b audit/production-readiness <repo> rag-eval && cd rag-eval
python3.12 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"   # exit 0
source .venv/bin/activate
rag-eval examples/sample.jsonl -k 3 --per-case        # matches the README example exactly
rag-eval examples/sample.jsonl -k 3 --format json --output ../re.json   # exit 0
rag-eval examples/sample.jsonl -k 3 --fail-under-recall 0.6 --fail-under-groundedness 0.4
#   exit 1: recall@k 0.5625 < 0.6 and groundedness 0.2924 < 0.4 (the sample is meant to fail)
pytest -q            # 120 passed
ruff check . ; mypy  # clean; no issues in 9 files
```

## Commands and outcomes

| Step | Command | Exit | Result |
|---|---|---:|---|
| Baseline (main `c2c84aa`) | `pytest -q` | 0 | 113 passed, 0 failed, 0 skipped |
| Baseline lint/types | `ruff check .`, `mypy` | 0, 0 | clean |
| Final (3.10, 3.12, 3.13) | `pytest -q` | 0 | 120 passed, 0 failed, 0 skipped |
| New fixtures on original code | `PYTHONPATH=<main>/src pytest tests/test_hand_computed_fixtures.py` | 1 | 3 failed, 4 passed |
| Dependency audit | `uvx pip-audit … --strict -s osv` | 0 | no known vulnerabilities |
| Secret scan (tree + history) | detect-secrets; credential-pattern `git grep` | – | none |
| Stress | groundedness, 3000-token answer vs 3000-token heavily overlapping context | – | 0.19 s |

The aggregate numbers in the README example were re-derived by hand from
the per-case table (e.g. recall@3 = (1+0+1+0.5+1+0+1+0)/8 = 0.5625; q4
nDCG@3 = 1 / (1 + 1/log2 3) = 0.613) and match.

CI: default branch `main`: **success** (`c2c84aa`). Audit branch: pending
until pushed.

## Bugs found and fixed

| # | Severity | Defect | Regression test |
|---|---|---|---|
| 1 | High (wrong metric math) | Duplicate retrieved ids counted repeatedly: `["c1","c1","c2"]` vs gold `["c1"]` gave recall@3 = 2.0 and nDCG@3 = 1.63; duplicates also pushed gold chunks down the attribution ranking | `test_duplicate_retrieved_ids_count_once`, `test_attribution_ignores_duplicates_when_ranking` |
| 2 | High (wrong score) | ASCII-only tokenizer: non-English words dropped; a fabricated all-Chinese answer scored groundedness 1.0 | `test_non_ascii_answer_is_not_vacuously_grounded` |
| 3 | Low (docs) | Install commands Windows-only; no warning that `pip install rag-eval` fetches an unrelated package | README |

## Security review

No network, no subprocess, reads only the dataset path, writes only
`--output`; `json` parsing only. CI token now `contents: read`. No secrets
in tree or history.

## Known limitations

Groundedness is lexical, not entailment (see README); binary relevance only;
scripts without spaces tokenize coarsely.

## Acceptance checklist

| Item | Expected | Command / test | Result | Actual | Reference |
|---|---|---|---|---|---|
| A1 | Install from source | README | pass | exit 0 | clean clone |
| A2 | Metric math equals hand computation | tests | pass | yes | `test_retrieval_metrics.py`, `test_hand_computed_fixtures.py` |
| A3 | Metrics bounded in [0, 1] incl. duplicates | tests | pass | yes | `test_duplicate_retrieved_ids_count_once` |
| A4 | Groundedness/utilization by hand | tests | pass | 3/8, 3/6 | `test_groundedness_by_hand` |
| A5 | Attribution: all three labels + success | tests | pass | yes | `test_attribution.py` |
| A6 | README example output reproduced | clean clone | pass | identical | README |
| A7 | Gates exit 1, dataset errors exit 2 | tests + clean clone | pass | yes | `test_cli.py` |

## Classification proposal

**Partially Verified** — metric math verified against hand-computed
fixtures on macOS with Python 3.10–3.13, clean clone passes. Not verified:
Linux/Windows CI on the audit branch (pending push).
