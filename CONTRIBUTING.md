# Contributing

Contributions are welcome. This project is small and dependency-free on
purpose; please keep changes in that spirit.

## Development setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# source .venv/bin/activate && pip install -e ".[dev]"  # macOS / Linux
```

## Running the checks

All three must pass before a change is merged; CI runs the same commands.

```bash
pytest
ruff check .
mypy
```

## Guidelines

- **No new runtime dependencies without discussion.** The evaluation logic is
  stdlib-only by design so it stays trivially auditable and safe to run
  offline. `numpy` may be considered for performance-sensitive paths but is
  not currently required.
- **Metric changes need hand-computed tests.** If you touch
  `retrieval_metrics.py` or `groundedness.py`, add a test with a
  hand-computed expected value in a comment, not just a regression snapshot.
- **Document edge cases.** Empty gold sets, empty answers, ties, and
  `k` larger than the result set are all real inputs users will hit. If you
  change behavior for one of them, update the docstring and the README's
  metrics reference.
- **Keep the groundedness limitation honest.** Anything touching
  `groundedness.py` should preserve (or strengthen) the documentation that
  this is a lexical proxy, not an entailment check.

## Reporting issues

Please include a minimal JSONL snippet reproducing the problem when filing a
bug — it makes triage much faster.
