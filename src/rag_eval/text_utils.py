"""Shared tokenization helpers.

Tokenization is intentionally simple (lowercased word-boundary regex split) so
behaviour is fully deterministic and dependency-free. This is a lexical proxy,
not a linguistically aware tokenizer.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer: alphanumeric runs, keeping internal apostrophes."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def find_token_spans(text: str) -> list[tuple[int, int]]:
    """Character (start, end) spans for each token, in the same order as tokenize()."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
