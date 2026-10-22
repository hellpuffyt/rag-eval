"""Shared tokenization helpers.

Tokenization is intentionally simple (lowercased word-boundary regex split) so
behaviour is fully deterministic and dependency-free. This is a lexical proxy,
not a linguistically aware tokenizer.
"""

from __future__ import annotations

import re

# Unicode letters and digits (``[^\W_]``), not just ASCII: an ASCII-only
# pattern silently dropped every non-ASCII word, so an answer written entirely
# in, say, Chinese tokenized to nothing and scored as vacuously grounded.
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer: Unicode alphanumeric runs, keeping internal
    apostrophes. Scripts written without spaces (e.g. Chinese, Japanese)
    become one token per run, so their groundedness is very coarse."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def find_token_spans(text: str) -> list[tuple[int, int]]:
    """Character (start, end) spans for each token, in the same order as tokenize()."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
