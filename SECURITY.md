# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's advisory form:
<https://github.com/hellpuffyt/rag-eval/security/advisories/new>

Please do not open a public issue for a vulnerability.

In scope, for example: network access or process execution by the tool,
reading files other than the dataset given, writing anywhere other than
`--output`, or a crafted dataset that makes it hang or exhaust memory.

A metric that is computed incorrectly is a correctness bug; please open a
normal issue with a minimal dataset and the hand-computed expected value.

## Supported versions

Only the latest release on the default branch receives fixes.
