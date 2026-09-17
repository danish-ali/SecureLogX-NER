# Template and Text Leakage Report

## Gate results

| Gate | Target | Result | Status |
|---|---:|---:|---|
| Exact duplicate text leakage | 0 | 0 | PASS |
| Synthetic template-family leakage | 0 | 0 | PASS |
| Synthetic records missing template family | 0 | 0 | PASS |

## Exact-text pair checks

- Train/dev: 0
- Train/test: 0
- Dev/test: 0

Synthetic template families checked: **266**.

External-source records have no template-family field and therefore cannot be evaluated by the template-family gate; they remain protected by stable-record grouping and exact-text deduplication.
