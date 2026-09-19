# ML-v1.2 Template Leakage Report

| Check | Observed | Required | Result |
|---|---:|---:|---|
| New cross-split exact-text overlap | 0 | 0 | PASS |
| New cross-split source-ID overlap | 0 | 0 | PASS |
| New cross-split template-family overlap | 0 | 0 | PASS |
| New cross-split focus-value hash overlap | 0 | 0 | PASS |
| Exact text shared with any ML-v1 split | 0 | 0 | PASS |
| Exact text shared with the original ML-v1 test | 0 | 0 | PASS |
| Source ID shared with any ML-v1 split | 0 | 0 | PASS |
| Family ID shared with any ML-v1 split | 0 | 0 | PASS |
| ML-v1.1 training addition overlap (text/id/family/value/skeleton) | 0 | 0 | PASS |
| ML-v1.1 dev challenge overlap (text/id/family/value/skeleton) | 0 | 0 | PASS |
| Sealed final challenge overlap (text/id/family/value/skeleton) | 0 | 0 | PASS |
| New cross-split normalized template skeletons | 0 | 0 | PASS |
| New skeletons equal to ML-v1.1 catalog templates | 0 | 0 | PASS |
| New family names shared with the ML-v1.1 catalog | 0 | 0 | PASS |
| New skeletons equal to parent generator templates | 0 | 0 | PASS |
| Forbidden historical failed family/wording | 0 | 0 | PASS |

ML-v1.1 comparisons were executed against hash-verified partitions: readable files are checked on disk, while the sealed challenge (and any v1.1 file that is not readable in this environment) is regenerated deterministically in memory by the frozen v1.1 generator, and the canonical serialization must reproduce the byte SHA-256 recorded in `configs/ml_v1_2_parent_freeze.json` before any comparison is trusted. The sealed challenge file itself was never opened.

## Descriptive near-template diagnostics

- Maximum cross-new-split literal token 3-gram Jaccard: **0.181818**
- Maximum cross-new-split sequence ratio: **0.875274**
- Maximum new-vs-ML-v1.1 literal token 3-gram Jaccard: **0.000000**
- Maximum new-vs-ML-v1.1 sequence ratio: **0.627551**
- Maximum new-vs-parent literal token 3-gram Jaccard: **0.000000**
- Maximum new-vs-parent sequence ratio: **0.514469**

These lexical ratios are diagnostics, not proof of semantic independence. The v1.2 train and dev families intentionally share the COUNTERCHECK surface scaffolding while using disjoint field vocabularies, semantic signatures, and value namespaces, so their sequence ratios are elevated; exact normalized-skeleton overlap is what the gate enforces, and it is zero everywhere. The historical `order_reference_ssn_shape_warn_v1` family and its failed wording remain forbidden.
