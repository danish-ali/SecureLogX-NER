# ML-v1.1 Template Leakage Report

| Check | Observed | Required | Result |
|---|---:|---:|---|
| New cross-split exact-text overlap | 0 | 0 | PASS |
| New cross-split source-ID overlap | 0 | 0 | PASS |
| New cross-split template-family overlap | 0 | 0 | PASS |
| New cross-split target-value hash overlap | 0 | 0 | PASS |
| Exact text shared with any ML-v1 split | 0 | 0 | PASS |
| Source ID shared with any ML-v1 split | 0 | 0 | PASS |
| Family ID shared with any ML-v1 split | 0 | 0 | PASS |
| New cross-split normalized template skeletons | 0 | 0 | PASS |
| New skeletons equal to parent generator templates | 0 | 0 | PASS |
| Forbidden historical failed family/wording | 0 | 0 | PASS |

The original ML-v1 test has no exact text, source record ID, or template family copied into the v1.1 training addition or dev challenge. Every contrast group remains wholly within one split.

## Descriptive near-template diagnostics

- Maximum cross-new-split literal token 3-gram Jaccard: **0.142857**
- Maximum cross-new-split sequence ratio: **0.873362**
- Maximum new-vs-parent literal token 3-gram Jaccard: **0.066667**
- Maximum new-vs-parent sequence ratio: **0.746269**

These lexical ratios are diagnostics, not proof of semantic independence. The sealed scenarios were separately authored; zero exact normalized skeleton overlap and explicit catalog review provide the enforceable evidence. The historical `order_reference_ssn_shape_warn_v1` family and its failed sentence are forbidden.
