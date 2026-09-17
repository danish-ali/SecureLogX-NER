# ML-v1.1 SSN Morphology-Conflict Analysis

**Scope: selected-checkpoint development challenge only.**

| Diagnostic | ML-v1 historical immutable test evidence | ML-v1.1 dev challenge |
|---|---:|---:|
| True SSN support | 105 | 120 |
| True SSNs correctly recognized | 104 | 120 |
| BUSINESS_ID values incorrectly classified as SSN | 93 in historical failed family | 10 |
| Other non-SSN values incorrectly classified as SSN | 3 aggregate residual SSN false positives | 0 |
| SSNs missed as BUSINESS_ID | not re-derived after stop | 0 |
| SSN boundary errors | not re-derived after stop | 0 |

ML-v1.1 challenge SSN precision/recall/F1 is **0.923077 / 1.000000 / 0.960000**. BUSINESS_ID recall is **0.935000**, so SSN recall was not improved merely by suppressing SSN output.

The historical and dev-challenge rows use different datasets and are diagnostic, not a direct regression comparison.
