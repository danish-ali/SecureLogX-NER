# ML-v1.3 DeBERTa Collapse Audit

**Scope: development only. Sealed challenge and original test were not accessed.**

Primary diagnosis: **IMPLEMENTATION/TOKENIZATION DEFECT IDENTIFIED**

**DeBERTa remains BLOCKED FROM SEALED VALIDATION pending review of this audit.**

## Exact-metric reproduction

| Model/view | Micro F1 | Macro F1 |
|---|---:|---:|
| BERT standard | 0.911475 | 0.933544 |
| DeBERTa standard | 0.262521 | 0.319832 |
| BERT challenge | 0.979592 | 0.995349 |
| DeBERTa challenge | 0.994152 | 0.998665 |

## Whitespace-boundary diagnostic

| Model | Raw standard micro F1 | Whitespace-trimmed micro F1 | Delta |
|---|---:|---:|---:|
| BERT | 0.911475 | 0.911475 | +0.000000 |
| DeBERTa | 0.262521 | 0.888943 | +0.626421 |

- DeBERTa standard boundary-adjusted spans: 2875/4020 (71.52%).
- DeBERTa challenge boundary-adjusted spans: 0/860 (0.00%).
- Shared exact account-like O -> BUSINESS_ID errors: 94.

This trimming test is diagnostic only. It does not change historical metrics or the predeclared selection result.
