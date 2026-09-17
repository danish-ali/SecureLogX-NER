# BUSINESS_ID Subtype Analysis

The model predicts only `BUSINESS_ID`; original subtypes are recovered from frozen span provenance for analysis.

- Overall BUSINESS_ID precision: **88.81%**
- Overall BUSINESS_ID recall: **54.98%**
- Overall BUSINESS_ID F1: **67.91%**
- Records containing multiple BUSINESS_ID subtypes: **0**

| Original subtype | Records | Support | Exact TP | FN | Exact-span recall | Records with any BUSINESS_ID prediction | Context detection rate | Predictions/context record | Ambiguous context records |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACCOUNT_ID | 14 | 14 | 14 | 0 | 100.00% | 14 | 100.00% | 1.0000 | 0 |
| APPLICATION_ID | 11 | 11 | 11 | 0 | 100.00% | 11 | 100.00% | 1.0000 | 0 |
| CUSTOMER_ID | 33 | 40 | 32 | 8 | 80.00% | 31 | 93.94% | 1.1818 | 0 |
| LOAN_NUMBER | 16 | 16 | 16 | 0 | 100.00% | 16 | 100.00% | 1.0000 | 0 |
| ORDER_ID | 94 | 94 | 0 | 94 | 0.00% | 0 | 0.00% | 0.0000 | 0 |
| TRANSACTION_ID | 19 | 19 | 19 | 0 | 100.00% | 19 | 100.00% | 1.0000 | 0 |
| USER_ID | 32 | 37 | 35 | 2 | 94.59% | 31 | 96.88% | 1.3125 | 0 |

Subtype exact-span recall is well-defined because every gold BUSINESS_ID span retains its subtype. Predicted false positives do not have a gold subtype, so subtype precision and subtype F1 are intentionally not reported.

Context detection rate and predictions per context record are descriptive diagnostics only. In multi-subtype records, the same record-level predictions can appear in more than one subtype context; the ambiguity count makes that overlap explicit.
