# ML-v1.3 BERT Training Summary

- Model: `bert-base-cased` revision `cd5ef92a9fb2f889e972770a36d4ed042daf221e`
- Selected epoch: **3**
- Selection score (harmonic mean of supported-entity macro F1): **0.963456**
- Training records: **30462**
- Standard micro/macro F1: **0.911475 / 0.933544**
- Challenge micro/macro F1: **0.979592 / 0.995349**
- Challenge record error / worst-category error: **0.022917 / 0.083333**
- BUSINESS_ID P/R/F1 (standard): **0.550847 / 0.962963 / 0.700809**
- AUTH_TOKEN P/R/F1 (standard): **1.000000 / 1.000000 / 1.000000**
- SSN P/R/F1 (standard): **0.952381 / 0.970874 / 0.961538**
- ITIN P/R/F1 (standard): **0.979167 / 1.000000 / 0.989474**
- Challenge security-critical miss/partial: **4 / 15**
- Combined (standard+challenge) miss/partial: **112 / 112**
- High-risk full-mask recall (challenge): **1.000000**
- Account-like O→BUSINESS_ID: standard **94**, challenge **0**
- AUTH_TOKEN vs BUSINESS_ID errors (challenge): **0**
- SSN→ITIN: standard **0**, challenge **0**
- Checkpoint SHA-256: `31a460720152a9b555150a55eb474d67cbfa2f236254d9ce464e930f8e2a0f44`
- Train minutes: **62.06**; peak GPU GiB: **3.32**
- Model bytes: **431058932**
- 64-record challenge latency seconds: **0.305**
- Sealed challenge was not opened.

## Epoch history

| Epoch | Train loss | Std micro | Std macro | Ch micro | Ch macro | Selection |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.229515 | 0.880291 | 0.894482 | 0.987059 | 0.997035 | 0.942979 |
| 2 | 0.014849 | 0.901495 | 0.913234 | 0.988235 | 0.997305 | 0.953420 |
| 3 ← selected | 0.007824 | 0.911475 | 0.933544 | 0.979592 | 0.995349 | 0.963456 |

## Challenge failure-mode security

| Failure mode | Miss | Partial | Full-mask recall | High-risk full-mask recall |
|---|---:|---:|---:|---:|
| `accountlike_o_vs_business_id` | 4 | 14 | 0.900000 | 1.000000 |
| `auth_token_vs_business_id` | 0 | 0 | 1.000000 | 1.000000 |
| `ssn_vs_itin` | 0 | 0 | 1.000000 | 1.000000 |
| `ip_vs_technical_reference` | 0 | 1 | 0.991667 | 1.000000 |
| `high_risk_identifier` | 0 | 0 | 1.000000 | 1.000000 |
