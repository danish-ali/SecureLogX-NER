# ML-v1.3 DEBERTA Training Summary

- Model: `microsoft/deberta-v3-base` revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`
- Selected epoch: **3**
- Selection score (harmonic mean of supported-entity macro F1): **0.484499**
- Training records: **30462**
- Standard micro/macro F1: **0.262521 / 0.319832**
- Challenge micro/macro F1: **0.994152 / 0.998665**
- Challenge record error / worst-category error: **0.020833 / 0.083333**
- BUSINESS_ID P/R/F1 (standard): **0.428571 / 0.733333 / 0.540984**
- AUTH_TOKEN P/R/F1 (standard): **1.000000 / 1.000000 / 1.000000**
- SSN P/R/F1 (standard): **0.037383 / 0.038835 / 0.038095**
- ITIN P/R/F1 (standard): **0.000000 / 0.000000 / 0.000000**
- Challenge security-critical miss/partial: **0 / 0**
- Combined (standard+challenge) miss/partial: **108 / 84**
- High-risk full-mask recall (challenge): **1.000000**
- Account-like O→BUSINESS_ID: standard **94**, challenge **0**
- AUTH_TOKEN vs BUSINESS_ID errors (challenge): **0**
- SSN→ITIN: standard **54**, challenge **0**
- Checkpoint SHA-256: `ec8229d7f3b1de4d4d2d3a9fc2b465edba194ffc5b06391a46c4645756e473b6`
- Train minutes: **202.68**; peak GPU GiB: **4.17**
- Model bytes: **735507468**
- 64-record challenge latency seconds: **0.521**
- Sealed challenge was not opened.

## Epoch history

| Epoch | Train loss | Std micro | Std macro | Ch micro | Ch macro | Selection |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.208124 | 0.257579 | 0.313117 | 0.954003 | 0.989678 | 0.475723 |
| 2 | 0.014138 | 0.263067 | 0.317757 | 0.987741 | 0.997204 | 0.481944 |
| 3 ← selected | 0.008830 | 0.262521 | 0.319832 | 0.994152 | 0.998665 | 0.484499 |

## Challenge failure-mode security

| Failure mode | Miss | Partial | Full-mask recall | High-risk full-mask recall |
|---|---:|---:|---:|---:|
| `accountlike_o_vs_business_id` | 0 | 0 | 1.000000 | 1.000000 |
| `auth_token_vs_business_id` | 0 | 0 | 1.000000 | 1.000000 |
| `ssn_vs_itin` | 0 | 0 | 1.000000 | 1.000000 |
| `ip_vs_technical_reference` | 0 | 0 | 1.000000 | 1.000000 |
| `high_risk_identifier` | 0 | 0 | 1.000000 | 1.000000 |
