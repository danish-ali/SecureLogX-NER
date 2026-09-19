# ML-v1.3 BERT vs DeBERTa Model Comparison

Selection used development data only. The sealed challenge was not opened.

| Metric | BERT | DeBERTa | Delta |
|---|---:|---:|---:|
| standard-dev micro F1 | 0.911475 | 0.262521 | -0.648954 |
| standard-dev macro F1 | 0.933544 | 0.319832 | -0.613712 |
| challenge-dev micro F1 | 0.979592 | 0.994152 | +0.014560 |
| challenge-dev macro F1 | 0.995349 | 0.998665 | +0.003316 |
| BUSINESS_ID precision (standard) | 0.550847 | 0.428571 | -0.122276 |
| BUSINESS_ID recall (standard) | 0.962963 | 0.733333 | -0.229630 |
| BUSINESS_ID F1 (standard) | 0.700809 | 0.540984 | -0.159825 |
| BUSINESS_ID P/R/F1 challenge-P | 0.971963 | 1.000000 | +0.028037 |
| BUSINESS_ID recall (challenge) | 0.962963 | 0.981481 | +0.018519 |
| SSN recall (standard) | 0.970874 | 0.038835 | -0.932039 |
| ITIN recall (standard) | 1.000000 | 0.000000 | -1.000000 |
| AUTH_TOKEN precision (standard) | 1.000000 | 1.000000 | +0.000000 |
| AUTH_TOKEN recall (standard) | 1.000000 | 1.000000 | +0.000000 |
| AUTH_TOKEN F1 (standard) | 1.000000 | 1.000000 | +0.000000 |
| API_KEY recall (standard) | 0.915888 | 0.000000 | -0.915888 |
| IP_ADDRESS F1 (standard) | 0.937984 | 0.077220 | -0.860764 |
| high-risk recall | 0.938557 | 0.350846 | -0.587711 |
| sensitive-span recall | 0.995349 | 1.000000 | +0.004651 |
| full-mask recall | 0.977907 | 1.000000 | +0.022093 |
| high-risk full-mask recall | 1.000000 | 1.000000 | +0.000000 |
| SECURITY_CRITICAL_MISS | 112.000000 | 108.000000 | -4.000000 |
| SECURITY_CRITICAL_PARTIAL | 112.000000 | 84.000000 | -28.000000 |
| OVERMASKING | 251.000000 | 245.000000 | -6.000000 |
| account-like O→BUSINESS_ID | 94.000000 | 94.000000 | +0.000000 |
| AUTH_TOKEN vs BUSINESS_ID errors | 0.000000 | 0.000000 | +0.000000 |
| challenge record error | 0.022917 | 0.020833 | -0.002083 |
| worst-category error | 0.083333 | 0.083333 | +0.000000 |

- BERT train min / peak GiB / 64-lat s: 62.06 / 3.32 / 0.305
- DeBERTa train min / peak GiB / 64-lat s: 202.68 / 4.17 / 0.521
- BERT model bytes: 431058932; DeBERTa model bytes: 735507468
- BERT selected epoch 3 SHA-256 `31a460720152a9b555150a55eb474d67cbfa2f236254d9ce464e930f8e2a0f44`
- DeBERTa selected epoch 3 SHA-256 `ec8229d7f3b1de4d4d2d3a9fc2b465edba194ffc5b06391a46c4645756e473b6`
- Wrong-class predictions that would still be masked remain POLICY_SAFE_WRONG_CLASS, not NER successes.
