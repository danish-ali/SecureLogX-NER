# ML-v1.3 Security-Outcome Model Comparison

Exact NER metrics are preserved separately. A wrong class that would still be
masked is POLICY_SAFE_WRONG_CLASS, not an NER success.

| Outcome | BERT standard | BERT challenge | DeBERTa standard | DeBERTa challenge |
|---|---:|---:|---:|---:|
| SECURITY_CRITICAL_MISS | 108.000000 | 4.000000 | 108.000000 | 0.000000 |
| SECURITY_CRITICAL_PARTIAL | 97.000000 | 15.000000 | 84.000000 | 0.000000 |
| POLICY_SAFE_WRONG_CLASS | 41.000000 | 0.000000 | 127.000000 | 0.000000 |
| OVERMASKING | 251.000000 | 0.000000 | 245.000000 | 0.000000 |
| account-like O→BUSINESS_ID | 94.000000 | 0.000000 | 94.000000 | 0.000000 |
| SSN→ITIN | 0.000000 | 0.000000 | 54.000000 | 0.000000 |
| AUTH_TOKEN vs BUSINESS_ID | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| API_KEY-related errors | 15.000000 | 0.000000 | 9.000000 | 0.000000 |
| full-mask recall | 0.949005 | 0.977907 | 0.952239 | 1.000000 |
| high-risk full-mask recall | 0.968833 | 1.000000 | 0.962600 | 1.000000 |
