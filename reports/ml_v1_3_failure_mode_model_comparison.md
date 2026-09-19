# ML-v1.3 Failure-Mode Model Comparison

| Failure mode | BERT full-mask recall | DeBERTa full-mask recall | BERT miss | DeBERTa miss |
|---|---:|---:|---:|---:|
| `accountlike_o_vs_business_id` | 0.900000 | 1.000000 | 4 | 0 |
| `auth_token_vs_business_id` | 1.000000 | 1.000000 | 0 | 0 |
| `ssn_vs_itin` | 1.000000 | 1.000000 | 0 | 0 |
| `ip_vs_technical_reference` | 0.991667 | 1.000000 | 0 | 0 |
| `high_risk_identifier` | 1.000000 | 1.000000 | 0 | 0 |

Categories follow the frozen ML-v1.3 challenge `failure_mode` field:
account-like O vs BUSINESS_ID, AUTH_TOKEN vs BUSINESS_ID, SSN vs ITIN,
IP vs technical-reference, and high-risk identifiers (including API_KEY / AUTH_TOKEN / SSN / ITIN).
Exact NER metrics are preserved separately; a policy-safe wrong class is not counted as an NER success.

DeBERTa does not reduce account-like O→BUSINESS_ID errors (94 vs 94 on
standard). Challenge AUTH_TOKEN vs BUSINESS_ID is already clean for both.
DeBERTa adds 54 standard SSN→ITIN confusions. Encoder switching therefore does
not resolve the persistent account-like failure; that points to data/task
ambiguity and/or a deterministic context resolver.
