# SecureLogX ML-v1.2 Development Gate

**FAIL DEVELOPMENT GATE**

No threshold was changed after results were observed. The sealed challenge and original test were not opened or inferred.

## Targeted diagnostics

- Account-like gold-O → BUSINESS_ID: **94 spans / 94 records**.
- CREDIT_CARD_NUMBER → BUSINESS_ID: **0/36 (0.000000)**.
- BUSINESS_ID → SSN: **0/54 (0.000000)**.
- Standard-dev SSN → ITIN: **5**.
- IP_ADDRESS on technical references: **0/54 (0.000000)**.
- CUSTOMER_ID / REQUEST_ID recall: **1.000000 / 0.800000**.
- Challenge record/context-target error: **0.131481 / 0.065741**.
- Worst category: **business_id_vs_auth_token (0.500000)**.

## Gate results

| Gate | Actual | Operator | Threshold | Result |
|---|---:|:---:|---:|---|
| `standard_dev_micro_f1_minimum` | 0.903912 | >= | 0.893486 | PASS |
| `standard_dev_supported_macro_f1_minimum` | 0.913050 | >= | 0.901667 | PASS |
| `standard_dev_high_risk_recall_minimum` | 0.896705 | >= | 0.889377 | PASS |
| `standard_dev_business_id_precision_minimum` | 0.557447 | >= | 0.650000 | FAIL |
| `standard_dev_business_id_recall_minimum` | 0.970370 | >= | 0.900000 | PASS |
| `standard_dev_business_id_f1_minimum` | 0.708108 | >= | 0.700000 | PASS |
| `standard_dev_ssn_precision_minimum` | 0.970000 | >= | 0.930000 | PASS |
| `standard_dev_ssn_recall_minimum` | 0.941748 | >= | 0.900000 | PASS |
| `standard_dev_ssn_f1_minimum` | 0.955665 | >= | 0.915000 | PASS |
| `standard_dev_ip_address_precision_minimum` | 0.960317 | >= | 0.940000 | PASS |
| `standard_dev_ip_address_recall_minimum` | 0.937984 | >= | 0.900000 | PASS |
| `standard_dev_ip_address_f1_minimum` | 0.949020 | >= | 0.920000 | PASS |
| `challenge_dev_micro_f1_minimum` | 0.920926 | >= | 0.900000 | PASS |
| `challenge_dev_supported_macro_f1_minimum` | 0.934822 | >= | 0.850000 | PASS |
| `challenge_dev_business_id_precision_minimum` | 0.963753 | >= | 0.900000 | PASS |
| `challenge_dev_business_id_recall_minimum` | 0.865900 | >= | 0.900000 | FAIL |
| `challenge_dev_ssn_recall_minimum` | 1.000000 | >= | 0.900000 | PASS |
| `challenge_dev_ip_address_recall_minimum` | 1.000000 | >= | 0.900000 | PASS |
| `challenge_dev_record_error_rate_maximum` | 0.131481 | <= | 0.100000 | FAIL |
| `challenge_dev_context_target_error_rate_maximum` | 0.065741 | <= | 0.100000 | PASS |
| `challenge_dev_maximum_category_error_rate_maximum` | 0.500000 | <= | 0.200000 | FAIL |
| `standard_dev_accountlike_o_to_business_id_count_maximum` | 94.000000 | <= | 47.000000 | FAIL |
| `standard_dev_ssn_to_itin_count_maximum` | 5.000000 | <= | 4.000000 | FAIL |
| `challenge_dev_credit_card_to_business_id_rate_maximum` | 0.000000 | <= | 0.100000 | PASS |
| `challenge_dev_business_id_to_ssn_rate_maximum` | 0.000000 | <= | 0.100000 | PASS |
| `challenge_dev_ip_on_technical_reference_rate_maximum` | 0.000000 | <= | 0.200000 | PASS |
| `challenge_dev_customer_id_recall_minimum` | 1.000000 | >= | 0.800000 | PASS |
| `challenge_dev_request_id_recall_minimum` | 0.800000 | >= | 0.800000 | PASS |
| `unexplained_alignment_failures_maximum` | 0.000000 | <= | 0.000000 | PASS |
| `truncated_entity_spans_maximum` | 0.000000 | <= | 0.000000 | PASS |
