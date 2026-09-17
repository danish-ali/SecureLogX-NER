# Data Lineage Report

**Merge Date (UTC)**: 2026-09-13T14:02:58.566808+00:00

## Inputs

- `data\processed\gretel_finance_securelogx.jsonl`
- `data\custom\generated_logs\securelogx_generated_labeled.jsonl`

## Merge Summary

- **Input Records**: 30000
- **Merged Records**: 29541
- **Merged Canonical Entity Spans**: 40581
- **Merged Negative Records (zero entities)**: 2059
- **Invalid Records Rejected**: 0
- **Exact Duplicate Text Records Detected**: 0
- **Exact Duplicate Text Records Removed**: 0
- **Training-Excluded Records**: 459
- **Canonical Spans Removed With Training-Excluded Records**: 2003
- **All Excluded/Trace Spans Observed**: 17631
- **Training-Excluding Metadata Spans**: 641
- **Benign Trace-Only Metadata Spans**: 16990
- **Records With Deferred ML-v1 Metadata**: 358
- **Deferred ML-v1 Metadata Spans**: 449
- **Records With Healthcare-Extension Metadata**: 0
- **Healthcare-Extension Metadata Spans**: 0
- **Duplicate Texts With Conflicting Annotations**: 0

A record is omitted when `meta.training_exclusion=true` or any `meta.excluded_spans` entry has `training_exclusion=true`. This prevents an intentionally excluded sensitive value from becoming an implicit `O` label.

## Source Summary

| Source | Input | Merged | Negative | Entities | Invalid | Training-Excluded | Canonical Spans Removed | Duplicates | Excluded/Trace Spans | Usage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gretel_finance_pii | 5000 | 4541 | 1217 | 13885 | 0 | 459 | 2003 | 0 | 17631 | 15.4% |
| securelogx_custom_logs | 25000 | 25000 | 842 | 26696 | 0 | 0 | 0 | 0 | 0 | 84.6% |

## Accepted Source Mix

| Source Kind | Records | Canonical Spans | Record Percentage |
|---|---:|---:|---:|
| Synthetic | 25000 | 26696 | 84.6% |
| External | 4541 | 13885 | 15.4% |

## Canonical Entity Distribution in Merged Data

| Entity | Count | Percentage |
|---|---:|---:|
| PERSON_NAME | 7505 | 18.49% |
| EMAIL | 4565 | 11.25% |
| STREET_ADDRESS | 4183 | 10.31% |
| PHONE | 1945 | 4.79% |
| BUSINESS_ID | 1726 | 4.25% |
| IP_ADDRESS | 1278 | 3.15% |
| DOB | 1177 | 2.90% |
| CREDIT_CARD_NUMBER | 1162 | 2.86% |
| SWIFT_BIC | 1099 | 2.71% |
| PASSPORT_NUMBER | 1068 | 2.63% |
| BANK_ACCOUNT_NUMBER | 1067 | 2.63% |
| IBAN | 1059 | 2.61% |
| DRIVER_LICENSE | 1057 | 2.60% |
| ROUTING_NUMBER | 1053 | 2.59% |
| SSN | 1051 | 2.59% |
| AUTH_TOKEN | 1034 | 2.55% |
| API_KEY | 1032 | 2.54% |
| AGE | 940 | 2.32% |
| CITY | 940 | 2.32% |
| COUNTRY | 940 | 2.32% |
| DEVICE_ID | 940 | 2.32% |
| ITIN | 940 | 2.32% |
| POSTAL_CODE | 940 | 2.32% |
| STATE_PROVINCE | 940 | 2.32% |
| TAX_ID | 940 | 2.32% |

## Excluded and Deferred Source Labels

| Source Label | All Trace Spans | Training-Excluding | Benign Trace-Only | Deferred ML-v1 | Healthcare Extension |
|---|---:|---:|---:|---:|---:|
| account_pin | 162 | 162 | 0 | 162 | 0 |
| company | 6030 | 8 | 6022 | 0 | 0 |
| credit_card_number | 1 | 1 | 0 | 0 | 0 |
| credit_card_security_code | 120 | 120 | 0 | 120 | 0 |
| date | 9305 | 2 | 9303 | 0 | 0 |
| date_of_birth | 2 | 2 | 0 | 0 | 0 |
| date_time | 88 | 0 | 88 | 0 | 0 |
| driver_license_number | 4 | 4 | 0 | 0 | 0 |
| email | 10 | 10 | 0 | 0 | 0 |
| ipv6 | 1 | 1 | 0 | 0 | 0 |
| last_name | 13 | 13 | 0 | 0 | 0 |
| local_latlng | 83 | 83 | 0 | 83 | 0 |
| name | 122 | 122 | 0 | 0 | 0 |
| passport_number | 1 | 1 | 0 | 0 | 0 |
| password | 84 | 84 | 0 | 84 | 0 |
| phone_number | 5 | 5 | 0 | 0 | 0 |
| street_address | 19 | 19 | 0 | 0 | 0 |
| time | 1578 | 1 | 1577 | 0 | 0 |
| user_name | 3 | 3 | 0 | 0 | 0 |

## Excluded Span Dispositions

| Source | Source Label | Category | Reason | Training Exclusion | Count |
|---|---|---|---|---:|---:|
| gretel_finance_pii | date | benign_unmapped | not_sensitive_or_not_semantically_safe_for_ml_v1 | false | 9303 |
| gretel_finance_pii | company | benign_unmapped | not_sensitive_or_not_semantically_safe_for_ml_v1 | false | 6022 |
| gretel_finance_pii | time | benign_unmapped | not_sensitive_or_not_semantically_safe_for_ml_v1 | false | 1577 |
| gretel_finance_pii | account_pin | deferred_ml_v1 | reserved_product_taxonomy_label | true | 162 |
| gretel_finance_pii | credit_card_security_code | deferred_ml_v1 | reserved_product_taxonomy_label | true | 120 |
| gretel_finance_pii | name | overlap_conflict | deterministic_overlap_resolution | true | 95 |
| gretel_finance_pii | date_time | benign_unmapped | not_sensitive_or_not_semantically_safe_for_ml_v1 | false | 88 |
| gretel_finance_pii | password | deferred_ml_v1 | reserved_product_taxonomy_label | true | 84 |
| gretel_finance_pii | local_latlng | deferred_ml_v1 | reserved_product_taxonomy_label | true | 83 |
| gretel_finance_pii | name | invalid_source_span | offsets_outside_source_text | true | 27 |
| gretel_finance_pii | street_address | invalid_source_span | offsets_outside_source_text | true | 13 |
| gretel_finance_pii | last_name | overlap_conflict | deterministic_overlap_resolution | true | 10 |
| gretel_finance_pii | email | overlap_conflict | deterministic_overlap_resolution | true | 9 |
| gretel_finance_pii | company | invalid_source_span | offsets_outside_source_text | true | 8 |
| gretel_finance_pii | street_address | overlap_conflict | deterministic_overlap_resolution | true | 6 |
| gretel_finance_pii | driver_license_number | invalid_source_span | offsets_outside_source_text | true | 4 |
| gretel_finance_pii | last_name | invalid_source_span | offsets_outside_source_text | true | 3 |
| gretel_finance_pii | phone_number | invalid_source_span | offsets_outside_source_text | true | 3 |
| gretel_finance_pii | user_name | overlap_conflict | deterministic_overlap_resolution | true | 3 |
| gretel_finance_pii | date | invalid_source_span | offsets_outside_source_text | true | 2 |
| gretel_finance_pii | date_of_birth | invalid_source_span | offsets_outside_source_text | true | 2 |
| gretel_finance_pii | phone_number | overlap_conflict | deterministic_overlap_resolution | true | 2 |
| gretel_finance_pii | credit_card_number | overlap_conflict | deterministic_overlap_resolution | true | 1 |
| gretel_finance_pii | email | invalid_source_span | offsets_outside_source_text | true | 1 |
| gretel_finance_pii | ipv6 | invalid_source_span | offsets_outside_source_text | true | 1 |
| gretel_finance_pii | passport_number | overlap_conflict | deterministic_overlap_resolution | true | 1 |
| gretel_finance_pii | time | invalid_source_span | offsets_outside_source_text | true | 1 |
