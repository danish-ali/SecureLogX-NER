# SecureLogX ML-v1 Dataset Split Summary

## Record and span counts

| Split | Records | Record % | Entity spans |
|---|---:|---:|---:|
| train | 23558 | 79.75% | 32203 |
| dev | 2994 | 10.14% | 4020 |
| test | 2989 | 10.12% | 4358 |
| **Total** | **29541** | **100.00%** | **40581** |

## Per-label span distribution

| Entity | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| PERSON_NAME | 5964 | 759 | 782 | 7505 |
| DOB | 939 | 122 | 116 | 1177 |
| AGE | 752 | 94 | 94 | 940 |
| SSN | 843 | 103 | 105 | 1051 |
| ITIN | 752 | 94 | 94 | 940 |
| TAX_ID | 752 | 94 | 94 | 940 |
| EMAIL | 3320 | 493 | 752 | 4565 |
| PHONE | 1607 | 173 | 165 | 1945 |
| STREET_ADDRESS | 3354 | 417 | 412 | 4183 |
| CITY | 752 | 94 | 94 | 940 |
| STATE_PROVINCE | 752 | 94 | 94 | 940 |
| POSTAL_CODE | 752 | 94 | 94 | 940 |
| COUNTRY | 752 | 94 | 94 | 940 |
| CREDIT_CARD_NUMBER | 949 | 112 | 101 | 1162 |
| BANK_ACCOUNT_NUMBER | 855 | 105 | 107 | 1067 |
| ROUTING_NUMBER | 843 | 103 | 107 | 1053 |
| IBAN | 857 | 100 | 102 | 1059 |
| SWIFT_BIC | 884 | 105 | 110 | 1099 |
| BUSINESS_ID | 1360 | 135 | 231 | 1726 |
| PASSPORT_NUMBER | 867 | 105 | 96 | 1068 |
| DRIVER_LICENSE | 849 | 106 | 102 | 1057 |
| IP_ADDRESS | 1030 | 129 | 119 | 1278 |
| DEVICE_ID | 752 | 94 | 94 | 940 |
| AUTH_TOKEN | 846 | 94 | 94 | 1034 |
| API_KEY | 820 | 107 | 105 | 1032 |

## Source distribution

| Source | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| gretel_finance_pii | 3632 | 457 | 452 | 4541 |
| securelogx_custom_logs | 19926 | 2537 | 2537 | 25000 |

## Synthetic/external ratio

- Synthetic records: **25000** (84.63%)
- External records: **4541** (15.37%)

## Independence gates

- Exact duplicate text leakage: **0**
- Synthetic template-family leakage: **0**
- Synthetic records missing template family: **0**

## Unsupported/deferred metadata retained in final data

- Spans marked as training-excluding: **0** (must be 0 in final splits)
- company: 5558
- date: 8523
- date_time: 82
- time: 1446
- Categories: benign_unmapped=15609

## External-source limitation

External datasets do not provide SecureLogX template-family metadata, so they are split by stable source record ID while retaining exact-text deduplication.
Records requiring a text-hash fallback because source_record_id was absent: **0**.
