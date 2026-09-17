# Dataset Quality Report

## Summary

- **Total Records**: 5000
- **Valid Records**: 5000
- **Invalid Records**: 0
- **Validity Rate**: 100.0%
- **Negative Records (zero entities)**: 1236
- **Total Canonical Entities**: 15888
- **Average Text Length**: 1401.0 characters
- **Exact Duplicate Records**: 2
- **Distinct Duplicated Text Values**: 1
- **Duplicate Entity Spans**: 0
- **Training-Entity Overlap Conflicts**: 0

## Excluded and Deferred Span Metadata

- **Records With Excluded Spans**: 4005
- **All Excluded/Trace Spans**: 17631
- **Training-Excluding Spans**: 641
- **Benign Trace-Only Spans**: 16990
- **Records Excluded From Training**: 459
- **Records With Deferred ML-v1 Spans**: 358
- **Deferred ML-v1 Spans**: 449
- **Records With Healthcare-Extension Spans**: 0
- **Healthcare-Extension Spans**: 0

## Canonical Entity Distribution

| Entity | Count | Percentage |
|---|---:|---:|
| PERSON_NAME | 7609 | 47.89% |
| DOB | 254 | 1.60% |
| AGE | 0 | 0.00% |
| SSN | 118 | 0.74% |
| ITIN | 0 | 0.00% |
| TAX_ID | 0 | 0.00% |
| EMAIL | 1464 | 9.21% |
| PHONE | 914 | 5.75% |
| STREET_ADDRESS | 3711 | 23.36% |
| CITY | 0 | 0.00% |
| STATE_PROVINCE | 0 | 0.00% |
| POSTAL_CODE | 0 | 0.00% |
| COUNTRY | 0 | 0.00% |
| CREDIT_CARD_NUMBER | 142 | 0.89% |
| BANK_ACCOUNT_NUMBER | 142 | 0.89% |
| ROUTING_NUMBER | 126 | 0.79% |
| IBAN | 136 | 0.86% |
| SWIFT_BIC | 172 | 1.08% |
| BUSINESS_ID | 453 | 2.85% |
| PASSPORT_NUMBER | 146 | 0.92% |
| DRIVER_LICENSE | 131 | 0.82% |
| IP_ADDRESS | 272 | 1.71% |
| DEVICE_ID | 0 | 0.00% |
| AUTH_TOKEN | 0 | 0.00% |
| API_KEY | 98 | 0.62% |

## Excluded/Trace Source Labels

| Source Label | All Spans | Training-Excluding | Deferred ML-v1 | Healthcare Extension |
|---|---:|---:|---:|---:|
| account_pin | 162 | 162 | 162 | 0 |
| company | 6030 | 8 | 0 | 0 |
| credit_card_number | 1 | 1 | 0 | 0 |
| credit_card_security_code | 120 | 120 | 120 | 0 |
| date | 9305 | 2 | 0 | 0 |
| date_of_birth | 2 | 2 | 0 | 0 |
| date_time | 88 | 0 | 0 | 0 |
| driver_license_number | 4 | 4 | 0 | 0 |
| email | 10 | 10 | 0 | 0 |
| ipv6 | 1 | 1 | 0 | 0 |
| last_name | 13 | 13 | 0 | 0 |
| local_latlng | 83 | 83 | 83 | 0 |
| name | 122 | 122 | 0 | 0 |
| passport_number | 1 | 1 | 0 | 0 |
| password | 84 | 84 | 84 | 0 |
| phone_number | 5 | 5 | 0 | 0 |
| street_address | 19 | 19 | 0 | 0 |
| time | 1578 | 1 | 0 | 0 |
| user_name | 3 | 3 | 0 | 0 |

## Warnings (2)

- Line 1690: Exact duplicate text; first seen on line 604
- Line 2572: Exact duplicate text; first seen on line 604
