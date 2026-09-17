# SecureLogX ML-v1 Label Readiness

Coverage gates: train >= 300, dev >= 40, test >= 40 spans.

| Entity | Train spans | Dev spans | Test spans | Template diversity | Readiness | Notes |
|---|---:|---:|---:|---|---|---|
| PERSON_NAME | 5964 | 759 | 782 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| DOB | 939 | 122 | 116 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| AGE | 752 | 94 | 94 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| SSN | 843 | 103 | 105 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| ITIN | 752 | 94 | 94 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| TAX_ID | 752 | 94 | 94 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| EMAIL | 3320 | 493 | 752 | 35 total (24/4/7) | PASS | Meets provisional span gates |
| PHONE | 1607 | 173 | 165 | 12 total (10/1/1) | PASS | Meets provisional span gates |
| STREET_ADDRESS | 3354 | 417 | 412 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| CITY | 752 | 94 | 94 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| STATE_PROVINCE | 752 | 94 | 94 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| POSTAL_CODE | 752 | 94 | 94 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| COUNTRY | 752 | 94 | 94 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| CREDIT_CARD_NUMBER | 949 | 112 | 101 | 11 total (9/1/1) | PASS | Meets provisional span gates |
| BANK_ACCOUNT_NUMBER | 855 | 105 | 107 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| ROUTING_NUMBER | 843 | 103 | 107 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| IBAN | 857 | 100 | 102 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| SWIFT_BIC | 884 | 105 | 110 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| BUSINESS_ID | 1360 | 135 | 231 | 14 total (11/1/2) | PASS | Meets provisional span gates |
| PASSPORT_NUMBER | 867 | 105 | 96 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| DRIVER_LICENSE | 849 | 106 | 102 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| IP_ADDRESS | 1030 | 129 | 119 | 11 total (9/1/1) | PASS | Meets provisional span gates |
| DEVICE_ID | 752 | 94 | 94 | 10 total (8/1/1) | PASS | Meets provisional span gates |
| AUTH_TOKEN | 846 | 94 | 94 | 11 total (9/1/1) | PASS | Meets provisional span gates |
| API_KEY | 820 | 107 | 105 | 10 total (8/1/1) | PASS | Meets provisional span gates |

## Recommendation

**READY FOR TRAINING**

This is a data-coverage and split-independence recommendation only. It is not model evaluation or production-readiness evidence.
