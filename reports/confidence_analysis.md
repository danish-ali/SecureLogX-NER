# Model Confidence Analysis

Span confidence is the mean maximum softmax probability across predicted tokens. For false negatives, the reported value is the mean probability assigned to the gold BIO labels over the gold-token region.

| Group | Count | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| boundary_shifted | 1 | 0.849426 | 0.849426 | 0.849426 | 0.849426 |
| boundary_too_long | 25 | 0.878941 | 0.880078 | 0.804066 | 0.977561 |
| boundary_too_short | 64 | 0.755425 | 0.769111 | 0.609423 | 0.912150 |
| correct | 3994 | 0.993822 | 0.998489 | 0.997146 | 0.999076 |
| false_negative_gold_probability | 158 | 0.066556 | 0.012697 | 0.001047 | 0.078205 |
| false_positive | 157 | 0.691370 | 0.656244 | 0.555834 | 0.825357 |
| wrong_class | 116 | 0.940076 | 0.995347 | 0.980335 | 0.997040 |

## Error-versus-correct diagnostic

The diagnostic low-confidence cutoff is the correct-span p25 (0.997146); it is not a runtime threshold.

| Error group | Count | Median confidence/probability | At or below cutoff | Compared with correct spans |
|---|---:|---:|---:|---|
| boundary_error | 90 | 0.815405 | 96.67% | higher low-confidence share |
| false positive | 157 | 0.656244 | 98.73% | higher low-confidence share |
| false_negative_gold_probability | 158 | 0.012697 | 99.37% | higher low-confidence share |
| wrong entity class | 116 | 0.995347 | 75.86% | higher low-confidence share |

## Per-label confidence and support

Rare-class cohort: lowest supported-label quartile by gold support (support <= 94): `AGE`, `AUTH_TOKEN`, `CITY`, `COUNTRY`, `DEVICE_ID`, `ITIN`, `POSTAL_CODE`, `STATE_PROVINCE`, `TAX_ID`.

| Entity | Cohort | Support | F1 | Correct predictions | Correct median | Incorrect predictions | Incorrect median | FN probabilities | FN median gold probability |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PERSON_NAME | other | 782 | 83.43% | 622 | 0.997814 | 87 | 0.711855 | 114 | 0.007054 |
| DOB | other | 116 | 95.00% | 114 | 0.997817 | 10 | 0.641781 | 2 | 0.037397 |
| AGE | rare | 94 | 100.00% | 94 | 0.996155 | 0 | n/a | 0 | n/a |
| SSN | other | 105 | 68.20% | 104 | 0.998452 | 96 | 0.996022 | 0 | n/a |
| ITIN | rare | 94 | 100.00% | 94 | 0.998642 | 0 | n/a | 0 | n/a |
| TAX_ID | rare | 94 | 100.00% | 94 | 0.998561 | 0 | n/a | 0 | n/a |
| EMAIL | other | 752 | 99.07% | 746 | 0.999401 | 8 | 0.881521 | 3 | 0.106860 |
| PHONE | other | 165 | 95.47% | 158 | 0.998382 | 8 | 0.726499 | 6 | 0.153322 |
| STREET_ADDRESS | other | 412 | 83.18% | 356 | 0.997721 | 88 | 0.762874 | 24 | 0.041454 |
| CITY | rare | 94 | 99.47% | 94 | 0.997240 | 1 | 0.494879 | 0 | n/a |
| STATE_PROVINCE | rare | 94 | 100.00% | 94 | 0.995398 | 0 | n/a | 0 | n/a |
| POSTAL_CODE | rare | 94 | 100.00% | 94 | 0.996787 | 0 | n/a | 0 | n/a |
| COUNTRY | rare | 94 | 100.00% | 94 | 0.996817 | 0 | n/a | 0 | n/a |
| CREDIT_CARD_NUMBER | other | 101 | 94.29% | 99 | 0.998986 | 10 | 0.945910 | 0 | n/a |
| BANK_ACCOUNT_NUMBER | other | 107 | 96.68% | 102 | 0.998454 | 2 | 0.502722 | 1 | 0.159457 |
| ROUTING_NUMBER | other | 107 | 97.17% | 103 | 0.998535 | 2 | 0.556441 | 4 | 0.008313 |
| IBAN | other | 102 | 97.58% | 101 | 0.998994 | 4 | 0.774277 | 0 | n/a |
| SWIFT_BIC | other | 110 | 95.07% | 106 | 0.998500 | 7 | 0.697173 | 0 | n/a |
| BUSINESS_ID | other | 231 | 67.91% | 127 | 0.999017 | 16 | 0.494221 | 1 | 0.079655 |
| PASSPORT_NUMBER | other | 96 | 97.44% | 95 | 0.998362 | 4 | 0.510148 | 0 | n/a |
| DRIVER_LICENSE | other | 102 | 98.54% | 101 | 0.998532 | 2 | 0.632440 | 0 | n/a |
| IP_ADDRESS | other | 119 | 95.40% | 114 | 0.999127 | 6 | 0.852773 | 1 | 0.000410 |
| DEVICE_ID | rare | 94 | 100.00% | 94 | 0.999348 | 0 | n/a | 0 | n/a |
| AUTH_TOKEN | rare | 94 | 100.00% | 94 | 0.999605 | 0 | n/a | 0 | n/a |
| API_KEY | other | 105 | 92.17% | 100 | 0.999456 | 12 | 0.645033 | 2 | 0.188158 |

## Rare-class comparison

- Rare prediction low-confidence share: **40.50%**
- Other-label prediction low-confidence share: **28.09%**
- Rare incorrect-prediction share: **0.12%**
- Other-label incorrect-prediction share: **10.31%**

False-negative values are gold-label probabilities, while other error values are predicted-span confidences; compare them cautiously. No runtime confidence threshold is selected in this phase.
