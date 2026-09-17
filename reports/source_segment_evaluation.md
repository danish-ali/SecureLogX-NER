# Source-Segment Test Evaluation

All segment metrics were derived from the same single held-out test inference pass.

| Segment | Records | Gold entities | Micro P | Micro R | Micro F1 | Supported-label macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 2989 | 4358 | 91.67% | 91.65% | 91.66% | 94.24% |
| Gretel-only | 452 | 1350 | 80.12% | 80.00% | 80.06% | 71.23% |
| Synthetic-only | 2537 | 3008 | 96.84% | 96.88% | 96.86% | 97.30% |

Synthetic test records are unseen-template examples: the frozen split audit proves each synthetic template family belongs to exactly one split.

Gretel has no template-family metadata, so the Gretel segment is protected by stable source-record grouping and exact-text independence rather than a template-family claim.

## Gretel-only per-label metrics (supported labels)

| Entity | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| PERSON_NAME | 85.85% | 76.74% | 81.04% | 688 |
| DOB | 71.43% | 90.91% | 80.00% | 22 |
| SSN | 76.92% | 90.91% | 83.33% | 11 |
| EMAIL | 91.67% | 93.62% | 92.63% | 94 |
| PHONE | 88.89% | 90.14% | 89.51% | 71 |
| STREET_ADDRESS | 74.86% | 82.39% | 78.44% | 318 |
| CREDIT_CARD_NUMBER | 33.33% | 71.43% | 45.45% | 7 |
| BANK_ACCOUNT_NUMBER | 80.00% | 61.54% | 69.57% | 13 |
| ROUTING_NUMBER | 81.82% | 69.23% | 75.00% | 13 |
| IBAN | 63.64% | 87.50% | 73.68% | 8 |
| SWIFT_BIC | 63.16% | 75.00% | 68.57% | 16 |
| BUSINESS_ID | 67.35% | 76.74% | 71.74% | 43 |
| PASSPORT_NUMBER | 20.00% | 50.00% | 28.57% | 2 |
| DRIVER_LICENSE | 77.78% | 87.50% | 82.35% | 8 |
| IP_ADDRESS | 76.92% | 80.00% | 78.43% | 25 |
| API_KEY | 33.33% | 54.55% | 41.38% | 11 |
