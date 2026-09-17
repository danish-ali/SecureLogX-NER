# Sequence-Length Analysis

Tokenizer: **bert-base-cased fast WordPiece tokenizer**. Counts include special tokens.

| Split | Median | p90 | p95 | p99 | Maximum | >384 | >512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 65 | 378 | 470 | 572.4 | 8178 | 2278 | 571 |
| dev | 64 | 375.7 | 465 | 552.1 | 7742 | 288 | 63 |
| test | 78 | 378.2 | 465 | 596.0 | 1562 | 293 | 64 |
| all | 66 | 378 | 469 | 572.6 | 8178 | 2859 | 698 |

## Naive truncation risk

A single 512-token window would fail to cover these gold spans:

- train: **342 spans** (API_KEY=2, BANK_ACCOUNT_NUMBER=3, BUSINESS_ID=5, CREDIT_CARD_NUMBER=3, DRIVER_LICENSE=1, EMAIL=23, IP_ADDRESS=7, PASSPORT_NUMBER=3, PERSON_NAME=255, PHONE=4, ROUTING_NUMBER=2, SSN=1, STREET_ADDRESS=28, SWIFT_BIC=5)
- dev: **27 spans** (BUSINESS_ID=1, EMAIL=2, PERSON_NAME=22, STREET_ADDRESS=2)
- test: **40 spans** (BUSINESS_ID=2, EMAIL=2, IP_ADDRESS=3, PERSON_NAME=30, STREET_ADDRESS=3)

## Selected policy

- Maximum window length: **384 tokens**
- Overflow stride: **128 tokens**
- Padding: dynamic to the longest sequence in each batch, rounded to a multiple of eight

The selected window is near the observed p90 and avoids wasteful fixed 512-token padding. Overlapping windows preserve long-document context and provide a complete window for ordinary entity spans instead of silently right-truncating them. Predictions from overlapping windows are averaged per unique character-offset token before BIO decoding.
