# Model Error Analysis

- Exact span matches: **3994**
- Total false positives: **363**
- Total false negatives: **364**
- Boundary-too-short pairs: **64**
- Boundary-too-long pairs: **25**
- Boundary-shifted pairs: **1**
- Wrong-class pairs: **116**

## Error categories

- boundary-shifted: 1
- boundary-too-long: 25
- boundary-too-short: 64
- false negative: 158
- false positive: 157
- wrong entity class: 116

## Most affected labels

- PERSON_NAME: 204 classified error rows
- BUSINESS_ID: 116 classified error rows
- STREET_ADDRESS: 111 classified error rows
- API_KEY: 14 classified error rows
- PHONE: 14 classified error rows
- DOB: 10 classified error rows
- EMAIL: 9 classified error rows
- BANK_ACCOUNT_NUMBER: 7 classified error rows
- SWIFT_BIC: 7 classified error rows
- IP_ADDRESS: 7 classified error rows
- CREDIT_CARD_NUMBER: 6 classified error rows
- ROUTING_NUMBER: 5 classified error rows
- SSN: 5 classified error rows
- IBAN: 3 classified error rows
- PASSPORT_NUMBER: 2 classified error rows

The full privacy-safe error inventory is in `reports/model_errors.csv`; the entity confusion matrix includes `<MISSED>`, `<SPURIOUS>`, and `<BOUNDARY>` buckets.
