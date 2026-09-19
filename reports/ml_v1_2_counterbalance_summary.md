# ML-v1.2 Counterbalance Summary

ML-v1.2 is a targeted counterbalance revision. Its objective is to reduce BUSINESS_ID overprediction while preserving the contextual BUSINESS_ID gains achieved in ML-v1.1. Every family is bidirectional: the same deterministic surface value appears once as a genuine BUSINESS_ID and once as the canonical sensitive entity (or, for the dedicated IP families, once as IP_ADDRESS and once as a non-network technical coordinate that stays `O`). It was produced without model training or inference.

## New artifacts

| Partition | Records | Entity spans | Contrast groups | Families | Multi-entity ratio |
|---|---:|---:|---:|---:|---:|
| Training addition | 1600 | 3080 | 800 | 40 | 0.925 |
| Dev challenge (v1.2) | 540 | 1026 | 270 | 15 | 0.900 |

## Bidirectional contrast coverage

| Category | Train records/families | Dev records/families |
|---|---:|---:|
| `business_id_vs_api_key` | 80/2 | 36/1 |
| `business_id_vs_auth_token` | 80/2 | 36/1 |
| `business_id_vs_bank_account` | 120/3 | 36/1 |
| `business_id_vs_credit_card` | 160/4 | 36/1 |
| `business_id_vs_driver_license` | 80/2 | 36/1 |
| `business_id_vs_ip_address` | 80/2 | 36/1 |
| `business_id_vs_itin` | 80/2 | 36/1 |
| `business_id_vs_passport` | 80/2 | 36/1 |
| `business_id_vs_routing_number` | 120/3 | 36/1 |
| `business_id_vs_ssn` | 400/10 | 72/2 |
| `business_id_vs_tax_id` | 80/2 | 36/1 |
| `ip_address_vs_technical_reference` | 240/6 | 108/3 |

## Sensitive-entity counterbalance spans (training addition)

| Label | Train spans |
|---|---:|
| BUSINESS_ID | 1560 |
| SSN | 400 |
| CREDIT_CARD_NUMBER | 160 |
| BANK_ACCOUNT_NUMBER | 120 |
| ROUTING_NUMBER | 120 |
| ITIN | 80 |
| TAX_ID | 80 |
| PASSPORT_NUMBER | 80 |
| DRIVER_LICENSE | 80 |
| API_KEY | 80 |
| AUTH_TOKEN | 80 |
| IP_ADDRESS | 200 |
| PERSON_NAME (companions) | 40 |

## True-SSN protection

The training addition contains **400** true SSN spans across hyphenated, compact, spaced, and dotted morphologies in JSON, nested JSON, key=value, query, CSV, bracketed, syslog, and audit message surfaces, with camelCase, snake_case, and abbreviated keys. Every SSN family is multi-entity: the SSN coexists with a genuine BUSINESS_ID or a PERSON_NAME companion, so the model must keep both readings apart inside one record.

## Weak-subtype strengthening (BUSINESS_ID provenance only)

| BUSINESS_ID subtype | Train spans | Dev spans |
|---|---:|---:|
| CUSTOMER_ID | 220 | 72 |
| ACCOUNT_ID | 120 | 36 |
| APPLICATION_ID | 200 | 36 |
| TRANSACTION_ID | 120 | 36 |
| USER_ID | 60 | 36 |
| ORDER_ID | 80 | 36 |
| CASE_ID | 120 | 36 |
| TICKET_ID | 80 | 0 |
| INVOICE_ID | 80 | 36 |
| REFERENCE_ID | 80 | 72 |
| REQUEST_ID | 240 | 90 |
| WORKFLOW_ID | 160 | 36 |

CUSTOMER_ID (50% challenge recall in ML-v1.1) and REQUEST_ID (85%) receive the largest dedicated allocations; all twelve subtypes remain represented so successful subtype diversity is preserved. Subtypes stay metadata only.

## Why the `ip_address_vs_technical_reference` families were added

The ML-v1.1 worst development category (record error rate 0.2625, gate limit 0.20) was `ip_address_vs_technical_reference`: valid dotted-quad values in non-network technical contexts were promoted to `IP_ADDRESS` on shape alone (21 of 21 category errors were morphology-over-context; see `reports/ml_v1_2_v1_1_error_diagnosis.md`). ML-v1.2 therefore adds 6 dedicated training families (and 3 dev families) in which the identical dotted-quad surface is `IP_ADDRESS` when context names a network endpoint and `O` when context names a release version, package coordinate, schema revision, build tuple, firmware revision, or dependency coordinate. No sensitive entity is ever converted to `O`: the O-role records keep their adjacent BUSINESS_ID annotated.

## Relationship to the frozen parents

Future training composition is the frozen ML-v1 train split plus the ML-v1.1 training addition plus this counterbalance addition. Future development scores the frozen ML-v1 dev split plus this new v1.2 dev challenge; the ML-v1.1 dev challenge becomes a historical diagnostic. The original ML-v1 test remains a regression benchmark, and the sealed final challenge remains sealed, unevaluated, and byte-identical.
