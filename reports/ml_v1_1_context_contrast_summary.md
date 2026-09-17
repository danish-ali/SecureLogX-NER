# ML-v1.1 Context-Contrast Summary

This additive dataset revision targets semantic context versus surface morphology. It does not replace, rewrite, or resplit ML-v1, and it was produced without model training or inference.

## New artifacts

| Partition | New records | Entity spans | Contrast groups | Families | Negative records |
|---|---:|---:|---:|---:|---:|
| Training addition | 3000 | 2700 | 1500 | 30 | 400 |
| Dev challenge | 480 | 440 | 240 | 12 | 40 |
| Sealed final challenge | 480 | 420 | 240 | 12 | 60 |

## Morphology-conflict coverage

| Category | Train records/families | Dev records/families | Sealed records/families |
|---|---:|---:|---:|
| `business_id_vs_bank_account` | 400/4 | 80/2 | 80/2 |
| `business_id_vs_credit_card` | 600/6 | 80/2 | 80/2 |
| `business_id_vs_ssn` | 1200/12 | 240/6 | 200/5 |
| `ip_address_vs_technical_reference` | 400/4 | 80/2 | 120/3 |
| `person_name_vs_service_name` | 200/2 | 0/0 | 0/0 |
| `street_address_vs_system_location` | 200/2 | 0/0 | 0/0 |

## Context and style design

Each contrast group has two records that reuse the same synthetic value morphology under different semantic contexts. BUSINESS_ID subtype is retained in provenance while the neural label remains only `BUSINESS_ID`. O-role technical values are recorded as audit targets but are not invented as entity spans.

Training uses **10** surface styles: `abbreviated_key`, `error_message`, `json_flat`, `json_nested`, `key_colon`, `key_equals`, `mixed_punctuation`, `natural_language`, `structured_audit`, `warn_message`.

True SSN contexts include explicit fields, JSON/nested JSON, natural-language references, WARN/ERROR messages, compact and standard formatting, masked-last-four examples, punctuation variation, and PERSON_NAME companions in selected families. Card numbers are Luhn-valid; IP contrasts reuse valid dotted quads in explicitly non-network technical contexts.

No new external dataset was downloaded. This controlled synthetic revision increases contextual and stylistic coverage but is not claimed to equal real-world data.

## Relationship to ML-v1

Future training composition is the frozen ML-v1 train split plus the training addition. Future development may score the frozen ML-v1 dev split and the independent dev challenge. The original ML-v1 test remains a historical regression benchmark; the new sealed challenge is reserved for a single post-selection evaluation in a later phase.
