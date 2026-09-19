# ML-v1.2 Diagnosis of ML-v1.1 Development Errors

This diagnosis uses only frozen ML-v1.1 development artifacts. No model was
loaded, no inference was executed, and the sealed challenge was neither parsed
nor predicted. All numbers below were recomputed from the cached epoch-3
prediction files by `scripts/data/analyze_securelogx_ml_v1_1_dev_errors.py`;
the machine-readable evidence is `reports/ml_v1_2_v1_1_dev_error_analysis.json`.

## Evidence provenance

| Split | Records | Records SHA-256 | Cached predictions SHA-256 |
|---|---:|---|---|
| Standard dev (`data/split/dev.jsonl`) | 2,994 | `126d65a63e0b37bff8462b2f0f56e8cdceab2ec54ad97b868fb1981a4bc8053a` | `e5f3a7a912effd6b4ee60bcac71504d3cdc197287a636b6fb13af5cfde4d5dee` |
| Challenge dev (`data/ml_v1_1/context_contrast/dev_challenge.jsonl`) | 480 | `215c2133c05a3199b5c54b145406f4ed1571214b30eb6c37d00bb7783c96a09e` | `737c079258bb33633588db3216d9730b099050acda1fb5160754380ced683eac` |

Both caches are pinned to the frozen selected epoch 3 and to the exact input
file hashes recorded above.

## Frozen gate outcome being diagnosed

From `output_securelogx/ml-v1.1/bert-base-cased/dev_gate_decision.json`
(decision: `NOT READY FOR SEALED CHALLENGE EVALUATION`), the failed gates were:

| Gate | Actual | Threshold | Result |
|---|---:|---:|---|
| `dev_challenge_micro_f1_minimum` | 0.892544 | >= 0.900000 | FAIL |
| `dev_challenge_context_target_error_rate_maximum` | 0.112500 | <= 0.100000 | FAIL |
| `dev_challenge_maximum_category_error_rate` | 0.262500 | <= 0.200000 | FAIL |

## Actual worst category behind the 0.2625 error rate

The category responsible for `dev_challenge_maximum_category_error_rate =
0.2625` is **`ip_address_vs_technical_reference`**:

- records in category: 80
- records with at least one error: 21 (record error rate 0.2625)
- morphology-over-context errors: 21 of 21 target errors (share 1.0)
- dominant confusion direction: **technical-reference values that are gold `O`
  were predicted as `IP_ADDRESS`** (27 of the 28 spurious `IP_ADDRESS`
  predictions on the challenge dev sit in `technical_reference` role records;
  mean confidence 0.716)

This is a pure morphology-over-context failure: valid dotted-quad shapes used
as non-network technical coordinates (versions, coordinates, revisions) are
promoted to `IP_ADDRESS` on shape alone. It is the same class of failure that
sank ML-v1 (`orderId=123-45-6789` → SSN), transposed onto IPv4 morphology.

## Principal confusion directions (epoch 3, both dev splits)

Challenge dev (480 records):

| Direction | Count | Mean confidence | Typical context |
|---|---:|---:|---|
| `CREDIT_CARD_NUMBER` → `BUSINESS_ID` | 20 | 0.9961 | `business_id_vs_credit_card` / `payment_card_number` (20) |
| spurious → `IP_ADDRESS` | 28 | 0.7159 | `ip_address_vs_technical_reference` / `technical_reference` (27) |
| `BUSINESS_ID` → `SSN` | 10 | 0.7965 | `business_id_vs_ssn` / `business_identifier` (10) |
| spurious → `BUSINESS_ID` | 4 | 0.5217 | mixed |
| `BUSINESS_ID` → `IP_ADDRESS` | 2 | 0.6924 | `business_id_vs_ssn` / `business_identifier` (2) |

Standard dev (2,994 records), largest directions:

| Direction | Count | Mean confidence | Typical context |
|---|---:|---:|---|
| spurious → `BUSINESS_ID` | 99 | 0.9790 | `key_value_accountlike_batch_v1` (94); `gretel_finance_pii` (5) |
| `ROUTING_NUMBER` → `BANK_ACCOUNT_NUMBER` | 94 | 0.9952 | `routing_number_free_b_v1` (94) |
| spurious → `STREET_ADDRESS` | 58 | 0.7348 | `gretel_finance_pii` (58) |
| spurious → `PERSON_NAME` | 51 | 0.7582 | `gretel_finance_pii` (51) |
| `SSN` → `ITIN` | 5 | 0.6463 | `ssn_free_b_v1` (5) |

## Interpretation for ML-v1.2 targeting

1. **BUSINESS_ID overprediction is real and has two distinct faces.** On the
   standard dev it is almost entirely *spurious* spans: account-like `O`
   values in parent `key_value_accountlike_batch_v1` records are asserted as
   `BUSINESS_ID` with near-1.0 confidence (94 of 100 wrong-class-or-spurious
   cases). On the challenge dev it is *wrong-class* promotion: genuine
   `CREDIT_CARD_NUMBER` values adjacent to business context are absorbed into
   `BUSINESS_ID` (20 cases at 0.996 mean confidence). See
   `reports/ml_v1_2_business_id_false_positive_analysis.md`.
2. **SSN recall regression (0.9903 → 0.9223 on standard dev) is not caused by
   SSN → BUSINESS_ID.** That direction has zero standard-dev instances at
   epoch 3. The visible directional loss is `SSN` → `ITIN` (5 cases in
   `ssn_free_b_v1`); the remainder are outright misses. ML-v1.2 must therefore
   strengthen *positive* SSN evidence in varied contexts rather than only
   suppressing a confusion pair.
3. **The worst category needs O-preserving counter-evidence.** The model needs
   families where valid dotted quads are `O` because context says
   "version/coordinate/revision", alongside families where they are
   `IP_ADDRESS` because context says "endpoint/client/peer".
4. `ROUTING_NUMBER` → `BANK_ACCOUNT_NUMBER` (94, parent family) is the largest
   non-BUSINESS_ID confusion on the standard dev and motivates the dedicated
   routing-number contrast families in the v1.2 catalog.

No checkpoint selection is revisited here; epoch 3 remains the frozen
selection. See `reports/ml_v1_2_checkpoint_tradeoff_analysis.md` for the
epoch-wise analysis.
