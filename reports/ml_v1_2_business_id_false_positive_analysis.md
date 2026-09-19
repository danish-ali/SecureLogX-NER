# ML-v1.2 BUSINESS_ID False-Positive Analysis (ML-v1.1 epoch 3)

Source of truth: cached frozen epoch-3 development predictions, recomputed by
`scripts/data/analyze_securelogx_ml_v1_1_dev_errors.py`
(evidence: `reports/ml_v1_2_v1_1_dev_error_analysis.json`). Matching uses
exact span signatures first, then best-IoU overlap pairing; "spurious" means a
BUSINESS_ID prediction that overlaps no gold entity at all (gold `O` text).

## Accounting

| Split | BUSINESS_ID predictions | True positives | False positives | Wrong-class or spurious | Boundary-only errors |
|---|---:|---:|---:|---:|---:|
| Standard dev | 230 | 129 | 101 | 100 | 1 |
| Challenge dev | 212 | 187 | 25 | 24 | 1 |

Standard-dev BUSINESS_ID precision 129/230 = 0.560870 — this table is the
complete causal account of the 0.5609 precision collapse.

## Gold label → BUSINESS_ID error table

### Standard dev

| Gold label | Count | % of wrong-class/spurious | Typical context | Confidence (median / min / max) |
|---|---:|---:|---|---|
| `O` (spurious) | 99 | 99.0% | `key_value_accountlike_batch_v1` (94); `gretel_finance_pii` (5) | 0.9993 / 0.4799 / 0.9994 |
| `PERSON_NAME` | 1 | 1.0% | `gretel_finance_pii` (1) | 0.6960 |

### Challenge dev

| Gold label | Count | % of wrong-class/spurious | Typical context | Confidence (median / min / max) |
|---|---:|---:|---|---|
| `CREDIT_CARD_NUMBER` | 20 | 83.3% | `business_id_vs_credit_card` / `payment_card_number` role (20) | 0.9976 / 0.9849 / 0.9987 |
| `O` (spurious) | 4 | 16.7% | `business_id_vs_ssn` (2); `ip_address_vs_technical_reference` (2) | 0.4901 / 0.4299 / 0.6767 |

## Explicitly requested directions (epoch 3 counts)

| Direction | Standard dev | Challenge dev |
|---|---:|---:|
| SSN → BUSINESS_ID | 0 | 0 |
| CREDIT_CARD_NUMBER → BUSINESS_ID | 0 | **20** |
| BANK_ACCOUNT_NUMBER → BUSINESS_ID | 0 | 0 |
| ROUTING_NUMBER → BUSINESS_ID | 0 | 0 |
| ITIN / TAX_ID → BUSINESS_ID | 0 | 0 |
| API_KEY / AUTH_TOKEN → BUSINESS_ID | 0 | 0 |
| PERSON_NAME → BUSINESS_ID | 1 | 0 |
| `O` (spurious) → BUSINESS_ID | 99 | 4 |

## Diagnosis

1. **The dominant standard-dev failure is spurious assertion, not label
   swapping.** 94 of the 100 non-boundary false positives are account-like
   `O` values inside the parent ML-v1 family `key_value_accountlike_batch_v1`,
   predicted BUSINESS_ID at a median confidence of 0.9993. After the v1.1
   augmentation, the model treats "identifier-shaped token in key=value log"
   as sufficient evidence for BUSINESS_ID — precisely the overprediction
   ML-v1.2 must counterbalance.
2. **The dominant challenge failure is high-risk absorption.** All 20
   wrong-class cases are genuine `CREDIT_CARD_NUMBER` values (in
   `payment_card_number` context roles) absorbed into BUSINESS_ID at ~0.996
   confidence. This is the dangerous direction for a masking system: a real
   card number labeled as a business identifier could be under-protected.
3. **Related high-risk directions remain visible even where BUSINESS_ID is not
   the predicted class**: `BUSINESS_ID → SSN` (10, challenge) shows the
   opposite-direction instability on the same value shapes, and
   `SSN → ITIN` (5, standard) plus outright SSN misses explain the SSN recall
   regression. These motivate the bidirectional (not one-way) design of the
   v1.2 counterbalance families.

## Consequences implemented in the ML-v1.2 catalog

- `business_id_vs_credit_card` train families (4) attack finding 2 directly:
  the same Luhn-valid surface appears once as a true card number and once as a
  genuine business/transaction key, with the semantics carried by context.
- Every contrast family pairs an identifier-shaped surface with an explicit
  companion entity, so BUSINESS_ID is never the only plausible reading of a
  record ("identifier-shaped ≠ BUSINESS_ID automatically"), countering
  finding 1 without converting any sensitive entity to `O`.
- `ip_address_vs_technical_reference` families (6 train / 3 dev) supply
  O-preserving evidence for the worst category; their technical-role records
  keep the dotted-quad focus value as `O` while still annotating the adjacent
  business key, so O-evidence never suppresses a real entity.
