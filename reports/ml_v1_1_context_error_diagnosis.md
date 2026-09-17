# ML-v1.1 Context-Error Diagnosis

## Scope and evidence

This is a diagnosis of the frozen ML-v1 held-out evaluation. It does not rerun inference, revise frozen annotations, alter the model, or authorize training. The evidence comes from these six historical artifacts:

- `reports/model_errors.csv`
- `reports/model_error_analysis.md`
- `reports/hard_negative_model_evaluation.md`
- `reports/business_id_subtype_analysis.md`
- `reports/per_label_test_metrics.csv`
- `reports/source_segment_evaluation.md`

All counts below describe that one historical test inference pass.

## How to read `model_errors.csv`

The evaluator first removes exact span-and-label matches. It then pairs remaining overlapping gold and predicted spans one-to-one, in descending intersection-over-union order. Consequently:

- a row with both `gold_entity` and `predicted_entity` populated is one paired overlap, classified as either a wrong entity class or a boundary error;
- a row with only `gold_entity` populated is a gold span left unmatched after overlap pairing (`false negative` in the CSV category);
- a row with only `predicted_entity` populated is a prediction left unmatched after overlap pairing (`false positive` in the CSV category).

The per-label metrics use exact span-and-label matching. A wrong-class or boundary pair therefore contributes one false negative to the gold label and one false positive to the predicted label, even though it is represented by one paired CSV row rather than duplicate standalone false-negative and false-positive rows. The literal CSV category `false negative` counts only unmatched gold rows; it is not the full exact-match false-negative total.

The "most affected labels" counts in `reports/model_error_analysis.md` use the gold label when a row has one, otherwise the predicted label. Those primary-attribution counts are not the same as counting every row in which a label appears on either side.

## BUSINESS_ID diagnosis

The historical per-label result is 127 exact true positives from support 231, leaving 104 exact-match false negatives. Precision is 88.81%, recall is 54.98%, and F1 is 67.91%.

All 104 BUSINESS_ID false negatives reconcile to the CSV as follows:

| Gold BUSINESS_ID outcome | Rows | Exact-match interpretation |
|---|---:|---|
| Predicted as SSN | 92 | Wrong-class pair; BUSINESS_ID FN and SSN FP |
| Predicted as PASSPORT_NUMBER | 3 | Wrong-class pair |
| Predicted as DOB | 2 | Wrong-class pair |
| Predicted as PERSON_NAME | 1 | Wrong-class pair |
| Predicted as DRIVER_LICENSE | 1 | Wrong-class pair |
| Predicted as EMAIL | 1 | Wrong-class pair |
| Predicted as BUSINESS_ID with a too-short boundary | 3 | Same-label boundary pair, not an exact match |
| No overlapping prediction remained | 1 | Standalone unmatched false negative |
| **Total** | **104** | **All BUSINESS_ID exact-match false negatives** |

Thus, only one BUSINESS_ID span was completely unmatched under the CSV pairing algorithm, but all 104 failed exact BUSINESS_ID recovery. In addition, the subtype analysis shows that all 94 ORDER_ID spans had zero exact BUSINESS_ID true positives and that none of those 94 records contained any BUSINESS_ID prediction. This is a failure to recover the correct class in context, not evidence that the model emitted no entity of any kind.

The 94 ORDER_ID cases are the single historical hard-negative family `order_reference_ssn_shape_warn_v1`. That family produced zero exact matches, 94 false negatives, 95 false positives, and an error on every record. Its 95 predictions were 93 SSN and 2 DOB. The paired CSV view contains 92 BUSINESS_ID-to-SSN rows and 2 BUSINESS_ID-to-DOB rows; the remaining SSN prediction is an unmatched false positive. This explains why the hard-negative report's SSN prediction count is 93 while the paired BUSINESS_ID-to-SSN count is 92.

The directionality and the scope both matter. BUSINESS_ID predicted as SSN occurred 92 times; the reverse, gold SSN predicted as BUSINESS_ID, occurred 0 times. The severe result is strong evidence of context-versus-morphology confusion for one held-out ORDER_ID family, but that single family is not broad enough to establish behavior across all BUSINESS_ID contexts.

The subtype report reinforces that distinction: ACCOUNT_ID, APPLICATION_ID, LOAN_NUMBER, and TRANSACTION_ID had 100% exact recall; USER_ID had 94.59%; CUSTOMER_ID had 80.00%; and ORDER_ID had 0.00%. Subtypes remain provenance metadata only. The neural label remains `BUSINESS_ID`.

## Card and bank-account contrasts are prospective gaps

The historical error inventory contains no direct paired confusions in either direction for these label pairs:

| Direction | Historical paired rows |
|---|---:|
| BUSINESS_ID to CREDIT_CARD_NUMBER | 0 |
| CREDIT_CARD_NUMBER to BUSINESS_ID | 0 |
| BUSINESS_ID to BANK_ACCOUNT_NUMBER | 0 |
| BANK_ACCOUNT_NUMBER to BUSINESS_ID | 0 |

These zeros do not prove contextual robustness. They show that direct BUSINESS_ID/card and BUSINESS_ID/bank-account confusion was not observed in this frozen test pass. Morphology-matched card-versus-business and account-versus-business examples are therefore prospective coverage gaps for ML-v1.1, not remediations of measured historical confusion. New families must test these concepts independently without copying or mutating historical test text.

## IP_ADDRESS diagnosis

`model_errors.csv` has 7 rows involving IP_ADDRESS across 5 records, all from `gretel_finance_pii`:

- 4 paired IP_ADDRESS boundary errors: 2 too long, 1 too short, and 1 shifted;
- 1 unmatched gold IP_ADDRESS false negative;
- 2 unmatched IP_ADDRESS false positives, both in the same record.

This row representation is consistent with the per-label totals: support 119, 114 exact true positives, 5 false negatives, 120 predictions, and 6 false positives. Each of the four boundary rows counts once on both the exact-match false-negative and false-positive sides.

No historical IP_ADDRESS error row establishes that a version, build number, or reference identifier was falsely classified as IP_ADDRESS. The frozen evidence therefore does not prove a build/version false-positive defect. Technical/reference values with IP-like morphology are a prospective ML-v1.1 challenge category intended to measure an untested contextual distinction.

## PERSON_NAME and STREET_ADDRESS diagnosis

The primary-attribution counts in `reports/model_error_analysis.md` are 204 classified rows for PERSON_NAME and 111 for STREET_ADDRESS. Every one of those rows is from `gretel_finance_pii`.

| Label | Support | Exact TP | Exact FN | Predicted | Exact FP | F1 | Primary-attributed CSV rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| PERSON_NAME | 782 | 622 | 160 | 709 | 87 | 83.43% | 204 |
| STREET_ADDRESS | 412 | 356 | 56 | 444 | 88 | 83.18% | 111 |

For PERSON_NAME, the 204 primary-attributed rows comprise 160 rows with PERSON_NAME as gold plus 44 unmatched PERSON_NAME predictions. Three additional rows have another gold label paired to a PERSON_NAME prediction, so 207 rows involve PERSON_NAME on either side across 80 records. For STREET_ADDRESS, the 111 primary-attributed rows comprise 56 rows with STREET_ADDRESS as gold plus 55 unmatched STREET_ADDRESS predictions. Five additional PERSON_NAME-to-STREET_ADDRESS pairs bring total involvement to 116 rows across 69 records.

The Gretel-only metrics are consistent with a difficult external-style segment: PERSON_NAME F1 is 81.04% and STREET_ADDRESS F1 is 78.44%. However, Gretel has no template-family metadata, and its source-supplied spans can expose boundary and annotation-policy differences as well as genuine model errors. The frozen annotations remain the evaluation gold, but this diagnosis does not assume that every name/address disagreement independently proves a context-classification defect. ML-v1.1 should add stylistic and contextual diversity; it must not rewrite the historical test labels to improve the score.

## Source-generalization gap

The same held-out inference pass scored 96.86% micro F1 on synthetic unseen-template records and 80.06% on Gretel-only records, a 16.80 percentage-point gap. Supported-label macro F1 was 97.30% for synthetic and 71.23% for Gretel, a 26.07-point gap. This supports increasing contextual and stylistic diversity rather than merely adding synthetic volume. It does not justify claiming that new synthetic data represents real-world source coverage.

The BUSINESS_ID hard-negative result is an important exception to a simplistic source-only explanation: the dominant 94-case ORDER_ID failure is in `securelogx_custom_logs`, while the remaining 10 BUSINESS_ID error rows are in Gretel. ML-v1.1 therefore needs both morphology-controlled context contrasts and broader log-style variation.

## Diagnosis and ML-v1.1 constraints

Observed historical defects:

1. A strong, directional BUSINESS_ID-to-SSN confusion concentrated in one ORDER_ID hard-negative family.
2. Uneven BUSINESS_ID subtype generalization despite good performance on several other subtypes.
3. A substantial aggregate synthetic-to-Gretel generalization gap.
4. Concentrated PERSON_NAME and STREET_ADDRESS errors in Gretel-style records, with boundary and annotation-policy ambiguity requiring cautious interpretation.

Prospective, not historically proven, challenge gaps:

1. BUSINESS_ID versus CREDIT_CARD_NUMBER in both directions.
2. BUSINESS_ID versus BANK_ACCOUNT_NUMBER in both directions.
3. IP-like morphology used explicitly for build, version, or reference semantics.

The canonical ontology is unchanged: all 25 entity types and all 51 BIO labels remain exactly as defined for ML-v1. BUSINESS_ID subtypes remain provenance only and do not become neural output labels.

Historical ML-v1 train/dev/test data, reports, receipt, and checkpoint remain immutable. The findings above may guide conceptual contrast axes only. No exact historical test wording, template wording, or textual mutation of a historical test record may be reused in ML-v1.1 training or development data. New training, development-challenge, and sealed challenge-test families must be independently authored and kept family-disjoint.
