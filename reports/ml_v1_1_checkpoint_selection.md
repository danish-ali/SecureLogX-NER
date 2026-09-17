# SecureLogX ML-v1.1 Checkpoint Selection

Selection used development data only. The sealed challenge was not parsed or inferred.

Score = harmonic mean of original-dev supported-entity macro F1 and dev-challenge supported-entity macro F1. Scores within 0.001 are treated as tied and the earlier epoch wins.

| Epoch | Original dev macro F1 | Challenge dev macro F1 | Harmonic score | Original micro F1 | Challenge micro F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.897264 | 0.859988 | 0.878231 | 0.881001 | 0.853448 |
| 2 | 0.910520 | 0.835223 | 0.871247 | 0.901053 | 0.866091 |
| 3 | 0.911667 | 0.852341 | 0.881007 | 0.903486 | 0.892544 |

Selected epoch **3** at `output_securelogx/ml-v1.1/bert-base-cased/best-checkpoint` with harmonic score **0.881007**.

Development unlock decision: **NOT READY FOR SEALED CHALLENGE EVALUATION**.

| Development gate | Threshold | Actual | Status |
|---|---:|---:|---|
| `original_dev_micro_f1_minimum` | 0.889022 | 0.903486 | PASS |
| `original_dev_supported_macro_f1_minimum` | 0.893732 | 0.911667 | PASS |
| `original_dev_high_risk_recall_minimum` | 0.892939 | 0.899377 | PASS |
| `dev_challenge_micro_f1_minimum` | 0.900000 | 0.892544 | FAIL |
| `dev_challenge_supported_macro_f1_minimum` | 0.850000 | 0.852341 | PASS |
| `dev_challenge_business_id_recall_minimum` | 0.800000 | 0.935000 | PASS |
| `dev_challenge_business_id_recall_improvement_over_ml_v1_minimum` | 0.200000 | 0.385216 | PASS |
| `dev_challenge_ssn_recall_minimum` | 0.900000 | 1.000000 | PASS |
| `dev_challenge_record_error_rate_maximum` | 0.150000 | 0.112500 | PASS |
| `dev_challenge_context_target_error_rate_maximum` | 0.100000 | 0.112500 | FAIL |
| `dev_challenge_morphology_over_context_error_rate_maximum` | 0.100000 | 0.064583 | PASS |
| `dev_challenge_maximum_category_error_rate` | 0.200000 | 0.262500 | FAIL |
| `unexplained_alignment_failures_maximum` | 0.000000 | 0.000000 | PASS |
| `truncated_entity_spans_maximum` | 0.000000 | 0.000000 | PASS |

Because three required gates failed, the sealed challenge remained unopened and the ordered experiment stopped before original-test regression inference. No sealed metrics, prediction cache, or receipt exists.
