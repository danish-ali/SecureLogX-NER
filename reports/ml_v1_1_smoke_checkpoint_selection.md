# SecureLogX ML-v1.1 Checkpoint Selection

Selection used development data only. The sealed challenge was not parsed or inferred.

Score = harmonic mean of original-dev supported-entity macro F1 and dev-challenge supported-entity macro F1. Scores within 0.001 are treated as tied and the earlier epoch wins.

| Epoch | Original dev macro F1 | Challenge dev macro F1 | Harmonic score | Original micro F1 | Challenge micro F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Selected epoch **1** at `output_securelogx/ml-v1.1/smoke-20260917/best-checkpoint` with harmonic score **0.000000**.

Development unlock decision: **NOT READY FOR SEALED CHALLENGE EVALUATION**.

| Development gate | Threshold | Actual | Status |
|---|---:|---:|---|
| `original_dev_micro_f1_minimum` | 0.889022 | 0.000000 | FAIL |
| `original_dev_supported_macro_f1_minimum` | 0.893732 | 0.000000 | FAIL |
| `original_dev_high_risk_recall_minimum` | 0.892939 | 0.000000 | FAIL |
| `dev_challenge_micro_f1_minimum` | 0.900000 | 0.000000 | FAIL |
| `dev_challenge_supported_macro_f1_minimum` | 0.850000 | 0.000000 | FAIL |
| `dev_challenge_business_id_recall_minimum` | 0.800000 | 0.000000 | FAIL |
| `dev_challenge_business_id_recall_improvement_over_ml_v1_minimum` | 0.200000 | -0.549784 | FAIL |
| `dev_challenge_ssn_recall_minimum` | 0.900000 | 0.000000 | FAIL |
| `dev_challenge_record_error_rate_maximum` | 0.150000 | 1.000000 | FAIL |
| `dev_challenge_context_target_error_rate_maximum` | 0.100000 | 1.000000 | FAIL |
| `dev_challenge_morphology_over_context_error_rate_maximum` | 0.100000 | 0.000000 | PASS |
| `dev_challenge_maximum_category_error_rate` | 0.200000 | 1.000000 | FAIL |
| `unexplained_alignment_failures_maximum` | 0.000000 | 0.000000 | PASS |
| `truncated_entity_spans_maximum` | 0.000000 | 0.000000 | PASS |
