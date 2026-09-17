# ML-v1 ONNX Validation Decision

**NOT READY FOR ONNX VALIDATION**

These gates were committed to configuration before the training/test results were observed.

| Gate | Threshold | Actual | Status |
|---|---:|---:|---|
| overall_test_micro_f1_minimum | 0.900000 | 0.916581 | PASS |
| overall_test_macro_f1_minimum | 0.850000 | 0.942423 | PASS |
| high_risk_test_recall_minimum | 0.950000 | 0.981933 | PASS |
| minimum_individual_label_recall | 0.700000 | 0.549784 | FAIL |
| gretel_test_micro_f1_minimum | 0.800000 | 0.800593 | PASS |
| unseen_template_synthetic_micro_f1_minimum | 0.900000 | 0.968589 | PASS |
| hard_negative_records_with_false_positive_rate_maximum | 0.100000 | 1.000000 | FAIL |
| unexplained_alignment_failures_maximum | 0.000000 | 0.000000 | PASS |
| truncated_entity_spans_maximum | 0.000000 | 0.000000 | PASS |

Worst-recall supported label: `BUSINESS_ID`.

This decision authorizes only a later ONNX validation phase when READY. It does not export ONNX, approve licensing, or establish production readiness.
