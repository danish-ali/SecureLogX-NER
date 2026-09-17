# ML-v1.1 Model Validation Decision

**NOT READY FOR ONNX VALIDATION**

The selected checkpoint failed the immutable development unlock. No threshold was changed after results were observed.

## Failed development gates

| Gate | Actual | Operator | Threshold |
|---|---:|:---:|---:|
| `dev_challenge_context_target_error_rate_maximum` | 0.112500 | <= | 0.100000 |
| `dev_challenge_maximum_category_error_rate` | 0.262500 | <= | 0.200000 |
| `dev_challenge_micro_f1_minimum` | 0.892544 | >= | 0.900000 |

- Selected epoch/checkpoint: **3** / `output_securelogx/ml-v1.1/bert-base-cased/best-checkpoint`
- Model SHA-256: `28ec51cbe7d46423ef9e72467994e43b2d47f7d5465593b69d7d1c427975f571`
- Sealed challenge: **NOT OPENED; no metrics, cache, or receipt created**
- Original ML-v1 regression benchmark: **NOT RUN after Step 9 stop**
- ONNX export: **NOT PERFORMED**

A later regression or sealed-evaluation phase would require explicit authorization; this stopped run cannot be represented as having those results.

Gretel `license_reviewed=false` remains a separate production/commercial release blocker.
