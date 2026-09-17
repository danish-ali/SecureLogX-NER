# ML-v1.1 Error Analysis

**Scope: selected-checkpoint development challenge only.**

- False positives: **65**
- False negatives: **33**
- Boundary too short: **1**
- Boundary too long: **0**
- Shifted boundary: **0**
- Wrong class: **32**

## Context-conflict tags

- ambiguous-key: **5**
- context-over-morphology: **20**
- morphology-over-context: **37**
- structured-field-confusion: **39**
- unseen-business-subtype: **5**

Detailed rows are in `reports/ml_v1_1_model_errors.csv`. ML-v1.1 original-test and sealed errors do not exist because their inference phases were not run.
