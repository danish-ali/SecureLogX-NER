# ML-v1.1 Generalization Analysis

The Step 9 development gate failed, so post-gate model inference stopped. The following views remain deliberately separate.

| View | Interpretation | Result |
|---|---|---|
| Original standard dev | Ordinary NER regression signal | micro F1 0.903486; supported macro F1 0.911667 |
| New dev context challenge | Context-conflict robustness | micro F1 0.892544; supported macro F1 0.852341; record error 0.112500 |
| Historical ML-v1 Gretel | Historical external/public-source evidence only | micro F1 0.800593; ML-v1.1 NOT RUN |
| Historical ML-v1 synthetic unseen templates | Historical synthetic evidence only | micro F1 0.968589; ML-v1.1 NOT RUN |
| Historical failed hard-negative family | Required new-model regression evidence | NOT RUN because Step 9 stopped the phase |
| New sealed challenge families | Independent context-conflict evidence | NOT OPENED / NOT RUN |

## Dev conflict categories

| Category | Micro F1 | Supported macro F1 | Record error rate |
|---|---:|---:|---:|
| business_id_vs_bank_account | 1.000000 | 1.000000 | 0.000000 |
| business_id_vs_credit_card | 0.750000 | 0.733333 | 0.250000 |
| business_id_vs_ssn | 0.939959 | 0.945217 | 0.054167 |
| ip_address_vs_technical_reference | 0.733945 | 0.747664 | 0.262500 |

External/public-source generalization, synthetic generalization, and context-conflict robustness are not combined into one headline metric. Gretel remains subject to `license_reviewed=false`.
