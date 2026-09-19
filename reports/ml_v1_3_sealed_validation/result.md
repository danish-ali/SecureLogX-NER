# ML-v1.3 Final Sealed Validation

**SEALED VALIDATION COMPLETE**

Model selection was frozen before this evaluation and may not change based on these results.

- Candidate: BERT epoch 3 (bert-base-cased)
- Model SHA-256: 31a460720152a9b555150a55eb474d67cbfa2f236254d9ce464e930f8e2a0f44
- Sealed records: **480**
- Sealed SHA-256 verified: 6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a

## Sealed metrics

- Micro P/R/F1: **0.957245 / 0.959524 / 0.958383**
- Macro P/R/F1: **0.977832 / 0.981111 / 0.979458**
- High-risk recall: **1.000000**
- BUSINESS_ID P/R/F1: **0.905556 / 0.905556 / 0.905556**
- SSN P/R/F1: **1.000000 / 1.000000 / 1.000000**

## Security outcomes

- Security-critical miss: **0**
- Security-critical partial: **17**
- Policy-safe wrong class: **0**
- Overmasking: **1**
- Sensitive-span recall: **1.000000**
- Full-mask recall: **0.959524**
- High-risk full-mask recall: **1.000000**

No original-test inference, retraining, threshold tuning, ONNX export, or Java changes occurred.
