# ML-v1.3 Original-Test Regression

**Historical regression only. Model selection remains frozen.**

- Candidate: BERT epoch 3
- Model SHA-256: 31a460720152a9b555150a55eb474d67cbfa2f236254d9ce464e930f8e2a0f44
- Records: **2989**

## Exact NER

- Micro P/R/F1: **0.916513 / 0.914410 / 0.915461**
- Macro P/R/F1: **0.941720 / 0.953105 / 0.941946**
- High-risk recall: **0.980126**
- BUSINESS_ID P/R/F1: **0.946565 / 0.536797 / 0.685083**
- SSN P/R/F1: **0.515000 / 0.980952 / 0.675410**
- ITIN P/R/F1: **1.000000 / 1.000000 / 1.000000**
- AUTH_TOKEN P/R/F1: **1.000000 / 1.000000 / 1.000000**
- API_KEY P/R/F1: **0.844828 / 0.933333 / 0.886878**

## Security outcomes

- Security-critical miss: **158**
- Security-critical partial: **79**
- Policy-safe wrong class: **104**
- Overmasking: **115**
- Sensitive-span recall: **0.963745**
- Full-mask recall: **0.945617**
- High-risk full-mask recall: **0.984643**

No sealed inference, retraining, threshold tuning, ONNX export, or Java changes occurred.
