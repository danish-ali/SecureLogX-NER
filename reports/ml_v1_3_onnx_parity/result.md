# ML-v1.3 ONNX Export and Python Runtime Parity

**ONNX PARITY PASSED**

- Frozen BERT SHA-256: 31a460720152a9b555150a55eb474d67cbfa2f236254d9ce464e930f8e2a0f44
- ONNX SHA-256: 5953acd999083fa08c17c537e1d88c38f3c7137bc518022d8f3246c1133997c2
- Opset: **17**
- Dynamic batch/sequence axes: **yes**
- Exact decoded-span parity across all sampled records: **True**
- Maximum absolute logit delta: **0.0000240803**
- Weighted mean absolute logit delta: **0.0000004479**

| View | Records | Exact span parity | Max abs logit delta | Mean abs logit delta |
|---|---:|---:|---:|---:|
| standard_dev | 48 | 100.00% | 0.0000166893 | 0.0000004464 |
| v1_3_challenge_dev | 48 | 100.00% | 0.0000152588 | 0.0000004436 |
| original_test_regression | 48 | 100.00% | 0.0000240803 | 0.0000004537 |

No training, sealed inference, threshold tuning, or Java changes occurred.
