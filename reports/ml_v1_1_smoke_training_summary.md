# SecureLogX ML-v1.1 Training Run Summary

**Status: SMOKE TEST PASSED**

## Controlled configuration

- Architecture/base/revision: BERT token classification / `bert-base-cased` / `cd5ef92a9fb2f889e972770a36d4ed042daf221e`
- Epochs / learning rate / weight decay / warmup: 1 / 2e-05 / 0.01 / 0.1
- Max length / stride / dynamic padding: 384 / 128 / yes
- Physical train batch / accumulation / effective batch / eval batch: 4 / 4 / 16 / 8
- Gradient clipping / BF16 / seed: 1.0 / True / 42
- Training records: 64 (32 parent + 32 additions)
- Training gate SHA-256: `cf9f217bbe08a81d79634f649d3a157c4b59a30d9f8168355621e2cdb4bfb273`
- Architecture, tokenizer, learning rate, epochs, and sequence length are unchanged from ML-v1.

## Epoch development results

| Epoch | Train loss | Standard loss | Standard micro F1 | Standard macro F1 | Challenge micro F1 | Challenge macro F1 | BID recall | SSN recall | Challenge record error | Selection score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.488893 | 2.890344 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |

- Selected epoch: **1**
- Best checkpoint: `output_securelogx/ml-v1.1/smoke-20260917/best-checkpoint`
- Elapsed train plus development evaluation time: **0.09 minutes**
- Checkpoint selection used no test or sealed-challenge prediction.

## Environment

- Python / PyTorch / Transformers: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]` / `2.6.0+cu124` / `4.49.0`
- CUDA / GPU: `True` / `NVIDIA GeForce RTX 4070 Laptop GPU`

## Smoke assertions

- First optimization-group mean loss: **3.975657**
- Last optimization-group mean loss: **3.174310**
- Loss decreased: **True**
- Finite loss, decoded predictions, and serialized checkpoint reload: **PASS**
