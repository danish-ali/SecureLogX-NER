# SecureLogX ML-v1.1 Training Run Summary

**Status: TRAINING COMPLETE**

## Controlled configuration

- Architecture/base/revision: BERT token classification / `bert-base-cased` / `cd5ef92a9fb2f889e972770a36d4ed042daf221e`
- Epochs / learning rate / weight decay / warmup: 3 / 2e-05 / 0.01 / 0.1
- Max length / stride / dynamic padding: 384 / 128 / yes
- Physical train batch / accumulation / effective batch / eval batch: 4 / 4 / 16 / 8
- Gradient clipping / BF16 / seed: 1.0 / True / 42
- Training records: 26558 (23558 parent + 3000 additions)
- Training gate SHA-256: `cf9f217bbe08a81d79634f649d3a157c4b59a30d9f8168355621e2cdb4bfb273`
- Architecture, tokenizer, learning rate, epochs, and sequence length are unchanged from ML-v1.

## Epoch development results

| Epoch | Train loss | Standard loss | Standard micro F1 | Standard macro F1 | Challenge micro F1 | Challenge macro F1 | BID recall | SSN recall | Challenge record error | Selection score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.239170 | 0.049982 | 0.881001 | 0.897264 | 0.853448 | 0.859988 | 0.835000 | 0.925000 | 0.095833 | 0.878231 |
| 2 | 0.016594 | 0.043336 | 0.901053 | 0.910520 | 0.866091 | 0.835223 | 0.905000 | 1.000000 | 0.125000 | 0.871247 |
| 3 | 0.008618 | 0.051218 | 0.903486 | 0.911667 | 0.892544 | 0.852341 | 0.935000 | 1.000000 | 0.112500 | 0.881007 |

- Selected epoch: **3**
- Best checkpoint: `output_securelogx/ml-v1.1/bert-base-cased/best-checkpoint`
- Elapsed train plus development evaluation time: **51.95 minutes**
- Checkpoint selection used no test or sealed-challenge prediction.

## Environment

- Python / PyTorch / Transformers: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]` / `2.6.0+cu124` / `4.49.0`
- CUDA / GPU: `True` / `NVIDIA GeForce RTX 4070 Laptop GPU`

## Post-training boundary

The selected checkpoint did not pass the predeclared development unlock. The phase therefore stopped with **NOT READY FOR SEALED CHALLENGE EVALUATION**. No sealed-challenge or original-test regression inference was run, and no ONNX export was performed.
