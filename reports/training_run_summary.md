# SecureLogX ML-v1 Training Run Summary

**Status: TRAINING COMPLETE**

## Architecture and configuration

- Base model: `bert-base-cased`
- Base revision: `cd5ef92a9fb2f889e972770a36d4ed042daf221e`
- Task: BERT token classification with the frozen 51-label BIO head
- Maximum window length / stride: 384 / 128
- Epochs: 3
- Learning rate: 2e-05
- Weight decay: 0.01
- Warmup ratio: 0.1
- Train batch / gradient accumulation / effective batch: 4 / 4 / 16
- Eval batch: 8
- Gradient clipping: 1.0
- Mixed precision: BF16
- Seed: 42
- Frozen manifest SHA-256: `ac397beba394b0904fc0350ec0912291c7b2a47057e6490d59fff4b51e8b43b9`
- Predeclared validation-gate SHA-256: `1d8249e1d4f052d0ba4cb64c607e6178d75f34ae160af17e0060907d02e56176`
- Checkpoint selection metric: dev exact-span macro F1

## Alignment used for this run

- Train: 32197/32203 aligned; 29 boundary-adjusted; 0 truncated; 6 explicitly unrepresentable.
- Dev: 4017/4020 aligned; 0 boundary-adjusted; 0 truncated; 3 explicitly unrepresentable.

## Per-epoch dev selection

| Epoch | Train loss | Dev loss | Dev micro F1 | Dev macro F1 | High-risk recall | Optimizer steps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.246986 | 0.043161 | 0.882382 | 0.888613 | 0.882458 | 1629 |
| 2 | 0.017212 | 0.063021 | 0.895618 | 0.899796 | 0.902048 | 1629 |
| 3 | 0.009247 | 0.058070 | 0.899022 | 0.903732 | 0.902939 | 1629 |

## Selection

- Selected epoch: **3**
- Best checkpoint: `output_securelogx/ml-v1/bert-base-cased/best-checkpoint`
- Elapsed training/evaluation time: 42.70 minutes
- The held-out test set was not used for model inference, metrics, training decisions, or checkpoint selection; it was read only by the frozen-manifest integrity check.

## Environment

- Python: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
- PyTorch: `2.6.0+cu124`
- Transformers: `4.49.0`
- Datasets: `4.8.5`
- CUDA available/runtime: `True` / `12.4`
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- CPU: `Intel64 Family 6 Model 170 Stepping 4, GenuineIntel` (16 physical / 22 logical cores)
- PYTHONHASHSEED: `42`
- CUBLAS workspace configuration: `:4096:8`

The frozen manifest was validated immediately before and after the run.
