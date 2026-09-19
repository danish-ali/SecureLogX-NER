# ML-v1.2 Checkpoint Tradeoff Analysis (ML-v1.1 epochs 1-3)

Analysis only: the frozen ML-v1.1 checkpoint selection (epoch 3) is **not**
revisited or changed. All values come from the frozen per-epoch
`epoch_metrics.json` files under
`output_securelogx/ml-v1.1/bert-base-cased/checkpoints/epoch-{1,2,3}/` and are
reproduced in `reports/ml_v1_2_v1_1_dev_error_analysis.json`
(`checkpoint_tradeoffs`). No model was executed.

## Standard dev (frozen `data/split/dev.jsonl`)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Micro F1 | 0.881001 | 0.901053 | 0.903486 |
| Macro F1 (supported) | 0.897264 | 0.910520 | 0.911667 |
| BUSINESS_ID precision | 0.479167 | 0.549784 | 0.560870 |
| BUSINESS_ID recall | 0.851852 | 0.940741 | 0.955556 |
| BUSINESS_ID F1 | 0.613333 | 0.693989 | 0.706849 |
| SSN recall | 0.970874 | 0.990291 | 0.922330 |
| SSN F1 | 0.956938 | 0.985507 | 0.935961 |
| High-risk recall | 0.894034 | 0.900267 | 0.899377 |

## Challenge dev (frozen ML-v1.1 dev challenge)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Micro F1 | 0.853448 | 0.866091 | 0.892544 |
| Macro F1 (supported) | 0.859988 | 0.835223 | 0.852341 |
| BUSINESS_ID precision | 0.835000 | 0.837963 | 0.882075 |
| BUSINESS_ID recall | 0.835000 | 0.905000 | 0.935000 |
| BUSINESS_ID F1 | 0.835000 | 0.870192 | 0.907767 |
| High-risk recall | 0.945000 | 0.900000 | 0.900000 |
| Record error rate | 0.095833 | 0.125000 | 0.112500 |
| Context-target error rate | 0.095833 | 0.125000 | 0.112500 |
| Worst-category error rate | 0.312500 | 0.262500 | 0.262500 |

## Findings

1. **BUSINESS_ID overprediction did increase with training.** Standard-dev
   BUSINESS_ID recall climbed monotonically (0.8519 → 0.9407 → 0.9556) while
   precision stayed collapsed (0.4792 → 0.5498 → 0.5609). The model widened
   its BUSINESS_ID decision boundary each epoch far faster than it learned to
   restrict it; the false-positive base (spurious account-like `O` spans)
   persisted at every epoch.
2. **SSN recall peaked at epoch 2 (0.9903) and dropped sharply at epoch 3
   (0.9223).** The final epoch traded SSN sensitivity for BUSINESS_ID/context
   gains — direct motivation for ML-v1.2's true-SSN protection families.
3. **No epoch satisfied all predeclared gates**, so the failure was not an
   artifact of checkpoint choice:
   - Epoch 1 passes the context-target error gate (0.0958 <= 0.10) but is far
     below the challenge micro-F1 gate (0.8534 < 0.90) and above the
     worst-category gate (0.3125 > 0.20).
   - Epoch 2 fails all three (0.8661 < 0.90; 0.1250 > 0.10; 0.2625 > 0.20).
   - Epoch 3 fails all three (0.8925 < 0.90; 0.1125 > 0.10; 0.2625 > 0.20).
4. **The worst-category error plateaued at 0.2625 after epoch 2** — additional
   training on the v1.1 mixture could not fix `ip_address_vs_technical_reference`
   because the training data lacked O-preserving dotted-quad counter-evidence.
   This is a data gap, not an optimization artifact.

## Conclusion

The tradeoff pattern (recall-heavy BUSINESS_ID growth, late SSN regression,
plateaued worst category) supports the ML-v1.2 decision to counterbalance the
*data* rather than re-pick a checkpoint: no ML-v1.1 epoch is gate-clean, and
the deficits are attributable to missing bidirectional contrast evidence.
Checkpoint selection policy is unchanged.
