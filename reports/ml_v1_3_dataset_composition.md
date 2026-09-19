# ML-v1.3 Dataset Composition

Future combined training size: **30462** records.

| Component | Records | Percentage |
|---|---:|---:|
| original synthetic | 19926 | 65.41% |
| Gretel/external | 3632 | 11.92% |
| other original train | 0 | 0.00% |
| ML-v1.1 synthetic additions | 3000 | 9.85% |
| ML-v1.2 synthetic additions | 1600 | 5.25% |
| ML-v1.3 real-structure additions | 2304 | 7.56% |

The v1.3 additions are smaller than the original synthetic generator mass on purpose.
They reduce dependence on a few generator skeletons by introducing 96 independent
public-format log structures instead of more COUNTERCHECK/key=value clones.
Gretel records remain `license_reviewed=false` and were not used as structure donors.
