# SecureLogX ML-v1 vs ML-v1.1 Comparison

**Partial comparison: ML-v1.1 post-gate regression was NOT RUN. Metrics from different dev/challenge views are not treated as interchangeable test results.**

## Direct original-dev comparison

| Metric | ML-v1 original dev | ML-v1.1 original dev | Delta |
|---|---:|---:|---:|
| Micro F1 | 0.899022 | 0.903486 | +0.004464 |
| Macro F1 | 0.903732 | 0.911667 | +0.007935 |
| High-risk recall | 0.902939 | 0.899377 | -0.003562 |
| BUSINESS_ID recall | 0.955556 | 0.955556 | +0.000000 |
| SSN recall | 0.990291 | 0.922330 | -0.067961 |

## Known-failure diagnostic (not a direct same-split comparison)

| Indicator | ML-v1 historical test/family | ML-v1.1 dev challenge |
|---|---:|---:|
| BUSINESS_ID recall | 0.549784 | 0.935000 |
| Hard/context record error | 1.000000 | 0.112500 |

The apparent targeted improvement is encouraging but did not satisfy all predeclared challenge gates. Sealed, Gretel, synthetic-unseen, historical hard-negative, and original-test ML-v1.1 comparisons remain unavailable.
