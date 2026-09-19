# SecureLogX ML-v1.2 Training Preflight

**FULL PREFLIGHT PASSED**

- Dataset manifest SHA-256: `67e5d32e40444f7399793b79e0dca4f2e788afd70706c4496f1c6a35bf6c6ead`
- Predeclared training gate SHA-256: `9b37b291c24364695b986a542e76aa7b3a977b64be818dd67035de7dc94fd668`
- Parent freeze: **129 files verified; 0 mismatches**
- Sealed challenge: **not opened**, regenerated hash `6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a`
- Ontology: **25 entities / 51 BIO labels**

## Dataset composition

- Original ML-v1 train: **23558**
- ML-v1.1 addition: **3000** (deterministic_in_memory_regeneration)
- ML-v1.2 addition: **1600**
- Combined in-memory train: **28158**
- Standard dev / ML-v1.2 challenge dev: **2994 / 540**

## Alignment

| View | Spans | Aligned | Adjusted | Truncated | Failed |
|---|---:|---:|---:|---:|---:|
| train | 37983 | 37977 | 29 | 0 | 6 |
| standard_dev | 4020 | 4017 | 0 | 0 | 3 |
| challenge_dev | 1026 | 1026 | 0 | 0 | 0 |
