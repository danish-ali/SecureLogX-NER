# SecureLogX ML-v1.1 Training Preflight

**Status: FULL PREFLIGHT PASSED**

The sealed challenge was not parsed or evaluated. Its frozen bytes were SHA-256 checked only.

## Frozen evidence

- Verified file/hash checks: **33**
- ML-v1.1 manifest SHA-256: `cf4a2c60c0b0fafa0a570b57af8335c8ca8fbefd4a9f84175e0da2cbc5b5ba63`
- Predeclared training-gate SHA-256: `cf9f217bbe08a81d79634f649d3a157c4b59a30d9f8168355621e2cdb4bfb273`
- Frozen ML-v1 checkpoint fingerprint: `922474e5af6819c7cdc172d5cb279d7207f914fda709bb5aa39f0ebf512787cc`
- Ontology: **25 entities / 51 BIO labels**

## Dataset composition

- Original ML-v1 train records: **23558**
- ML-v1.1 context additions: **3000**
- Combined in-memory training records: **26558**
- Original standard-dev records: **2994**
- Independent dev-challenge records: **480**
- No derived JSONL was needed; concatenation is in memory and every source record retains its original `meta` provenance.

## Alignment

| View | Spans | Aligned | Boundary adjusted | Truncated | Explicitly unrepresentable |
|---|---:|---:|---:|---:|---:|
| train | 34903 | 34897 | 29 | 0 | 6 |
| standard_dev | 4020 | 4017 | 0 | 0 | 3 |
| challenge_dev | 440 | 440 | 0 | 0 | 0 |
