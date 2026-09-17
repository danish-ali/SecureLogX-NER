# SecureLogX ML-v1 Training Preflight

**Status: PASS**

The frozen inputs passed canonical-label, count, SHA-256, span, provenance, and split-independence checks. No dataset file was modified.

## Canonical ontology

- Entity types: **25**
- BIO labels: **51**
- First label: **O=0**
- Final label: **I-API_KEY=50**
- `label_to_id` and `id_to_label`: exact deterministic inverses

## Frozen split checks

| Split | Records | Entity spans | Negative records | SHA-256 |
|---|---:|---:|---:|---|
| train | 23558 | 32203 | 1535 | `b36926d148c3672c463f9530aac77f78f4ccd6e864f2cd70b8c8577a766a83b5` |
| dev | 2994 | 4020 | 309 | `126d65a63e0b37bff8462b2f0f56e8cdceab2ec54ad97b868fb1981a4bc8053a` |
| test | 2989 | 4358 | 215 | `32c0dc0a259fbc4c51c4a7ce74e3258dedb18ed5dcc48898e182d0991b06fbb7` |

- Total records: **29541**
- Total entity spans: **40581**
- Exact text leakage: **0**
- Synthetic template-family leakage: **0**
- External source-record-ID leakage: **0**
- Invalid spans, overlaps, unsupported labels, or provenance mismatches: **0**
- Training-excluded records in frozen splits: **0**
- Frozen readiness failures: **0** (all 25 labels pass 300/40/40)

## Source and hard-negative coverage

- train: Gretel 3632, synthetic 19926
- dev: Gretel 457, synthetic 2537
- test: Gretel 452, synthetic 2537

Held-out hard-negative coverage is deliberately limited: test contains one complete 94-record family. Conclusions from that segment must remain qualified.

## Reproducibility caveats

- The frozen split/config artifacts are currently untracked by Git; this manifest supplies immutable SHA-256 fingerprints for this run.
- Gretel metadata remains `license_reviewed=false`. This does not block local technical training, but it blocks any publication/commercial-release claim pending review.
- `meta.split` on Gretel records is upstream provenance. SecureLogX split membership is determined by the frozen file containing the record.
- The older profile names DistilBERT and ONNX. The active phase instruction overrides it with `bert-base-cased` and stops before ONNX; the frozen profile remains unchanged.
