# Proposed gold annotation correction — bank account field

**Status: proposed for human review, not merged, not a retrospective evaluation pass.**

## Reproduced evidence

- Frozen public dataset: `data/split/dev.jsonl` in `SecureLogX-NER`.
- Git blob SHA-1: `13cd609d67180be2f2306709290cfdc4dcb616bb`. Developer confirmed the same SHA on local Windows clone via `git hash-object`.
- Zero-based record index: `2027`, Gretel source record ID `36428`.
- Metadata: `source=gretel_finance_pii`, `review_status=auto_converted`, document type loan application, expanded type income verification.
- Original annotations: six gold spans, including one unrelated `BANK_ACCOUNT_NUMBER` span at `[129,147)`.
- Independently identifiable field: `Bank Account Number: [nine digits]`, value span `[490,499)` (exclusive end), ASCII only. It is **not** gold-annotated or included among the existing source's eight benign exclusions.
- Deterministic detector: matches this value with `explicit-bank-account-field` and emits `MASK`. The baseline and Phase 2 branches of SecureLogX share the identical scanner code blob.
- Original 6,463-case dataset audit: `bypass_uncovered_gold=0`, `deterministic_overmask=1`, `allow_gold_conflict=0`; the single overmask is in this exact dev record.

**Proposed new gold annotation:** `[490, 499, "BANK_ACCOUNT_NUMBER"]`. This appears semantically correct for the source text but requires dataset-owner review of annotation policy/source provenance before promotion to canonical gold. Do not modify the deterministic detector to make the mismatch disappear.

## How the proposed candidate is prepared

`scripts/data/propose_bank_account_gold_amendment.py` is a guarded, local-only tool. It checks the complete original Git blob, index, source ID, annotation state, exact field label, digit format, existing labels and exclusions. A dry run writes nothing. With `--write-candidate`, it creates a **new** versioned evaluation folder **outside** the NER checkout:

- `data/split/dev.jsonl`: identical source except for one appended gold span at record 2027.
- `data/split/test.jsonl`: identical byte copy of original regression test.
- `data/ml_v1_3/real_structure/dev_challenge.jsonl`: identical byte copy of original challenge dev.
- `annotation-candidate-manifest.json`: payload-free source/candidate SHA256, one-record change and review state.

It never changes the original frozen inputs, source text, training split, ONNX model, tokenizer, or the baseline report. It will refuse to overwrite an existing output folder.

### Windows procedure (run from SecureLogX-NER)

```powershell
git fetch origin
git switch --track origin/chatgpt/review-gretel-bank-account-gold-span
python -m unittest discover -s tests -p test_bank_account_gold_amendment.py -v

python scripts/data/propose_bank_account_gold_amendment.py --ner-root . --output-root ../SecureLogX/reports/parallel-onnx/.local-steady-runs/phase2-gold-candidate-v1
# Dry run only; verify proposed label against source annotation policy.
python scripts/data/propose_bank_account_gold_amendment.py --ner-root . --output-root ../SecureLogX/reports/parallel-onnx/.local-steady-runs/phase2-gold-candidate-v1 --write-candidate
```

After dataset-owner approval, use `SecureLogX` branch `chatgpt/phase2-request-scoped-scan-reuse` and pass `-NerRoot ".\\reports\\parallel-onnx\\.local-steady-runs\\phase2-gold-candidate-v1"` to the `-Stage Corpus` validator. **The unchanged original frozen audit must still be reported separately as failed until a formally versioned canonical dataset is approved.**

```powershell
cd ../SecureLogX
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-phase2-validation.ps1 -Stage Corpus -NerRoot ".\reports\parallel-onnx\.local-steady-runs\phase2-gold-candidate-v1"
```

Passing this candidate audit does NOT prove a previously published result had zero overmask, does NOT retroactively change prior training/evaluation metrics, does NOT prove full-corpus ONNX quality, and does NOT authorize merging the SecureLogX Phase 2 PR. Once approved, amend the authoritative conversion/merged source and regenerate the versioned evaluation datasets (with lineage and metrics), rather than silently overwriting frozen `dev.jsonl`.

## Required review gates

1. Confirm original Gretel source record's intended annotations and why this explicitly keyed bank-account value was missed.
2. Review whether the proposed new span is correct, including whether related generated inputs and original model quality measurements must be revised.
3. Independently test SHA256/count invariants and the old-vs-new dataset diff (one `entities` entry only).
4. Keep the strict gold overmask threshold at **zero**.
5. Rerun Phase 2 request-scoped scan parity and matched real-ONNX protected-output A/B; keep full-corpus ONNX privacy accuracy as a separate pending gate.

## Executed validation evidence (2026-11-01)

[GitHub Actions run 41308569120](https://github.com/danish-ali/SecureLogX-NER/actions/runs/41308569120)
completed successfully on Python 3.13 / Java 25 using the proposed candidate
**without modifying original frozen datasets**:

| Verification | Result |
|---|---|
| Python synthetic unit tests | 6 passed |
| Exactly one record modified in candidate | Passed |
| Other two locked input files byte-identical | Passed |
| Original 6,463-record frozen gold audit | Fails as expected: 1 overmask, 0 uncovered gold, 0 ALLOW conflicts |
| Proposed 6,463-record candidate gold audit | Passed: 0 overmask, 0 uncovered gold, 0 ALLOW conflicts |
| Phase 2 model-free deterministic scan reuse audit | Passed: 5,545 reused and 918 worker fallback scans |
| Real ONNX A/B model-output parity | **NOT TESTED** |
| Dataset owner review and canonical versioning | **NOT APPROVED** |

This CI result establishes a reproducible **candidate**, not permission to
overwrite frozen gold, revise published baseline measurements, retrain on a
changed dev split, or merge/activate Phase 2.
