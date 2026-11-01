#!/usr/bin/env python3
"""Create a proposed, versioned read-only-input gold amendment for audit.

Never edits frozen data/split/dev.jsonl, processed data, source metadata,
training files, or model artifacts. Output is a new local evaluation copy.
It MUST be independently reviewed before being treated as canonical gold.
No sensitive source text is emitted to stdout or the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

EXPECTED_DEV_GIT_BLOB = "13cd609d67180be2f2306709290cfdc4dcb616bb"
TARGET_INDEX = 2027
TARGET_SOURCE_ID = 36428
TARGET_START = 490
TARGET_END = 499
TARGET_LABEL = "BANK_ACCOUNT_NUMBER"
EXPECTED_RECORD_COUNT = 2994
OTHER_INPUTS = (
    ("data/split/test.jsonl", 2989),
    ("data/ml_v1_3/real_structure/dev_challenge.jsonl", 480),
)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii")
                        + b"\0" + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def propose_record(record: dict) -> dict:
    """Return an amended copy; do not change the caller's dictionary."""
    if not isinstance(record, dict):
        raise ValueError("Target record must be a JSON object")
    text = record.get("text")
    meta = record.get("meta")
    entities = record.get("entities")
    if not isinstance(text, str) or not isinstance(meta, dict) or not isinstance(entities, list):
        raise ValueError("Target record schema mismatch")
    if (meta.get("source") != "gretel_finance_pii"
            or str(meta.get("source_record_id")) != str(TARGET_SOURCE_ID)
            or meta.get("review_status") != "auto_converted"):
        raise ValueError("Target source identity/review state mismatch")
    if len(entities) != 6:
        raise ValueError("Unexpected target gold annotation count")
    # The field name and value are both ASCII, so Python codepoint and Java
    # UTF-16 offsets agree at the target span only if the prefix is BMP.
    # Fail on any non-BMP prefix instead of silently changing offsets.
    if any(ord(c) > 0xFFFF for c in text[:TARGET_END]):
        raise ValueError("Non-BMP text would invalidate Java span offsets")
    if text[TARGET_START - 21:TARGET_START] != "Bank Account Number: ":
        raise ValueError("Expected explicit bank-account field is absent")
    if re.fullmatch(r"[0-9]{9}", text[TARGET_START:TARGET_END]) is None:
        raise ValueError("Expected nine-digit bank-account value is absent")
    if not any(isinstance(s, list) and len(s) == 3
               and s[2] == TARGET_LABEL and s[0] == 129 and s[1] == 147
               for s in entities):
        raise ValueError("Original bank-account gold annotation missing")
    for s in entities:
        if (not isinstance(s, list) or len(s) != 3
                or not isinstance(s[0], int) or not isinstance(s[1], int)):
            raise ValueError("Malformed gold annotation")
        if max(s[0], TARGET_START) < min(s[1], TARGET_END):
            raise ValueError("Target already overlaps a gold annotation")
    for e in meta.get("excluded_spans", []):
        if (isinstance(e, dict) and isinstance(e.get("start"), int)
                and isinstance(e.get("end"), int)
                and max(e["start"], TARGET_START) < min(e["end"], TARGET_END)):
            raise ValueError("Target overlaps a source exclusion")
    amended = dict(record)
    amended["entities"] = sorted(
        [list(s) for s in entities] + [[TARGET_START, TARGET_END, TARGET_LABEL]],
        key=lambda e: (e[0], e[1], e[2]),
    )
    return amended


def count_lines(data: bytes) -> int:
    return sum(bool(line.strip()) for line in data.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="NEW local directory outside the NER Git checkout")
    parser.add_argument("--write-candidate", action="store_true",
                        help="Explicit permission to write a NEW unapproved evaluation copy")
    args = parser.parse_args()
    source_root = args.ner_root.resolve(strict=True)
    out_root = args.output_root.resolve()
    if not (source_root / "data/split/dev.jsonl").is_file():
        raise RuntimeError("Expected NER development split not found")
    if out_root == source_root or source_root in out_root.parents:
        raise RuntimeError("Refusing to create candidate inside source NER checkout")
    source = (source_root / "data/split/dev.jsonl").read_bytes()
    if git_blob_sha1(source) != EXPECTED_DEV_GIT_BLOB:
        raise RuntimeError("Frozen development dataset Git blob differs; refusing patch")
    lines = source.splitlines(keepends=True)
    if len(lines) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Frozen dev record count mismatch")
    original = lines[TARGET_INDEX]
    old = json.loads(original.decode("utf-8"))
    patched = propose_record(old)
    old_ending = b"\r\n" if original.endswith(b"\r\n") else (
        b"\n" if original.endswith(b"\n") else b""
    )
    lines[TARGET_INDEX] = (json.dumps(
        patched, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8") + old_ending)
    new_dev = b"".join(lines)
    if count_lines(new_dev) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Amended dev record count mismatch")
    if out_root.exists():
        raise RuntimeError("Output candidate directory exists; refusing overwrite")
    other_payloads: dict[str, bytes] = {}
    for rel, expected_records in OTHER_INPUTS:
        value = (source_root / rel).read_bytes()
        if count_lines(value) != expected_records:
            raise RuntimeError("Frozen comparison corpus count mismatch: " + rel)
        other_payloads[rel] = value
    manifest = {
        "kind": "securelogx-gold-annotation-amendment-candidate-v1",
        "review_status": "PROPOSED_NOT_APPROVED",
        "source_development_git_blob_sha1": git_blob_sha1(source),
        "source_development_sha256": sha256(source),
        "candidate_development_sha256": sha256(new_dev),
        "source_record_count": EXPECTED_RECORD_COUNT,
        "target_zero_based_index": TARGET_INDEX,
        "source_record_id": TARGET_SOURCE_ID,
        "target_span": [TARGET_START, TARGET_END, TARGET_LABEL],
        "added_gold_spans": 1,
        "changed_records": 1,
        "full_corpus_onnx_parity": "NOT_TESTED",
        "other_dataset_sha256": {k: sha256(v) for k, v in other_payloads.items()},
    }
    if not args.write_candidate:
        print("DRY RUN: frozen Git blob verified; candidate passes identity and annotation checks.")
        print("Add --write-candidate to write a NEW, unapproved evaluation copy.")
        return
    out_root.mkdir(parents=True, exist_ok=False)
    dev_path = out_root / "data/split/dev.jsonl"
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.write_bytes(new_dev)
    for rel, payload in other_payloads.items():
        path = out_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (out_root / "annotation-candidate-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("PROPOSED GOLD CANDIDATE CREATED; NOT YET APPROVED")
    print("Changed exactly 1 / 2994 dev records; appended 1 labeled bank-account span.")
    print("No original corpus file, training file, or model has been modified.")
    print("Candidate directory: " + str(out_root))


if __name__ == "__main__":
    main()
