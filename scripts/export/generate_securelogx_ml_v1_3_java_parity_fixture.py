#!/usr/bin/env python3
"""Generate deterministic Python/ONNX reference fixtures for Java ML-v1.3 parity.

The fixture uses development and already-observed historical-test records only.
It never reads the sealed challenge. Each case records exact Hugging Face token
IDs, attention mask, character offsets, ONNX argmax label IDs, and decoded
entity spans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from securelogx_training_common import (  # noqa: E402
    decode_bio_spans,
    load_canonical_labels,
    load_jsonl,
    normalize_decoded_spans,
    sha256_file,
)

STANDARD_DEV = "data/split/dev.jsonl"
CHALLENGE_DEV = "data/ml_v1_3/real_structure/dev_challenge.jsonl"
ORIGINAL_TEST = "data/split/test.jsonl"
ONNX_DIR = "onnx-model/ml-v1.3/bert-base-cased"
ONNX_MODEL = "onnx-model/ml-v1.3/bert-base-cased/model.onnx"
OUTPUT = "reports/ml_v1_3_java_parity/fixture.json"
MAX_LENGTH = 384
PER_VIEW = 32


def stable_sample(
    records: Sequence[Mapping[str, Any]],
    size: int,
) -> list[Mapping[str, Any]]:
    if len(records) <= size:
        return list(records)
    return [
        records[round(i * (len(records) - 1) / (size - 1))]
        for i in range(size)
    ]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def span_payload(spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": int(span["start"]),
            "end": int(span["end"]),
            "label": str(span["label"]),
        }
        for span in spans
    ]


def build_case(
    source: str,
    source_index: int,
    text: str,
    tokenizer: Any,
    session: Any,
    id_to_label: Mapping[int, str],
) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        return_tensors="np",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    token_type_ids = encoded.get(
        "token_type_ids",
        np.zeros_like(input_ids, dtype=np.int64),
    ).astype(np.int64)

    logits = session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0][0]

    predicted_ids = logits.argmax(axis=-1).astype(int).tolist()

    valid_offsets: list[tuple[int, int]] = []
    valid_ids: list[int] = []
    valid_confidences: list[float] = []
    for offset, predicted_id, row in zip(offsets, predicted_ids, logits):
        start, end = int(offset[0]), int(offset[1])
        if start == end:
            continue
        shifted = row - np.max(row)
        probs = np.exp(shifted)
        probs = probs / probs.sum()
        valid_offsets.append((start, end))
        valid_ids.append(int(predicted_id))
        valid_confidences.append(float(probs[predicted_id]))

    spans, invalid_bio = decode_bio_spans(
        valid_offsets,
        valid_ids,
        valid_confidences,
        id_to_label,
    )
    spans = normalize_decoded_spans(text, spans)

    return {
        "source": source,
        "source_index": source_index,
        "text": text,
        "text_sha256": text_sha256(text),
        "input_ids": input_ids[0].astype(int).tolist(),
        "attention_mask": attention_mask[0].astype(int).tolist(),
        "token_type_ids": token_type_ids[0].astype(int).tolist(),
        "offsets": [[int(start), int(end)] for start, end in offsets],
        "predicted_label_ids": predicted_ids,
        "decoded_spans": span_payload(spans),
        "invalid_bio_transitions": int(invalid_bio),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required") from exc

    model_path = root / ONNX_MODEL
    tokenizer_dir = root / ONNX_DIR
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not (tokenizer_dir / "tokenizer.json").exists():
        raise FileNotFoundError(tokenizer_dir / "tokenizer.json")

    labels = load_canonical_labels(root / "configs/securelogx_labels.json")
    id_to_label = {int(key): str(value) for key, value in labels["id_to_label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=True,
        local_files_only=True,
    )
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    cases: list[dict[str, Any]] = []
    views = (
        ("standard_dev", load_jsonl(root / STANDARD_DEV)),
        ("v1_3_challenge_dev", load_jsonl(root / CHALLENGE_DEV)),
        ("original_test_regression", load_jsonl(root / ORIGINAL_TEST)),
    )

    for source, records in views:
        sample = stable_sample(records, PER_VIEW)
        sampled_indices = [
            round(i * (len(records) - 1) / (PER_VIEW - 1))
            for i in range(PER_VIEW)
        ]
        for source_index, record in zip(sampled_indices, sample):
            cases.append(
                build_case(
                    source,
                    source_index,
                    str(record["text"]),
                    tokenizer,
                    session,
                    id_to_label,
                )
            )

    probes = [
        'INFO customerId=CUST-123 auth="Bearer abc.DEF-123" ip=10.20.30.40',
        '{"email":"Jane.Doe@example.com","account":"001234567890","status":"ok"}',
        "WARN ssn=123-45-6789 itin=912-70-1234 requestId=REQ-2026-0001",
        "ERROR card=4111 1111 1111 1111 apiKey=sk_test_ABC123XYZ service=payments",
    ]
    for index, text in enumerate(probes):
        cases.append(
            build_case(
                "manual_probe",
                index,
                text,
                tokenizer,
                session,
                id_to_label,
            )
        )

    payload = {
        "phase": "ML-v1.3 Java parity reference",
        "sealed_challenge_accessed": False,
        "case_count": len(cases),
        "max_length": MAX_LENGTH,
        "onnx_sha256": sha256_file(model_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
        "ontology_sha256": sha256_file(root / "configs/securelogx_labels.json"),
        "bio_label_count": 51,
        "cases": cases,
    }

    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("JAVA PARITY FIXTURE READY")
    print("Cases: " + str(len(cases)))
    print("ONNX SHA-256: " + payload["onnx_sha256"])
    print("Tokenizer SHA-256: " + payload["tokenizer_json_sha256"])
    print("Wrote: " + OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
