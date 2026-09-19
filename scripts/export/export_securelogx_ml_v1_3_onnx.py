#!/usr/bin/env python3
"""Export the frozen SecureLogX ML-v1.3 BERT candidate to ONNX and verify parity.

This phase is deployment validation only:
- no training
- no model/data changes
- no sealed-challenge inference
- no threshold tuning
- no Java changes

The script exports the frozen BERT epoch-3 checkpoint, validates the ONNX graph
with ONNX Runtime, and compares PyTorch vs ONNX on a deterministic development
sample at both logit and decoded-span levels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from securelogx_training_common import (  # noqa: E402
    build_aligned_features,
    decode_bio_spans,
    load_canonical_labels,
    load_jsonl,
    normalize_decoded_spans,
    sha256_file,
)

FREEZE_PATH = "configs/ml_v1_3_final_model_freeze.json"
GATE_PATH = "configs/ml_v1_3_model_comparison_gate.json"
STANDARD_DEV = "data/split/dev.jsonl"
CHALLENGE_DEV = "data/ml_v1_3/real_structure/dev_challenge.jsonl"
ORIGINAL_TEST = "data/split/test.jsonl"
SEALED_MARKER = "reports/ml_v1_3_sealed_validation/SEALED_EVALUATION_COMPLETE"

OUTPUT_DIR = "onnx-model/ml-v1.3/bert-base-cased"
MODEL_ONNX = "onnx-model/ml-v1.3/bert-base-cased/model.onnx"
MANIFEST_JSON = "reports/ml_v1_3_onnx_parity/manifest.json"
REPORT_JSON = "reports/ml_v1_3_onnx_parity/result.json"
REPORT_MD = "reports/ml_v1_3_onnx_parity/result.md"

SAMPLE_SIZE_PER_VIEW = 48
MAX_ABS_LOGIT_TOLERANCE = 1e-4
MEAN_ABS_LOGIT_TOLERANCE = 1e-5


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_sample(
    records: Sequence[Mapping[str, Any]],
    size: int,
) -> list[Mapping[str, Any]]:
    """Choose a stable spread across the source order without randomness."""
    if len(records) <= size:
        return list(records)
    if size <= 1:
        return [records[0]]
    indices = [
        round(i * (len(records) - 1) / (size - 1))
        for i in range(size)
    ]
    return [records[index] for index in indices]


class BertTokenClassificationForOnnx(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits


def export_onnx(
    model: Any,
    tokenizer: Any,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = BertTokenClassificationForOnnx(model).eval()

    probe = tokenizer(
        "INFO customerId=CUST-123 request completed successfully",
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    input_ids = probe["input_ids"]
    attention_mask = probe["attention_mask"]
    token_type_ids = probe.get("token_type_ids", torch.zeros_like(input_ids))

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, token_type_ids),
        output_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def decode_one(
    text: str,
    offsets: Sequence[Sequence[int]],
    logits: np.ndarray,
    id_to_label: Mapping[int, str],
) -> list[dict[str, Any]]:
    valid_offsets: list[tuple[int, int]] = []
    valid_logits: list[np.ndarray] = []
    for offset, token_logits in zip(offsets, logits):
        start, end = int(offset[0]), int(offset[1])
        if start == end:
            continue
        valid_offsets.append((start, end))
        valid_logits.append(token_logits)

    if not valid_logits:
        return []

    matrix = np.stack(valid_logits)
    shifted = matrix - np.max(matrix, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    ids = probs.argmax(axis=-1).tolist()
    conf = probs.max(axis=-1).tolist()
    spans, _ = decode_bio_spans(valid_offsets, ids, conf, id_to_label)
    return normalize_decoded_spans(text, spans)


def span_signature(spans: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, str]]:
    return sorted(
        (int(span["start"]), int(span["end"]), str(span["label"]))
        for span in spans
    )


def compare_view(
    name: str,
    records: Sequence[Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    session: Any,
    id_to_label: Mapping[int, str],
    max_length: int,
) -> dict[str, Any]:
    max_abs = 0.0
    abs_sum = 0.0
    element_count = 0
    exact_span_records = 0
    total_records = 0
    mismatches: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        text = str(record["text"])
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get(
            "token_type_ids",
            torch.zeros_like(input_ids),
        )

        with torch.no_grad():
            pt_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits.detach().cpu().numpy()

        ort_logits = session.run(
            ["logits"],
            {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
                "token_type_ids": token_type_ids.cpu().numpy().astype(np.int64),
            },
        )[0]

        diff = np.abs(pt_logits - ort_logits)
        max_abs = max(max_abs, float(diff.max(initial=0.0)))
        abs_sum += float(diff.sum())
        element_count += int(diff.size)

        pt_spans = decode_one(text, offsets, pt_logits[0], id_to_label)
        ort_spans = decode_one(text, offsets, ort_logits[0], id_to_label)
        pt_sig = span_signature(pt_spans)
        ort_sig = span_signature(ort_spans)
        total_records += 1
        if pt_sig == ort_sig:
            exact_span_records += 1
        elif len(mismatches) < 20:
            mismatches.append(
                {
                    "record_index": index,
                    "text_sha256": sha256_bytes(text.encode("utf-8")),
                    "pytorch": pt_sig,
                    "onnx": ort_sig,
                }
            )

    return {
        "view": name,
        "records": total_records,
        "exact_span_record_matches": exact_span_records,
        "exact_span_record_match_rate": (
            exact_span_records / total_records if total_records else 1.0
        ),
        "max_abs_logit_delta": max_abs,
        "mean_abs_logit_delta": (
            abs_sum / element_count if element_count else 0.0
        ),
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX parity requires Python packages 'onnx' and 'onnxruntime'. "
            "Install them in the active SecureLogX-NER environment and rerun."
        ) from exc

    freeze = load_json(root / FREEZE_PATH)
    selected = freeze["selected_model"]
    if freeze.get("development_selection_final") is not True:
        raise ValueError("Final model selection is not frozen")
    if selected["model_key"] != "bert" or int(selected["epoch"]) != 3:
        raise ValueError("Frozen candidate is not BERT epoch 3")

    if not (root / SEALED_MARKER).exists():
        raise RuntimeError("Sealed validation marker is missing")

    checkpoint = root / selected["checkpoint_path"]
    model_path = checkpoint / "model.safetensors"
    actual_model_sha = sha256_file(model_path)
    if actual_model_sha != selected["expected_model_sha256"]:
        raise ValueError("Frozen model SHA mismatch")

    labels = load_canonical_labels(root / "configs/securelogx_labels.json")
    id_to_label = {int(key): str(value) for key, value in labels["id_to_label"].items()}

    gate = load_json(root / GATE_PATH)
    max_length = int(gate["shared_training_configuration"]["max_length"])

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Frozen tokenizer is not fast")

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).cpu().eval()

    output_path = root / MODEL_ONNX
    if output_path.exists():
        output_path.unlink()
    export_onnx(model, tokenizer, output_path)

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    views = {
        "standard_dev": deterministic_sample(
            load_jsonl(root / STANDARD_DEV), SAMPLE_SIZE_PER_VIEW
        ),
        "v1_3_challenge_dev": deterministic_sample(
            load_jsonl(root / CHALLENGE_DEV), SAMPLE_SIZE_PER_VIEW
        ),
        "original_test_regression": deterministic_sample(
            load_jsonl(root / ORIGINAL_TEST), SAMPLE_SIZE_PER_VIEW
        ),
    }

    results = [
        compare_view(
            name,
            records,
            model,
            tokenizer,
            session,
            id_to_label,
            max_length,
        )
        for name, records in views.items()
    ]

    overall_max = max(item["max_abs_logit_delta"] for item in results)
    total_elements_weight = sum(item["records"] for item in results)
    overall_mean = (
        sum(item["mean_abs_logit_delta"] * item["records"] for item in results)
        / total_elements_weight
        if total_elements_weight
        else 0.0
    )
    span_match = all(
        item["exact_span_record_matches"] == item["records"] for item in results
    )
    passed = (
        span_match
        and overall_max <= MAX_ABS_LOGIT_TOLERANCE
        and overall_mean <= MEAN_ABS_LOGIT_TOLERANCE
    )

    output_sha = sha256_file(output_path)
    tokenizer_files = {}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ):
        source = checkpoint / name
        if source.exists():
            destination = root / OUTPUT_DIR / name
            shutil.copy2(source, destination)
            tokenizer_files[name] = sha256_file(destination)

    manifest = {
        "phase": "ML-v1.3 ONNX export",
        "frozen_model_sha256": actual_model_sha,
        "onnx_sha256": output_sha,
        "onnx_opset": 17,
        "inputs": ["input_ids", "attention_mask", "token_type_ids"],
        "output": "logits",
        "dynamic_batch": True,
        "dynamic_sequence": True,
        "ontology_entities": 25,
        "bio_labels": 51,
        "tokenizer_files": tokenizer_files,
        "decoder": "BIO decode plus surrounding-whitespace span normalization",
    }

    report = {
        "status": "ONNX PARITY PASSED" if passed else "ONNX PARITY FAILED",
        "passed": passed,
        "model_sha256": actual_model_sha,
        "onnx_sha256": output_sha,
        "max_abs_logit_tolerance": MAX_ABS_LOGIT_TOLERANCE,
        "mean_abs_logit_tolerance": MEAN_ABS_LOGIT_TOLERANCE,
        "overall_max_abs_logit_delta": overall_max,
        "overall_weighted_mean_abs_logit_delta": overall_mean,
        "exact_span_parity_all_records": span_match,
        "views": results,
        "scope": {
            "training": False,
            "sealed_inference": False,
            "threshold_tuning": False,
            "java_changes": False,
        },
    }

    report_dir = root / "reports/ml_v1_3_onnx_parity"
    report_dir.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_JSON).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / REPORT_JSON).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ML-v1.3 ONNX Export and Python Runtime Parity",
        "",
        "**" + report["status"] + "**",
        "",
        "- Frozen BERT SHA-256: " + actual_model_sha,
        "- ONNX SHA-256: " + output_sha,
        "- Opset: **17**",
        "- Dynamic batch/sequence axes: **yes**",
        "- Exact decoded-span parity across all sampled records: **"
        + str(span_match)
        + "**",
        "- Maximum absolute logit delta: **{:.10f}**".format(overall_max),
        "- Weighted mean absolute logit delta: **{:.10f}**".format(overall_mean),
        "",
        "| View | Records | Exact span parity | Max abs logit delta | Mean abs logit delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {} | {} | {:.2%} | {:.10f} | {:.10f} |".format(
                item["view"],
                item["records"],
                item["exact_span_record_match_rate"],
                item["max_abs_logit_delta"],
                item["mean_abs_logit_delta"],
            )
        )
    lines.extend(
        [
            "",
            "No training, sealed inference, threshold tuning, or Java changes occurred.",
        ]
    )
    (root / REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(report["status"])
    print("ONNX SHA-256: " + output_sha)
    print("Max abs logit delta: {:.10f}".format(overall_max))
    print("Exact decoded-span parity: " + str(span_match))
    print("Wrote: " + REPORT_JSON)
    print("Wrote: " + REPORT_MD)

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
