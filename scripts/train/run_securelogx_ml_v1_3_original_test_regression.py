#!/usr/bin/env python3
"""Evaluate the frozen ML-v1.3 BERT candidate on the historical original test set.

This is a post-selection regression check only. The original ML-v1 test set has
already been used historically, so it is not used for model selection or tuning.

This script:
- does NOT retrain
- does NOT change model selection
- does NOT read or rerun the sealed challenge
- does NOT tune thresholds
- does NOT export ONNX
- does NOT modify Java
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from securelogx_training_common import (  # noqa: E402
    build_aligned_features,
    load_canonical_labels,
    load_jsonl,
    sha256_file,
    write_json,
)
from train_securelogx_ml_v1_1 import _compact_metrics, _score_view  # noqa: E402
from train_securelogx_ml_v1_3_comparison import (  # noqa: E402
    label_bundle,
    security_view,
)

FREEZE_PATH = "configs/ml_v1_3_final_model_freeze.json"
TEST_PATH = "data/split/test.jsonl"
SEALED_MARKER = "reports/ml_v1_3_sealed_validation/SEALED_EVALUATION_COMPLETE"
RESULT_DIR = "reports/ml_v1_3_original_test_regression"
RESULT_JSON = "reports/ml_v1_3_original_test_regression/result.json"
RESULT_MD = "reports/ml_v1_3_original_test_regression/result.md"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    freeze = load_json(root / FREEZE_PATH)
    selected = freeze["selected_model"]
    if freeze.get("development_selection_final") is not True:
        raise ValueError("Final development selection is not frozen")
    if selected["model_key"] != "bert" or int(selected["epoch"]) != 3:
        raise ValueError("Frozen candidate is not BERT epoch 3")

    sealed_marker = root / SEALED_MARKER
    if not sealed_marker.exists():
        raise RuntimeError(
            "Expected completed sealed-validation marker before historical "
            "original-test regression."
        )

    checkpoint = root / selected["checkpoint_path"]
    model_path = checkpoint / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError("Frozen model is missing: " + str(model_path))
    model_sha = sha256_file(model_path)
    if model_sha != selected["expected_model_sha256"]:
        raise ValueError(
            "Frozen model SHA mismatch: "
            + model_sha
            + " != "
            + selected["expected_model_sha256"]
        )

    records = load_jsonl(root / TEST_PATH)
    labels = load_canonical_labels(root / "configs/securelogx_labels.json")
    entities = [str(value) for value in labels["entities"]]
    label_to_id = {str(key): int(value) for key, value in labels["label_to_id"].items()}
    id_to_label = {int(key): str(value) for key, value in labels["id_to_label"].items()}

    gate = load_json(root / "configs/ml_v1_3_model_comparison_gate.json")
    shared = gate["shared_training_configuration"]

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Frozen tokenizer is not fast")

    aligned = build_aligned_features(
        records,
        tokenizer,
        label_to_id,
        int(shared["max_length"]),
        int(shared["stride"]),
    )
    summary = aligned.summary
    if summary["truncated_spans"]:
        raise ValueError(
            "Original-test alignment has truncated spans: "
            + str(summary["truncated_spans"])
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to(device)

    metrics, predictions = _score_view(
        model,
        tokenizer,
        records,
        aligned.windows,
        entities,
        label_to_id,
        id_to_label,
        "ml_v1_3_original_test_regression",
        int(shared["eval_batch_size"]),
        device,
        use_bf16,
        False,
    )
    security = security_view("original_test_regression", records, predictions)
    bundle = label_bundle(metrics)

    payload = {
        "status": "ORIGINAL TEST REGRESSION COMPLETE",
        "purpose": "historical post-selection regression only",
        "model_selection_unchanged": True,
        "selected_model": selected,
        "actual_model_sha256": model_sha,
        "records": len(records),
        "alignment": {
            "total_spans": summary["total_spans"],
            "successfully_aligned_spans": summary["successfully_aligned_spans"],
            "boundary_adjusted_spans": summary["boundary_adjusted_spans"],
            "failed_alignments": summary["failed_alignments"],
            "truncated_spans": summary["truncated_spans"],
            "reason_counts": dict(summary.get("reason_counts", {})),
        },
        "metrics": _compact_metrics(metrics),
        "bundle": bundle,
        "security": security,
        "device": str(device),
        "bf16": use_bf16,
        "scope": {
            "training": False,
            "sealed_inference": False,
            "threshold_tuning": False,
            "onnx_export": False,
            "java_changes": False,
        },
    }

    out_dir = root / RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(root / RESULT_JSON, payload)

    lines = [
        "# ML-v1.3 Original-Test Regression",
        "",
        "**Historical regression only. Model selection remains frozen.**",
        "",
        "- Candidate: BERT epoch " + str(selected["epoch"]),
        "- Model SHA-256: " + model_sha,
        "- Records: **" + str(len(records)) + "**",
        "",
        "## Exact NER",
        "",
        "- Micro P/R/F1: **{:.6f} / {:.6f} / {:.6f}**".format(
            bundle["micro_p"], bundle["micro_r"], bundle["micro_f1"]
        ),
        "- Macro P/R/F1: **{:.6f} / {:.6f} / {:.6f}**".format(
            bundle["macro_p"], bundle["macro_r"], bundle["macro_f1"]
        ),
        "- High-risk recall: **{:.6f}**".format(bundle["high_risk_recall"]),
        "- BUSINESS_ID P/R/F1: **{:.6f} / {:.6f} / {:.6f}**".format(
            bundle["business_id_p"], bundle["business_id_r"], bundle["business_id_f1"]
        ),
        "- SSN P/R/F1: **{:.6f} / {:.6f} / {:.6f}**".format(
            bundle["ssn_p"], bundle["ssn_r"], bundle["ssn_f1"]
        ),
        "- ITIN P/R/F1: **{:.6f} / {:.6f} / {:.6f}**".format(
            bundle["itin_p"], bundle["itin_r"], bundle["itin_f1"]
        ),
        "- AUTH_TOKEN P/R/F1: **{:.6f} / {:.6f} / {:.6f}**".format(
            bundle["auth_token_p"], bundle["auth_token_r"], bundle["auth_token_f1"]
        ),
        "- API_KEY P/R/F1: **{:.6f} / {:.6f} / {:.6f}**".format(
            bundle["api_key_p"], bundle["api_key_r"], bundle["api_key_f1"]
        ),
        "",
        "## Security outcomes",
        "",
        "- Security-critical miss: **" + str(security["security_critical_miss"]) + "**",
        "- Security-critical partial: **" + str(security["security_critical_partial"]) + "**",
        "- Policy-safe wrong class: **" + str(security["policy_safe_wrong_class"]) + "**",
        "- Overmasking: **" + str(security["overmasking"]) + "**",
        "- Sensitive-span recall: **{:.6f}**".format(security["sensitive_span_recall"]),
        "- Full-mask recall: **{:.6f}**".format(security["full_mask_recall"]),
        "- High-risk full-mask recall: **{:.6f}**".format(
            security["high_risk_full_mask_recall"]
        ),
        "",
        "No sealed inference, retraining, threshold tuning, ONNX export, or Java changes occurred.",
    ]
    (root / RESULT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("ORIGINAL TEST REGRESSION COMPLETE")
    print("Micro F1: {:.6f}".format(bundle["micro_f1"]))
    print("Macro F1: {:.6f}".format(bundle["macro_f1"]))
    print(
        "Security miss/partial: {}/{}".format(
            security["security_critical_miss"],
            security["security_critical_partial"],
        )
    )
    print("Wrote: " + RESULT_JSON)
    print("Wrote: " + RESULT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
