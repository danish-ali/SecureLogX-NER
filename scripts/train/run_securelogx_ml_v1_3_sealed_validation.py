#!/usr/bin/env python3
"""Run the single post-selection ML-v1.3 sealed validation.

The selected BERT epoch-3 candidate is already frozen. This script regenerates
the sealed v1.1 challenge in memory, verifies its canonical SHA-256, evaluates
exactly once, and writes immutable result artifacts.

It does not retrain, tune thresholds, read the original test set, export ONNX,
or modify Java.
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
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from generate_securelogx_ml_v1_1_context_contrast import generate_partition  # noqa: E402
from securelogx_training_common import (  # noqa: E402
    build_aligned_features,
    load_canonical_labels,
    sha256_file,
    write_json,
)
from train_securelogx_ml_v1_1 import _compact_metrics, _score_view  # noqa: E402
from train_securelogx_ml_v1_2 import _canonical_jsonl_sha256  # noqa: E402
from train_securelogx_ml_v1_3_comparison import (  # noqa: E402
    label_bundle,
    security_view,
    verify_frozen_state,
)

FREEZE_PATH = "configs/ml_v1_3_final_model_freeze.json"
RESULT_DIR = "reports/ml_v1_3_sealed_validation"
RESULT_JSON = "reports/ml_v1_3_sealed_validation/result.json"
RESULT_MD = "reports/ml_v1_3_sealed_validation/result.md"
MARKER = "reports/ml_v1_3_sealed_validation/SEALED_EVALUATION_COMPLETE"


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

    marker = root / MARKER
    if marker.exists():
        raise RuntimeError(
            "Sealed validation has already been executed in this workspace. "
            + "Marker exists: "
            + str(marker)
        )

    freeze = load_json(root / FREEZE_PATH)
    if freeze.get("development_selection_final") is not True:
        raise ValueError("Final development selection is not frozen")

    selected = freeze["selected_model"]
    if selected["model_key"] != "bert" or int(selected["epoch"]) != 3:
        raise ValueError("Frozen candidate is not BERT epoch 3")
    if int(freeze["policy"]["sealed_evaluation_count"]) != 1:
        raise ValueError("Freeze policy does not require one sealed evaluation")

    evidence = verify_frozen_state(root)
    if evidence["dataset_manifest_sha256"] != freeze["frozen_dataset_manifest_sha256"]:
        raise ValueError("ML-v1.3 dataset manifest changed after final freeze")

    expected_sealed_sha = freeze["sealed_challenge_sha256"]
    parent_sealed = evidence["parent_result"]["sealed_challenge"]
    if parent_sealed["actual_sha256"] != expected_sealed_sha:
        raise ValueError("Historical sealed SHA mismatch")
    if parent_sealed["file_opened"] is not False:
        raise ValueError("Sealed challenge was opened before final model freeze")

    checkpoint = root / selected["checkpoint_path"]
    model_path = checkpoint / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError("Frozen model is missing: " + str(model_path))

    actual_model_sha = sha256_file(model_path)
    if actual_model_sha != selected["expected_model_sha256"]:
        raise ValueError(
            "Frozen model SHA mismatch: "
            + actual_model_sha
            + " != "
            + selected["expected_model_sha256"]
        )

    # Single intentional sealed opening. Generate in memory and verify exact
    # canonical bytes before inference.
    sealed_records = generate_partition("sealed_challenge", seed=42)
    actual_sealed_sha = _canonical_jsonl_sha256(sealed_records)
    if actual_sealed_sha != expected_sealed_sha:
        raise ValueError(
            "Regenerated sealed SHA mismatch: "
            + actual_sealed_sha
            + " != "
            + expected_sealed_sha
        )

    labels = load_canonical_labels(root / "configs/securelogx_labels.json")
    entities = [str(value) for value in labels["entities"]]
    label_to_id = {str(key): int(value) for key, value in labels["label_to_id"].items()}
    id_to_label = {int(key): str(value) for key, value in labels["id_to_label"].items()}

    shared = evidence["gate"]["shared_training_configuration"]
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Frozen tokenizer is not fast")

    aligned = build_aligned_features(
        sealed_records,
        tokenizer,
        label_to_id,
        int(shared["max_length"]),
        int(shared["stride"]),
    )
    summary = aligned.summary
    if summary["truncated_spans"]:
        raise ValueError(
            "Sealed alignment has truncated spans: "
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
        sealed_records,
        aligned.windows,
        entities,
        label_to_id,
        id_to_label,
        "ml_v1_3_final_sealed",
        int(shared["eval_batch_size"]),
        device,
        use_bf16,
        True,
    )
    security = security_view("sealed_final", sealed_records, predictions)
    bundle = label_bundle(metrics)

    payload = {
        "status": "SEALED VALIDATION COMPLETE",
        "selection_frozen_before_sealed": True,
        "model_selection_may_not_change_after_sealed_results": True,
        "selected_model": selected,
        "actual_model_sha256": actual_model_sha,
        "sealed": {
            "records": len(sealed_records),
            "expected_sha256": expected_sealed_sha,
            "actual_sha256": actual_sealed_sha,
            "hash_match": actual_sealed_sha == expected_sealed_sha,
        },
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
            "threshold_tuning": False,
            "original_test_inference": False,
            "onnx_export": False,
            "java_changes": False,
        },
    }

    out_dir = root / RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(root / RESULT_JSON, payload)

    lines = [
        "# ML-v1.3 Final Sealed Validation",
        "",
        "**SEALED VALIDATION COMPLETE**",
        "",
        "Model selection was frozen before this evaluation and may not change based on these results.",
        "",
        "- Candidate: BERT epoch " + str(selected["epoch"]) + " (" + selected["base_model"] + ")",
        "- Model SHA-256: " + actual_model_sha,
        "- Sealed records: **" + str(len(sealed_records)) + "**",
        "- Sealed SHA-256 verified: " + actual_sealed_sha,
        "",
        "## Sealed metrics",
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
        "No original-test inference, retraining, threshold tuning, ONNX export, or Java changes occurred.",
    ]
    (root / RESULT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")

    marker.write_text(
        json.dumps(
            {
                "status": "SEALED EVALUATION EXECUTED",
                "sealed_sha256": actual_sealed_sha,
                "model_sha256": actual_model_sha,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("SEALED VALIDATION COMPLETE")
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
