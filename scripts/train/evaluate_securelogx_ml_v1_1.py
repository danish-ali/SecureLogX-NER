"""Run the two protected final evaluations for SecureLogX ML-v1.1.

``sealed`` is the one permitted inference pass over the independent ML-v1.1
challenge.  ``regression`` is the subsequent, explicitly non-unseen benchmark
against the immutable ML-v1 test split.  Both phases acquire an exclusive
receipt immediately before inference; a STARTED, COMPLETE, or FAILED receipt
is terminal and prevents a second pass.

This program never trains, retunes, exports ONNX, or changes Java artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from securelogx_ml_v1_1_metrics import (
    context_diagnostics,
    evaluate_named_gates,
    supported_macro,
    tag_context_errors,
)
from securelogx_training_common import (
    build_aligned_features,
    evaluate_model,
    load_canonical_labels,
    load_json,
    load_jsonl,
    score_predictions,
    set_reproducible_seed,
    sha256_file,
    subset_score,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path("output_securelogx/ml-v1.1/bert-base-cased")
MODEL_PATH = RUN_ROOT / "best-checkpoint"
REPORTS = Path("reports")

DATASET_MANIFEST = Path("configs/ml_v1_1_dataset_manifest.json")
DATASET_MANIFEST_HASH = Path("configs/ml_v1_1_dataset_manifest.sha256")
PARENT_FREEZE = Path("configs/ml_v1_1_parent_freeze.json")
TRAINING_GATE = Path("configs/ml_v1_1_training_gate.json")
TRAINING_GATE_HASH = Path("configs/ml_v1_1_training_gate.sha256")
V1_GATE = Path("configs/ml_v1_onnx_validation_gate.json")
LABELS = Path("configs/securelogx_labels.json")
DEV_GATE = RUN_ROOT / "dev_gate_decision.json"

SEALED_INPUT = Path("data/ml_v1_1/challenge/sealed_test.jsonl")
REGRESSION_INPUT = Path("data/split/test.jsonl")
DEV_CHALLENGE_INPUT = Path("data/ml_v1_1/context_contrast/dev_challenge.jsonl")
DEV_CHALLENGE_CACHE = RUN_ROOT / "dev_challenge_predictions.json"

SEALED_RECEIPT = RUN_ROOT / "sealed_challenge_evaluation_receipt.json"
SEALED_CACHE = RUN_ROOT / "sealed_challenge_predictions.json"
REGRESSION_RECEIPT = RUN_ROOT / "regression_evaluation_receipt.json"
REGRESSION_CACHE = RUN_ROOT / "regression_test_predictions.json"

SEALED_METRICS = REPORTS / "ml_v1_1_sealed_challenge_metrics.json"
REGRESSION_METRICS = REPORTS / "ml_v1_1_regression_test_metrics.json"
BUSINESS_REPORT = REPORTS / "ml_v1_1_business_id_subtype_analysis.md"
SSN_REPORT = REPORTS / "ml_v1_1_ssn_context_analysis.md"
GENERALIZATION_REPORT = REPORTS / "ml_v1_1_generalization_analysis.md"
ERROR_REPORT = REPORTS / "ml_v1_1_error_analysis.md"
ERROR_CSV = REPORTS / "ml_v1_1_model_errors.csv"
COMPARISON_REPORT = REPORTS / "ml_v1_vs_v1_1_comparison.md"
DECISION_REPORT = REPORTS / "ml_v1_1_model_validation_decision.md"

EXPECTED_DATASET_MANIFEST_SHA256 = (
    "cf4a2c60c0b0fafa0a570b57af8335c8ca8fbefd4a9f84175e0da2cbc5b5ba63"
)
EXPECTED_TRAINING_GATE_SHA256 = (
    "cf9f217bbe08a81d79634f649d3a157c4b59a30d9f8168355621e2cdb4bfb273"
)
EXPECTED_V1_GATE_SHA256 = (
    "1d8249e1d4f052d0ba4cb64c607e6178d75f34ae160af17e0060907d02e56176"
)
EXPECTED_LABEL_SHA256 = (
    "60361655594004a36c154725a79d43f64896014e8f8b445552b57305b3ba24e5"
)

EVALUATION_SETTINGS = {"max_length": 384, "stride": 128, "batch_size": 8, "seed": 42}
CORE_TRAINING_SETTINGS = {
    "base_model": "bert-base-cased",
    "revision": "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
    "max_length": 384,
    "stride": 128,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "train_batch_size": 4,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 16,
    "seed": 42,
    "parent_train_records": 23_558,
    "addition_records": 3_000,
    "training_records": 26_558,
    "standard_dev_records": 2_994,
    "challenge_dev_records": 480,
    "smoke_test": False,
}
EXPECTED_COUNTS = {"sealed": 480, "regression": 2_989}
EXPLAINED_ALIGNMENT_FAILURES = frozenset(
    {
        "tokenizer_emitted_no_token_for_span",
        "token_boundary_adjustment_exceeds_three_characters",
    }
)
BUSINESS_SUBTYPES = (
    "CUSTOMER_ID",
    "ACCOUNT_ID",
    "APPLICATION_ID",
    "TRANSACTION_ID",
    "USER_ID",
    "ORDER_ID",
    "CASE_ID",
    "TICKET_ID",
    "INVOICE_ID",
    "REFERENCE_ID",
    "REQUEST_ID",
    "WORKFLOW_ID",
)


def _path(path: Path, root: Path) -> Path:
    return (root / path).resolve()


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json_value(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mapping_fingerprint(values: Mapping[str, str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _detached_hash(path: Path, expected_name: str) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or len(fields[0]) != 64 or fields[1] != expected_name:
        raise ValueError(f"Malformed detached SHA-256 declaration: {path}")
    try:
        int(fields[0], 16)
    except ValueError as exc:
        raise ValueError(f"Malformed detached SHA-256 declaration: {path}") from exc
    return fields[0].lower()


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _guard_absent(paths: Sequence[Path]) -> None:
    existing = [_display(path, REPOSITORY_ROOT) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Protected final-evaluation artifacts already exist; refusing a second pass: "
            + ", ".join(existing)
        )


def _verify_file(root: Path, relative: str, expected_sha256: str) -> str:
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen artifact is missing: {relative}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Frozen artifact SHA-256 mismatch for {relative}: "
            f"expected {expected_sha256}, received {actual}"
        )
    return actual


def _verify_jsonl_count(path: Path, expected: int) -> None:
    count = sum(1 for _ in path.open(encoding="utf-8"))
    if count != expected:
        raise ValueError(f"Expected {expected} records in {path}, received {count}")


def _verify_frozen_evidence(root: Path, *, include_sealed: bool) -> dict[str, Any]:
    """Verify every manifest-bound input without parsing sealed JSONL early."""

    manifest_path = _path(DATASET_MANIFEST, root)
    manifest_hash_path = _path(DATASET_MANIFEST_HASH, root)
    gate_path = _path(TRAINING_GATE, root)
    gate_hash_path = _path(TRAINING_GATE_HASH, root)
    parent_path = _path(PARENT_FREEZE, root)

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("ML-v1.1 dataset manifest differs from its frozen digest")
    if _detached_hash(manifest_hash_path, DATASET_MANIFEST.name) != manifest_sha:
        raise ValueError("ML-v1.1 dataset manifest detached digest does not match")

    gate_sha = sha256_file(gate_path)
    if gate_sha != EXPECTED_TRAINING_GATE_SHA256:
        raise ValueError("ML-v1.1 training gate differs from its predeclared digest")
    if _detached_hash(gate_hash_path, TRAINING_GATE.name) != gate_sha:
        raise ValueError("ML-v1.1 training-gate detached digest does not match")

    manifest = load_json(manifest_path)
    gate = load_json(gate_path)
    parent = load_json(parent_path)
    if gate.get("declared_before_training") is not True:
        raise ValueError("ML-v1.1 gates were not declared before training")
    frozen_inputs = gate.get("frozen_inputs", {})
    if frozen_inputs.get("ml_v1_1_dataset_manifest", {}).get("sha256") != manifest_sha:
        raise ValueError("Training gate is not bound to the frozen ML-v1.1 manifest")
    if frozen_inputs.get("ml_v1_onnx_validation_gate", {}).get("sha256") != EXPECTED_V1_GATE_SHA256:
        raise ValueError("Training gate is not bound to the original ML-v1 gate")
    if frozen_inputs.get("ontology", {}).get("sha256") != EXPECTED_LABEL_SHA256:
        raise ValueError("Training gate is not bound to the canonical ontology")

    if manifest.get("dataset_version") != "ML-v1.1":
        raise ValueError("Unexpected ML-v1.1 manifest dataset version")
    if manifest.get("sealed_challenge", {}).get("development_predictions_forbidden") is not True:
        raise ValueError("Manifest does not preserve the sealed-development boundary")
    if manifest.get("sealed_challenge", {}).get("model_predictions_run") is not False:
        raise ValueError("Frozen manifest indicates prior sealed predictions")

    _verify_file(root, V1_GATE.as_posix(), EXPECTED_V1_GATE_SHA256)
    _verify_file(root, LABELS.as_posix(), EXPECTED_LABEL_SHA256)
    _verify_file(root, manifest["parent"]["manifest_path"], manifest["parent"]["manifest_sha256"])
    _verify_file(root, manifest["parent"]["snapshot_path"], manifest["parent"]["snapshot_sha256"])

    parent_entries: list[tuple[str, Mapping[str, Any]]] = [
        (parent["parent_manifest"]["path"], parent["parent_manifest"]),
        (parent["ontology"]["path"], parent["ontology"]),
    ]
    for section in ("parent_configs", "parent_splits", "evaluation_evidence"):
        parent_entries.extend(parent[section].items())
    for relative, details in parent_entries:
        _verify_file(root, relative, str(details["sha256"]))

    checkpoint = parent["checkpoint"]
    checkpoint_root = (root / checkpoint["path"]).resolve()
    actual_names = sorted(path.name for path in checkpoint_root.iterdir() if path.is_file())
    if actual_names != sorted(checkpoint["files"]):
        raise ValueError("Frozen ML-v1 checkpoint file set changed")
    parent_checkpoint_hashes = {
        name: _verify_file(root, f"{checkpoint['path']}/{name}", expected)
        for name, expected in sorted(checkpoint["files"].items())
    }
    if _mapping_fingerprint(parent_checkpoint_hashes) != checkpoint["fingerprint_sha256"]:
        raise ValueError("Frozen ML-v1 checkpoint fingerprint changed")

    for relative, details in manifest["new_data_artifacts"].items():
        if relative == SEALED_INPUT.as_posix() and not include_sealed:
            continue
        _verify_file(root, relative, str(details["sha256"]))
    for relative, details in manifest["reports"].items():
        _verify_file(root, relative, str(details["sha256"]))
    _verify_file(root, manifest["generation"]["profile_path"], manifest["generation"]["profile_sha256"])
    for relative, expected in manifest["generation"]["script_sha256"].items():
        _verify_file(root, relative, expected)

    return {
        "dataset_manifest_sha256": manifest_sha,
        "training_gate_sha256": gate_sha,
        "parent_freeze_sha256": sha256_file(parent_path),
        "v1_gate_sha256": EXPECTED_V1_GATE_SHA256,
        "label_config_sha256": EXPECTED_LABEL_SHA256,
        "parent_checkpoint_fingerprint_sha256": checkpoint["fingerprint_sha256"],
        "manifest": manifest,
        "gate": gate,
        "parent": parent,
    }


def _validate_checkpoint(root: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    model_path = _path(MODEL_PATH, root)
    required = (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "training_args.json",
        "environment.json",
        "epoch_metrics.json",
        "selected_checkpoint.json",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Selected ML-v1.1 checkpoint is incomplete: {missing}")

    arguments = load_json(model_path / "training_args.json")
    mismatches = [
        f"{name}: expected {expected!r}, received {arguments.get(name)!r}"
        for name, expected in CORE_TRAINING_SETTINGS.items()
        if arguments.get(name) != expected
    ]
    if arguments.get("dataset_manifest_sha256") != evidence["dataset_manifest_sha256"]:
        mismatches.append("dataset_manifest_sha256 does not match frozen evidence")
    if arguments.get("training_gate_sha256") != evidence["training_gate_sha256"]:
        mismatches.append("training_gate_sha256 does not match frozen evidence")
    if mismatches:
        raise ValueError("ML-v1.1 checkpoint training metadata mismatch: " + "; ".join(mismatches))

    selected = load_json(model_path / "selected_checkpoint.json")
    selected_epoch = selected.get("selected_epoch")
    if selected_epoch not in {1, 2, 3}:
        raise ValueError(f"Invalid selected ML-v1.1 epoch: {selected_epoch!r}")
    if "harmonic mean" not in str(selected.get("selection_metric", "")).lower():
        raise ValueError("ML-v1.1 checkpoint was not selected by the predeclared harmonic mean")
    if selected.get("tie_break") != "earlier epoch":
        raise ValueError("ML-v1.1 checkpoint does not preserve the predeclared tie break")

    declared_hashes = selected.get("artifact_sha256")
    if not isinstance(declared_hashes, dict) or "model.safetensors" not in declared_hashes:
        raise ValueError("Selected checkpoint lacks its frozen artifact hash mapping")
    actual_declared_hashes = {
        name: sha256_file(model_path / name)
        for name in sorted(declared_hashes)
        if (model_path / name).is_file()
    }
    if actual_declared_hashes != declared_hashes:
        raise ValueError("Selected checkpoint artifact hashes changed after selection")
    selection_fingerprint = _mapping_fingerprint(actual_declared_hashes)
    if selection_fingerprint != selected.get("checkpoint_fingerprint_sha256"):
        raise ValueError("Selected checkpoint fingerprint disagrees with its artifacts")
    if selected.get("model_sha256") != actual_declared_hashes["model.safetensors"]:
        raise ValueError("Selected checkpoint model hash disagrees with model.safetensors")

    source = (root / str(selected.get("source_checkpoint", ""))).resolve()
    expected_source = _path(RUN_ROOT / "checkpoints" / f"epoch-{selected_epoch}", root)
    if source != expected_source or not source.is_dir():
        raise ValueError("Selected checkpoint source epoch is not canonical")
    source_mismatch = [
        name
        for name, expected in actual_declared_hashes.items()
        if not (source / name).is_file() or sha256_file(source / name) != expected
    ]
    if source_mismatch:
        raise ValueError("Selected checkpoint differs from its source epoch: " + ", ".join(source_mismatch))

    history_path = _path(RUN_ROOT / "training_history.json", root)
    history = _load_json_value(history_path)
    if not isinstance(history, list) or [row.get("epoch") for row in history] != [1, 2, 3]:
        raise ValueError("ML-v1.1 training history must contain exactly epochs 1, 2, and 3")
    selection_value = float(selected["selection_value"])
    if not math.isclose(
        float(history[selected_epoch - 1]["selection_score"]),
        selection_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Selected checkpoint score disagrees with training history")

    full_hashes = {name: sha256_file(model_path / name) for name in required}
    return {
        "selected_epoch": selected_epoch,
        "selection_metric": selected["selection_metric"],
        "selection_value": selection_value,
        "source_checkpoint": _display(source, root),
        "model_sha256": full_hashes["model.safetensors"],
        "selection_checkpoint_fingerprint_sha256": selection_fingerprint,
        "full_artifact_fingerprint_sha256": _mapping_fingerprint(full_hashes),
        "artifact_sha256": full_hashes,
        "training_history_sha256": sha256_file(history_path),
    }


def _validate_dev_gate(root: Path, evidence: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    path = _path(DEV_GATE, root)
    gate_result = load_json(path)
    if gate_result.get("status") != "PASS" or gate_result.get("decision") != "READY FOR SEALED CHALLENGE EVALUATION":
        raise ValueError("Development gate did not authorize sealed challenge evaluation")
    expected = {
        "gate_sha256": evidence["training_gate_sha256"],
        "dataset_manifest_sha256": evidence["dataset_manifest_sha256"],
        "checkpoint_fingerprint_sha256": checkpoint["selection_checkpoint_fingerprint_sha256"],
        "selected_epoch": checkpoint["selected_epoch"],
    }
    mismatches = [
        f"{name}: expected {value!r}, received {gate_result.get(name)!r}"
        for name, value in expected.items()
        if gate_result.get(name) != value
    ]
    if mismatches:
        raise ValueError("Development gate provenance mismatch: " + "; ".join(mismatches))
    if not gate_result.get("gates") or not all(
        item.get("passed") is True for item in gate_result["gates"].values()
    ):
        raise ValueError("Development gate contains a failed or missing predeclared gate")
    return {"path": DEV_GATE.as_posix(), "sha256": sha256_file(path), **gate_result}


def _alignment_blockers(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    blockers = [
        item
        for item in summary.get("diagnostics", [])
        if item.get("status") == "failed"
        and item.get("reason") not in EXPLAINED_ALIGNMENT_FAILURES
    ]
    if summary.get("truncated_spans"):
        blockers.append({"reason": "truncated_entity_spans", "count": summary["truncated_spans"]})
    return blockers


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"errors", "confusion"}}


def _score(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
    split: str,
) -> dict[str, Any]:
    metrics = score_predictions(records, predictions, labels, split)
    metrics["supported_macro"] = supported_macro(metrics)
    return metrics


def _context_groups(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
    field: str,
    split: str,
) -> dict[str, Any]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        value = str(record.get("meta", {}).get(field) or "<UNSPECIFIED>")
        indices[value].append(index)
    result: dict[str, Any] = {}
    for value, selected in sorted(indices.items()):
        view_records = [records[index] for index in selected]
        view_predictions = [predictions[index] for index in selected]
        metrics = _score(view_records, view_predictions, labels, f"{split}:{value}")
        diagnostics = context_diagnostics(
            view_records, view_predictions, labels, f"{split}:{value}"
        )
        result[value] = {
            "score": _compact(metrics),
            "record_error_rate": diagnostics["record_error_rate"],
            "context_target_error_rate": diagnostics["context_target_error_rate"],
            "morphology_over_context_error_rate": diagnostics[
                "morphology_over_context_error_rate"
            ],
        }
    return result


def _business_subtype(record: Mapping[str, Any], start: int, end: int) -> str:
    for entry in record.get("meta", {}).get("entity_provenance", []):
        if (
            int(entry.get("start", -1)) == start
            and int(entry.get("end", -1)) == end
            and entry.get("label") == "BUSINESS_ID"
        ):
            return str(
                entry.get("business_id_subtype")
                or entry.get("source_subtype")
                or entry.get("source_label")
                or "BUSINESS_ID"
            ).upper()
    return "BUSINESS_ID"


def _business_analysis(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    support: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    for record, predicted in zip(records, predictions):
        signatures = {
            (int(span["start"]), int(span["end"]), str(span["label"]))
            for span in predicted
        }
        for start, end, label in record["entities"]:
            if label != "BUSINESS_ID":
                continue
            subtype = _business_subtype(record, int(start), int(end))
            support[subtype] += 1
            if (int(start), int(end), "BUSINESS_ID") in signatures:
                true_positive[subtype] += 1
    rows = []
    for subtype in BUSINESS_SUBTYPES:
        gold = support[subtype]
        tp = true_positive[subtype]
        rows.append(
            {
                "subtype": subtype,
                "support": gold,
                "true_positives": tp,
                "false_negatives": gold - tp,
                "exact_span_recall": tp / gold if gold else None,
            }
        )
    extra = sorted(set(support) - set(BUSINESS_SUBTYPES))
    for subtype in extra:
        gold = support[subtype]
        tp = true_positive[subtype]
        rows.append(
            {
                "subtype": subtype,
                "support": gold,
                "true_positives": tp,
                "false_negatives": gold - tp,
                "exact_span_recall": tp / gold if gold else None,
            }
        )
    return {"support": sum(support.values()), "true_positives": sum(true_positive.values()), "subtypes": rows}


def _overlap(start: int, end: int, span: Mapping[str, Any]) -> bool:
    return int(span["start"]) < end and int(span["end"]) > start


def _ssn_analysis(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record, predicted in zip(records, predictions):
        gold = [
            {"start": int(start), "end": int(end), "label": str(label)}
            for start, end, label in record["entities"]
        ]
        exact_predictions = {
            (int(span["start"]), int(span["end"]), str(span["label"]))
            for span in predicted
        }
        ssn_gold = [span for span in gold if span["label"] == "SSN"]
        business_gold = [span for span in gold if span["label"] == "BUSINESS_ID"]
        counts["true_ssn_support"] += len(ssn_gold)
        for span in ssn_gold:
            signature = (span["start"], span["end"], "SSN")
            if signature in exact_predictions:
                counts["true_ssns_correctly_recognized"] += 1
            if any(
                prediction["label"] == "BUSINESS_ID"
                and _overlap(span["start"], span["end"], prediction)
                for prediction in predicted
            ):
                counts["ssns_missed_as_business_id"] += 1
            if any(
                prediction["label"] == "SSN"
                and _overlap(span["start"], span["end"], prediction)
                and (
                    int(prediction["start"]), int(prediction["end"]), "SSN"
                )
                != signature
                for prediction in predicted
            ):
                counts["ssn_boundary_errors"] += 1
        for prediction in (span for span in predicted if span["label"] == "SSN"):
            if any(
                int(prediction["start"]) == span["start"]
                and int(prediction["end"]) == span["end"]
                for span in ssn_gold
            ):
                continue
            if any(
                _overlap(span["start"], span["end"], prediction)
                for span in ssn_gold
            ):
                continue
            if any(
                _overlap(span["start"], span["end"], prediction)
                for span in business_gold
            ):
                counts["business_id_values_incorrectly_classified_as_ssn"] += 1
            else:
                counts["other_non_ssn_values_incorrectly_classified_as_ssn"] += 1
    return dict(counts)


def _prediction_payload(path: Path) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    payload = load_json(path)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Prediction cache has no prediction list: {path}")
    return payload, predictions


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _write_tagged_errors(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "evaluation_view",
        "split",
        "record_index",
        "record_id",
        "source",
        "template_family",
        "error_category",
        "context_error_tags",
        "contrast_category",
        "context_role",
        "morphology_class",
        "business_id_subtype",
        "text",
        "gold_entity",
        "predicted_entity",
        "gold_span",
        "predicted_span",
        "gold_text",
        "predicted_text",
        "prediction_confidence",
        "gold_label_probability",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            if isinstance(serialized.get("context_error_tags"), list):
                serialized["context_error_tags"] = ",".join(serialized["context_error_tags"])
            writer.writerow(serialized)


def _load_model_and_data(
    root: Path,
    input_path: Path,
    expected_records: int,
    label_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, Any, list[dict[str, Any]], Any, torch.device, bool]:
    model_path = _path(MODEL_PATH, root)
    set_reproducible_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    if device.type != "cuda" or not use_bf16:
        raise RuntimeError("Final ML-v1.1 evaluation requires canonical CUDA/BF16 execution")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("Final evaluation requires the saved fast tokenizer")
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    label_to_id = {str(key): int(value) for key, value in label_config["label_to_id"].items()}
    id_to_label = {int(key): str(value) for key, value in label_config["id_to_label"].items()}
    checkpoint_id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    if checkpoint_id2label != id_to_label or model.config.label2id != label_to_id:
        raise ValueError("ML-v1.1 checkpoint labels differ from the canonical 51-label mapping")
    model.to(device)
    records = load_jsonl(input_path)
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} records, received {len(records)}")
    build = build_aligned_features(records, tokenizer, label_to_id, args.max_length, args.stride)
    blockers = _alignment_blockers(build.summary)
    if blockers:
        raise ValueError(f"Final-evaluation alignment blockers: {blockers[:10]}")
    return model, tokenizer, records, build, device, use_bf16


def _base_receipt(
    *,
    phase: str,
    root: Path,
    input_path: Path,
    output_cache: Path,
    checkpoint: Mapping[str, Any],
    evidence: Mapping[str, Any],
    args: argparse.Namespace,
    records: int,
    use_bf16: bool,
    device: torch.device,
) -> dict[str, Any]:
    settings = {
        "max_length": args.max_length,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "bf16": use_bf16,
    }
    return {
        "receipt_version": 1,
        "status": "STARTED",
        "phase": phase,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "input": _display(input_path, root),
        "input_sha256": sha256_file(input_path),
        "records": records,
        "model": MODEL_PATH.as_posix(),
        "model_sha256": checkpoint["model_sha256"],
        "checkpoint_fingerprint_sha256": checkpoint[
            "selection_checkpoint_fingerprint_sha256"
        ],
        "prediction_cache": _display(output_cache, root),
        "configuration": settings,
        "configuration_sha256": hashlib.sha256(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "provenance": {
            "dataset_manifest_sha256": evidence["dataset_manifest_sha256"],
            "training_gate_sha256": evidence["training_gate_sha256"],
            "v1_gate_sha256": evidence["v1_gate_sha256"],
            "label_config_sha256": evidence["label_config_sha256"],
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def _run_inference(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    build: Any,
    label_config: Mapping[str, Any],
    cache_path: Path,
    input_path: Path,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
    use_bf16: bool,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    id_to_label = {int(key): str(value) for key, value in label_config["id_to_label"].items()}
    label_to_id = {str(key): int(value) for key, value in label_config["label_to_id"].items()}
    started = time.perf_counter()
    evaluation = evaluate_model(
        model,
        tokenizer,
        records,
        build.windows,
        id_to_label,
        args.batch_size,
        device,
        use_bf16,
    )
    inference_seconds = time.perf_counter() - started
    write_json(
        cache_path,
        {
            "split": split,
            "input_sha256": sha256_file(input_path),
            "records": len(records),
            "model_sha256": provenance["model_sha256"],
            "checkpoint_fingerprint_sha256": provenance[
                "selection_checkpoint_fingerprint_sha256"
            ],
            "predictions": evaluation["predictions"],
            "loss": evaluation["loss"],
            "invalid_bio_transitions": evaluation["invalid_bio_transitions"],
            "inference_seconds": inference_seconds,
        },
    )
    metrics = score_predictions(
        records,
        evaluation["predictions"],
        label_config["entities"],
        split,
        token_details=evaluation["token_details"],
        label_to_id=label_to_id,
    )
    metrics["supported_macro"] = supported_macro(metrics)
    metrics["loss"] = evaluation["loss"]
    metrics["invalid_bio_transitions"] = evaluation["invalid_bio_transitions"]
    metrics["inference_seconds"] = inference_seconds
    return evaluation, metrics


def _sealed_actual(metrics: Mapping[str, Any], context: Mapping[str, Any], alignment: Mapping[str, Any]) -> dict[str, Any]:
    per_label = metrics["per_label"]
    return {
        "micro_f1": metrics["micro"]["f1"],
        "supported_macro_f1": metrics["supported_macro"]["f1"],
        "business_id_recall": per_label["BUSINESS_ID"]["recall"],
        "ssn_recall": per_label["SSN"]["recall"],
        "record_error_rate": context["record_error_rate"],
        "context_target_error_rate": context["context_target_error_rate"],
        "morphology_over_context_error_rate": context[
            "morphology_over_context_error_rate"
        ],
        "maximum_category_error_rate": context["maximum_category_error_rate"],
        "unexplained_alignment_failures": len(_alignment_blockers(alignment)),
        "truncated_entity_spans": alignment["truncated_spans"],
    }


def _run_sealed(args: argparse.Namespace, root: Path) -> int:
    receipt_path = _path(SEALED_RECEIPT, root)
    cache_path = _path(SEALED_CACHE, root)
    metrics_path = _path(SEALED_METRICS, root)
    protected = [receipt_path, cache_path, metrics_path]
    _guard_absent(protected)

    # Do not even hash sealed bytes until the development-only gate is verified.
    evidence = _verify_frozen_evidence(root, include_sealed=False)
    checkpoint = _validate_checkpoint(root, evidence)
    dev_gate = _validate_dev_gate(root, evidence, checkpoint)
    sealed_details = evidence["manifest"]["new_data_artifacts"][SEALED_INPUT.as_posix()]
    input_path = _path(SEALED_INPUT, root)
    _verify_file(root, SEALED_INPUT.as_posix(), sealed_details["sha256"])
    _verify_jsonl_count(input_path, EXPECTED_COUNTS["sealed"])

    label_config = load_canonical_labels(_path(LABELS, root))
    model, tokenizer, records, build, device, use_bf16 = _load_model_and_data(
        root, input_path, EXPECTED_COUNTS["sealed"], label_config, args
    )
    receipt = _base_receipt(
        phase="sealed_challenge",
        root=root,
        input_path=input_path,
        output_cache=cache_path,
        checkpoint=checkpoint,
        evidence=evidence,
        args=args,
        records=len(records),
        use_bf16=use_bf16,
        device=device,
    )
    receipt["development_gate"] = {
        "path": DEV_GATE.as_posix(),
        "sha256": dev_gate["sha256"],
        "decision": dev_gate["decision"],
    }

    # Terminal exactly-once lock: it is created immediately before sole inference.
    _guard_absent(protected)
    _exclusive_json(receipt_path, receipt)
    try:
        evaluation, metrics = _run_inference(
            model=model,
            tokenizer=tokenizer,
            records=records,
            build=build,
            label_config=label_config,
            cache_path=cache_path,
            input_path=input_path,
            split="ml_v1_1_sealed_challenge",
            args=args,
            device=device,
            use_bf16=use_bf16,
            provenance=checkpoint,
        )
        diagnostics = context_diagnostics(
            records,
            evaluation["predictions"],
            label_config["entities"],
            "ml_v1_1_sealed_challenge",
        )
        actual = _sealed_actual(metrics, diagnostics, build.summary)
        acceptance = evaluate_named_gates(
            actual,
            evidence["gate"]["sealed_challenge_acceptance"]["all_required"],
        )
        payload = {
            "evaluation_label": "SEALED ML-v1.1 CHALLENGE - SINGLE FINAL PASS",
            "overall": _compact(metrics),
            "context_diagnostics": diagnostics,
            "per_family": _context_groups(
                records,
                evaluation["predictions"],
                label_config["entities"],
                "template_family",
                "ml_v1_1_sealed_challenge",
            ),
            "per_conflict_category": _context_groups(
                records,
                evaluation["predictions"],
                label_config["entities"],
                "contrast_category",
                "ml_v1_1_sealed_challenge",
            ),
            "business_id_subtypes": _business_analysis(records, evaluation["predictions"]),
            "ssn_context": _ssn_analysis(records, evaluation["predictions"]),
            "alignment": build.summary,
            "acceptance": acceptance,
            "provenance": receipt["provenance"],
        }
        write_json(metrics_path, payload)
        # Recheck all frozen evidence after result generation.
        _verify_frozen_evidence(root, include_sealed=True)
        results = {
            _display(cache_path, root): sha256_file(cache_path),
            _display(metrics_path, root): sha256_file(metrics_path),
        }
        receipt.update(
            {
                "status": "COMPLETE",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "single_inference_pass": True,
                "inference_seconds": metrics["inference_seconds"],
                "acceptance_passed": acceptance["all_passed"],
                "result_sha256": results,
            }
        )
        _replace_json(receipt_path, receipt)
    except BaseException as exc:
        receipt.update(
            {
                "status": "FAILED",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        try:
            _replace_json(receipt_path, receipt)
        except Exception:
            pass
        raise

    print(
        "sealed_challenge "
        f"micro_f1={metrics['micro']['f1']:.6f} "
        f"supported_macro_f1={metrics['supported_macro']['f1']:.6f} "
        f"record_error_rate={diagnostics['record_error_rate']:.6f}",
        flush=True,
    )
    return 0


def _validate_sealed_receipt(root: Path, checkpoint: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    path = _path(SEALED_RECEIPT, root)
    receipt = load_json(path)
    if receipt.get("status") != "COMPLETE" or receipt.get("single_inference_pass") is not True:
        raise ValueError("Regression phase requires a COMPLETE one-pass sealed receipt")
    expected = {
        "input_sha256": evidence["manifest"]["new_data_artifacts"][SEALED_INPUT.as_posix()]["sha256"],
        "model_sha256": checkpoint["model_sha256"],
        "checkpoint_fingerprint_sha256": checkpoint["selection_checkpoint_fingerprint_sha256"],
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"Sealed receipt {name} does not match the frozen regression model/input")
    results = receipt.get("result_sha256")
    if not isinstance(results, dict) or not results:
        raise ValueError("Sealed receipt does not bind result files")
    for relative, expected_hash in results.items():
        _verify_file(root, relative, expected_hash)
    return {"sha256": sha256_file(path), **receipt}


def _record_error_rate(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> float:
    errors = 0
    for record, predicted in zip(records, predictions):
        gold = Counter((int(start), int(end), str(label)) for start, end, label in record["entities"])
        guess = Counter(
            (int(span["start"]), int(span["end"]), str(span["label"]))
            for span in predicted
        )
        errors += gold != guess
    return errors / len(records) if records else 0.0


def _record_false_positive_rate(metrics: Mapping[str, Any], record_count: int) -> float:
    """Match the immutable ML-v1 gate's record-level FP/error definition."""

    records_with_nonexact_prediction = {
        int(row["record_index"])
        for row in metrics["errors"]
        if row.get("predicted_entity")
    }
    return len(records_with_nonexact_prediction) / record_count if record_count else 0.0


def _segment(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
    selected: Sequence[int],
    split: str,
) -> dict[str, Any]:
    metrics = subset_score(records, predictions, labels, split, selected)
    metrics["supported_macro"] = supported_macro(metrics)
    return metrics


def _write_business_report(path: Path, views: Mapping[str, Mapping[str, Any]]) -> None:
    lines = [
        "# ML-v1.1 BUSINESS_ID Subtype Analysis",
        "",
        "The classifier still emits only `BUSINESS_ID`. Subtypes are recovered from frozen gold provenance, so subtype recall is well-defined; subtype precision/F1 are not assigned to predictions without gold subtype provenance.",
        "",
    ]
    for name, analysis in views.items():
        lines.extend(
            [
                f"## {name}",
                "",
                "| Subtype | Support | Exact TP | FN | Exact-span recall |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in analysis["subtypes"]:
            lines.append(
                f"| {row['subtype']} | {row['support']} | {row['true_positives']} | "
                f"{row['false_negatives']} | {_pct(row['exact_span_recall'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "ORDER_ID and REFERENCE_ID are directly challenged in the sealed view. ACCOUNT_ID and WORKFLOW_ID are represented in the development challenge; their results are reported from the frozen selected-checkpoint development cache and were not recomputed after sealed evaluation.",
        ]
    )
    _write_markdown(path, lines)


def _write_ssn_report(path: Path, current: Mapping[str, int], baseline: Mapping[str, int], sealed: Mapping[str, int]) -> None:
    keys = (
        "true_ssn_support",
        "true_ssns_correctly_recognized",
        "business_id_values_incorrectly_classified_as_ssn",
        "other_non_ssn_values_incorrectly_classified_as_ssn",
        "ssns_missed_as_business_id",
        "ssn_boundary_errors",
    )
    lines = [
        "# ML-v1.1 SSN Morphology-Conflict Analysis",
        "",
        "| Diagnostic | ML-v1 historical regression | ML-v1.1 regression | ML-v1.1 sealed challenge |",
        "|---|---:|---:|---:|",
    ]
    for key in keys:
        lines.append(
            f"| {key.replace('_', ' ')} | {baseline.get(key, 0)} | {current.get(key, 0)} | {sealed.get(key, 0)} |"
        )
    lines.extend(
        [
            "",
            "Counts distinguish semantic BUSINESS_ID-versus-SSN confusions from ordinary spurious SSN predictions and from SSN boundary errors. This prevents an apparent gain produced only by suppressing SSN output.",
        ]
    )
    _write_markdown(path, lines)


def _write_generalization_report(
    path: Path,
    gretel: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    hard_negative: Mapping[str, Any],
    hard_negative_error_rate: float,
    dev_challenge: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> None:
    dev_view = dev_challenge.get("overall", dev_challenge)
    sealed_view = sealed["overall"]
    lines = [
        "# ML-v1.1 Generalization Analysis",
        "",
        "These segments are intentionally separate; they answer different generalization questions and are not combined into one headline score.",
        "",
        "| Generalization view | Interpretation | Micro F1 | Supported macro F1 | Record error rate |",
        "|---|---|---:|---:|---:|",
        f"| Original Gretel records | External/public-source generalization | {gretel['micro']['f1']:.6f} | {gretel['supported_macro']['f1']:.6f} | N/A |",
        f"| Original synthetic unseen templates | Synthetic generalization | {synthetic['micro']['f1']:.6f} | {synthetic['supported_macro']['f1']:.6f} | N/A |",
        f"| ML-v1.1 dev challenge | Development context-conflict robustness | {dev_view['micro']['f1']:.6f} | {dev_view['supported_macro']['f1']:.6f} | {dev_view['context_diagnostics']['record_error_rate']:.6f} |",
        f"| Historical hard-negative family | Known-failure regression | {hard_negative['micro']['f1']:.6f} | {hard_negative['supported_macro']['f1']:.6f} | {hard_negative_error_rate:.6f} |",
        f"| ML-v1.1 sealed challenge | Independent context-conflict robustness | {sealed_view['micro']['f1']:.6f} | {sealed_view['supported_macro']['f1']:.6f} | {sealed['context_diagnostics']['record_error_rate']:.6f} |",
        "",
        "The Gretel result remains subject to `license_reviewed=false`; it is technical evidence, not production/commercial release authorization.",
    ]
    _write_markdown(path, lines)


def _write_error_report(path: Path, regression: Mapping[str, Any], sealed: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    tag_counts: Counter[str] = Counter()
    for row in rows:
        tag_counts.update(row.get("context_error_tags", []))
    lines = [
        "# ML-v1.1 Error Analysis",
        "",
        "## Regression benchmark - not unseen test",
        "",
        *(f"- {name.replace('_', ' ')}: **{value}**" for name, value in regression["error_counts"].items()),
        "",
        "## Sealed context challenge",
        "",
        *(f"- {name.replace('_', ' ')}: **{value}**" for name, value in sealed["overall"]["error_counts"].items()),
        "",
        "## Context-conflict tags",
        "",
        *(f"- {name}: **{count}**" for name, count in sorted(tag_counts.items())),
        "",
        "Detailed, span-level rows are in `reports/ml_v1_1_model_errors.csv`.",
    ]
    _write_markdown(path, lines)


def _write_comparison_report(
    path: Path,
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    baseline_hard_error: float,
    current_hard_error: float,
) -> None:
    old = baseline["overall"]
    rows = [
        ("Original-test micro F1", old["micro"]["f1"], current["overall"]["micro"]["f1"]),
        ("Original-test macro F1", old["macro"]["f1"], current["overall"]["macro"]["f1"]),
        ("High-risk recall", old["high_risk"]["recall"], current["overall"]["high_risk"]["recall"]),
        ("BUSINESS_ID F1", old["per_label"]["BUSINESS_ID"]["f1"], current["overall"]["per_label"]["BUSINESS_ID"]["f1"]),
        ("BUSINESS_ID recall", old["per_label"]["BUSINESS_ID"]["recall"], current["overall"]["per_label"]["BUSINESS_ID"]["recall"]),
        ("SSN F1", old["per_label"]["SSN"]["f1"], current["overall"]["per_label"]["SSN"]["f1"]),
        ("Gretel-only micro F1", baseline["segments"]["gretel"]["micro"]["f1"], current["segments"]["gretel"]["micro"]["f1"]),
        ("Synthetic unseen-template micro F1", baseline["segments"]["synthetic_unseen_template"]["micro"]["f1"], current["segments"]["synthetic_unseen_template"]["micro"]["f1"]),
        ("Historical hard-negative record error rate", baseline_hard_error, current_hard_error),
    ]
    lines = [
        "# SecureLogX ML-v1 vs ML-v1.1",
        "",
        "**REGRESSION BENCHMARK - NOT UNSEEN TEST**",
        "",
        "| Metric | ML-v1 | ML-v1.1 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name, before, after in rows:
        lines.append(f"| {name} | {before:.6f} | {after:.6f} | {after - before:+.6f} |")
    _write_markdown(path, lines)


def _regression_gate_actual(
    overall: Mapping[str, Any],
    gretel: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    hard_false_positive_rate: float,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    supported_recalls = [
        values["recall"] for values in overall["per_label"].values() if values["support"] > 0
    ]
    return {
        "overall_test_micro_f1": overall["micro"]["f1"],
        "overall_test_macro_f1": overall["macro"]["f1"],
        "high_risk_test_recall": overall["high_risk"]["recall"],
        "minimum_individual_label_recall": min(supported_recalls, default=0.0),
        "gretel_test_micro_f1": gretel["micro"]["f1"],
        "unseen_template_synthetic_micro_f1": synthetic["micro"]["f1"],
        "hard_negative_records_with_false_positive_rate": hard_false_positive_rate,
        "unexplained_alignment_failures": len(_alignment_blockers(alignment)),
        "truncated_entity_spans": alignment["truncated_spans"],
    }


def _write_decision_report(
    path: Path,
    decision: str,
    sealed_acceptance: Mapping[str, Any],
    regression_gates: Mapping[str, Any],
    dev_gate: Mapping[str, Any],
) -> None:
    lines = [
        "# ML-v1.1 Model Validation Decision",
        "",
        f"**{decision}**",
        "",
        "All thresholds were predeclared. None were lowered after observing results.",
        "",
        f"- Development gate: **{dev_gate['decision']}**",
        f"- Sealed challenge acceptance: **{'PASS' if sealed_acceptance['all_passed'] else 'FAIL'}**",
        f"- Original ML-v1 ONNX gate applied to regression benchmark: **{'PASS' if regression_gates['all_passed'] else 'FAIL'}**",
        "",
        "## Sealed gates",
        "",
        "| Gate | Actual | Operator | Threshold | Result |",
        "|---|---:|:---:|---:|---|",
    ]
    for name, result in sealed_acceptance["gates"].items():
        lines.append(
            f"| {name} | {result['actual']:.6f} | {result['operator']} | {result['threshold']:.6f} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Original ML-v1 gates on regression benchmark",
            "",
            "| Gate | Actual | Operator | Threshold | Result |",
            "|---|---:|:---:|---:|---|",
        ]
    )
    for name, result in regression_gates["gates"].items():
        lines.append(
            f"| {name} | {result['actual']:.6f} | {result['operator']} | {result['threshold']:.6f} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "This is a technical ONNX-validation decision only. Gretel provenance still has `license_reviewed=false`, which remains a separate production/commercial release blocker.",
            "",
            "No ONNX export was performed.",
        ]
    )
    _write_markdown(path, lines)


def _run_regression(args: argparse.Namespace, root: Path) -> int:
    receipt_path = _path(REGRESSION_RECEIPT, root)
    cache_path = _path(REGRESSION_CACHE, root)
    report_paths = [
        _path(path, root)
        for path in (
            REGRESSION_METRICS,
            BUSINESS_REPORT,
            SSN_REPORT,
            GENERALIZATION_REPORT,
            ERROR_REPORT,
            ERROR_CSV,
            COMPARISON_REPORT,
            DECISION_REPORT,
        )
    ]
    protected = [receipt_path, cache_path, *report_paths]
    _guard_absent(protected)

    evidence = _verify_frozen_evidence(root, include_sealed=True)
    checkpoint = _validate_checkpoint(root, evidence)
    dev_gate = _validate_dev_gate(root, evidence, checkpoint)
    sealed_receipt = _validate_sealed_receipt(root, checkpoint, evidence)
    input_path = _path(REGRESSION_INPUT, root)
    expected_test = evidence["parent"]["parent_splits"][REGRESSION_INPUT.as_posix()]
    _verify_file(root, REGRESSION_INPUT.as_posix(), expected_test["sha256"])
    _verify_jsonl_count(input_path, EXPECTED_COUNTS["regression"])

    label_config = load_canonical_labels(_path(LABELS, root))
    model, tokenizer, records, build, device, use_bf16 = _load_model_and_data(
        root, input_path, EXPECTED_COUNTS["regression"], label_config, args
    )
    receipt = _base_receipt(
        phase="original_ml_v1_regression_benchmark",
        root=root,
        input_path=input_path,
        output_cache=cache_path,
        checkpoint=checkpoint,
        evidence=evidence,
        args=args,
        records=len(records),
        use_bf16=use_bf16,
        device=device,
    )
    receipt["benchmark_label"] = "REGRESSION BENCHMARK - NOT UNSEEN TEST"
    receipt["sealed_receipt_sha256"] = sealed_receipt["sha256"]

    _guard_absent(protected)
    _exclusive_json(receipt_path, receipt)
    try:
        evaluation, overall = _run_inference(
            model=model,
            tokenizer=tokenizer,
            records=records,
            build=build,
            label_config=label_config,
            cache_path=cache_path,
            input_path=input_path,
            split="ml_v1_original_test_regression_benchmark",
            args=args,
            device=device,
            use_bf16=use_bf16,
            provenance=checkpoint,
        )
        predictions = evaluation["predictions"]
        labels = label_config["entities"]
        gretel_indices = [
            index
            for index, record in enumerate(records)
            if record.get("meta", {}).get("source") == "gretel_finance_pii"
        ]
        synthetic_indices = [
            index
            for index, record in enumerate(records)
            if record.get("meta", {}).get("source") == "securelogx_custom_logs"
        ]
        hard_indices = [
            index
            for index, record in enumerate(records)
            if record.get("meta", {}).get("template_family")
            == "order_reference_ssn_shape_warn_v1"
        ]
        gretel = _segment(records, predictions, labels, gretel_indices, "regression:gretel")
        synthetic = _segment(records, predictions, labels, synthetic_indices, "regression:synthetic")
        hard = _segment(records, predictions, labels, hard_indices, "regression:hard_negative")
        hard_records = [records[index] for index in hard_indices]
        hard_predictions = [predictions[index] for index in hard_indices]
        hard_error_rate = _record_error_rate(hard_records, hard_predictions)
        hard_false_positive_rate = _record_false_positive_rate(hard, len(hard_records))

        sealed_payload, sealed_predictions = _prediction_payload(_path(SEALED_CACHE, root))
        sealed_records = load_jsonl(_path(SEALED_INPUT, root))
        sealed_metrics = load_json(_path(SEALED_METRICS, root))
        if sealed_payload.get("input_sha256") != sha256_file(_path(SEALED_INPUT, root)):
            raise ValueError("Sealed prediction cache is not bound to the sealed input")
        sealed_scored = _score(
            sealed_records, sealed_predictions, labels, "ml_v1_1_sealed_challenge"
        )

        dev_records = load_jsonl(_path(DEV_CHALLENGE_INPUT, root))
        dev_payload, dev_predictions = _prediction_payload(_path(DEV_CHALLENGE_CACHE, root))
        if dev_payload.get("input_sha256") != sha256_file(_path(DEV_CHALLENGE_INPUT, root)):
            raise ValueError("Selected-checkpoint dev challenge cache has the wrong input hash")
        if dev_payload.get("checkpoint_fingerprint_sha256") != checkpoint[
            "selection_checkpoint_fingerprint_sha256"
        ]:
            raise ValueError("Dev challenge cache is not bound to the selected checkpoint")

        historical_cache = _path(
            Path("output_securelogx/ml-v1/bert-base-cased/test_predictions.json"), root
        )
        historical_payload, historical_predictions = _prediction_payload(historical_cache)
        if historical_payload.get("input_sha256") != expected_test["sha256"]:
            raise ValueError("Historical ML-v1 prediction cache is not bound to the frozen test")
        historical_metrics = load_json(_path(Path("reports/test_metrics.json"), root))

        business_views = {
            "Original ML-v1 test regression": _business_analysis(records, predictions),
            "Selected-checkpoint development challenge": _business_analysis(
                dev_records, dev_predictions
            ),
            "Sealed challenge": _business_analysis(sealed_records, sealed_predictions),
        }
        ssn_current = _ssn_analysis(records, predictions)
        ssn_baseline = _ssn_analysis(records, historical_predictions)
        ssn_sealed = _ssn_analysis(sealed_records, sealed_predictions)

        dev_challenge_metrics = load_json(
            _path(Path("reports/ml_v1_1_dev_challenge_metrics.json"), root)
        )
        regression_gate_actual = _regression_gate_actual(
            overall, gretel, synthetic, hard_false_positive_rate, build.summary
        )
        regression_gates = evaluate_named_gates(
            regression_gate_actual, load_json(_path(V1_GATE, root))["required"]
        )
        sealed_acceptance = sealed_metrics["acceptance"]
        final_decision = (
            evidence["gate"]["final_onnx_readiness"]["ready_value"]
            if sealed_acceptance["all_passed"] and regression_gates["all_passed"]
            else evidence["gate"]["final_onnx_readiness"]["failure_value"]
        )

        historical_hard_indices = hard_indices
        historical_hard_predictions = [historical_predictions[index] for index in historical_hard_indices]
        historical_hard_error_rate = _record_error_rate(
            hard_records, historical_hard_predictions
        )
        payload = {
            "evaluation_label": "REGRESSION BENCHMARK - NOT UNSEEN TEST",
            "overall": _compact(overall),
            "segments": {
                "gretel": _compact(gretel),
                "synthetic_unseen_template": _compact(synthetic),
                "historical_hard_negative": {
                    **_compact(hard),
                    "record_error_rate": hard_error_rate,
                    "records_with_false_positive_rate": hard_false_positive_rate,
                },
            },
            "business_id_subtypes": business_views,
            "ssn_context": {
                "ml_v1_historical": ssn_baseline,
                "ml_v1_1_regression": ssn_current,
                "ml_v1_1_sealed": ssn_sealed,
            },
            "alignment": build.summary,
            "original_ml_v1_gates": regression_gates,
            "sealed_acceptance": sealed_acceptance,
            "final_decision": final_decision,
            "provenance": receipt["provenance"],
        }
        write_json(_path(REGRESSION_METRICS, root), payload)

        _write_business_report(_path(BUSINESS_REPORT, root), business_views)
        _write_ssn_report(_path(SSN_REPORT, root), ssn_current, ssn_baseline, ssn_sealed)
        _write_generalization_report(
            _path(GENERALIZATION_REPORT, root),
            gretel,
            synthetic,
            hard,
            hard_error_rate,
            dev_challenge_metrics,
            sealed_metrics,
        )
        regression_tagged = tag_context_errors(overall["errors"], records)
        for row in regression_tagged:
            row["evaluation_view"] = "original_ml_v1_test_regression"
        sealed_tagged = tag_context_errors(sealed_scored["errors"], sealed_records)
        for row in sealed_tagged:
            row["evaluation_view"] = "ml_v1_1_sealed_challenge"
        error_rows = [*regression_tagged, *sealed_tagged]
        _write_tagged_errors(_path(ERROR_CSV, root), error_rows)
        _write_error_report(
            _path(ERROR_REPORT, root), overall, sealed_metrics, error_rows
        )
        _write_comparison_report(
            _path(COMPARISON_REPORT, root),
            historical_metrics,
            payload,
            historical_hard_error_rate,
            hard_error_rate,
        )
        _write_decision_report(
            _path(DECISION_REPORT, root),
            final_decision,
            sealed_acceptance,
            regression_gates,
            dev_gate,
        )

        _verify_frozen_evidence(root, include_sealed=True)
        result_hashes = {
            _display(cache_path, root): sha256_file(cache_path),
            **{
                _display(path, root): sha256_file(path)
                for path in report_paths
            },
        }
        receipt.update(
            {
                "status": "COMPLETE",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "single_inference_pass": True,
                "inference_seconds": overall["inference_seconds"],
                "final_decision": final_decision,
                "result_sha256": result_hashes,
            }
        )
        _replace_json(receipt_path, receipt)
    except BaseException as exc:
        receipt.update(
            {
                "status": "FAILED",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        try:
            _replace_json(receipt_path, receipt)
        except Exception:
            pass
        raise

    print(
        "REGRESSION BENCHMARK - NOT UNSEEN TEST "
        f"micro_f1={overall['micro']['f1']:.6f} "
        f"macro_f1={overall['macro']['f1']:.6f}",
        flush=True,
    )
    print(final_decision, flush=True)
    return 0


def _validate_cli(args: argparse.Namespace, root: Path) -> None:
    if root != REPOSITORY_ROOT:
        raise ValueError(f"--root must resolve to {REPOSITORY_ROOT}")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("PYTHONHASHSEED must be exported as 42 before final evaluation")
    for name, expected in EVALUATION_SETTINGS.items():
        if getattr(args, name) != expected:
            raise ValueError(f"{name} must be {expected}, received {getattr(args, name)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("sealed", "regression"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(args.root).resolve()
    _validate_cli(args, root)
    if args.phase == "sealed":
        return _run_sealed(args, root)
    return _run_regression(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
