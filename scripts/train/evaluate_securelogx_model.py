"""Evaluate one selected SecureLogX checkpoint in a single held-out test pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from securelogx_training_common import (
    HIGH_RISK_ENTITIES,
    build_aligned_features,
    load_canonical_labels,
    load_json,
    load_jsonl,
    score_predictions,
    set_reproducible_seed,
    sha256_file,
    subset_score,
    validate_manifest,
    write_confusion_csv,
    write_error_csv,
    write_json,
    evaluate_model,
)


CANONICAL_MANIFEST = Path("configs/ml_v1_frozen_manifest.json")
CANONICAL_LABELS = Path("configs/securelogx_labels.json")
CANONICAL_GATES = Path("configs/ml_v1_onnx_validation_gate.json")
CANONICAL_TEST = Path("data/split/test.jsonl")
CANONICAL_MODEL = Path(
    "output_securelogx/ml-v1/bert-base-cased/best-checkpoint"
)
CANONICAL_RUN_ROOT = Path("output_securelogx/ml-v1/bert-base-cased")
CANONICAL_REPORTS = Path("reports")
CANONICAL_TEST_RECORDS = 2_989
CANONICAL_EVALUATION_SETTINGS = {
    "max_length": 384,
    "stride": 128,
    "batch_size": 8,
    "seed": 42,
}
CANONICAL_TRAINING_ARGUMENTS = {
    "base_model": "bert-base-cased",
    "revision": "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
    "max_length": 384,
    "stride": 128,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "warmup_steps": 489,
    "max_grad_norm": 1.0,
    "train_batch_size": 4,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 16,
    "seed": 42,
    "use_bf16": True,
    "device": "cuda",
    "planned_optimizer_steps": 4_887,
    "train_records": 23_558,
    "dev_records": 2_994,
    "train_windows": 26_051,
    "dev_windows": 3_324,
    "smoke_test": False,
}
FINAL_REPORT_FILENAMES = (
    "per_label_test_metrics.csv",
    "high_risk_entity_metrics.md",
    "source_segment_evaluation.md",
    "hard_negative_model_evaluation.md",
    "business_id_subtype_analysis.md",
    "model_error_analysis.md",
    "confidence_analysis.md",
    "model_errors.csv",
    "entity_confusion_matrix.csv",
    "model_validation_decision.md",
    "test_metrics.json",
)
EXPLAINED_ALIGNMENT_FAILURE_REASONS = frozenset(
    {
        "tokenizer_emitted_no_token_for_span",
        "token_boundary_adjustment_exceeds_three_characters",
    }
)


def _path_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json_value(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mapping_fingerprint(values: Mapping[str, str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _unexplained_alignment_failures(
    alignment: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return [
        diagnostic
        for diagnostic in alignment.get("diagnostics", [])
        if diagnostic.get("status") == "failed"
        and diagnostic.get("reason") not in EXPLAINED_ALIGNMENT_FAILURE_REASONS
    ]


def _validate_mode_contract(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    paths = {
        "manifest": (root / args.manifest).resolve(),
        "labels": (root / args.labels).resolve(),
        "gates": (root / args.gates).resolve(),
        "model": (root / args.model).resolve(),
        "input": (root / args.input).resolve(),
        "reports": (root / args.reports_dir).resolve(),
    }
    canonical = {
        "manifest": (root / CANONICAL_MANIFEST).resolve(),
        "labels": (root / CANONICAL_LABELS).resolve(),
        "gates": (root / CANONICAL_GATES).resolve(),
        "model": (root / CANONICAL_MODEL).resolve(),
        "input": (root / CANONICAL_TEST).resolve(),
        "reports": (root / CANONICAL_REPORTS).resolve(),
    }
    if args.smoke_test:
        if paths["input"] == canonical["input"] or args.split_name == "test":
            raise ValueError("Smoke evaluation may not inspect the frozen test split")
        if paths["reports"] == canonical["reports"]:
            raise ValueError("Smoke evaluation may not write canonical final reports")
        return paths

    violations: list[str] = []
    for name in ("manifest", "labels", "gates", "model", "input", "reports"):
        if paths[name] != canonical[name]:
            violations.append(
                f"{name} must be {_path_display(canonical[name], root)}, "
                f"received {_path_display(paths[name], root)}"
            )
    if args.split_name != "test":
        violations.append("split_name must be test")
    if args.max_records is not None:
        violations.append("max_records must be omitted for final evaluation")
    for name, expected in CANONICAL_EVALUATION_SETTINGS.items():
        actual = getattr(args, name)
        if actual != expected:
            violations.append(f"{name} must be {expected}, received {actual}")
    if violations:
        raise ValueError("Final-evaluation contract violation: " + "; ".join(violations))
    return paths


def _guard_absent(paths: Sequence[Path]) -> None:
    existing = [path.as_posix() for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing evaluation because protected output artifacts already exist: "
            + ", ".join(existing)
        )


def _validate_gate_provenance(
    manifest: Mapping[str, Any], gate_path: Path, gates: Mapping[str, Any]
) -> str:
    expected = manifest["files"][CANONICAL_GATES.as_posix()]["sha256"]
    actual = sha256_file(gate_path)
    if actual != expected:
        raise ValueError(
            f"Decision-gate SHA-256 mismatch: expected {expected}, received {actual}"
        )
    if gates.get("declared_before_training") is not True:
        raise ValueError("Decision gates must be explicitly declared before training")
    return actual


def _validate_final_checkpoint(root: Path, model_path: Path) -> dict[str, Any]:
    run_root = (root / CANONICAL_RUN_ROOT).resolve()
    required_names = (
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
    missing = [name for name in required_names if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Selected checkpoint is missing: {missing}")

    training_arguments = load_json(model_path / "training_args.json")
    expected_protected_metadata = {
        "frozen_manifest_path": CANONICAL_MANIFEST.as_posix(),
        "frozen_manifest_sha256": sha256_file((root / CANONICAL_MANIFEST).resolve()),
        "validation_gate_path": CANONICAL_GATES.as_posix(),
        "validation_gate_sha256": sha256_file((root / CANONICAL_GATES).resolve()),
    }
    expected_training_arguments = {
        **CANONICAL_TRAINING_ARGUMENTS,
        **expected_protected_metadata,
    }
    mismatches = [
        f"{name}: expected {expected!r}, received {training_arguments.get(name)!r}"
        for name, expected in expected_training_arguments.items()
        if training_arguments.get(name) != expected
    ]
    if mismatches:
        raise ValueError("Final training metadata mismatch: " + "; ".join(mismatches))

    selected = load_json(model_path / "selected_checkpoint.json")
    selected_epoch = selected.get("selected_epoch")
    if selected_epoch not in {1, 2, 3}:
        raise ValueError(f"Invalid selected epoch: {selected_epoch!r}")
    if selected.get("selection_metric") != "dev exact-span macro F1":
        raise ValueError("Checkpoint was not selected by dev exact-span macro F1")
    source_checkpoint = (root / str(selected.get("source_checkpoint", ""))).resolve()
    expected_source = (run_root / "checkpoints" / f"epoch-{selected_epoch}").resolve()
    if source_checkpoint != expected_source or not source_checkpoint.is_dir():
        raise ValueError("Selected checkpoint source does not match its canonical epoch")

    epoch_metrics = load_json(model_path / "epoch_metrics.json")
    epoch_history = epoch_metrics.get("history", {})
    selection_value = float(selected.get("selection_value"))
    if epoch_history.get("epoch") != selected_epoch or not math.isclose(
        float(epoch_history.get("dev_macro_f1")), selection_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Selected checkpoint metadata disagrees with epoch metrics")
    stored_macro = float(epoch_metrics.get("dev_metrics", {}).get("macro", {}).get("f1"))
    if not math.isclose(stored_macro, selection_value, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Selected checkpoint value disagrees with stored dev metrics")

    environment = load_json(model_path / "environment.json")
    expected_environment = {
        "base_model": CANONICAL_TRAINING_ARGUMENTS["base_model"],
        "base_model_revision": CANONICAL_TRAINING_ARGUMENTS["revision"],
        "seed": CANONICAL_TRAINING_ARGUMENTS["seed"],
        "cuda_available": True,
        "bf16_supported": True,
    }
    environment_mismatches = [
        f"{name}: expected {expected!r}, received {environment.get(name)!r}"
        for name, expected in expected_environment.items()
        if environment.get(name) != expected
    ]
    if environment_mismatches:
        raise ValueError(
            "Final environment metadata mismatch: " + "; ".join(environment_mismatches)
        )

    copied_names = tuple(name for name in required_names if name != "selected_checkpoint.json")
    artifact_hashes = {
        name: sha256_file(model_path / name) for name in required_names
    }
    source_hashes = {
        name: sha256_file(source_checkpoint / name) for name in copied_names
    }
    copied_mismatches = [
        name for name in copied_names if artifact_hashes[name] != source_hashes[name]
    ]
    if copied_mismatches:
        raise ValueError(
            "Best checkpoint differs from selected source epoch: "
            + ", ".join(copied_mismatches)
        )

    history_path = run_root / "training_history.json"
    alignment_path = run_root / "alignment_summary.json"
    dev_metrics_path = (root / "reports/dev_metrics.json").resolve()
    for path in (history_path, alignment_path, dev_metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing full-run metadata: {path}")
    history = _load_json_value(history_path)
    if not isinstance(history, list) or [row.get("epoch") for row in history] != [1, 2, 3]:
        raise ValueError("Full-run history must contain exactly epochs 1, 2, and 3")
    selected_history = history[selected_epoch - 1]
    if not math.isclose(
        float(selected_history.get("dev_macro_f1")), selection_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Run history disagrees with selected checkpoint value")
    dev_metrics = load_json(dev_metrics_path)
    if not math.isclose(
        float(dev_metrics.get("macro", {}).get("f1")), selection_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Canonical dev metrics disagree with selected checkpoint")

    metadata_hashes = {
        "training_history.json": sha256_file(history_path),
        "alignment_summary.json": sha256_file(alignment_path),
        "reports/dev_metrics.json": sha256_file(dev_metrics_path),
    }
    return {
        "selected_epoch": selected_epoch,
        "selection_value": selection_value,
        "source_checkpoint": _path_display(source_checkpoint, root),
        "artifact_sha256": artifact_hashes,
        "source_artifact_sha256": source_hashes,
        "run_metadata_sha256": metadata_hashes,
        "checkpoint_fingerprint_sha256": _mapping_fingerprint(artifact_hashes),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _supported_macro(metrics: Mapping[str, Any]) -> dict[str, float]:
    supported = [
        values for values in metrics["per_label"].values() if values["support"] > 0
    ]
    if not supported:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        name: float(statistics.fmean(values[name] for values in supported))
        for name in ("precision", "recall", "f1")
    }


def _metrics_without_rows(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"errors", "confusion"}
    }


def _write_per_label_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "entity",
                "precision",
                "recall",
                "f1",
                "support",
                "predicted",
                "true_positives",
                "false_positives",
                "false_negatives",
            ]
        )
        for label, values in metrics["per_label"].items():
            writer.writerow(
                [
                    label,
                    f"{values['precision']:.10f}",
                    f"{values['recall']:.10f}",
                    f"{values['f1']:.10f}",
                    values["support"],
                    values["predicted"],
                    values["true_positives"],
                    values["false_positives"],
                    values["false_negatives"],
                ]
            )


def _write_high_risk_report(path: Path, metrics: Mapping[str, Any]) -> None:
    high = metrics["high_risk"]
    lines = [
        "# High-Risk Entity Metrics",
        "",
        "The frozen high-risk group is: "
        + ", ".join(f"`{label}`" for label in sorted(HIGH_RISK_ENTITIES))
        + ".",
        "",
        f"- Precision: **{_pct(high['precision'])}**",
        f"- Recall: **{_pct(high['recall'])}**",
        f"- F1: **{_pct(high['f1'])}**",
        f"- False negatives: **{high['false_negatives']}**",
        f"- Support: **{high['support']}**",
        "",
        "| Entity | Precision | Recall | F1 | Support | False negatives |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in sorted(HIGH_RISK_ENTITIES):
        values = metrics["per_label"][label]
        lines.append(
            f"| {label} | {_pct(values['precision'])} | {_pct(values['recall'])} | "
            f"{_pct(values['f1'])} | {values['support']} | {values['false_negatives']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_source_report(
    path: Path,
    overall: Mapping[str, Any],
    gretel: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    gretel_records: Sequence[Mapping[str, Any]],
    synthetic_records: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Source-Segment Test Evaluation",
        "",
        "All segment metrics were derived from the same single held-out test inference pass.",
        "",
        "| Segment | Records | Gold entities | Micro P | Micro R | Micro F1 | Supported-label macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in (
        ("Overall", overall),
        ("Gretel-only", gretel),
        ("Synthetic-only", synthetic),
    ):
        supported = _supported_macro(values)
        lines.append(
            f"| {name} | {values['records']} | {values['gold_entities']} | "
            f"{_pct(values['micro']['precision'])} | {_pct(values['micro']['recall'])} | "
            f"{_pct(values['micro']['f1'])} | {_pct(supported['f1'])} |"
        )
    lines.extend(
        [
            "",
            "Synthetic test records are unseen-template examples: the frozen split audit proves each synthetic template family belongs to exactly one split.",
            "",
            "Gretel has no template-family metadata, so the Gretel segment is protected by stable source-record grouping and exact-text independence rather than a template-family claim.",
            "",
            "## Gretel-only per-label metrics (supported labels)",
            "",
            "| Entity | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, values in gretel["per_label"].items():
        if values["support"]:
            lines.append(
                f"| {label} | {_pct(values['precision'])} | {_pct(values['recall'])} | "
                f"{_pct(values['f1'])} | {values['support']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hard_negative_report(
    path: Path,
    metrics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> float:
    families = sorted(
        {str(record["meta"].get("template_family")) for record in records}
    )
    predicted_error_records = {
        int(row["record_index"])
        for row in metrics["errors"]
        if row["predicted_entity"]
    }
    record_fp_rate = len(predicted_error_records) / len(records) if records else 0.0
    incorrect_entities = Counter(
        row["predicted_entity"]
        for row in metrics["errors"]
        if row["predicted_entity"]
    )
    lines = [
        "# Hard-Negative Model Evaluation",
        "",
        f"- Records: **{len(records)}**",
        f"- Complete template families: **{len(families)}**",
        f"- Family: `{families[0] if families else 'none'}`",
        f"- Expected gold entities: **{metrics['gold_entities']}**",
        f"- Exact span matches: **{metrics['exact_span_matches']}**",
        f"- False-positive count: **{metrics['error_counts']['false_positives']}**",
        f"- False-negative count: **{metrics['error_counts']['false_negatives']}**",
        f"- Records with any non-exact predicted entity: **{len(predicted_error_records)}**",
        f"- Record-level false-positive/error rate: **{_pct(record_fp_rate)}**",
        "",
        "Incorrectly detected entity types: "
        + (", ".join(f"{label}={count}" for label, count in sorted(incorrect_entities.items())) or "none"),
        "",
        "This subset contains only one held-out hard-negative family. It is useful evidence against pattern-only behavior but is not broad enough to support a general hard-negative robustness claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record_fp_rate


def _business_subtype(
    record: Mapping[str, Any], gold: Mapping[str, Any]
) -> str:
    for entry in record.get("meta", {}).get("entity_provenance", []):
        if (
            int(entry.get("start", -1)) == int(gold["start"])
            and int(entry.get("end", -1)) == int(gold["end"])
            and entry.get("label") == "BUSINESS_ID"
        ):
            return str(
                entry.get("source_subtype")
                or entry.get("source_label")
                or "BUSINESS_ID"
            )
    return "BUSINESS_ID"


def _business_id_analysis(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    subtype_gold: Counter[str] = Counter()
    subtype_tp: Counter[str] = Counter()
    subtype_record_indices: dict[str, set[int]] = defaultdict(set)
    record_subtypes: dict[int, set[str]] = defaultdict(set)
    for record_index, record in enumerate(records):
        predicted_signatures = {
            (int(span["start"]), int(span["end"]), str(span["label"]))
            for span in predictions[record_index]
        }
        for start, end, label in record["entities"]:
            if label != "BUSINESS_ID":
                continue
            gold = {"start": int(start), "end": int(end), "label": str(label)}
            subtype = _business_subtype(record, gold)
            subtype_gold[subtype] += 1
            subtype_record_indices[subtype].add(record_index)
            record_subtypes[record_index].add(subtype)
            if (int(start), int(end), str(label)) in predicted_signatures:
                subtype_tp[subtype] += 1

    rows: list[dict[str, Any]] = []
    for subtype in sorted(subtype_gold):
        indices = subtype_record_indices[subtype]
        predicted_business = sum(
            1
            for index in indices
            for span in predictions[index]
            if span["label"] == "BUSINESS_ID"
        )
        records_with_prediction = sum(
            any(span["label"] == "BUSINESS_ID" for span in predictions[index])
            for index in indices
        )
        ambiguous_context_records = sum(
            len(record_subtypes[index]) > 1 for index in indices
        )
        tp = subtype_tp[subtype]
        support = subtype_gold[subtype]
        recall = tp / support if support else 0.0
        rows.append(
            {
                "subtype": subtype,
                "records": len(indices),
                "support": support,
                "true_positives": tp,
                "false_negatives": support - tp,
                "predicted_business_id_in_context_records": predicted_business,
                "records_with_any_business_id_prediction": records_with_prediction,
                "context_record_detection_rate": (
                    records_with_prediction / len(indices) if indices else 0.0
                ),
                "business_id_predictions_per_context_record": (
                    predicted_business / len(indices) if indices else 0.0
                ),
                "ambiguous_multi_subtype_context_records": ambiguous_context_records,
                "exact_span_recall": recall,
            }
        )
    total_support = sum(subtype_gold.values())
    total_tp = sum(subtype_tp.values())
    total_predicted = sum(
        span["label"] == "BUSINESS_ID"
        for record_predictions in predictions
        for span in record_predictions
    )
    overall_precision = total_tp / total_predicted if total_predicted else 0.0
    overall_recall = total_tp / total_support if total_support else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision + overall_recall
        else 0.0
    )
    multi_subtype_records = sum(len(subtypes) > 1 for subtypes in record_subtypes.values())
    lines = [
        "# BUSINESS_ID Subtype Analysis",
        "",
        "The model predicts only `BUSINESS_ID`; original subtypes are recovered from frozen span provenance for analysis.",
        "",
        f"- Overall BUSINESS_ID precision: **{_pct(overall_precision)}**",
        f"- Overall BUSINESS_ID recall: **{_pct(overall_recall)}**",
        f"- Overall BUSINESS_ID F1: **{_pct(overall_f1)}**",
        f"- Records containing multiple BUSINESS_ID subtypes: **{multi_subtype_records}**",
        "",
        "| Original subtype | Records | Support | Exact TP | FN | Exact-span recall | Records with any BUSINESS_ID prediction | Context detection rate | Predictions/context record | Ambiguous context records |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subtype']} | {row['records']} | {row['support']} | "
            f"{row['true_positives']} | {row['false_negatives']} | "
            f"{_pct(row['exact_span_recall'])} | "
            f"{row['records_with_any_business_id_prediction']} | "
            f"{_pct(row['context_record_detection_rate'])} | "
            f"{row['business_id_predictions_per_context_record']:.4f} | "
            f"{row['ambiguous_multi_subtype_context_records']} |"
        )
    lines.extend(
        [
            "",
            "Subtype exact-span recall is well-defined because every gold BUSINESS_ID span retains its subtype. Predicted false positives do not have a gold subtype, so subtype precision and subtype F1 are intentionally not reported.",
            "",
            "Context detection rate and predictions per context record are descriptive diagnostics only. In multi-subtype records, the same record-level predictions can appear in more than one subtype context; the ambiguity count makes that overlap explicit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "overall": {
            "support": total_support,
            "predicted": total_predicted,
            "true_positives": total_tp,
            "false_positives": total_predicted - total_tp,
            "false_negatives": total_support - total_tp,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        },
        "multi_subtype_records": multi_subtype_records,
        "subtypes": rows,
    }


def _write_error_analysis(path: Path, metrics: Mapping[str, Any]) -> None:
    categories = Counter(row["error_category"] for row in metrics["errors"])
    labels = Counter(
        row["gold_entity"] or row["predicted_entity"] for row in metrics["errors"]
    )
    lines = [
        "# Model Error Analysis",
        "",
        f"- Exact span matches: **{metrics['exact_span_matches']}**",
        f"- Total false positives: **{metrics['error_counts']['false_positives']}**",
        f"- Total false negatives: **{metrics['error_counts']['false_negatives']}**",
        f"- Boundary-too-short pairs: **{metrics['error_counts']['boundary_too_short']}**",
        f"- Boundary-too-long pairs: **{metrics['error_counts']['boundary_too_long']}**",
        f"- Boundary-shifted pairs: **{metrics['error_counts'].get('boundary_shifted', 0)}**",
        f"- Wrong-class pairs: **{metrics['error_counts']['wrong_entity_class']}**",
        "",
        "## Error categories",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(categories.items()))
    lines.extend(["", "## Most affected labels", ""])
    lines.extend(
        f"- {label}: {count} classified error rows"
        for label, count in labels.most_common(15)
    )
    lines.extend(
        [
            "",
            "The full privacy-safe error inventory is in `reports/model_errors.csv`; the entity confusion matrix includes `<MISSED>`, `<SPURIOUS>`, and `<BOUNDARY>` buckets.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_confidence_report(
    path: Path,
    metrics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    def median(values: Sequence[float]) -> float | None:
        return float(statistics.median(values)) if values else None

    def low_share(values: Sequence[float], cutoff: float | None) -> float | None:
        if not values or cutoff is None:
            return None
        return sum(value <= cutoff for value in values) / len(values)

    def show(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.6f}"

    correct_by_label: dict[str, list[float]] = defaultdict(list)
    incorrect_by_label: dict[str, list[float]] = defaultdict(list)
    for record, record_predictions in zip(records, predictions):
        gold = {
            (int(start), int(end), str(label))
            for start, end, label in record["entities"]
        }
        for prediction in record_predictions:
            label = str(prediction["label"])
            confidence = float(prediction["confidence"])
            signature = (
                int(prediction["start"]),
                int(prediction["end"]),
                label,
            )
            target = correct_by_label if signature in gold else incorrect_by_label
            target[label].append(confidence)

    false_negative_by_label: dict[str, list[float]] = defaultdict(list)
    error_prediction_groups: dict[str, list[float]] = defaultdict(list)
    for row in metrics["errors"]:
        prediction_confidence = row.get("prediction_confidence")
        if prediction_confidence is not None:
            category = str(row["error_category"])
            if category.startswith("boundary-"):
                category = "boundary_error"
            error_prediction_groups[category].append(float(prediction_confidence))
        gold_probability = row.get("gold_label_probability")
        if row.get("error_category") == "false negative" and gold_probability is not None:
            false_negative_by_label[str(row["gold_entity"])].append(
                float(gold_probability)
            )

    supported = {
        label: values
        for label, values in metrics["per_label"].items()
        if values["support"] > 0
    }
    ordered_support = sorted(int(values["support"]) for values in supported.values())
    rare_rank = max(1, math.ceil(len(ordered_support) / 4))
    rare_support_max = ordered_support[rare_rank - 1] if ordered_support else 0
    rare_labels = {
        label
        for label, values in supported.items()
        if int(values["support"]) <= rare_support_max
    }
    correct_values = [value for values in correct_by_label.values() for value in values]
    correct_summary = metrics.get("confidence", {}).get("correct", {})
    diagnostic_cutoff = correct_summary.get("p25")
    if diagnostic_cutoff is None and correct_values:
        diagnostic_cutoff = sorted(correct_values)[len(correct_values) // 4]
    diagnostic_cutoff = (
        float(diagnostic_cutoff) if diagnostic_cutoff is not None else None
    )

    per_label_rows: list[dict[str, Any]] = []
    for label, values in supported.items():
        correct = correct_by_label[label]
        incorrect = incorrect_by_label[label]
        false_negative = false_negative_by_label[label]
        per_label_rows.append(
            {
                "entity": label,
                "support": int(values["support"]),
                "cohort": "rare" if label in rare_labels else "other",
                "f1": float(values["f1"]),
                "correct_predictions": len(correct),
                "incorrect_predictions": len(incorrect),
                "correct_median_confidence": median(correct),
                "incorrect_median_confidence": median(incorrect),
                "false_negative_count_with_probability": len(false_negative),
                "false_negative_median_gold_probability": median(false_negative),
            }
        )

    rare_prediction_values = [
        value
        for label in rare_labels
        for value in correct_by_label[label] + incorrect_by_label[label]
    ]
    other_prediction_values = [
        value
        for label in supported
        if label not in rare_labels
        for value in correct_by_label[label] + incorrect_by_label[label]
    ]
    rare_incorrect = sum(len(incorrect_by_label[label]) for label in rare_labels)
    other_incorrect = sum(
        len(incorrect_by_label[label]) for label in supported if label not in rare_labels
    )
    rare_analysis = {
        "definition": "lowest supported-label quartile by gold support",
        "maximum_support": rare_support_max,
        "labels": sorted(rare_labels),
        "prediction_count": len(rare_prediction_values),
        "low_confidence_share": low_share(rare_prediction_values, diagnostic_cutoff),
        "incorrect_prediction_share": (
            rare_incorrect / len(rare_prediction_values) if rare_prediction_values else None
        ),
        "other_prediction_count": len(other_prediction_values),
        "other_low_confidence_share": low_share(
            other_prediction_values, diagnostic_cutoff
        ),
        "other_incorrect_prediction_share": (
            other_incorrect / len(other_prediction_values)
            if other_prediction_values
            else None
        ),
    }

    lines = [
        "# Model Confidence Analysis",
        "",
        "Span confidence is the mean maximum softmax probability across predicted tokens. For false negatives, the reported value is the mean probability assigned to the gold BIO labels over the gold-token region.",
        "",
        "| Group | Count | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics["confidence"].items():
        lines.append(
            f"| {name} | {values['count']} | {show(values['mean'])} | "
            f"{show(values['median'])} | {show(values['p25'])} | {show(values['p75'])} |"
        )
    lines.extend(["", "## Error-versus-correct diagnostic", ""])
    if diagnostic_cutoff is None:
        lines.append("No correct-span confidence values were available for comparison.")
    else:
        baseline = low_share(correct_values, diagnostic_cutoff)
        lines.append(
            f"The diagnostic low-confidence cutoff is the correct-span p25 ({diagnostic_cutoff:.6f}); "
            "it is not a runtime threshold."
        )
        lines.extend(["", "| Error group | Count | Median confidence/probability | At or below cutoff | Compared with correct spans |", "|---|---:|---:|---:|---|"])
        comparison_groups = dict(error_prediction_groups)
        comparison_groups["false_negative_gold_probability"] = [
            value for values in false_negative_by_label.values() for value in values
        ]
        for name, values in sorted(comparison_groups.items()):
            share = low_share(values, diagnostic_cutoff)
            relation = "n/a"
            if share is not None and baseline is not None:
                relation = "higher" if share > baseline else "lower" if share < baseline else "equal"
            lines.append(
                f"| {name} | {len(values)} | {show(median(values))} | "
                f"{'n/a' if share is None else _pct(share)} | {relation} low-confidence share |"
            )

    lines.extend(
        [
            "",
            "## Per-label confidence and support",
            "",
            f"Rare-class cohort: lowest supported-label quartile by gold support (support <= {rare_support_max}): "
            + ", ".join(f"`{label}`" for label in sorted(rare_labels))
            + ".",
            "",
            "| Entity | Cohort | Support | F1 | Correct predictions | Correct median | Incorrect predictions | Incorrect median | FN probabilities | FN median gold probability |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_label_rows:
        lines.append(
            f"| {row['entity']} | {row['cohort']} | {row['support']} | {_pct(row['f1'])} | "
            f"{row['correct_predictions']} | {show(row['correct_median_confidence'])} | "
            f"{row['incorrect_predictions']} | {show(row['incorrect_median_confidence'])} | "
            f"{row['false_negative_count_with_probability']} | "
            f"{show(row['false_negative_median_gold_probability'])} |"
        )
    lines.extend(
        [
            "",
            "## Rare-class comparison",
            "",
            f"- Rare prediction low-confidence share: **{'n/a' if rare_analysis['low_confidence_share'] is None else _pct(rare_analysis['low_confidence_share'])}**",
            f"- Other-label prediction low-confidence share: **{'n/a' if rare_analysis['other_low_confidence_share'] is None else _pct(rare_analysis['other_low_confidence_share'])}**",
            f"- Rare incorrect-prediction share: **{'n/a' if rare_analysis['incorrect_prediction_share'] is None else _pct(rare_analysis['incorrect_prediction_share'])}**",
            f"- Other-label incorrect-prediction share: **{'n/a' if rare_analysis['other_incorrect_prediction_share'] is None else _pct(rare_analysis['other_incorrect_prediction_share'])}**",
            "",
            "False-negative values are gold-label probabilities, while other error values are predicted-span confidences; compare them cautiously. No runtime confidence threshold is selected in this phase.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "diagnostic_low_confidence_cutoff": diagnostic_cutoff,
        "rare_class_analysis": rare_analysis,
        "per_label": per_label_rows,
    }


def _evaluate_gates(
    gates: Mapping[str, Any],
    overall: Mapping[str, Any],
    gretel: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    hard_negative_record_fp_rate: float,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    required = gates["required"]
    individual_recalls = {
        label: values["recall"]
        for label, values in overall["per_label"].items()
        if values["support"] > 0
    }
    actual = {
        "overall_test_micro_f1_minimum": overall["micro"]["f1"],
        "overall_test_macro_f1_minimum": overall["macro"]["f1"],
        "high_risk_test_recall_minimum": overall["high_risk"]["recall"],
        "minimum_individual_label_recall": min(individual_recalls.values()),
        "gretel_test_micro_f1_minimum": gretel["micro"]["f1"],
        "unseen_template_synthetic_micro_f1_minimum": synthetic["micro"]["f1"],
        "hard_negative_records_with_false_positive_rate_maximum": hard_negative_record_fp_rate,
        "unexplained_alignment_failures_maximum": len(
            _unexplained_alignment_failures(alignment)
        ),
        "truncated_entity_spans_maximum": alignment["truncated_spans"],
    }
    maximum_keys = {
        "hard_negative_records_with_false_positive_rate_maximum",
        "unexplained_alignment_failures_maximum",
        "truncated_entity_spans_maximum",
    }
    results = {}
    for name, threshold in required.items():
        value = actual[name]
        passed = value <= threshold if name in maximum_keys else value >= threshold
        results[name] = {"threshold": threshold, "actual": value, "passed": passed}
    return {
        "decision": (
            "READY FOR ONNX VALIDATION"
            if all(value["passed"] for value in results.values())
            else "NOT READY FOR ONNX VALIDATION"
        ),
        "gates": results,
        "worst_label": min(individual_recalls, key=individual_recalls.get),
    }


def _write_decision_report(path: Path, decision: Mapping[str, Any]) -> None:
    lines = [
        "# ML-v1 ONNX Validation Decision",
        "",
        f"**{decision['decision']}**",
        "",
        "These gates were committed to configuration before the training/test results were observed.",
        "",
        "| Gate | Threshold | Actual | Status |",
        "|---|---:|---:|---|",
    ]
    for name, values in decision["gates"].items():
        lines.append(
            f"| {name} | {values['threshold']:.6f} | {values['actual']:.6f} | "
            f"{'PASS' if values['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Worst-recall supported label: `{decision['worst_label']}`.",
            "",
            "This decision authorizes only a later ONNX validation phase when READY. It does not export ONNX, approve licensing, or establish production readiness.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute_evaluation(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    build: Any,
    id_to_label: Mapping[int, str],
    label_config: Mapping[str, Any],
    label_to_id: Mapping[str, int],
    batch_size: int,
    device: Any,
    use_bf16: bool,
    split_name: str,
    reports: Path,
    prediction_cache: Path,
    gates: Mapping[str, Any],
    smoke_test: bool,
    started: str,
    input_path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    start_time = time.perf_counter()
    evaluation = evaluate_model(
        model,
        tokenizer,
        records,
        build.windows,
        id_to_label,
        batch_size,
        device,
        use_bf16,
    )
    inference_seconds = time.perf_counter() - start_time
    write_json(
        prediction_cache,
        {
            "split": split_name,
            "input_sha256": sha256_file(input_path),
            "records": len(records),
            "predictions": evaluation["predictions"],
            "loss": evaluation["loss"],
            "invalid_bio_transitions": evaluation["invalid_bio_transitions"],
            "inference_seconds": inference_seconds,
            "provenance": provenance,
        },
    )
    overall = score_predictions(
        records,
        evaluation["predictions"],
        label_config["entities"],
        split_name,
        token_details=evaluation["token_details"],
        label_to_id=label_to_id,
    )
    overall["loss"] = evaluation["loss"]
    overall["invalid_bio_transitions"] = evaluation["invalid_bio_transitions"]
    overall["inference_seconds"] = inference_seconds

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
    hard_negative_indices = [
        index
        for index, record in enumerate(records)
        if record.get("meta", {}).get("scenario_kind") == "hard_negative"
    ]
    gretel = subset_score(
        records, evaluation["predictions"], label_config["entities"], split_name,
        gretel_indices,
    )
    synthetic = subset_score(
        records, evaluation["predictions"], label_config["entities"], split_name,
        synthetic_indices,
    )
    hard_negative = subset_score(
        records, evaluation["predictions"], label_config["entities"], split_name,
        hard_negative_indices,
    )

    reports.mkdir(parents=True, exist_ok=True)
    _write_per_label_csv(reports / "per_label_test_metrics.csv", overall)
    _write_high_risk_report(reports / "high_risk_entity_metrics.md", overall)
    _write_source_report(
        reports / "source_segment_evaluation.md",
        overall,
        gretel,
        synthetic,
        [records[index] for index in gretel_indices],
        [records[index] for index in synthetic_indices],
    )
    hard_negative_rate = _write_hard_negative_report(
        reports / "hard_negative_model_evaluation.md",
        hard_negative,
        [records[index] for index in hard_negative_indices],
    )
    business = _business_id_analysis(
        reports / "business_id_subtype_analysis.md",
        records,
        evaluation["predictions"],
    )
    _write_error_analysis(reports / "model_error_analysis.md", overall)
    confidence_analysis = _write_confidence_report(
        reports / "confidence_analysis.md",
        overall,
        records,
        evaluation["predictions"],
    )
    write_error_csv(reports / "model_errors.csv", overall["errors"])
    write_confusion_csv(
        reports / "entity_confusion_matrix.csv",
        overall["confusion"],
        label_config["entities"],
    )
    decision = _evaluate_gates(
        gates,
        overall,
        gretel,
        synthetic,
        hard_negative_rate,
        build.summary,
    )
    _write_decision_report(reports / "model_validation_decision.md", decision)

    test_metrics = {
        "overall": _metrics_without_rows(overall),
        "segments": {
            "gretel": _metrics_without_rows(gretel),
            "synthetic_unseen_template": _metrics_without_rows(synthetic),
            "hard_negative": _metrics_without_rows(hard_negative),
        },
        "business_id": business,
        "confidence_analysis": confidence_analysis,
        "alignment": build.summary,
        "decision": decision,
        "provenance": provenance,
        "evaluation": {
            "started_utc": started,
            "inference_seconds": inference_seconds,
            "device": str(device),
            "bf16": use_bf16,
            "records_per_second": (
                len(records) / inference_seconds if inference_seconds else 0.0
            ),
            "single_test_pass": not smoke_test,
        },
    }
    write_json(reports / "test_metrics.json", test_metrics)
    report_hashes = {
        name: sha256_file(reports / name) for name in FINAL_REPORT_FILENAMES
    }
    return {
        "overall": overall,
        "decision": decision,
        "inference_seconds": inference_seconds,
        "prediction_cache_sha256": sha256_file(prediction_cache),
        "report_sha256": report_hashes,
    }


def _run_main(args: argparse.Namespace) -> int:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not args.smoke_test and os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("PYTHONHASHSEED must be exported as 42 before final evaluation")
    root = Path(args.root).resolve()
    paths = _validate_mode_contract(args, root)
    model_path = paths["model"]
    reports = paths["reports"]
    run_root = (root / CANONICAL_RUN_ROOT).resolve()
    if args.smoke_test:
        receipt_path = reports / "smoke_evaluation_receipt.json"
        prediction_cache = reports / "smoke_predictions.json"
        legacy_artifacts: list[Path] = []
    else:
        receipt_path = run_root / "test_evaluation_receipt.json"
        prediction_cache = run_root / "test_predictions.json"
        legacy_artifacts = [
            model_path / "test_evaluation_receipt.json",
            model_path / "test_predictions.json",
        ]
    protected_artifacts = [
        receipt_path,
        prediction_cache,
        *(reports / name for name in FINAL_REPORT_FILENAMES),
        *legacy_artifacts,
    ]
    _guard_absent(protected_artifacts)

    manifest = validate_manifest(root, paths["manifest"])
    gates = load_json(paths["gates"])
    gate_sha256 = _validate_gate_provenance(manifest, paths["gates"], gates)
    label_config = load_canonical_labels(paths["labels"])
    label_to_id = {
        str(key): int(value) for key, value in label_config["label_to_id"].items()
    }
    id_to_label = {
        int(key): str(value) for key, value in label_config["id_to_label"].items()
    }
    checkpoint_provenance: dict[str, Any]
    if args.smoke_test:
        smoke_names = (
            "model.safetensors",
            "config.json",
            "training_args.json",
            "environment.json",
            "epoch_metrics.json",
            "selected_checkpoint.json",
        )
        hashes = {
            name: sha256_file(model_path / name)
            for name in smoke_names
            if (model_path / name).is_file()
        }
        checkpoint_provenance = {
            "artifact_sha256": hashes,
            "checkpoint_fingerprint_sha256": _mapping_fingerprint(hashes),
        }
    else:
        checkpoint_provenance = _validate_final_checkpoint(root, model_path)

    set_reproducible_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    if not args.smoke_test and (device.type != "cuda" or not use_bf16):
        raise RuntimeError("Final evaluation requires the canonical CUDA/BF16 environment")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Evaluation requires the saved fast tokenizer")
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, local_files_only=True
    )
    checkpoint_id2label = {
        int(key): value for key, value in model.config.id2label.items()
    }
    if checkpoint_id2label != id_to_label or model.config.label2id != label_to_id:
        raise ValueError("Checkpoint labels do not exactly match frozen ML-v1 labels")
    model.to(device)

    records = load_jsonl(paths["input"], args.max_records)
    if not args.smoke_test and len(records) != CANONICAL_TEST_RECORDS:
        raise ValueError(
            f"Final evaluation requires {CANONICAL_TEST_RECORDS} records, "
            f"received {len(records)}"
        )
    build = build_aligned_features(
        records, tokenizer, label_to_id, args.max_length, args.stride
    )
    if build.summary["truncated_spans"]:
        raise ValueError("Evaluation alignment contains truncated entity spans")
    unexpected = _unexplained_alignment_failures(build.summary)
    if unexpected:
        raise ValueError(f"Unexplained evaluation alignments: {unexpected[:10]}")

    input_sha256 = sha256_file(paths["input"])
    provenance = {
        "manifest_sha256": sha256_file(paths["manifest"]),
        "label_config_sha256": sha256_file(paths["labels"]),
        "decision_gate_sha256": gate_sha256,
        "input_sha256": input_sha256,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": checkpoint_provenance,
    }
    started = datetime.now(timezone.utc).isoformat()
    receipt = {
        "status": "STARTED",
        "started_utc": started,
        "split": args.split_name,
        "input": _path_display(paths["input"], root),
        "model": _path_display(model_path, root),
        "reports": _path_display(reports, root),
        "prediction_cache": _path_display(prediction_cache, root),
        "records": len(records),
        "settings": {
            "max_length": args.max_length,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "bf16": use_bf16,
            "device": str(device),
            "smoke_test": args.smoke_test,
        },
        "provenance": provenance,
    }
    # Repeat the guard after expensive setup, then acquire the exclusive lock
    # immediately before the sole inference call.
    _guard_absent(protected_artifacts)
    _write_exclusive_json(receipt_path, receipt)
    try:
        result = _execute_evaluation(
            model=model,
            tokenizer=tokenizer,
            records=records,
            build=build,
            id_to_label=id_to_label,
            label_config=label_config,
            label_to_id=label_to_id,
            batch_size=args.batch_size,
            device=device,
            use_bf16=use_bf16,
            split_name=args.split_name,
            reports=reports,
            prediction_cache=prediction_cache,
            gates=gates,
            smoke_test=args.smoke_test,
            started=started,
            input_path=paths["input"],
            provenance=provenance,
        )
        validate_manifest(root, paths["manifest"])
        receipt.update(
            {
                "status": "COMPLETE",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "decision": result["decision"]["decision"],
                "inference_seconds": result["inference_seconds"],
                "prediction_cache_sha256": result["prediction_cache_sha256"],
                "report_sha256": result["report_sha256"],
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

    overall = result["overall"]
    print(
        f"{args.split_name} micro_f1={overall['micro']['f1']:.6f} "
        f"macro_f1={overall['macro']['f1']:.6f} "
        f"high_risk_recall={overall['high_risk']['recall']:.6f}",
        flush=True,
    )
    print(result["decision"]["decision"], flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default="configs/ml_v1_frozen_manifest.json")
    parser.add_argument("--labels", default="configs/securelogx_labels.json")
    parser.add_argument(
        "--gates", default="configs/ml_v1_onnx_validation_gate.json"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="data/split/test.jsonl")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    return _run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
