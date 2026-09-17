#!/usr/bin/env python3
"""Analyze frozen ML-v1.1 development predictions without model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = {
    "standard_dev": (
        Path("data/split/dev.jsonl"),
        Path("output_securelogx/ml-v1.1/bert-base-cased/dev_standard_predictions.json"),
    ),
    "challenge_dev": (
        Path("data/ml_v1_1/context_contrast/dev_challenge.jsonl"),
        Path("output_securelogx/ml-v1.1/bert-base-cased/dev_challenge_predictions.json"),
    ),
}
EPOCH_METRICS = Path(
    "output_securelogx/ml-v1.1/bert-base-cased/checkpoints/epoch-{epoch}/epoch_metrics.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _signature(span: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["label"])


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    return max(
        0,
        min(int(left["end"]), int(right["end"]))
        - max(int(left["start"]), int(right["start"])),
    )


def _context(record: Mapping[str, Any]) -> str:
    meta = record.get("meta", {})
    if not isinstance(meta, Mapping):
        return "unclassified"
    category = str(meta.get("contrast_category") or "").strip()
    role = str(meta.get("context_role") or "").strip()
    if category:
        return f"{category}/{role}" if role else category
    family = str(meta.get("template_family") or "").strip()
    source = str(meta.get("source") or "").strip()
    return family or source or "unclassified"


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _load_cached_split(
    root: Path, name: str, record_path: Path, cache_path: Path
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], dict[str, Any]]:
    records = load_jsonl(root / record_path)
    cache = load_json(root / cache_path)
    if not isinstance(cache, dict):
        raise ValueError(f"{cache_path} must contain an object")
    predictions = cache.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"{cache_path} has no predictions list")
    if len(records) != len(predictions) or cache.get("records") != len(records):
        raise ValueError(f"{name} cached record count mismatch")
    actual_input_hash = sha256_file(root / record_path)
    if cache.get("input_sha256") != actual_input_hash:
        raise ValueError(f"{name} cached input hash mismatch")
    if cache.get("selected_epoch") != 3:
        raise ValueError(f"{name} cache is not the frozen selected epoch 3 output")
    normalized: list[list[dict[str, Any]]] = []
    for index, row in enumerate(predictions):
        if not isinstance(row, list):
            raise ValueError(f"{name} prediction row {index} is not a list")
        normalized.append([dict(span) for span in row])
    return records, normalized, cache


def analyze_split(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    directional: Counter[tuple[str, str]] = Counter()
    directional_confidence: dict[tuple[str, str], list[float]] = defaultdict(list)
    directional_contexts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    business_fp: Counter[str] = Counter()
    business_fp_confidence: dict[str, list[float]] = defaultdict(list)
    business_fp_contexts: dict[str, Counter[str]] = defaultdict(Counter)
    business_boundary_errors = 0
    business_predictions = 0
    business_true_positives = 0

    for record, raw_predictions in zip(records, predictions):
        gold = [
            {"start": int(start), "end": int(end), "label": str(label)}
            for start, end, label in record["entities"]
        ]
        predicted = [dict(span) for span in raw_predictions]
        business_predictions += sum(
            1 for span in predicted if str(span["label"]) == "BUSINESS_ID"
        )
        unmatched_gold = set(range(len(gold)))
        unmatched_predicted = set(range(len(predicted)))
        by_signature: dict[tuple[int, int, str], list[int]] = defaultdict(list)
        for pred_index, span in enumerate(predicted):
            by_signature[_signature(span)].append(pred_index)
        for gold_index, span in enumerate(gold):
            pred_index = next(
                (
                    candidate
                    for candidate in by_signature.get(_signature(span), [])
                    if candidate in unmatched_predicted
                ),
                None,
            )
            if pred_index is None:
                continue
            unmatched_gold.remove(gold_index)
            unmatched_predicted.remove(pred_index)
            if span["label"] == "BUSINESS_ID":
                business_true_positives += 1

        candidates: list[tuple[float, int, int]] = []
        for gold_index in unmatched_gold:
            for pred_index in unmatched_predicted:
                overlap = _overlap(gold[gold_index], predicted[pred_index])
                if not overlap:
                    continue
                union = max(
                    int(gold[gold_index]["end"]), int(predicted[pred_index]["end"])
                ) - min(
                    int(gold[gold_index]["start"]), int(predicted[pred_index]["start"])
                )
                candidates.append((overlap / union, gold_index, pred_index))
        for _iou, gold_index, pred_index in sorted(candidates, reverse=True):
            if gold_index not in unmatched_gold or pred_index not in unmatched_predicted:
                continue
            unmatched_gold.remove(gold_index)
            unmatched_predicted.remove(pred_index)
            gold_span = gold[gold_index]
            prediction = predicted[pred_index]
            gold_label = str(gold_span["label"])
            predicted_label = str(prediction["label"])
            confidence = float(prediction["confidence"])
            context = _context(record)
            if gold_label != predicted_label:
                key = (gold_label, predicted_label)
                directional[key] += 1
                directional_confidence[key].append(confidence)
                directional_contexts[key][context] += 1
                if predicted_label == "BUSINESS_ID":
                    business_fp[gold_label] += 1
                    business_fp_confidence[gold_label].append(confidence)
                    business_fp_contexts[gold_label][context] += 1
            elif predicted_label == "BUSINESS_ID":
                business_boundary_errors += 1

        for pred_index in sorted(unmatched_predicted):
            prediction = predicted[pred_index]
            predicted_label = str(prediction["label"])
            confidence = float(prediction["confidence"])
            context = _context(record)
            key = ("<SPURIOUS>", predicted_label)
            directional[key] += 1
            directional_confidence[key].append(confidence)
            directional_contexts[key][context] += 1
            if predicted_label == "BUSINESS_ID":
                business_fp["<SPURIOUS>"] += 1
                business_fp_confidence["<SPURIOUS>"].append(confidence)
                business_fp_contexts["<SPURIOUS>"][context] += 1

    false_positive_count = business_predictions - business_true_positives
    classified_fp_count = sum(business_fp.values()) + business_boundary_errors
    if false_positive_count != classified_fp_count:
        raise AssertionError(
            f"BUSINESS_ID false-positive accounting mismatch: {false_positive_count} "
            f"!= {classified_fp_count}"
        )
    table: list[dict[str, Any]] = []
    wrong_class_or_spurious = sum(business_fp.values())
    for gold_label, count in business_fp.most_common():
        contexts = business_fp_contexts[gold_label]
        table.append(
            {
                "gold_label": gold_label,
                "predicted_business_id_count": count,
                "percentage_of_wrong_class_or_spurious_business_id": (
                    100.0 * count / wrong_class_or_spurious
                    if wrong_class_or_spurious
                    else 0.0
                ),
                "typical_context": "; ".join(
                    f"{name} ({value})" for name, value in contexts.most_common(3)
                ),
                "confidence": _distribution(business_fp_confidence[gold_label]),
            }
        )
    return {
        "records": len(records),
        "business_id_predictions": business_predictions,
        "business_id_true_positives": business_true_positives,
        "business_id_false_positives": false_positive_count,
        "business_id_wrong_class_or_spurious": wrong_class_or_spurious,
        "business_id_boundary_errors": business_boundary_errors,
        "business_id_false_positive_table": table,
        "directional_confusions": {
            f"{gold}->{predicted_label}": {
                "count": count,
                "confidence": _distribution(
                    directional_confidence[(gold, predicted_label)]
                ),
                "typical_context": "; ".join(
                    f"{name} ({value})"
                    for name, value in directional_contexts[
                        (gold, predicted_label)
                    ].most_common(3)
                ),
            }
            for (gold, predicted_label), count in sorted(directional.items())
        },
    }


def _metric(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def checkpoint_tradeoffs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in (1, 2, 3):
        value = load_json(root / Path(str(EPOCH_METRICS).format(epoch=epoch)))
        standard = value["standard_dev_metrics"]
        challenge = value["challenge_dev_metrics"]
        context = challenge["context_diagnostics"]
        rows.append(
            {
                "epoch": epoch,
                "standard_macro_f1": _metric(standard, "supported_macro", "f1"),
                "standard_micro_f1": _metric(standard, "micro", "f1"),
                "standard_business_id_precision": _metric(
                    standard, "per_label", "BUSINESS_ID", "precision"
                ),
                "standard_business_id_recall": _metric(
                    standard, "per_label", "BUSINESS_ID", "recall"
                ),
                "standard_business_id_f1": _metric(
                    standard, "per_label", "BUSINESS_ID", "f1"
                ),
                "standard_ssn_recall": _metric(
                    standard, "per_label", "SSN", "recall"
                ),
                "standard_ssn_f1": _metric(standard, "per_label", "SSN", "f1"),
                "standard_high_risk_recall": _metric(
                    standard, "high_risk", "recall"
                ),
                "challenge_micro_f1": _metric(challenge, "micro", "f1"),
                "challenge_macro_f1": _metric(
                    challenge, "supported_macro", "f1"
                ),
                "challenge_business_id_precision": _metric(
                    challenge, "per_label", "BUSINESS_ID", "precision"
                ),
                "challenge_business_id_recall": _metric(
                    challenge, "per_label", "BUSINESS_ID", "recall"
                ),
                "challenge_business_id_f1": _metric(
                    challenge, "per_label", "BUSINESS_ID", "f1"
                ),
                "challenge_high_risk_recall": _metric(
                    challenge, "high_risk", "recall"
                ),
                "challenge_record_error_rate": context["record_error_rate"],
                "challenge_context_target_error_rate": context[
                    "context_target_error_rate"
                ],
                "challenge_worst_category_error_rate": context[
                    "maximum_category_error_rate"
                ],
            }
        )
    return rows


def analyze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    splits: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for name, (record_path, cache_path) in SPLITS.items():
        records, predictions, cache = _load_cached_split(
            root, name, record_path, cache_path
        )
        splits[name] = analyze_split(records, predictions)
        evidence[name] = {
            "records_path": record_path.as_posix(),
            "records_sha256": sha256_file(root / record_path),
            "cache_path": cache_path.as_posix(),
            "cache_sha256": sha256_file(root / cache_path),
            "selected_epoch": cache["selected_epoch"],
        }

    selected = load_json(
        root / "output_securelogx/ml-v1.1/bert-base-cased/checkpoints/epoch-3/epoch_metrics.json"
    )
    categories = selected["challenge_dev_metrics"]["context_diagnostics"][
        "by_category"
    ]
    worst_name, worst = max(
        categories.items(), key=lambda item: float(item[1]["record_error_rate"])
    )
    return {
        "analysis_version": 1,
        "method": "frozen cached development predictions_only_no_model_execution",
        "evidence": evidence,
        "splits": splits,
        "checkpoint_tradeoffs": checkpoint_tradeoffs(root),
        "actual_worst_category": {
            "name": worst_name,
            "record_error_rate": worst["record_error_rate"],
            "records": worst["records"],
            "records_with_errors": worst["records_with_errors"],
            "context_target_error_rate": worst["context_target_error_rate"],
            "morphology_over_context_errors": worst[
                "morphology_over_context_errors"
            ],
            "morphology_over_context_share_of_target_errors": worst[
                "morphology_over_context_share_of_target_errors"
            ],
            "dominant_confusion_direction": "technical reference O -> IP_ADDRESS",
        },
        "sealed_challenge": {
            "dataset_parsed": False,
            "predictions_accessed": False,
            "model_executed": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze(Path(args.repo_root))
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
