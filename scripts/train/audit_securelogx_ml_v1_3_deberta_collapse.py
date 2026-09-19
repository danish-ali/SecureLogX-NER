#!/usr/bin/env python3
"""Development-only audit for the ML-v1.3 DeBERTa standard-dev collapse.

This script intentionally uses only:
- frozen standard/challenge development data
- cached BERT/DeBERTa development predictions
- committed v1.3 alignment/security reports

It does NOT read the sealed challenge or original test set, does NOT retrain,
does NOT export ONNX, and does NOT modify Java.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from securelogx_training_common import (  # noqa: E402
    load_canonical_labels,
    load_jsonl,
    score_predictions,
    supported_macro,
)

SEALED_SHA256 = "6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a"

PATHS = {
    "labels": "configs/securelogx_labels.json",
    "manifest": "configs/ml_v1_3_dataset_manifest.json",
    "standard_dev": "data/split/dev.jsonl",
    "challenge_dev": "data/ml_v1_3/real_structure/dev_challenge.jsonl",
    "bert_standard_predictions": "output_securelogx/ml-v1.3/bert-base-cased/dev_standard_predictions.json",
    "deberta_standard_predictions": "output_securelogx/ml-v1.3/deberta-v3-base/dev_standard_predictions.json",
    "bert_challenge_predictions": "output_securelogx/ml-v1.3/bert-base-cased/dev_challenge_predictions.json",
    "deberta_challenge_predictions": "output_securelogx/ml-v1.3/deberta-v3-base/dev_challenge_predictions.json",
    "bert_standard_metrics": "reports/ml_v1_3_bert_dev_standard_metrics.json",
    "deberta_standard_metrics": "reports/ml_v1_3_deberta_dev_standard_metrics.json",
    "bert_alignment": "reports/ml_v1_3_bert_tokenizer_alignment.json",
    "deberta_alignment": "reports/ml_v1_3_deberta_tokenizer_alignment.json",
    "bert_security": "reports/ml_v1_3_bert_security_outcomes.json",
    "deberta_security": "reports/ml_v1_3_deberta_security_outcomes.json",
}

REPORT_JSON = "reports/ml_v1_3_deberta_collapse_audit.json"
REPORT_MD = "reports/ml_v1_3_deberta_collapse_audit.md"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_predictions(path: Path) -> list[list[dict[str, Any]]]:
    payload = load_json(path)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Missing predictions array: {path}")
    return predictions


def gold_entities(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for entity in record.get("entities", []):
        if isinstance(entity, Mapping):
            entities.append(
                {
                    "start": int(entity["start"]),
                    "end": int(entity["end"]),
                    "label": str(entity["label"]),
                }
            )
        else:
            start, end, label = entity
            entities.append({"start": int(start), "end": int(end), "label": str(label)})
    return entities


def trim_span(text: str, span: Mapping[str, Any]) -> dict[str, Any]:
    """Trim whitespace only; preserve every non-whitespace character."""
    result = dict(span)
    start = max(0, int(result["start"]))
    end = min(len(text), int(result["end"]))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    result["start"] = start
    result["end"] = end
    return result


def trim_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> list[list[dict[str, Any]]]:
    if len(records) != len(predictions):
        raise ValueError("Record/prediction length mismatch")
    return [
        [trim_span(str(record["text"]), span) for span in record_predictions]
        for record, record_predictions in zip(records, predictions)
    ]


def score(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
    split: str,
) -> dict[str, Any]:
    metrics = score_predictions(records, predictions, labels, split)
    metrics["supported_macro"] = supported_macro(metrics)
    return metrics


def source_bucket(record: Mapping[str, Any]) -> str:
    meta = record.get("meta", {})
    if not isinstance(meta, Mapping):
        meta = {}
    for key in ("source", "source_name", "dataset_source", "origin"):
        value = meta.get(key)
        if value:
            lower = str(value).lower()
            if "gretel" in lower:
                return "gretel"
            if "synthetic" in lower or "securelogx" in lower or "generated" in lower:
                return "synthetic"
            return str(value)
    family = str(meta.get("template_family") or "").lower()
    if "gretel" in family:
        return "gretel"
    if family:
        return "synthetic"
    return "unknown"


def source_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
    prefix: str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[source_bucket(record)].append(index)

    result: dict[str, Any] = {}
    for bucket, indices in sorted(groups.items()):
        subset_records = [records[i] for i in indices]
        subset_predictions = [predictions[i] for i in indices]
        metrics = score(subset_records, subset_predictions, labels, f"{prefix}:{bucket}")
        result[bucket] = {
            "records": len(indices),
            "entities": sum(len(gold_entities(record)) for record in subset_records),
            "micro_f1": metrics["micro"]["f1"],
            "macro_f1": metrics["supported_macro"]["f1"],
        }
    return result


def exact_confusion(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    wrong_class: Counter[tuple[str, str]] = Counter()
    boundary_overlap: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    predicted: Counter[str] = Counter()

    for record, record_predictions in zip(records, predictions):
        gold = gold_entities(record)
        prediction_by_exact = {
            (int(pred["start"]), int(pred["end"])): str(pred["label"])
            for pred in record_predictions
        }
        predicted.update(str(pred["label"]) for pred in record_predictions)

        for entity in gold:
            key = (entity["start"], entity["end"])
            predicted_label = prediction_by_exact.get(key)
            if predicted_label is not None:
                if predicted_label != entity["label"]:
                    wrong_class[(entity["label"], predicted_label)] += 1
                continue

            overlapping = any(
                max(entity["start"], int(pred["start"]))
                < min(entity["end"], int(pred["end"]))
                for pred in record_predictions
            )
            if overlapping:
                boundary_overlap[entity["label"]] += 1
            else:
                missed[entity["label"]] += 1

    return {
        "wrong_class": [
            {"gold": gold, "predicted": pred, "count": count}
            for (gold, pred), count in wrong_class.most_common()
        ],
        "boundary_overlap_gold": dict(boundary_overlap),
        "missed_gold": dict(missed),
        "predicted_label_frequency": dict(predicted),
    }


def accountlike_false_positive_keys(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()

    for index, (record, record_predictions) in enumerate(zip(records, predictions)):
        meta = record.get("meta", {})
        if not isinstance(meta, Mapping):
            meta = {}
        family = str(meta.get("template_family") or "")
        role = str(meta.get("context_role") or "")
        if family != "key_value_accountlike_batch_v1" and role != "technical_account":
            continue

        gold = gold_entities(record)
        for pred in record_predictions:
            if str(pred["label"]) != "BUSINESS_ID":
                continue
            start = int(pred["start"])
            end = int(pred["end"])
            overlaps_gold = any(
                max(start, entity["start"]) < min(end, entity["end"])
                for entity in gold
            )
            if not overlaps_gold:
                result.add((index, start, end))
    return result


def reproduce(
    name: str,
    committed: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    expected = float(committed["micro"]["f1"])
    actual = float(recomputed["micro"]["f1"])
    if abs(expected - actual) > 1e-12:
        raise ValueError(f"{name} metric mismatch: recomputed={actual}, committed={expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    manifest = load_json(root / PATHS["manifest"])
    sealed = manifest.get("sealed_challenge", {})
    if sealed.get("sha256") != SEALED_SHA256 or sealed.get("file_opened") is not False:
        raise ValueError("Sealed-challenge guard does not match frozen ML-v1.3 manifest")

    ontology = load_canonical_labels(root / PATHS["labels"])
    labels = [str(value) for value in ontology["entities"]]
    if len(labels) != 25 or len(ontology["bio_labels"]) != 51:
        raise ValueError("Canonical ontology changed")

    standard_records = load_jsonl(root / PATHS["standard_dev"])
    challenge_records = load_jsonl(root / PATHS["challenge_dev"])

    bert_std_predictions = load_predictions(root / PATHS["bert_standard_predictions"])
    deberta_std_predictions = load_predictions(root / PATHS["deberta_standard_predictions"])
    bert_ch_predictions = load_predictions(root / PATHS["bert_challenge_predictions"])
    deberta_ch_predictions = load_predictions(root / PATHS["deberta_challenge_predictions"])

    for name, records, predictions in (
        ("BERT standard", standard_records, bert_std_predictions),
        ("DeBERTa standard", standard_records, deberta_std_predictions),
        ("BERT challenge", challenge_records, bert_ch_predictions),
        ("DeBERTa challenge", challenge_records, deberta_ch_predictions),
    ):
        if len(records) != len(predictions):
            raise ValueError(f"{name} record/prediction count mismatch")

    bert_std_raw = score(standard_records, bert_std_predictions, labels, "audit_bert_standard")
    deberta_std_raw = score(standard_records, deberta_std_predictions, labels, "audit_deberta_standard")
    bert_ch_raw = score(challenge_records, bert_ch_predictions, labels, "audit_bert_challenge")
    deberta_ch_raw = score(challenge_records, deberta_ch_predictions, labels, "audit_deberta_challenge")

    reproduce(
        "BERT standard",
        load_json(root / PATHS["bert_standard_metrics"]),
        bert_std_raw,
    )
    reproduce(
        "DeBERTa standard",
        load_json(root / PATHS["deberta_standard_metrics"]),
        deberta_std_raw,
    )

    deberta_std_trimmed_predictions = trim_predictions(
        standard_records, deberta_std_predictions
    )
    bert_std_trimmed_predictions = trim_predictions(
        standard_records, bert_std_predictions
    )
    deberta_std_trimmed = score(
        standard_records,
        deberta_std_trimmed_predictions,
        labels,
        "audit_deberta_standard_trimmed",
    )
    bert_std_trimmed = score(
        standard_records,
        bert_std_trimmed_predictions,
        labels,
        "audit_bert_standard_trimmed",
    )

    bert_alignment = load_json(root / PATHS["bert_alignment"])["alignments"]
    deberta_alignment = load_json(root / PATHS["deberta_alignment"])["alignments"]
    d_std_alignment = deberta_alignment["standard_dev"]
    d_ch_alignment = deberta_alignment["challenge_dev"]

    d_std_adjusted_ratio = (
        d_std_alignment["boundary_adjusted"] / d_std_alignment["total_spans"]
    )
    d_ch_adjusted_ratio = (
        d_ch_alignment["boundary_adjusted"] / d_ch_alignment["total_spans"]
    )

    gain = (
        deberta_std_trimmed["micro"]["f1"] - deberta_std_raw["micro"]["f1"]
    )
    if gain >= 0.35 and d_std_adjusted_ratio >= 0.50 and d_ch_adjusted_ratio <= 0.10:
        if deberta_std_trimmed["micro"]["f1"] >= 0.75:
            diagnosis = "IMPLEMENTATION/TOKENIZATION DEFECT IDENTIFIED"
        else:
            diagnosis = "MIXED IMPLEMENTATION AND DOMAIN EFFECT"
    elif gain >= 0.15 and d_std_adjusted_ratio >= 0.30:
        diagnosis = "MIXED IMPLEMENTATION AND DOMAIN EFFECT"
    else:
        diagnosis = "CAUSE NOT YET ESTABLISHED"

    bert_security = load_json(root / PATHS["bert_security"])
    deberta_security = load_json(root / PATHS["deberta_security"])

    bert_accountlike = accountlike_false_positive_keys(
        standard_records, bert_std_predictions
    )
    deberta_accountlike = accountlike_false_positive_keys(
        standard_records, deberta_std_predictions
    )

    source_analysis = {
        "bert_raw": source_metrics(
            standard_records, bert_std_predictions, labels, "bert_source"
        ),
        "deberta_raw": source_metrics(
            standard_records, deberta_std_predictions, labels, "deberta_source"
        ),
        "deberta_trimmed": source_metrics(
            standard_records,
            deberta_std_trimmed_predictions,
            labels,
            "deberta_source_trimmed",
        ),
    }

    payload = {
        "scope": "development-only; sealed challenge and original test not accessed",
        "sealed_manifest_sha256": SEALED_SHA256,
        "raw": {
            "bert_standard_micro_f1": bert_std_raw["micro"]["f1"],
            "bert_standard_macro_f1": bert_std_raw["supported_macro"]["f1"],
            "deberta_standard_micro_f1": deberta_std_raw["micro"]["f1"],
            "deberta_standard_macro_f1": deberta_std_raw["supported_macro"]["f1"],
            "bert_challenge_micro_f1": bert_ch_raw["micro"]["f1"],
            "deberta_challenge_micro_f1": deberta_ch_raw["micro"]["f1"],
        },
        "whitespace_trim_diagnostic": {
            "bert_standard_micro_f1": bert_std_trimmed["micro"]["f1"],
            "deberta_standard_micro_f1": deberta_std_trimmed["micro"]["f1"],
            "deberta_micro_f1_gain": gain,
        },
        "alignment": {
            "deberta_standard_boundary_adjusted": d_std_alignment["boundary_adjusted"],
            "deberta_standard_total_spans": d_std_alignment["total_spans"],
            "deberta_standard_adjusted_ratio": d_std_adjusted_ratio,
            "deberta_challenge_boundary_adjusted": d_ch_alignment["boundary_adjusted"],
            "deberta_challenge_total_spans": d_ch_alignment["total_spans"],
            "deberta_challenge_adjusted_ratio": d_ch_adjusted_ratio,
            "bert_standard": bert_alignment["standard_dev"],
        },
        "source_analysis": source_analysis,
        "deberta_confusion_raw": exact_confusion(
            standard_records, deberta_std_predictions
        ),
        "deberta_confusion_trimmed": exact_confusion(
            standard_records, deberta_std_trimmed_predictions
        ),
        "accountlike_o_to_business_id": {
            "bert": len(bert_accountlike),
            "deberta": len(deberta_accountlike),
            "shared_exact_record_span": len(bert_accountlike & deberta_accountlike),
        },
        "security": {
            "bert_standard": bert_security["standard"],
            "deberta_standard": deberta_security["standard"],
            "bert_challenge": bert_security["challenge"],
            "deberta_challenge": deberta_security["challenge"],
        },
        "diagnosis": diagnosis,
        "sealed_validation_status": "BLOCKED FROM SEALED VALIDATION",
    }

    report_path = root / REPORT_JSON
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ML-v1.3 DeBERTa Collapse Audit",
        "",
        "**Scope: development only. Sealed challenge and original test were not accessed.**",
        "",
        f"Primary diagnosis: **{diagnosis}**",
        "",
        "**DeBERTa remains BLOCKED FROM SEALED VALIDATION pending review of this audit.**",
        "",
        "## Exact-metric reproduction",
        "",
        "| Model/view | Micro F1 | Macro F1 |",
        "|---|---:|---:|",
        f"| BERT standard | {bert_std_raw['micro']['f1']:.6f} | {bert_std_raw['supported_macro']['f1']:.6f} |",
        f"| DeBERTa standard | {deberta_std_raw['micro']['f1']:.6f} | {deberta_std_raw['supported_macro']['f1']:.6f} |",
        f"| BERT challenge | {bert_ch_raw['micro']['f1']:.6f} | {bert_ch_raw['supported_macro']['f1']:.6f} |",
        f"| DeBERTa challenge | {deberta_ch_raw['micro']['f1']:.6f} | {deberta_ch_raw['supported_macro']['f1']:.6f} |",
        "",
        "## Whitespace-boundary diagnostic",
        "",
        "| Model | Raw standard micro F1 | Whitespace-trimmed micro F1 | Delta |",
        "|---|---:|---:|---:|",
        f"| BERT | {bert_std_raw['micro']['f1']:.6f} | {bert_std_trimmed['micro']['f1']:.6f} | {bert_std_trimmed['micro']['f1']-bert_std_raw['micro']['f1']:+.6f} |",
        f"| DeBERTa | {deberta_std_raw['micro']['f1']:.6f} | {deberta_std_trimmed['micro']['f1']:.6f} | {gain:+.6f} |",
        "",
        f"- DeBERTa standard boundary-adjusted spans: {d_std_alignment['boundary_adjusted']}/{d_std_alignment['total_spans']} ({d_std_adjusted_ratio:.2%}).",
        f"- DeBERTa challenge boundary-adjusted spans: {d_ch_alignment['boundary_adjusted']}/{d_ch_alignment['total_spans']} ({d_ch_adjusted_ratio:.2%}).",
        f"- Shared exact account-like O -> BUSINESS_ID errors: {len(bert_accountlike & deberta_accountlike)}.",
        "",
        "This trimming test is diagnostic only. It does not change historical metrics or the predeclared selection result.",
    ]
    (root / REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Diagnosis: {diagnosis}")
    print("DeBERTa: BLOCKED FROM SEALED VALIDATION")
    print(f"Wrote: {REPORT_JSON}")
    print(f"Wrote: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
