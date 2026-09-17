"""Finalize the stopped ML-v1.1 experiment from selected dev evidence only.

This command is valid only after the predeclared development gate fails.  It
does not load a model, parse the sealed challenge, or parse/evaluate the
original ML-v1 test.  It renders honest dev-only analyses and a terminal NOT
READY decision, while recording the regression benchmark as NOT_RUN.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from securelogx_ml_v1_1_metrics import supported_macro, tag_context_errors
from securelogx_training_common import (
    load_canonical_labels,
    load_json,
    load_jsonl,
    score_predictions,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "output_securelogx/ml-v1.1/bert-base-cased"
REPORTS = ROOT / "reports"
DEV_GATE = RUN_ROOT / "dev_gate_decision.json"
DEV_CACHE = RUN_ROOT / "dev_challenge_predictions.json"
DEV_INPUT = ROOT / "data/ml_v1_1/context_contrast/dev_challenge.jsonl"
STANDARD_METRICS = REPORTS / "ml_v1_1_dev_standard_metrics.json"
CHALLENGE_METRICS = REPORTS / "ml_v1_1_dev_challenge_metrics.json"
HISTORICAL_DEV = REPORTS / "dev_metrics.json"
HISTORICAL_TEST = REPORTS / "test_metrics.json"
SELECTED = RUN_ROOT / "best-checkpoint/selected_checkpoint.json"
TRAINING_GATE = ROOT / "configs/ml_v1_1_training_gate.json"
DATASET_MANIFEST = ROOT / "configs/ml_v1_1_dataset_manifest.json"

REGRESSION_STATUS = REPORTS / "ml_v1_1_regression_test_metrics.json"
BUSINESS_REPORT = REPORTS / "ml_v1_1_business_id_subtype_analysis.md"
SSN_REPORT = REPORTS / "ml_v1_1_ssn_context_analysis.md"
GENERALIZATION_REPORT = REPORTS / "ml_v1_1_generalization_analysis.md"
ERROR_REPORT = REPORTS / "ml_v1_1_error_analysis.md"
ERROR_CSV = REPORTS / "ml_v1_1_model_errors.csv"
COMPARISON_REPORT = REPORTS / "ml_v1_vs_v1_1_comparison.md"
DECISION_REPORT = REPORTS / "ml_v1_1_model_validation_decision.md"

FORBIDDEN_RESULTS = (
    REPORTS / "ml_v1_1_sealed_challenge_metrics.json",
    RUN_ROOT / "sealed_challenge_evaluation_receipt.json",
    RUN_ROOT / "sealed_challenge_predictions.json",
    RUN_ROOT / "regression_evaluation_receipt.json",
    RUN_ROOT / "regression_test_predictions.json",
)
OUTPUTS = (
    REGRESSION_STATUS,
    BUSINESS_REPORT,
    SSN_REPORT,
    GENERALIZATION_REPORT,
    ERROR_REPORT,
    ERROR_CSV,
    COMPARISON_REPORT,
    DECISION_REPORT,
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


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _business_subtype(record: Mapping[str, Any], start: int, end: int) -> str:
    for item in record.get("meta", {}).get("entity_provenance", []):
        if (
            int(item.get("start", -1)) == start
            and int(item.get("end", -1)) == end
            and item.get("label") == "BUSINESS_ID"
        ):
            return str(
                item.get("business_id_subtype")
                or item.get("source_subtype")
                or item.get("source_label")
                or "BUSINESS_ID"
            ).upper()
    return "BUSINESS_ID"


def _business_analysis(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
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
            true_positive[subtype] += (
                int(start), int(end), "BUSINESS_ID"
            ) in signatures
    return [
        {
            "subtype": subtype,
            "support": support[subtype],
            "true_positives": true_positive[subtype],
            "false_negatives": support[subtype] - true_positive[subtype],
            "recall": (
                true_positive[subtype] / support[subtype]
                if support[subtype]
                else None
            ),
        }
        for subtype in BUSINESS_SUBTYPES
    ]


def _overlap(start: int, end: int, span: Mapping[str, Any]) -> bool:
    return int(span["start"]) < end and int(span["end"]) > start


def _ssn_analysis(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    result: Counter[str] = Counter()
    for record, predicted in zip(records, predictions):
        gold = [
            {"start": int(start), "end": int(end), "label": str(label)}
            for start, end, label in record["entities"]
        ]
        exact = {
            (int(span["start"]), int(span["end"]), str(span["label"]))
            for span in predicted
        }
        ssn_gold = [span for span in gold if span["label"] == "SSN"]
        business_gold = [span for span in gold if span["label"] == "BUSINESS_ID"]
        result["true_ssn_support"] += len(ssn_gold)
        for span in ssn_gold:
            signature = (span["start"], span["end"], "SSN")
            result["true_ssns_correctly_recognized"] += signature in exact
            result["ssns_missed_as_business_id"] += any(
                guess["label"] == "BUSINESS_ID"
                and _overlap(span["start"], span["end"], guess)
                for guess in predicted
            )
            result["ssn_boundary_errors"] += any(
                guess["label"] == "SSN"
                and _overlap(span["start"], span["end"], guess)
                and (int(guess["start"]), int(guess["end"]), "SSN") != signature
                for guess in predicted
            )
        for guess in (span for span in predicted if span["label"] == "SSN"):
            if any(
                int(guess["start"]) == span["start"]
                and int(guess["end"]) == span["end"]
                for span in ssn_gold
            ):
                continue
            if any(_overlap(span["start"], span["end"], guess) for span in ssn_gold):
                continue
            if any(
                _overlap(span["start"], span["end"], guess)
                for span in business_gold
            ):
                result["business_id_values_incorrectly_classified_as_ssn"] += 1
            else:
                result["other_non_ssn_values_incorrectly_classified_as_ssn"] += 1
    keys = (
        "true_ssn_support",
        "true_ssns_correctly_recognized",
        "business_id_values_incorrectly_classified_as_ssn",
        "other_non_ssn_values_incorrectly_classified_as_ssn",
        "ssns_missed_as_business_id",
        "ssn_boundary_errors",
    )
    return {key: result[key] for key in keys}


def _write_errors(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["context_error_tags"] = ",".join(value.get("context_error_tags", []))
            writer.writerow(value)


def main() -> int:
    existing = [str(path.relative_to(ROOT)) for path in OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite final artifacts: " + ", ".join(existing))
    forbidden = [str(path.relative_to(ROOT)) for path in FORBIDDEN_RESULTS if path.exists()]
    if forbidden:
        raise FileExistsError("Post-gate inference artifact unexpectedly exists: " + ", ".join(forbidden))

    gate = load_json(DEV_GATE)
    if gate.get("status") != "FAIL" or gate.get("decision") != "NOT READY FOR SEALED CHALLENGE EVALUATION":
        raise ValueError("This finalizer is permitted only for the failed development gate")
    selected = load_json(SELECTED)
    if gate.get("checkpoint_fingerprint_sha256") != selected.get("checkpoint_fingerprint_sha256"):
        raise ValueError("Development gate is not bound to the selected checkpoint")
    if sha256_file(TRAINING_GATE) != gate.get("gate_sha256"):
        raise ValueError("Predeclared training gate changed")
    if sha256_file(DATASET_MANIFEST) != gate.get("dataset_manifest_sha256"):
        raise ValueError("Frozen dataset manifest changed")
    if sha256_file(RUN_ROOT / "best-checkpoint/model.safetensors") != selected.get("model_sha256"):
        raise ValueError("Selected model changed")

    cache = load_json(DEV_CACHE)
    records = load_jsonl(DEV_INPUT)
    predictions = cache.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(records) or len(records) != 480:
        raise ValueError("Selected development cache is incomplete")
    if cache.get("input_sha256") != sha256_file(DEV_INPUT):
        raise ValueError("Selected development cache input hash mismatch")
    if cache.get("checkpoint_fingerprint_sha256") != selected.get("checkpoint_fingerprint_sha256"):
        raise ValueError("Selected development cache checkpoint mismatch")

    labels = load_canonical_labels(ROOT / "configs/securelogx_labels.json")["entities"]
    scored = score_predictions(records, predictions, labels, "ml_v1_1_dev_challenge")
    scored["supported_macro"] = supported_macro(scored)
    tagged = tag_context_errors(scored["errors"], records)
    for row in tagged:
        row["evaluation_view"] = "selected_checkpoint_dev_challenge"
    business = _business_analysis(records, predictions)
    ssn = _ssn_analysis(records, predictions)
    current_standard = load_json(STANDARD_METRICS)
    current_challenge = load_json(CHALLENGE_METRICS)
    old_dev = load_json(HISTORICAL_DEV)
    old_test = load_json(HISTORICAL_TEST)

    write_json(
        REGRESSION_STATUS,
        {
            "status": "NOT_RUN",
            "evaluation_performed": False,
            "evaluation_label": "REGRESSION BENCHMARK - NOT UNSEEN TEST",
            "reason": "Step 9 predeclared development gate failed; the experiment stopped before all post-gate inference.",
            "selected_epoch": selected["selected_epoch"],
            "model_sha256": selected["model_sha256"],
            "development_gate_sha256": sha256_file(DEV_GATE),
            "sealed_challenge_opened": False,
            "metric_fields_present": False,
        },
    )

    business_lines = [
        "# ML-v1.1 BUSINESS_ID Subtype Analysis",
        "",
        "**Scope: selected-checkpoint development challenge only. Post-gate regression and sealed inference were not run.**",
        "",
        "The classifier still emits only `BUSINESS_ID`; subtype is recovered from frozen gold provenance.",
        "",
        "| Subtype | Support | Exact TP | FN | Exact-span recall | Partition note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in business:
        note = "present in dev challenge" if row["support"] else "absent from dev challenge; not evaluated after gate stop"
        business_lines.append(
            f"| {row['subtype']} | {row['support']} | {row['true_positives']} | {row['false_negatives']} | {_pct(row['recall'])} | {note} |"
        )
    business_lines.extend(
        [
            "",
            "ACCOUNT_ID and WORKFLOW_ID are directly represented here. ORDER_ID and REFERENCE_ID were deliberately reserved for other partitions, so their generalization cannot be claimed from this stopped experiment.",
        ]
    )
    _write_markdown(BUSINESS_REPORT, business_lines)

    old_ssn = old_test["overall"]["per_label"]["SSN"]
    ssn_lines = [
        "# ML-v1.1 SSN Morphology-Conflict Analysis",
        "",
        "**Scope: selected-checkpoint development challenge only.**",
        "",
        "| Diagnostic | ML-v1 historical immutable test evidence | ML-v1.1 dev challenge |",
        "|---|---:|---:|",
        f"| True SSN support | {old_ssn['support']} | {ssn['true_ssn_support']} |",
        f"| True SSNs correctly recognized | {old_ssn['true_positives']} | {ssn['true_ssns_correctly_recognized']} |",
        f"| BUSINESS_ID values incorrectly classified as SSN | 93 in historical failed family | {ssn['business_id_values_incorrectly_classified_as_ssn']} |",
        f"| Other non-SSN values incorrectly classified as SSN | 3 aggregate residual SSN false positives | {ssn['other_non_ssn_values_incorrectly_classified_as_ssn']} |",
        f"| SSNs missed as BUSINESS_ID | not re-derived after stop | {ssn['ssns_missed_as_business_id']} |",
        f"| SSN boundary errors | not re-derived after stop | {ssn['ssn_boundary_errors']} |",
        "",
        f"ML-v1.1 challenge SSN precision/recall/F1 is **{current_challenge['per_label']['SSN']['precision']:.6f} / {current_challenge['per_label']['SSN']['recall']:.6f} / {current_challenge['per_label']['SSN']['f1']:.6f}**. BUSINESS_ID recall is **{current_challenge['per_label']['BUSINESS_ID']['recall']:.6f}**, so SSN recall was not improved merely by suppressing SSN output.",
        "",
        "The historical and dev-challenge rows use different datasets and are diagnostic, not a direct regression comparison.",
    ]
    _write_markdown(SSN_REPORT, ssn_lines)

    categories = current_challenge["context_diagnostics"]["by_category"]
    gen_lines = [
        "# ML-v1.1 Generalization Analysis",
        "",
        "The Step 9 development gate failed, so post-gate model inference stopped. The following views remain deliberately separate.",
        "",
        "| View | Interpretation | Result |",
        "|---|---|---|",
        f"| Original standard dev | Ordinary NER regression signal | micro F1 {current_standard['micro']['f1']:.6f}; supported macro F1 {current_standard['supported_macro']['f1']:.6f} |",
        f"| New dev context challenge | Context-conflict robustness | micro F1 {current_challenge['micro']['f1']:.6f}; supported macro F1 {current_challenge['supported_macro']['f1']:.6f}; record error {current_challenge['context_diagnostics']['record_error_rate']:.6f} |",
        f"| Historical ML-v1 Gretel | Historical external/public-source evidence only | micro F1 {old_test['segments']['gretel']['micro']['f1']:.6f}; ML-v1.1 NOT RUN |",
        f"| Historical ML-v1 synthetic unseen templates | Historical synthetic evidence only | micro F1 {old_test['segments']['synthetic_unseen_template']['micro']['f1']:.6f}; ML-v1.1 NOT RUN |",
        "| Historical failed hard-negative family | Required new-model regression evidence | NOT RUN because Step 9 stopped the phase |",
        "| New sealed challenge families | Independent context-conflict evidence | NOT OPENED / NOT RUN |",
        "",
        "## Dev conflict categories",
        "",
        "| Category | Micro F1 | Supported macro F1 | Record error rate |",
        "|---|---:|---:|---:|",
    ]
    for name, value in categories.items():
        gen_lines.append(
            f"| {name} | {value['micro']['f1']:.6f} | {value['supported_macro']['f1']:.6f} | {value['record_error_rate']:.6f} |"
        )
    gen_lines.extend(
        [
            "",
            "External/public-source generalization, synthetic generalization, and context-conflict robustness are not combined into one headline metric. Gretel remains subject to `license_reviewed=false`.",
        ]
    )
    _write_markdown(GENERALIZATION_REPORT, gen_lines)

    _write_errors(ERROR_CSV, tagged)
    error_tags: Counter[str] = Counter()
    for row in tagged:
        error_tags.update(row["context_error_tags"])
    error_lines = [
        "# ML-v1.1 Error Analysis",
        "",
        "**Scope: selected-checkpoint development challenge only.**",
        "",
        f"- False positives: **{scored['error_counts']['false_positives']}**",
        f"- False negatives: **{scored['error_counts']['false_negatives']}**",
        f"- Boundary too short: **{scored['error_counts']['boundary_too_short']}**",
        f"- Boundary too long: **{scored['error_counts']['boundary_too_long']}**",
        f"- Shifted boundary: **{scored['error_counts']['boundary_shifted']}**",
        f"- Wrong class: **{scored['error_counts']['wrong_entity_class']}**",
        "",
        "## Context-conflict tags",
        "",
    ]
    error_lines.extend(
        f"- {name}: **{count}**" for name, count in sorted(error_tags.items())
    )
    error_lines.extend(
        [
            "",
            "Detailed rows are in `reports/ml_v1_1_model_errors.csv`. ML-v1.1 original-test and sealed errors do not exist because their inference phases were not run.",
        ]
    )
    _write_markdown(ERROR_REPORT, error_lines)

    comparison_lines = [
        "# SecureLogX ML-v1 vs ML-v1.1 Comparison",
        "",
        "**Partial comparison: ML-v1.1 post-gate regression was NOT RUN. Metrics from different dev/challenge views are not treated as interchangeable test results.**",
        "",
        "## Direct original-dev comparison",
        "",
        "| Metric | ML-v1 original dev | ML-v1.1 original dev | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name, before, after in (
        ("Micro F1", old_dev["micro"]["f1"], current_standard["micro"]["f1"]),
        ("Macro F1", old_dev["macro"]["f1"], current_standard["supported_macro"]["f1"]),
        ("High-risk recall", old_dev["high_risk"]["recall"], current_standard["high_risk"]["recall"]),
        ("BUSINESS_ID recall", old_dev["per_label"]["BUSINESS_ID"]["recall"], current_standard["per_label"]["BUSINESS_ID"]["recall"]),
        ("SSN recall", old_dev["per_label"]["SSN"]["recall"], current_standard["per_label"]["SSN"]["recall"]),
    ):
        comparison_lines.append(
            f"| {name} | {before:.6f} | {after:.6f} | {after - before:+.6f} |"
        )
    comparison_lines.extend(
        [
            "",
            "## Known-failure diagnostic (not a direct same-split comparison)",
            "",
            "| Indicator | ML-v1 historical test/family | ML-v1.1 dev challenge |",
            "|---|---:|---:|",
            f"| BUSINESS_ID recall | {old_test['overall']['per_label']['BUSINESS_ID']['recall']:.6f} | {current_challenge['per_label']['BUSINESS_ID']['recall']:.6f} |",
            f"| Hard/context record error | {old_test['decision']['gates']['hard_negative_records_with_false_positive_rate_maximum']['actual']:.6f} | {current_challenge['context_diagnostics']['record_error_rate']:.6f} |",
            "",
            "The apparent targeted improvement is encouraging but did not satisfy all predeclared challenge gates. Sealed, Gretel, synthetic-unseen, historical hard-negative, and original-test ML-v1.1 comparisons remain unavailable.",
        ]
    )
    _write_markdown(COMPARISON_REPORT, comparison_lines)

    failed = [
        (name, value)
        for name, value in gate["gates"].items()
        if not value["passed"]
    ]
    decision_lines = [
        "# ML-v1.1 Model Validation Decision",
        "",
        "**NOT READY FOR ONNX VALIDATION**",
        "",
        "The selected checkpoint failed the immutable development unlock. No threshold was changed after results were observed.",
        "",
        "## Failed development gates",
        "",
        "| Gate | Actual | Operator | Threshold |",
        "|---|---:|:---:|---:|",
    ]
    for name, value in failed:
        decision_lines.append(
            f"| `{name}` | {value['actual']:.6f} | {value['operator']} | {value['threshold']:.6f} |"
        )
    decision_lines.extend(
        [
            "",
            f"- Selected epoch/checkpoint: **{selected['selected_epoch']}** / `output_securelogx/ml-v1.1/bert-base-cased/best-checkpoint`",
            f"- Model SHA-256: `{selected['model_sha256']}`",
            "- Sealed challenge: **NOT OPENED; no metrics, cache, or receipt created**",
            "- Original ML-v1 regression benchmark: **NOT RUN after Step 9 stop**",
            "- ONNX export: **NOT PERFORMED**",
            "",
            "A later regression or sealed-evaluation phase would require explicit authorization; this stopped run cannot be represented as having those results.",
            "",
            "Gretel `license_reviewed=false` remains a separate production/commercial release blocker.",
        ]
    )
    _write_markdown(DECISION_REPORT, decision_lines)

    if any(path.exists() for path in FORBIDDEN_RESULTS):
        raise RuntimeError("A forbidden post-gate inference artifact appeared during finalization")
    print("NOT READY FOR ONNX VALIDATION")
    print(f"dev_error_rows={len(tagged)} model_sha256={selected['model_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
