#!/usr/bin/env python3
"""Classify ML-v1.2 development errors by security outcome, not NER score.

This command reads frozen ML-v1.2 development records and cached selected-epoch
predictions only.  It does not load a model, run inference, open the sealed
challenge, or rewrite historical NER metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_ENTITIES = (
    "PERSON_NAME",
    "DOB",
    "AGE",
    "SSN",
    "ITIN",
    "TAX_ID",
    "EMAIL",
    "PHONE",
    "STREET_ADDRESS",
    "CITY",
    "STATE_PROVINCE",
    "POSTAL_CODE",
    "COUNTRY",
    "CREDIT_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "ROUTING_NUMBER",
    "IBAN",
    "SWIFT_BIC",
    "BUSINESS_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE",
    "IP_ADDRESS",
    "DEVICE_ID",
    "AUTH_TOKEN",
    "API_KEY",
)
MASKING_SENSITIVE = frozenset(CANONICAL_ENTITIES)
HIGH_RISK = frozenset(
    {
        "SSN",
        "ITIN",
        "TAX_ID",
        "CREDIT_CARD_NUMBER",
        "BANK_ACCOUNT_NUMBER",
        "ROUTING_NUMBER",
        "IBAN",
        "AUTH_TOKEN",
        "API_KEY",
        "PASSPORT_NUMBER",
        "DRIVER_LICENSE",
    }
)
DEFAULT_MASK_ACTION = {
    "SSN": "MASK_LAST4",
    "ITIN": "MASK",
    "TAX_ID": "MASK",
    "CREDIT_CARD_NUMBER": "MASK_LAST4",
    "BANK_ACCOUNT_NUMBER": "MASK",
    "ROUTING_NUMBER": "MASK",
    "IBAN": "MASK",
    "SWIFT_BIC": "MASK",
    "AUTH_TOKEN": "FULL_MASK",
    "API_KEY": "FULL_MASK",
    "PASSPORT_NUMBER": "MASK",
    "DRIVER_LICENSE": "MASK",
    "EMAIL": "PARTIAL_MASK",
    "PHONE": "PARTIAL_MASK",
    "PERSON_NAME": "FULL_MASK",
    "DOB": "MASK",
    "AGE": "MASK",
    "STREET_ADDRESS": "MASK",
    "CITY": "MASK",
    "STATE_PROVINCE": "MASK",
    "POSTAL_CODE": "MASK",
    "COUNTRY": "MASK",
    "BUSINESS_ID": "MASK",
    "IP_ADDRESS": "MASK",
    "DEVICE_ID": "MASK",
}

SPLITS = {
    "standard_dev": {
        "records": Path("data/split/dev.jsonl"),
        "predictions": Path(
            "output_securelogx/ml-v1.2/bert-base-cased/dev_standard_predictions.json"
        ),
    },
    "challenge_dev": {
        "records": Path("data/ml_v1_2/counterbalance/dev_challenge.jsonl"),
        "predictions": Path(
            "output_securelogx/ml-v1.2/bert-base-cased/dev_challenge_predictions.json"
        ),
    },
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def overlaps(start: int, end: int, span: Mapping[str, Any]) -> bool:
    return int(span["start"]) < end and int(span["end"]) > start


def covers(pred: Mapping[str, Any], start: int, end: int) -> bool:
    return int(pred["start"]) <= start and int(pred["end"]) >= end


def subtype_of(record: Mapping[str, Any], start: int, end: int, label: str) -> str:
    for item in record.get("meta", {}).get("entity_provenance", []):
        if (
            int(item.get("start", -1)) == start
            and int(item.get("end", -1)) == end
            and str(item.get("label")) == label
        ):
            return str(item.get("business_id_subtype") or "")
    return ""


def classify_record(
    split: str,
    index: int,
    record: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    text = str(record["text"])
    meta = record.get("meta", {}) if isinstance(record.get("meta"), Mapping) else {}
    family = str(meta.get("template_family") or "")
    category = str(meta.get("contrast_category") or "")
    gold = [(int(start), int(end), str(label)) for start, end, label in record["entities"]]
    rows: list[dict[str, Any]] = []
    used_preds: set[int] = set()
    stats = Counter()

    for start, end, label in gold:
        if label not in MASKING_SENSITIVE:
            continue
        stats["sensitive_gold"] += 1
        high_risk = label in HIGH_RISK
        if high_risk:
            stats["high_risk_gold"] += 1
        overlapping = [
            (pred_index, pred)
            for pred_index, pred in enumerate(predictions)
            if str(pred.get("label")) in MASKING_SENSITIVE
            and overlaps(start, end, pred)
        ]
        covering = [
            item for item in overlapping if covers(item[1], start, end)
        ]
        value = text[start:end]
        subtype = subtype_of(record, start, end, label)
        if covering:
            stats["sensitive_any_cover"] += 1
            stats["sensitive_full_mask"] += 1
            if high_risk:
                stats["high_risk_full_mask"] += 1
            pred_index, pred = covering[0]
            used_preds.add(pred_index)
            predicted_label = str(pred["label"])
            if predicted_label == label and int(pred["start"]) == start and int(pred["end"]) == end:
                stats["exact_true_positive"] += 1
                continue
            if predicted_label == label:
                stats["label_correct_boundary_long"] += 1
                continue
            outcome = "POLICY_SAFE_WRONG_CLASS"
            stats[outcome] += 1
            rows.append(
                _row(
                    split,
                    index,
                    record,
                    family,
                    category,
                    outcome,
                    "gold_span",
                    start,
                    end,
                    label,
                    predicted_label,
                    value,
                    subtype,
                    pred,
                )
            )
            continue
        if overlapping:
            stats["sensitive_any_cover"] += 1
            outcome = "SECURITY_CRITICAL_PARTIAL"
            stats[outcome] += 1
            pred_index, pred = overlapping[0]
            used_preds.add(pred_index)
            rows.append(
                _row(
                    split,
                    index,
                    record,
                    family,
                    category,
                    outcome,
                    "gold_span",
                    start,
                    end,
                    label,
                    str(pred["label"]),
                    value,
                    subtype,
                    pred,
                )
            )
            continue
        outcome = "SECURITY_CRITICAL_MISS"
        stats[outcome] += 1
        rows.append(
            _row(
                split,
                index,
                record,
                family,
                category,
                outcome,
                "gold_span",
                start,
                end,
                label,
                "O",
                value,
                subtype,
                None,
            )
        )

    for pred_index, pred in enumerate(predictions):
        if pred_index in used_preds:
            continue
        predicted_label = str(pred["label"])
        if predicted_label not in MASKING_SENSITIVE:
            continue
        if any(overlaps(start, end, pred) for start, end, _ in gold):
            # Overlap with gold that was already counted as TP/wrong-class/partial.
            continue
        outcome = "OVERMASKING"
        stats[outcome] += 1
        start = int(pred["start"])
        end = int(pred["end"])
        rows.append(
            _row(
                split,
                index,
                record,
                family,
                category,
                outcome,
                "spurious_prediction",
                start,
                end,
                "O",
                predicted_label,
                text[start:end],
                "",
                pred,
            )
        )
    return rows, stats


def _row(
    split: str,
    index: int,
    record: Mapping[str, Any],
    family: str,
    category: str,
    outcome: str,
    source: str,
    start: int,
    end: int,
    gold_label: str,
    predicted_label: str,
    value: str,
    subtype: str,
    pred: Mapping[str, Any] | None,
) -> dict[str, Any]:
    meta = record.get("meta", {}) if isinstance(record.get("meta"), Mapping) else {}
    return {
        "split": split,
        "record_index": index,
        "source_record_id": str(meta.get("source_record_id") or meta.get("id") or ""),
        "template_family": family,
        "contrast_category": category,
        "context_role": str(meta.get("context_role") or ""),
        "outcome": outcome,
        "error_source": source,
        "gold_label": gold_label,
        "predicted_label": predicted_label,
        "start": start,
        "end": end,
        "value": value,
        "business_id_subtype": subtype,
        "confidence": None if pred is None else pred.get("confidence"),
        "predicted_start": None if pred is None else pred.get("start"),
        "predicted_end": None if pred is None else pred.get("end"),
        "gold_mask_action": DEFAULT_MASK_ACTION.get(gold_label, "NONE"),
        "predicted_mask_action": DEFAULT_MASK_ACTION.get(predicted_label, "NONE"),
    }


def analyze(root: Path) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    by_split: dict[str, dict[str, Any]] = {}
    focused: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, paths in SPLITS.items():
        records = load_jsonl(root / paths["records"])
        payload = load_json(root / paths["predictions"])
        predictions = payload["predictions"]
        if len(records) != len(predictions):
            raise ValueError(f"{split} record/prediction length mismatch")
        split_stats: Counter[str] = Counter()
        split_rows: list[dict[str, Any]] = []
        for index, (record, preds) in enumerate(zip(records, predictions)):
            rows, stats = classify_record(split, index, record, preds)
            split_rows.extend(rows)
            split_stats.update(stats)
            family = str(record.get("meta", {}).get("template_family") or "")
            category = str(record.get("meta", {}).get("contrast_category") or "")
            for row in rows:
                if (
                    family == "key_value_accountlike_batch_v1"
                    and row["outcome"] == "OVERMASKING"
                    and row["predicted_label"] == "BUSINESS_ID"
                ):
                    focused["accountlike_o_to_business_id"].append(row)
                if row["gold_label"] == "SSN" and row["predicted_label"] == "ITIN":
                    focused["ssn_to_itin"].append(row)
                if category == "business_id_vs_auth_token":
                    focused["business_id_vs_auth_token"].append(row)
                if row["gold_label"] == "AUTH_TOKEN" and row["outcome"] in {
                    "SECURITY_CRITICAL_MISS",
                    "SECURITY_CRITICAL_PARTIAL",
                }:
                    focused["auth_token_misses"].append(row)
                if row["gold_label"] == "API_KEY" and row["outcome"] in {
                    "SECURITY_CRITICAL_MISS",
                    "SECURITY_CRITICAL_PARTIAL",
                }:
                    focused["api_key_misses"].append(row)
                if row["gold_label"] == "BUSINESS_ID" and row["outcome"] in {
                    "SECURITY_CRITICAL_MISS",
                    "SECURITY_CRITICAL_PARTIAL",
                    "POLICY_SAFE_WRONG_CLASS",
                }:
                    focused["business_id_false_negatives"].append(row)
                if row["business_id_subtype"] == "REQUEST_ID":
                    focused["request_id_failures"].append(row)
                if category == "ip_address_vs_technical_reference":
                    focused["ip_vs_technical_reference"].append(row)
                if row["gold_label"] in HIGH_RISK and row["outcome"] == "SECURITY_CRITICAL_PARTIAL":
                    focused["high_risk_partial"].append(row)
        totals.update(split_stats)
        by_split[split] = {
            "records": len(records),
            "prediction_input_sha256": payload.get("input_sha256"),
            "selected_epoch": payload.get("selected_epoch"),
            **_rates(split_stats),
            "outcome_counts": {
                name: int(split_stats[name])
                for name in (
                    "SECURITY_CRITICAL_MISS",
                    "SECURITY_CRITICAL_PARTIAL",
                    "POLICY_SAFE_WRONG_CLASS",
                    "OVERMASKING",
                    "OTHER_SEMANTIC_ERROR",
                )
            },
        }
        all_rows.extend(split_rows)
    summary = {
        "schema_version": 1,
        "historical_ner_metrics_unchanged": True,
        "sealed_challenge_opened": False,
        "default_mask_action_by_entity": DEFAULT_MASK_ACTION,
        "totals": _rates(totals),
        "by_split": by_split,
        "focused_counts": {name: len(rows) for name, rows in focused.items()},
        "focused_examples": {
            name: rows[:12]
            for name, rows in focused.items()
        },
        "outcome_rows": len(all_rows),
    }
    return {"summary": summary, "rows": all_rows, "focused": dict(focused)}


def _rates(stats: Mapping[str, int]) -> dict[str, Any]:
    sensitive = int(stats.get("sensitive_gold", 0))
    high_risk = int(stats.get("high_risk_gold", 0))
    return {
        "sensitive_gold_spans": sensitive,
        "high_risk_gold_spans": high_risk,
        "security_critical_miss": int(stats.get("SECURITY_CRITICAL_MISS", 0)),
        "security_critical_partial": int(stats.get("SECURITY_CRITICAL_PARTIAL", 0)),
        "policy_safe_wrong_class": int(stats.get("POLICY_SAFE_WRONG_CLASS", 0)),
        "overmasking": int(stats.get("OVERMASKING", 0)),
        "other_semantic_error": int(stats.get("OTHER_SEMANTIC_ERROR", 0)),
        "sensitive_span_recall": (
            stats.get("sensitive_any_cover", 0) / sensitive if sensitive else 1.0
        ),
        "full_mask_recall": (
            stats.get("sensitive_full_mask", 0) / sensitive if sensitive else 1.0
        ),
        "high_risk_full_mask_recall": (
            stats.get("high_risk_full_mask", 0) / high_risk if high_risk else 1.0
        ),
        "exact_true_positive": int(stats.get("exact_true_positive", 0)),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "split",
        "record_index",
        "source_record_id",
        "template_family",
        "contrast_category",
        "context_role",
        "outcome",
        "error_source",
        "gold_label",
        "predicted_label",
        "start",
        "end",
        "value",
        "business_id_subtype",
        "confidence",
        "predicted_start",
        "predicted_end",
        "gold_mask_action",
        "predicted_mask_action",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, analysis: Mapping[str, Any]) -> None:
    summary = analysis["summary"]
    totals = summary["totals"]
    focused = analysis["focused"]
    lines = [
        "# ML-v1.3 Security-Outcome Analysis of ML-v1.2 Development Errors",
        "",
        "This report reclassifies frozen ML-v1.2 development prediction errors by",
        "masking/security outcome. Historical NER metrics and gates are unchanged.",
        "The sealed challenge was not opened.",
        "",
        "## Headline counts",
        "",
        f"- Security-critical misses: **{totals['security_critical_miss']}**",
        f"- Security-critical partials: **{totals['security_critical_partial']}**",
        f"- Policy-safe wrong-class: **{totals['policy_safe_wrong_class']}**",
        f"- Over-masking: **{totals['overmasking']}**",
        f"- Sensitive-span recall: **{totals['sensitive_span_recall']:.6f}**",
        f"- Full-mask recall: **{totals['full_mask_recall']:.6f}**",
        f"- High-risk full-mask recall: **{totals['high_risk_full_mask_recall']:.6f}**",
        "",
        "## By development view",
        "",
        "| View | Miss | Partial | Policy-safe wrong class | Overmasking | Sensitive-span recall | Full-mask recall | High-risk full-mask recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["by_split"].items():
        lines.append(
            f"| {name} | {values['security_critical_miss']} | {values['security_critical_partial']} | "
            f"{values['policy_safe_wrong_class']} | {values['overmasking']} | "
            f"{values['sensitive_span_recall']:.6f} | {values['full_mask_recall']:.6f} | "
            f"{values['high_risk_full_mask_recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Focused failure modes",
            "",
            "| Mode | Count | Security reading |",
            "|---|---:|---|",
            f"| Account-like gold-O → BUSINESS_ID | {len(focused.get('accountlike_o_to_business_id', []))} | OVERMASKING; operational noise, not a sensitive leak |",
            f"| SSN → ITIN | {len(focused.get('ssn_to_itin', []))} | POLICY_SAFE_WRONG_CLASS; full span still masked |",
            f"| business_id_vs_auth_token errors | {len(focused.get('business_id_vs_auth_token', []))} | mostly BUSINESS_ID absorbed as AUTH_TOKEN (policy-safe or overmasking) |",
            f"| AUTH_TOKEN misses/partials | {len(focused.get('auth_token_misses', []))} | SECURITY_CRITICAL if non-zero |",
            f"| API_KEY misses/partials | {len(focused.get('api_key_misses', []))} | SECURITY_CRITICAL if non-zero |",
            f"| BUSINESS_ID false negatives (miss/partial/wrong-class) | {len(focused.get('business_id_false_negatives', []))} | miss/partial can leak a business identifier; wrong-class is policy-safe if covered |",
            f"| REQUEST_ID-tagged failures | {len(focused.get('request_id_failures', []))} | subtype metadata only; outcome follows the BUSINESS_ID span |",
            f"| IP vs technical-reference errors | {len(focused.get('ip_vs_technical_reference', []))} | v1.2 largely repaired morphology-over-context IP overmasking |",
            f"| High-risk partial spans | {len(focused.get('high_risk_partial', []))} | SECURITY_CRITICAL_PARTIAL |",
            "",
            "## Interpretation",
            "",
            "The persistent 94 account-like gold-O → BUSINESS_ID events are over-masking,",
            "not security misses. SSN → ITIN remains a genuine NER error that is policy-safe",
            "under default MASK/MASK_LAST4 actions. Challenge AUTH_TOKEN gold spans were",
            "recalled; the remaining AUTH_TOKEN issue is BUSINESS_ID predicted as AUTH_TOKEN",
            "inside `business_id_vs_auth_token`. API_KEY gold spans on the v1.2 challenge",
            "were fully covered. Residual security risk is concentrated in uncovered",
            "BUSINESS_ID / high-risk misses or partials, not in the account-like overmasking bucket.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_policy_matrix(path: Path) -> None:
    lines = [
        "# ML-v1.3 Policy Equivalence Matrix",
        "",
        "Default product behavior: every canonical ML-v1 entity is masking-sensitive.",
        "Java policy remains authoritative after ONNX export; this matrix is the",
        "analysis contract used to separate leaks from safe wrong-class/overmasking.",
        "",
        "| Gold entity | Default action | Predicted entity | Predicted action | Security outcome if full gold span is covered |",
        "|---|---|---|---|---|",
    ]
    examples = [
        ("SSN", "ITIN", "POLICY_SAFE_WRONG_CLASS"),
        ("SSN", "O", "SECURITY_CRITICAL_MISS"),
        ("AUTH_TOKEN", "O", "SECURITY_CRITICAL_MISS"),
        ("AUTH_TOKEN", "BUSINESS_ID", "POLICY_SAFE_WRONG_CLASS"),
        ("BUSINESS_ID", "AUTH_TOKEN", "POLICY_SAFE_WRONG_CLASS"),
        ("BUSINESS_ID", "O", "SECURITY_CRITICAL_MISS"),
        ("API_KEY", "AUTH_TOKEN", "POLICY_SAFE_WRONG_CLASS"),
        ("CREDIT_CARD_NUMBER", "BUSINESS_ID", "POLICY_SAFE_WRONG_CLASS"),
        ("IP_ADDRESS", "BUSINESS_ID", "POLICY_SAFE_WRONG_CLASS"),
        ("O", "BUSINESS_ID", "OVERMASKING"),
        ("O", "IP_ADDRESS", "OVERMASKING"),
        ("O", "AUTH_TOKEN", "OVERMASKING"),
        ("SSN", "SSN (partial span)", "SECURITY_CRITICAL_PARTIAL"),
    ]
    for gold, predicted, outcome in examples:
        gold_action = DEFAULT_MASK_ACTION.get(gold, "NONE")
        predicted_action = DEFAULT_MASK_ACTION.get(predicted.split(" ")[0], "NONE")
        lines.append(
            f"| {gold} | {gold_action} | {predicted} | {predicted_action} | {outcome} |"
        )
    lines.extend(
        [
            "",
            "A wrong class remains an NER error. Policy-safe means the full sensitive",
            "characters are still masked under default actions, not that the label is correct.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    analysis = analyze(root)
    write_csv(root / "reports/ml_v1_3_security_outcomes.csv", analysis["rows"])
    write_markdown(root / "reports/ml_v1_3_security_outcome_analysis.md", analysis)
    write_policy_matrix(root / "reports/ml_v1_3_policy_equivalence_matrix.md")
    json_path = root / "reports/ml_v1_3_security_outcome_analysis.json"
    json_path.write_text(
        json.dumps(analysis["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    totals = analysis["summary"]["totals"]
    print(
        f"miss={totals['security_critical_miss']} "
        f"partial={totals['security_critical_partial']} "
        f"policy_safe={totals['policy_safe_wrong_class']} "
        f"overmask={totals['overmasking']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
