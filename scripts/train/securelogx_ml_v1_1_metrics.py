"""Pure metric helpers for the SecureLogX ML-v1.1 context experiment.

The helpers in this module operate only on already-decoded character spans.
They do not load models or datasets and do not read or write the sealed
challenge.  Exact-span scoring remains delegated to the ML-v1 implementation
so the comparison keeps one metric contract.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from securelogx_training_common import score_predictions, subset_score


_CATEGORY_MORPHOLOGY_LABEL: dict[str, str] = {
    "business_id_vs_ssn": "SSN",
    "business_id_vs_credit_card": "CREDIT_CARD_NUMBER",
    "business_id_vs_bank_account": "BANK_ACCOUNT_NUMBER",
    "ip_address_vs_technical_reference": "IP_ADDRESS",
    "person_name_vs_service_name": "PERSON_NAME",
    "street_address_vs_system_location": "STREET_ADDRESS",
}

_HISTORICALLY_UNSEEN_BUSINESS_SUBTYPES = frozenset(
    {
        "ORDER_ID",
        "CASE_ID",
        "TICKET_ID",
        "INVOICE_ID",
        "REFERENCE_ID",
        "REQUEST_ID",
        "WORKFLOW_ID",
    }
)

_STRUCTURED_SURFACE_STYLES = frozenset(
    {
        "abbreviated_key",
        "json_flat",
        "json_nested",
        "key_colon",
        "key_equals",
        "structured_audit",
    }
)

_CONTEXT_TAG_ORDER = (
    "morphology-over-context",
    "context-over-morphology",
    "ambiguous-key",
    "unseen-business-subtype",
    "structured-field-confusion",
)


def supported_macro(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Average per-label P/R/F1 over labels with non-zero gold support."""

    per_label = metrics.get("per_label")
    if not isinstance(per_label, Mapping):
        raise ValueError("metrics.per_label must be a mapping")
    supported: list[Mapping[str, Any]] = []
    for label, values in per_label.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"metrics.per_label[{label!r}] must be a mapping")
        support = values.get("support")
        if not isinstance(support, (int, float)) or isinstance(support, bool):
            raise ValueError(f"metrics.per_label[{label!r}].support must be numeric")
        if support > 0:
            supported.append(values)
    if not supported:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        name: float(statistics.fmean(float(values[name]) for values in supported))
        for name in ("precision", "recall", "f1")
    }


def harmonic_mean(a: float, b: float) -> float:
    """Return the two-value harmonic mean used for checkpoint selection."""

    left = float(a)
    right = float(b)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("harmonic-mean inputs must be finite")
    if left < 0.0 or right < 0.0:
        raise ValueError("harmonic-mean inputs must be non-negative")
    if left == 0.0 or right == 0.0:
        return 0.0
    return 2.0 * left * right / (left + right)


def _span_signature(span: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["label"])


def _gold_signatures(record: Mapping[str, Any]) -> Counter[tuple[int, int, str]]:
    entities = record.get("entities")
    if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
        raise ValueError("record.entities must be a sequence")
    signatures: Counter[tuple[int, int, str]] = Counter()
    for entity in entities:
        if (
            not isinstance(entity, Sequence)
            or isinstance(entity, (str, bytes))
            or len(entity) != 3
        ):
            raise ValueError(f"malformed gold entity: {entity!r}")
        signatures[(int(entity[0]), int(entity[1]), str(entity[2]))] += 1
    return signatures


def _prediction_signatures(
    predicted_spans: Sequence[Mapping[str, Any]],
) -> Counter[tuple[int, int, str]]:
    return Counter(_span_signature(prediction) for prediction in predicted_spans)


def _overlaps(start: int, end: int, span: Mapping[str, Any]) -> bool:
    return int(span["start"]) < end and int(span["end"]) > start


def _contrast_targets(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    meta = record.get("meta", {})
    if not isinstance(meta, Mapping):
        raise ValueError("record.meta must be a mapping")
    raw_targets = meta.get("contrast_targets", [])
    if not isinstance(raw_targets, Sequence) or isinstance(
        raw_targets, (str, bytes)
    ):
        raise ValueError("record.meta.contrast_targets must be a sequence")
    targets: list[dict[str, Any]] = []
    for target in raw_targets:
        if not isinstance(target, Mapping):
            raise ValueError(f"malformed contrast target: {target!r}")
        start = int(target["start"])
        end = int(target["end"])
        expected_label = str(target["expected_label"])
        if start < 0 or end <= start or not expected_label:
            raise ValueError(f"invalid contrast target: {target!r}")
        normalized = dict(target)
        normalized.update(
            {"start": start, "end": end, "expected_label": expected_label}
        )
        targets.append(normalized)
    return targets


def _target_outcome(
    target: Mapping[str, Any],
    predicted_spans: Sequence[Mapping[str, Any]],
    category: str,
) -> tuple[bool, bool]:
    start = int(target["start"])
    end = int(target["end"])
    expected_label = str(target["expected_label"])
    overlapping = [
        prediction
        for prediction in predicted_spans
        if _overlaps(start, end, prediction)
    ]
    exact = [
        prediction
        for prediction in overlapping
        if _span_signature(prediction) == (start, end, expected_label)
    ]
    if expected_label == "O":
        target_error = bool(overlapping)
    else:
        # A duplicate or a second overlapping label makes the target ambiguous
        # even when one exact prediction is present.
        target_error = len(exact) != 1 or len(overlapping) != 1

    morphology_label = _CATEGORY_MORPHOLOGY_LABEL.get(category)
    morphology_over_context = bool(
        target_error
        and morphology_label is not None
        and expected_label in {"BUSINESS_ID", "O"}
        and any(str(prediction["label"]) == morphology_label for prediction in overlapping)
    )
    return target_error, morphology_over_context


def _compact_score(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "records": metrics["records"],
        "gold_entities": metrics["gold_entities"],
        "predicted_entities": metrics["predicted_entities"],
        "exact_span_matches": metrics["exact_span_matches"],
        "micro": metrics["micro"],
        "macro": metrics["macro"],
        "supported_macro": supported_macro(metrics),
        "per_label": metrics["per_label"],
        "high_risk": metrics["high_risk"],
        "error_counts": metrics["error_counts"],
    }


def context_diagnostics(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    label_order: Sequence[str],
    split: str,
) -> dict[str, Any]:
    """Score record and declared-target failures for context-contrast data.

    ``record_error_rate`` uses an exact multiset comparison of all gold and
    predicted spans in each record. ``context_target_error_rate`` is narrower:
    a positive target requires exactly one matching overlapping prediction,
    while an ``O`` target requires no overlapping prediction.  A
    morphology-over-context error is the directional failure of predicting the
    morphology label (SSN/card/account/IP/name/address) for a BUSINESS_ID or O
    target.
    """

    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")
    if not label_order:
        raise ValueError("label_order must not be empty")

    category_indices: dict[str, list[int]] = defaultdict(list)
    record_errors: list[bool] = []
    target_error_flags: list[list[bool]] = []
    morphology_flags: list[list[bool]] = []

    for index, (record, predicted_spans) in enumerate(zip(records, predictions)):
        meta = record.get("meta", {})
        if not isinstance(meta, Mapping):
            raise ValueError(f"record {index} meta must be a mapping")
        category = str(meta.get("contrast_category") or "<UNSPECIFIED>")
        category_indices[category].append(index)
        record_errors.append(
            _gold_signatures(record) != _prediction_signatures(predicted_spans)
        )
        outcomes = [
            _target_outcome(target, predicted_spans, category)
            for target in _contrast_targets(record)
        ]
        target_error_flags.append([outcome[0] for outcome in outcomes])
        morphology_flags.append([outcome[1] for outcome in outcomes])

    def summarize(indices: Sequence[int]) -> dict[str, int | float]:
        target_count = sum(len(target_error_flags[index]) for index in indices)
        target_errors = sum(
            sum(target_error_flags[index]) for index in indices
        )
        morphology_errors = sum(sum(morphology_flags[index]) for index in indices)
        records_with_errors = sum(record_errors[index] for index in indices)
        return {
            "records": len(indices),
            "records_with_errors": records_with_errors,
            "record_error_rate": (
                records_with_errors / len(indices) if indices else 0.0
            ),
            "context_targets": target_count,
            "context_target_errors": target_errors,
            "context_target_error_rate": (
                target_errors / target_count if target_count else 0.0
            ),
            "morphology_over_context_errors": morphology_errors,
            "morphology_over_context_error_rate": (
                morphology_errors / target_count if target_count else 0.0
            ),
            "morphology_over_context_share_of_target_errors": (
                morphology_errors / target_errors if target_errors else 0.0
            ),
        }

    all_indices = list(range(len(records)))
    overall_counts = summarize(all_indices)
    overall_score = score_predictions(records, predictions, label_order, split)
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted(category_indices):
        indices = category_indices[category]
        category_score = subset_score(
            records,
            predictions,
            label_order,
            f"{split}:{category}",
            selected_indices=indices,
        )
        by_category[category] = {
            **summarize(indices),
            **_compact_score(category_score),
        }

    maximum_category_error_rate = max(
        (
            float(category["record_error_rate"])
            for category in by_category.values()
        ),
        default=0.0,
    )
    return {
        "split": split,
        **overall_counts,
        "maximum_category_error_rate": maximum_category_error_rate,
        "by_category": by_category,
        "score": _compact_score(overall_score),
    }


def _comparison_operator(name: str) -> str:
    tokens = set(name.lower().split("_"))
    has_minimum = "minimum" in tokens or "min" in tokens
    has_maximum = "maximum" in tokens or "max" in tokens
    if has_minimum == has_maximum:
        raise ValueError(
            f"gate {name!r} must declare minimum/maximum or an explicit operator"
        )
    return ">=" if has_minimum else "<="


def _metric_name_from_gate(name: str) -> str:
    for suffix in ("_minimum", "_maximum", "_min", "_max"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def evaluate_named_gates(
    actual: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate named, predeclared thresholds without inferring policy later.

    Numeric threshold names must contain ``minimum``/``min`` or
    ``maximum``/``max``.  A mapping threshold may instead declare ``metric``,
    ``operator`` and ``threshold`` (or ``value``).  Exact-name actual values are
    preferred; for a suffixed gate such as ``macro_f1_minimum``, ``macro_f1`` is
    accepted as the actual metric name.
    """

    operators = {
        ">=": lambda value, threshold: value >= threshold,
        "<=": lambda value, threshold: value <= threshold,
        ">": lambda value, threshold: value > threshold,
        "<": lambda value, threshold: value < threshold,
        "==": lambda value, threshold: value == threshold,
    }
    results: dict[str, dict[str, Any]] = {}
    for gate_name, specification in thresholds.items():
        name = str(gate_name)
        if isinstance(specification, Mapping):
            metric_name = str(specification.get("metric") or _metric_name_from_gate(name))
            operator = str(specification.get("operator") or _comparison_operator(name))
            if "threshold" in specification:
                threshold = specification["threshold"]
            elif "value" in specification:
                threshold = specification["value"]
            else:
                raise ValueError(f"gate {name!r} has no threshold/value")
        else:
            metric_name = _metric_name_from_gate(name)
            operator = _comparison_operator(name)
            threshold = specification
        if operator not in operators:
            raise ValueError(f"gate {name!r} has unsupported operator {operator!r}")

        if name in actual:
            actual_name = name
        elif metric_name in actual:
            actual_name = metric_name
        else:
            raise KeyError(f"no actual value for gate {name!r} (metric {metric_name!r})")
        value = actual[actual_name]
        if isinstance(value, bool) or isinstance(threshold, bool):
            if operator != "==":
                raise ValueError("boolean gates require the == operator")
        else:
            try:
                value = float(value)
                threshold = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"gate {name!r} values must be numeric") from exc
            if not math.isfinite(value) or not math.isfinite(threshold):
                raise ValueError(f"gate {name!r} values must be finite")
        passed = bool(operators[operator](value, threshold))
        results[name] = {
            "metric": actual_name,
            "operator": operator,
            "threshold": threshold,
            "actual": value,
            "passed": passed,
        }
    return {
        "all_passed": all(result["passed"] for result in results.values()),
        "gates": results,
    }


def _parse_span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str) or not value:
        return None
    if not (value.startswith("[") and value.endswith(")") and "," in value):
        raise ValueError(f"invalid half-open span string: {value!r}")
    start, end = value[1:-1].split(",", 1)
    return int(start), int(end)


def _target_subtype(
    record: Mapping[str, Any], target: Mapping[str, Any]
) -> str:
    subtype = target.get("business_id_subtype")
    if subtype:
        return str(subtype).upper()
    meta = record.get("meta", {})
    provenance = meta.get("entity_provenance", []) if isinstance(meta, Mapping) else []
    if isinstance(provenance, Sequence) and not isinstance(provenance, (str, bytes)):
        for entry in provenance:
            if not isinstance(entry, Mapping):
                continue
            if (
                int(entry.get("start", -1)) == int(target["start"])
                and int(entry.get("end", -1)) == int(target["end"])
            ):
                raw_subtype = entry.get("business_id_subtype") or entry.get(
                    "source_subtype"
                )
                if raw_subtype:
                    return str(raw_subtype).upper()
    return ""


def tag_context_errors(
    rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return copied error rows enriched with deterministic context-error tags."""

    tagged_rows: list[dict[str, Any]] = []
    for row in rows:
        if "record_index" not in row:
            raise ValueError("error row has no record_index")
        record_index = int(row["record_index"])
        if record_index < 0 or record_index >= len(records):
            raise ValueError(f"error row record_index out of range: {record_index}")
        record = records[record_index]
        meta = record.get("meta", {})
        if not isinstance(meta, Mapping):
            raise ValueError(f"record {record_index} meta must be a mapping")
        category = str(meta.get("contrast_category") or "")
        role = str(meta.get("context_role") or "")
        morphology = str(meta.get("morphology_class") or "")
        morphology_label = _CATEGORY_MORPHOLOGY_LABEL.get(category)
        predicted_label = str(row.get("predicted_entity") or "")
        gold_span = _parse_span(row.get("gold_span"))
        predicted_span = _parse_span(row.get("predicted_span"))
        tags: set[str] = set()
        subtypes: set[str] = set()

        relevant_targets: list[dict[str, Any]] = []
        for target in _contrast_targets(record):
            start = int(target["start"])
            end = int(target["end"])
            if any(
                span is not None and span[0] < end and span[1] > start
                for span in (gold_span, predicted_span)
            ):
                relevant_targets.append(target)

        for target in relevant_targets:
            expected_label = str(target["expected_label"])
            subtype = _target_subtype(record, target)
            if subtype:
                subtypes.add(subtype)
            if (
                morphology_label is not None
                and expected_label in {"BUSINESS_ID", "O"}
                and predicted_label == morphology_label
            ):
                tags.add("morphology-over-context")
            if (
                morphology_label is not None
                and expected_label == morphology_label
                and predicted_label == "BUSINESS_ID"
            ):
                tags.add("context-over-morphology")
            if (
                expected_label == "BUSINESS_ID"
                and subtype in _HISTORICALLY_UNSEEN_BUSINESS_SUBTYPES
            ):
                tags.add("unseen-business-subtype")

        if relevant_targets and (
            meta.get("ambiguous_key") is True
            or str(meta.get("surface_style") or "") == "abbreviated_key"
        ):
            tags.add("ambiguous-key")
        if relevant_targets and (
            str(meta.get("format") or "") in {"json", "key_value"}
            or str(meta.get("surface_style") or "")
            in _STRUCTURED_SURFACE_STYLES
        ):
            tags.add("structured-field-confusion")

        enriched = dict(row)
        enriched.update(
            {
                "contrast_category": category,
                "context_role": role,
                "morphology_class": morphology,
                "business_id_subtype": ",".join(sorted(subtypes)),
                "context_error_tags": [
                    tag for tag in _CONTEXT_TAG_ORDER if tag in tags
                ],
            }
        )
        tagged_rows.append(enriched)
    return tagged_rows


__all__ = [
    "context_diagnostics",
    "evaluate_named_gates",
    "harmonic_mean",
    "supported_macro",
    "tag_context_errors",
]
