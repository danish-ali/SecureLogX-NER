#!/usr/bin/env python3
"""Validate SecureLogX canonical JSONL records and their span metadata.

The validator intentionally treats annotation defects as errors: an invalid
record must not be allowed to become an ``O``-label training example. Exact
duplicate text is always reported and becomes a failing quality gate when the
legacy ``--dedupe`` flag (or its clearer alias, ``--fail-on-duplicates``) is
used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


DEFAULT_LABELS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "securelogx_labels.json"
)

# Only reviewed/approved annotation states are valid inputs to a training-data
# quality gate. ``generated`` is the value emitted by the original pipeline and
# remains accepted as a backwards-compatible alias for ``synthetic_generated``.
CANONICAL_REVIEW_STATUSES = frozenset(
    {
        "auto_converted",
        "synthetic_generated",
        "manual_reviewed",
        "pseudolabeled_reviewed",
    }
)
LEGACY_REVIEW_STATUS_ALIASES = {"generated": "synthetic_generated"}
ALLOWED_REVIEW_STATUSES = CANONICAL_REVIEW_STATUSES | frozenset(
    LEGACY_REVIEW_STATUS_ALIASES
)

DEFERRED_CATEGORIES = frozenset(
    {
        "deferred",
        "deferred_ml_v1",
    }
)
HEALTHCARE_CATEGORIES = frozenset({"healthcare", "healthcare_extension"})


def _is_strict_int(value: Any) -> bool:
    """Return true for an integer offset, excluding bool (an int subclass)."""

    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _is_deferred_entry(entry: Mapping[str, Any]) -> bool:
    category = str(entry.get("category", "")).strip().lower()
    disposition = str(entry.get("disposition", "")).strip().lower()
    reason = str(entry.get("reason", "")).strip().lower()
    return (
        category in DEFERRED_CATEGORIES
        or disposition.startswith("defer")
        or "defer" in reason
    )


def _is_healthcare_entry(entry: Mapping[str, Any]) -> bool:
    category = str(entry.get("category", "")).strip().lower()
    disposition = str(entry.get("disposition", "")).strip().lower()
    reason = str(entry.get("reason", "")).strip().lower()
    return (
        category in HEALTHCARE_CATEGORIES
        or disposition.startswith("healthcare")
        or "healthcare" in reason
    )


class JSONLValidator:
    """Validate a SecureLogX JSONL file against a canonical label config."""

    def __init__(self, labels_path: str):
        self.labels_path = Path(labels_path)
        self.label_order, self.labels = self._load_labels()
        self._reset_results()

    def _reset_results(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "negative_records": 0,
            "total_entities": 0,
            "entities_by_label": Counter(),
            "text_lengths": [],
            "blank_lines": 0,
            "duplicate_records": 0,
            "duplicate_text_values": 0,
            "duplicate_texts": [],
            "duplicate_entities": 0,
            "overlap_conflicts": 0,
            "records_with_excluded_spans": 0,
            "excluded_spans": 0,
            "excluded_by_label": Counter(),
            "training_exclusion_records": 0,
            "training_exclusion_spans": 0,
            "training_exclusion_by_label": Counter(),
            "benign_excluded_spans": 0,
            "deferred_records": 0,
            "deferred_spans": 0,
            "deferred_by_label": Counter(),
            "healthcare_records": 0,
            "healthcare_spans": 0,
            "healthcare_by_label": Counter(),
        }

    def _load_labels(self) -> Tuple[List[str], Set[str]]:
        """Load and sanity-check the canonical entity label configuration."""

        with self.labels_path.open(encoding="utf-8") as handle:
            config = json.load(handle)

        if not isinstance(config, dict):
            raise ValueError("Label config must be a JSON object")
        entities = config.get("entities")
        if not isinstance(entities, list) or not entities:
            raise ValueError("Label config 'entities' must be a non-empty list")
        if any(not _nonempty_string(label) for label in entities):
            raise ValueError("Every canonical entity label must be a non-empty string")
        if any(label == "O" or label.startswith(("B-", "I-")) for label in entities):
            raise ValueError("Label config 'entities' must contain bare entity names")
        if len(set(entities)) != len(entities):
            raise ValueError("Label config contains duplicate entity names")

        expected_bio = ["O"]
        for entity in entities:
            expected_bio.extend((f"B-{entity}", f"I-{entity}"))

        bio_labels = config.get("bio_labels")
        if bio_labels is not None and bio_labels != expected_bio:
            raise ValueError(
                "Label config 'bio_labels' must be O followed by B-/I- labels "
                "for each entity in canonical order"
            )

        label_to_id = config.get("label_to_id")
        if label_to_id is not None:
            expected_label_to_id = {
                label: index for index, label in enumerate(expected_bio)
            }
            if label_to_id != expected_label_to_id:
                raise ValueError("Label config 'label_to_id' is inconsistent with bio_labels")

        id_to_label = config.get("id_to_label")
        if id_to_label is not None:
            expected_id_to_label = {
                str(index): label for index, label in enumerate(expected_bio)
            }
            if id_to_label != expected_id_to_label:
                raise ValueError("Label config 'id_to_label' is inconsistent with bio_labels")

        return list(entities), set(entities)

    def _error(self, line_num: int, message: str) -> None:
        self.errors.append(f"Line {line_num}: {message}")

    def _warning(self, line_num: int, message: str) -> None:
        self.warnings.append(f"Line {line_num}: {message}")

    def _validate_entities(
        self, line_num: int, text: str, entities: Any
    ) -> List[Tuple[int, int, str]]:
        if not isinstance(entities, list):
            self._error(line_num, "'entities' must be a list")
            return []

        valid_entities: List[Tuple[int, int, str]] = []
        seen: Set[Tuple[int, int, str]] = set()

        for index, entity in enumerate(entities):
            prefix = f"Entity {index}"
            if not isinstance(entity, (list, tuple)) or len(entity) != 3:
                self._error(
                    line_num,
                    f"{prefix} must be exactly [start, end, label]",
                )
                continue

            start, end, label = entity
            if not _is_strict_int(start) or not _is_strict_int(end):
                self._error(line_num, f"{prefix} offsets must be integers")
                continue
            if start < 0 or end > len(text) or start >= end:
                self._error(
                    line_num,
                    f"{prefix} has invalid offsets [{start}, {end}] for text length {len(text)}",
                )
                continue
            if not isinstance(label, str) or label not in self.labels:
                self._error(line_num, f"{prefix} has unsupported label {label!r}")
                continue
            if not text[start:end]:
                self._error(line_num, f"{prefix} extracts an empty string")
                continue

            normalized = (start, end, label)
            if normalized in seen:
                self._error(line_num, f"Duplicate entity span {list(normalized)!r}")
                self.stats["duplicate_entities"] += 1
                continue
            seen.add(normalized)
            valid_entities.append(normalized)

        ordered = sorted(valid_entities, key=lambda item: (item[0], item[1], item[2]))
        for previous, current in zip(ordered, ordered[1:]):
            if previous[1] > current[0]:
                self._error(
                    line_num,
                    "Overlapping training entities "
                    f"{list(previous)!r} and {list(current)!r}",
                )
                self.stats["overlap_conflicts"] += 1

        return valid_entities

    def _validate_entity_provenance(
        self,
        line_num: int,
        text: str,
        entity_set: Set[Tuple[int, int, str]],
        provenance: Any,
    ) -> None:
        if not isinstance(provenance, list):
            self._error(line_num, "meta.entity_provenance must be a list")
            return

        seen: Set[Tuple[Any, ...]] = set()
        for index, entry in enumerate(provenance):
            prefix = f"meta.entity_provenance[{index}]"
            if not isinstance(entry, dict):
                self._error(line_num, f"{prefix} must be an object")
                continue

            missing = [
                key for key in ("start", "end", "source_label") if key not in entry
            ]
            if "label" not in entry and "canonical_label" not in entry:
                missing.append("label")
            if missing:
                self._error(line_num, f"{prefix} missing required fields: {', '.join(missing)}")
                continue

            start = entry.get("start")
            end = entry.get("end")
            label = entry.get("label", entry.get("canonical_label"))
            source_label = entry.get("source_label")
            if "label" in entry and "canonical_label" in entry:
                if entry["label"] != entry["canonical_label"]:
                    self._error(line_num, f"{prefix} label aliases disagree")
                    continue

            if not _is_strict_int(start) or not _is_strict_int(end):
                self._error(line_num, f"{prefix} offsets must be integers")
                continue
            if start < 0 or end > len(text) or start >= end:
                self._error(line_num, f"{prefix} offsets are outside the record text")
                continue
            if not isinstance(label, str) or label not in self.labels:
                self._error(line_num, f"{prefix} has unsupported canonical label {label!r}")
                continue
            if not _nonempty_string(source_label):
                self._error(line_num, f"{prefix}.source_label must be a non-empty string")
                continue
            if (start, end, label) not in entity_set:
                self._error(
                    line_num,
                    f"{prefix} does not match a retained training entity",
                )

            if "source_span_index" in entry:
                source_index = entry["source_span_index"]
                if not _is_strict_int(source_index) or source_index < 0:
                    self._error(
                        line_num,
                        f"{prefix}.source_span_index must be a non-negative integer",
                    )
            if "source_field" in entry and not _nonempty_string(entry["source_field"]):
                self._error(line_num, f"{prefix}.source_field must be a non-empty string")
            if "action" in entry and not _nonempty_string(entry["action"]):
                self._error(line_num, f"{prefix}.action must be a non-empty string")
            if "training_exclusion" in entry:
                exclusion = entry["training_exclusion"]
                if not isinstance(exclusion, bool) or exclusion:
                    self._error(
                        line_num,
                        f"{prefix}.training_exclusion must be false for a retained entity",
                    )
            if "text" in entry:
                value = entry["text"]
                if not isinstance(value, str) or value != text[start:end]:
                    self._error(line_num, f"{prefix}.text does not match text[start:end]")

            signature = (
                start,
                end,
                label,
                source_label,
                entry.get("source_span_index"),
                entry.get("source_field"),
            )
            if signature in seen:
                self._error(line_num, f"Duplicate {prefix} entry")
            seen.add(signature)

    def _validate_excluded_spans(
        self,
        line_num: int,
        text: str,
        entity_set: Set[Tuple[int, int, str]],
        excluded_spans: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(excluded_spans, list):
            self._error(line_num, "meta.excluded_spans must be a list")
            return []

        validated: List[Dict[str, Any]] = []
        seen: Set[Tuple[Any, ...]] = set()
        for index, entry in enumerate(excluded_spans):
            prefix = f"meta.excluded_spans[{index}]"
            if not isinstance(entry, dict):
                self._error(line_num, f"{prefix} must be an object")
                continue

            required = ("start", "end", "source_label", "reason", "training_exclusion")
            missing = [key for key in required if key not in entry]
            if missing:
                self._error(line_num, f"{prefix} missing required fields: {', '.join(missing)}")
                continue

            start = entry.get("start")
            end = entry.get("end")
            source_label = entry.get("source_label")
            reason = entry.get("reason")
            training_exclusion = entry.get("training_exclusion")
            category = entry.get("category")

            if not _nonempty_string(source_label):
                self._error(line_num, f"{prefix}.source_label must be a non-empty string")
                continue
            if not _nonempty_string(reason):
                self._error(line_num, f"{prefix}.reason must be a non-empty string")
                continue
            if not isinstance(training_exclusion, bool):
                self._error(line_num, f"{prefix}.training_exclusion must be boolean")
                continue
            if "category" in entry and not _nonempty_string(category):
                self._error(line_num, f"{prefix}.category must be a non-empty string")

            # Invalid upstream coordinates are themselves trace evidence. They
            # are safe to retain only when explicitly categorized and blocked
            # from the training merge; they are not canonical entity spans.
            invalid_source_trace = (
                category == "invalid_source_span" and training_exclusion is True
            )
            coordinates_valid = _is_strict_int(start) and _is_strict_int(end)
            if not coordinates_valid:
                if not invalid_source_trace:
                    self._error(line_num, f"{prefix} offsets must be integers")
                    continue
            elif start < 0 or end > len(text) or start >= end:
                coordinates_valid = False
                if not invalid_source_trace:
                    self._error(line_num, f"{prefix} offsets are outside the record text")
                    continue

            mapped_label = entry.get("mapped_label")
            if mapped_label is not None:
                if not isinstance(mapped_label, str) or mapped_label not in self.labels:
                    self._error(
                        line_num,
                        f"{prefix}.mapped_label has unsupported canonical label {mapped_label!r}",
                    )
                elif coordinates_valid and (start, end, mapped_label) in entity_set:
                    self._error(
                        line_num,
                        f"{prefix} is also present as the same retained canonical entity",
                    )
            if "disposition" in entry and not _nonempty_string(entry["disposition"]):
                self._error(line_num, f"{prefix}.disposition must be a non-empty string")
            if "source_span_index" in entry:
                source_index = entry["source_span_index"]
                if not _is_strict_int(source_index) or source_index < 0:
                    self._error(
                        line_num,
                        f"{prefix}.source_span_index must be a non-negative integer",
                    )
            if "source_field" in entry and not _nonempty_string(entry["source_field"]):
                self._error(line_num, f"{prefix}.source_field must be a non-empty string")
            if "text" in entry and coordinates_valid:
                value = entry["text"]
                if not isinstance(value, str) or value != text[start:end]:
                    self._error(line_num, f"{prefix}.text does not match text[start:end]")

            signature = (
                repr(start),
                repr(end),
                source_label,
                entry.get("source_span_index"),
                entry.get("source_field"),
            )
            if signature in seen:
                self._error(line_num, f"Duplicate {prefix} entry")
            seen.add(signature)
            validated.append(entry)

        return validated

    def _validate_meta(
        self,
        line_num: int,
        text: str,
        entities: Sequence[Tuple[int, int, str]],
        meta: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(meta, dict):
            self._error(line_num, "'meta' must be an object")
            return []

        source = meta.get("source")
        if not _nonempty_string(source):
            self._error(line_num, "meta.source must be a non-empty string")

        review_status = meta.get("review_status")
        if not _nonempty_string(review_status):
            self._error(line_num, "meta.review_status must be a non-empty string")
        elif review_status not in ALLOWED_REVIEW_STATUSES:
            allowed = ", ".join(sorted(ALLOWED_REVIEW_STATUSES))
            self._error(
                line_num,
                f"meta.review_status {review_status!r} is not training-eligible; allowed: {allowed}",
            )
        elif review_status in LEGACY_REVIEW_STATUS_ALIASES:
            replacement = LEGACY_REVIEW_STATUS_ALIASES[review_status]
            self._warning(
                line_num,
                f"Legacy review_status {review_status!r}; migrate to {replacement!r}",
            )

        if "license_reviewed" not in meta:
            self._error(line_num, "Missing meta.license_reviewed")
        elif not isinstance(meta["license_reviewed"], bool):
            self._error(line_num, "meta.license_reviewed must be boolean")

        template_family = meta.get("template_family")
        if "template_family" in meta:
            if not _nonempty_string(template_family):
                self._error(line_num, "meta.template_family must be a non-empty string")
            elif template_family != template_family.strip() or any(
                character.isspace() for character in template_family
            ):
                self._error(
                    line_num,
                    "meta.template_family must be a stable identifier without whitespace",
                )
        if review_status == "synthetic_generated" and "template_family" not in meta:
            self._error(
                line_num,
                "synthetic_generated record is missing meta.template_family",
            )
        elif review_status == "generated" and "template_family" not in meta:
            self._warning(
                line_num,
                "Legacy generated record is missing meta.template_family",
            )

        if "training_exclusion" in meta and not isinstance(
            meta["training_exclusion"], bool
        ):
            self._error(line_num, "meta.training_exclusion must be boolean")
        if "training_exclusion_reasons" in meta:
            reasons = meta["training_exclusion_reasons"]
            if not isinstance(reasons, list) or any(
                not _nonempty_string(reason) for reason in reasons
            ):
                self._error(
                    line_num,
                    "meta.training_exclusion_reasons must be a list of non-empty strings",
                )

        entity_set = set(entities)
        if "entity_provenance" in meta:
            self._validate_entity_provenance(
                line_num,
                text,
                entity_set,
                meta["entity_provenance"],
            )

        excluded_entries: List[Dict[str, Any]] = []
        if "excluded_spans" in meta:
            excluded_entries = self._validate_excluded_spans(
                line_num,
                text,
                entity_set,
                meta["excluded_spans"],
            )

        if meta.get("training_exclusion") is False and any(
            entry.get("training_exclusion") is True for entry in excluded_entries
        ):
            self._error(
                line_num,
                "meta.training_exclusion=false conflicts with an excluded span marked true",
            )

        return excluded_entries

    def _commit_valid_record(
        self,
        text: str,
        entities: Sequence[Tuple[int, int, str]],
        meta: Mapping[str, Any],
        excluded_entries: Sequence[Mapping[str, Any]],
    ) -> None:
        self.stats["valid_records"] += 1
        self.stats["text_lengths"].append(len(text))
        self.stats["total_entities"] += len(entities)
        if not entities:
            self.stats["negative_records"] += 1
        for _, _, label in entities:
            self.stats["entities_by_label"][label] += 1

        if excluded_entries:
            self.stats["records_with_excluded_spans"] += 1
        record_training_exclusion = meta.get("training_exclusion") is True
        has_training_excluded_span = False
        has_deferred_span = False
        has_healthcare_span = False
        for entry in excluded_entries:
            source_label = entry["source_label"]
            self.stats["excluded_spans"] += 1
            self.stats["excluded_by_label"][source_label] += 1
            if entry.get("training_exclusion") is True:
                has_training_excluded_span = True
                self.stats["training_exclusion_spans"] += 1
                self.stats["training_exclusion_by_label"][source_label] += 1
            else:
                self.stats["benign_excluded_spans"] += 1
            if _is_deferred_entry(entry):
                has_deferred_span = True
                self.stats["deferred_spans"] += 1
                self.stats["deferred_by_label"][source_label] += 1
            if _is_healthcare_entry(entry):
                has_healthcare_span = True
                self.stats["healthcare_spans"] += 1
                self.stats["healthcare_by_label"][source_label] += 1

        if has_deferred_span:
            self.stats["deferred_records"] += 1
        if has_healthcare_span:
            self.stats["healthcare_records"] += 1

        if record_training_exclusion or has_training_excluded_span:
            self.stats["training_exclusion_records"] += 1

    def validate_record(
        self, record: Any, line_num: int = 1
    ) -> Tuple[
        bool,
        Optional[str],
        List[Tuple[int, int, str]],
        Optional[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Validate one decoded record without resetting accumulated results.

        The parsed components are returned so callers such as the merge stage
        can apply additional record-level policies without rewriting the schema
        checks implemented here.
        """

        error_count_before = len(self.errors)
        if not isinstance(record, dict):
            self._error(line_num, "Record must be a JSON object")
            return False, None, [], None, []

        missing = [key for key in ("text", "entities", "meta") if key not in record]
        if missing:
            self._error(line_num, f"Missing required keys: {', '.join(missing)}")
            return False, None, [], None, []

        text = record["text"]
        if not isinstance(text, str):
            self._error(line_num, "'text' must be a string")
            return False, None, [], None, []
        if not text.strip():
            self._error(line_num, "'text' must not be empty or whitespace-only")
            return False, text, [], None, []

        entities = self._validate_entities(line_num, text, record["entities"])
        excluded_entries = self._validate_meta(
            line_num,
            text,
            entities,
            record["meta"],
        )
        meta = record["meta"] if isinstance(record["meta"], dict) else None
        return (
            len(self.errors) == error_count_before,
            text,
            entities,
            meta,
            excluded_entries,
        )

    def validate_file(self, jsonl_path: str, dedupe: bool = False) -> bool:
        """Validate a JSONL file.

        ``dedupe`` retains the historical CLI/API name. The validator never
        rewrites input; when true it promotes exact duplicate text from a
        warning to an error so it can serve as a leakage quality gate.
        """

        self._reset_results()
        input_path = Path(jsonl_path)
        if not input_path.is_file():
            self.errors.append(f"File not found: {jsonl_path}")
            return False

        seen_texts: Dict[str, int] = {}
        duplicate_values: Set[str] = set()

        try:
            with input_path.open(encoding="utf-8") as handle:
                for line_num, raw_line in enumerate(handle, 1):
                    if not raw_line.strip():
                        self.stats["blank_lines"] += 1
                        continue

                    self.stats["total_records"] += 1
                    error_count_before = len(self.errors)

                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        self._error(line_num, f"Invalid JSON - {exc}")
                        self.stats["invalid_records"] += 1
                        continue

                    (
                        record_is_valid,
                        text,
                        entities,
                        meta,
                        excluded_entries,
                    ) = self.validate_record(record, line_num)

                    if text is None:
                        self.stats["invalid_records"] += 1
                        continue

                    if text in seen_texts:
                        first_line = seen_texts[text]
                        self.stats["duplicate_records"] += 1
                        duplicate_values.add(text)
                        duplicate_info = (line_num, first_line, text[:50])
                        self.stats["duplicate_texts"].append(duplicate_info)
                        message = f"Exact duplicate text; first seen on line {first_line}"
                        if dedupe:
                            self._error(line_num, message)
                        else:
                            self._warning(line_num, message)
                    else:
                        seen_texts[text] = line_num

                    if record_is_valid and len(self.errors) == error_count_before:
                        assert meta is not None
                        self._commit_valid_record(
                            text,
                            entities,
                            meta,
                            excluded_entries,
                        )
                    else:
                        self.stats["invalid_records"] += 1
        except OSError as exc:
            self.errors.append(f"Could not read {jsonl_path}: {exc}")
            return False

        self.stats["duplicate_text_values"] = len(duplicate_values)
        if self.stats["blank_lines"]:
            self.warnings.append(
                f"Ignored {self.stats['blank_lines']} blank JSONL line(s)"
            )
        return not self.errors

    def generate_quality_report(self, output_path: str) -> None:
        """Write a deterministic, audit-friendly Markdown quality report."""

        total_records = self.stats["total_records"]
        valid_records = self.stats["valid_records"]
        valid_pct = (valid_records / total_records * 100) if total_records else 0.0
        lengths = self.stats["text_lengths"]
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0

        lines = [
            "# Dataset Quality Report",
            "",
            "## Summary",
            "",
            f"- **Total Records**: {total_records}",
            f"- **Valid Records**: {valid_records}",
            f"- **Invalid Records**: {self.stats['invalid_records']}",
            f"- **Validity Rate**: {valid_pct:.1f}%",
            f"- **Negative Records (zero entities)**: {self.stats['negative_records']}",
            f"- **Total Canonical Entities**: {self.stats['total_entities']}",
            f"- **Average Text Length**: {avg_length:.1f} characters",
            f"- **Exact Duplicate Records**: {self.stats['duplicate_records']}",
            f"- **Distinct Duplicated Text Values**: {self.stats['duplicate_text_values']}",
            f"- **Duplicate Entity Spans**: {self.stats['duplicate_entities']}",
            f"- **Training-Entity Overlap Conflicts**: {self.stats['overlap_conflicts']}",
            "",
            "## Excluded and Deferred Span Metadata",
            "",
            f"- **Records With Excluded Spans**: {self.stats['records_with_excluded_spans']}",
            f"- **All Excluded/Trace Spans**: {self.stats['excluded_spans']}",
            f"- **Training-Excluding Spans**: {self.stats['training_exclusion_spans']}",
            f"- **Benign Trace-Only Spans**: {self.stats['benign_excluded_spans']}",
            f"- **Records Excluded From Training**: {self.stats['training_exclusion_records']}",
            f"- **Records With Deferred ML-v1 Spans**: {self.stats['deferred_records']}",
            f"- **Deferred ML-v1 Spans**: {self.stats['deferred_spans']}",
            f"- **Records With Healthcare-Extension Spans**: {self.stats['healthcare_records']}",
            f"- **Healthcare-Extension Spans**: {self.stats['healthcare_spans']}",
            "",
            "## Canonical Entity Distribution",
            "",
            "| Entity | Count | Percentage |",
            "|---|---:|---:|",
        ]

        total_entities = self.stats["total_entities"]
        for label in self.label_order:
            count = self.stats["entities_by_label"].get(label, 0)
            percentage = (count / total_entities * 100) if total_entities else 0.0
            lines.append(f"| {_markdown(label)} | {count} | {percentage:.2f}% |")

        lines.extend(
            [
                "",
                "## Excluded/Trace Source Labels",
                "",
                "| Source Label | All Spans | Training-Excluding | Deferred ML-v1 | Healthcare Extension |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        excluded_labels = sorted(self.stats["excluded_by_label"])
        if excluded_labels:
            for label in excluded_labels:
                lines.append(
                    f"| {_markdown(label)} | "
                    f"{self.stats['excluded_by_label'][label]} | "
                    f"{self.stats['training_exclusion_by_label'][label]} | "
                    f"{self.stats['deferred_by_label'][label]} | "
                    f"{self.stats['healthcare_by_label'][label]} |"
                )
        else:
            lines.append("| _None_ | 0 | 0 | 0 | 0 |")

        if self.warnings:
            lines.extend(["", f"## Warnings ({len(self.warnings)})", ""])
            lines.extend(f"- {_markdown(message)}" for message in self.warnings[:50])
            if len(self.warnings) > 50:
                lines.append(f"- ... and {len(self.warnings) - 50} more warnings")

        if self.errors:
            lines.extend(["", f"## Errors ({len(self.errors)})", ""])
            lines.extend(f"- {_markdown(message)}" for message in self.errors[:50])
            if len(self.errors) > 50:
                lines.append(f"- ... and {len(self.errors) - 50} more errors")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def generate_label_distribution(self, output_path: str) -> None:
        """Write canonical label counts, including zero-count labels, as CSV."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        total = self.stats["total_entities"]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("label", "count", "percentage"))
            for label in self.label_order:
                count = self.stats["entities_by_label"].get(label, 0)
                percentage = (count / total * 100) if total else 0.0
                writer.writerow((label, count, f"{percentage:.2f}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate SecureLogX canonical JSONL spans and metadata"
    )
    parser.add_argument("--in", dest="input_file", required=True, help="Input JSONL file")
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Canonical label config (default: configs/securelogx_labels.json)",
    )
    parser.add_argument("--report", required=True, help="Output Markdown quality report")
    parser.add_argument(
        "--label-dist", required=True, help="Output canonical label distribution CSV"
    )
    parser.add_argument(
        "--dedupe",
        "--fail-on-duplicates",
        dest="dedupe",
        action="store_true",
        help="Fail the quality gate when exact duplicate text records are present",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validator = JSONLValidator(args.labels)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Label configuration error: {exc}", file=sys.stderr)
        return 2

    is_valid = validator.validate_file(args.input_file, dedupe=args.dedupe)
    validator.generate_quality_report(args.report)
    validator.generate_label_distribution(args.label_dist)

    print(f"Validation {'PASSED' if is_valid else 'FAILED'}")
    print(
        f"Valid records: {validator.stats['valid_records']}/"
        f"{validator.stats['total_records']}"
    )
    print(f"Canonical entities: {validator.stats['total_entities']}")
    print(f"Negative records: {validator.stats['negative_records']}")
    print(f"Exact duplicate records: {validator.stats['duplicate_records']}")
    print(
        "Excluded spans: "
        f"{validator.stats['excluded_spans']} total, "
        f"{validator.stats['training_exclusion_spans']} training-excluding"
    )
    print(f"Errors: {len(validator.errors)}, Warnings: {len(validator.warnings)}")
    print(f"Report: {args.report}")
    print(f"Label dist: {args.label_dist}")
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
