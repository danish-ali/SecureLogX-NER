#!/usr/bin/env python3
"""Read-only validation for the SecureLogX ML-v1.1 context-contrast data.

The validator is deliberately independent from data generation.  It validates
the three new partitions as one dataset, checks them against the frozen ML-v1
train/dev/test files, and emits a JSON-serializable result.  It never writes a
report, manifest, or dataset.

The public entry points are :func:`validate_ontology`,
:func:`validate_split_records`, :func:`audit_template_catalogs`, and
:func:`validate_dataset`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_ENTITIES: Tuple[str, ...] = (
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

CANONICAL_BIO_LABELS: Tuple[str, ...] = (
    "O",
    *(label for entity in CANONICAL_ENTITIES for label in (f"B-{entity}", f"I-{entity}")),
)

NEW_SPLITS: Tuple[str, ...] = (
    "train_addition",
    "dev_challenge",
    "sealed_challenge",
)
PARENT_SPLITS: Tuple[str, ...] = ("parent_train", "parent_dev", "parent_test")

DEFAULT_RELATIVE_PATHS: Mapping[str, str] = {
    "labels": "configs/securelogx_labels.json",
    "train_addition": "data/ml_v1_1/context_contrast/train_additions.jsonl",
    "dev_challenge": "data/ml_v1_1/context_contrast/dev_challenge.jsonl",
    "sealed_challenge": "data/ml_v1_1/challenge/sealed_test.jsonl",
    "parent_train": "data/split/train.jsonl",
    "parent_dev": "data/split/dev.jsonl",
    "parent_test": "data/split/test.jsonl",
    "new_generator": "scripts/data/generate_securelogx_ml_v1_1_context_contrast.py",
    "parent_generator": "scripts/data/generate_securelogx_synthetic_logs.py",
}

EXPECTED_SOURCE = "securelogx_context_contrast_v1_1"
EXPECTED_REVIEW_STATUS = "synthetic_generated"
EXPECTED_SCENARIO_KIND = "context_contrast"
EXPECTED_GENERATOR_VERSION = "securelogx-ml-v1.1-context-contrast-v1"

ALLOWED_FORMATS: Set[str] = {"key_value", "json", "text"}
ALLOWED_SURFACE_STYLES: Set[str] = {
    "key_equals",
    "key_colon",
    "json_flat",
    "json_nested",
    "natural_language",
    "warn_message",
    "error_message",
    "structured_audit",
    "mixed_punctuation",
    "abbreviated_key",
}

CATEGORY_SPECS: Mapping[str, Mapping[str, Set[str]]] = {
    "business_id_vs_ssn": {
        "roles": {"business_identifier", "social_security_number"},
        "labels": {"BUSINESS_ID", "SSN"},
    },
    "business_id_vs_credit_card": {
        "roles": {"business_identifier", "payment_card_number"},
        "labels": {"BUSINESS_ID", "CREDIT_CARD_NUMBER"},
    },
    "business_id_vs_bank_account": {
        "roles": {"business_identifier", "bank_account_number"},
        "labels": {"BUSINESS_ID", "BANK_ACCOUNT_NUMBER"},
    },
    "ip_address_vs_technical_reference": {
        "roles": {"network_address", "technical_reference"},
        "labels": {"IP_ADDRESS", "O"},
    },
    "person_name_vs_service_name": {
        "roles": {"person_identity", "service_actor"},
        "labels": {"PERSON_NAME", "O"},
    },
    "street_address_vs_system_location": {
        "roles": {"street_location", "system_location"},
        "labels": {"STREET_ADDRESS", "O"},
    },
}

CORE_CHALLENGE_CATEGORIES: Set[str] = {
    "business_id_vs_ssn",
    "business_id_vs_credit_card",
    "business_id_vs_bank_account",
    "ip_address_vs_technical_reference",
}
TRAIN_ONLY_CATEGORIES: Set[str] = {
    "person_name_vs_service_name",
    "street_address_vs_system_location",
}
REQUIRED_SEALED_FORMATS: Set[str] = {"key_value", "json", "text"}
BUSINESS_ID_SUBTYPES: Tuple[str, ...] = (
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
EXPECTED_SPLIT_COUNTS: Mapping[str, Mapping[str, int]] = {
    "train_addition": {"records": 3000, "families": 30},
    "dev_challenge": {"records": 480, "families": 12},
    "sealed_challenge": {"records": 480, "families": 12},
}
MIN_TRAIN_BUSINESS_SUBTYPE_RECORDS = 40
MIN_PARENT_BASELINE_FRACTION = 0.25

FORBIDDEN_PARENT_FAMILY = "order_reference_ssn_shape_warn_v1"
FORBIDDEN_PARENT_WORDING_PARTS: Tuple[str, ...] = (
    "order reference",
    "was replayed after a warehouse timeout",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"<<[A-Za-z_][A-Za-z0-9_]*>>"),
    re.compile(r"\[\[[A-Za-z_][A-Za-z0-9_]*\]\]"),
    re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}"),
    re.compile(r"(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\})"),
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(
    errors: List[Dict[str, Any]],
    scope: str,
    code: str,
    message: str,
    line: Optional[int] = None,
) -> None:
    entry: Dict[str, Any] = {"scope": scope, "code": code, "message": message}
    if line is not None:
        entry["line"] = line
    errors.append(entry)


def _gate(passed: bool, actual: Any, expected: Any, details: Any = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }
    if details not in (None, [], {}):
        result["details"] = details
    return result


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file strictly, raising a useful exception on bad input."""

    input_path = Path(path)
    records: List[Dict[str, Any]] = []
    with input_path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise ValueError(f"{input_path}: blank JSONL line {line_number}")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path}: invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{input_path}: line {line_number} must be a JSON object"
                )
            records.append(record)
    return records


def validate_ontology(labels_path: str | Path) -> Dict[str, Any]:
    """Validate the label config against the exact frozen 25/51 ontology."""

    path = Path(labels_path)
    errors: List[str] = []
    config: Mapping[str, Any] = {}
    file_hash: Optional[str] = None
    try:
        with path.open(encoding="utf-8-sig") as handle:
            loaded = json.load(handle)
        file_hash = _sha256_file(path)
        if isinstance(loaded, dict):
            config = loaded
        else:
            errors.append("label config must be a JSON object")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"could not read label config: {exc}")

    entities = config.get("entities")
    bio_labels = config.get("bio_labels")
    expected_label_to_id = {
        label: index for index, label in enumerate(CANONICAL_BIO_LABELS)
    }
    expected_id_to_label = {
        str(index): label for index, label in enumerate(CANONICAL_BIO_LABELS)
    }

    if entities != list(CANONICAL_ENTITIES):
        errors.append("entities do not match the frozen ML-v1 order")
    if config.get("entity_count") != 25:
        errors.append("entity_count must be exactly 25")
    if bio_labels != list(CANONICAL_BIO_LABELS):
        errors.append("bio_labels do not match the frozen 51-label BIO order")
    if config.get("bio_label_count") != 51:
        errors.append("bio_label_count must be exactly 51")
    if config.get("label_to_id") != expected_label_to_id:
        errors.append("label_to_id is not the deterministic frozen mapping")
    if config.get("id_to_label") != expected_id_to_label:
        errors.append("id_to_label is not the deterministic frozen mapping")
    if config.get("ontology") != "SecureLogX ML-v1":
        errors.append("ontology must remain 'SecureLogX ML-v1'")
    if config.get("ontology_version") != "ML-v1":
        errors.append("ontology_version must remain 'ML-v1'")

    return {
        "passed": not errors,
        "path": str(path),
        "sha256": file_hash,
        "entity_count": len(entities) if isinstance(entities, list) else None,
        "bio_label_count": len(bio_labels) if isinstance(bio_labels, list) else None,
        "errors": errors,
    }


@dataclass
class _GroupItem:
    line: int
    family: str
    category: str
    morphology: str
    role: str
    expected_label: str
    value_hash: str
    value: str
    text_format: str
    surface_style: str


@dataclass
class _SplitFacts:
    texts: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    families: Set[str] = field(default_factory=set)
    value_hashes: Set[str] = field(default_factory=set)
    groups: Dict[str, List[_GroupItem]] = field(
        default_factory=lambda: defaultdict(list)
    )
    family_items: Dict[str, List[_GroupItem]] = field(
        default_factory=lambda: defaultdict(list)
    )


def _validate_split(
    records: Sequence[Mapping[str, Any]], split: str
) -> Tuple[Dict[str, Any], _SplitFacts]:
    if split not in NEW_SPLITS:
        raise ValueError(f"unsupported ML-v1.1 split: {split!r}")

    errors: List[Dict[str, Any]] = []
    facts = _SplitFacts()
    label_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    contrast_label_counts: Counter[str] = Counter()
    business_id_subtype_counts: Counter[str] = Counter()
    invalid_records = 0

    for line_number, record in enumerate(records, 1):
        before = len(errors)
        if not isinstance(record, Mapping):
            _error(errors, split, "record_shape", "record must be an object", line_number)
            invalid_records += 1
            continue

        text = record.get("text")
        if not _nonempty_string(text):
            _error(
                errors,
                split,
                "record_shape",
                "text must be a non-empty string",
                line_number,
            )
            text = "" if not isinstance(text, str) else text
        else:
            facts.texts.append(text)

        entities_raw = record.get("entities")
        entities: List[Tuple[int, int, str]] = []
        if not isinstance(entities_raw, list):
            _error(errors, split, "span", "entities must be a list", line_number)
        else:
            seen_entities: Set[Tuple[int, int, str]] = set()
            for entity_index, raw_entity in enumerate(entities_raw):
                if not isinstance(raw_entity, (list, tuple)) or len(raw_entity) != 3:
                    _error(
                        errors,
                        split,
                        "span",
                        f"entity {entity_index} must be [start, end, label]",
                        line_number,
                    )
                    continue
                start, end, label = raw_entity
                if (
                    not _strict_int(start)
                    or not _strict_int(end)
                    or start < 0
                    or end > len(text)
                    or start >= end
                ):
                    _error(
                        errors,
                        split,
                        "span",
                        f"entity {entity_index} has invalid offsets",
                        line_number,
                    )
                    continue
                if label not in CANONICAL_ENTITIES:
                    _error(
                        errors,
                        split,
                        "span",
                        f"entity {entity_index} has a non-canonical label",
                        line_number,
                    )
                    continue
                signature = (start, end, label)
                if signature in seen_entities:
                    _error(
                        errors,
                        split,
                        "span",
                        f"entity {entity_index} duplicates another entity",
                        line_number,
                    )
                    continue
                seen_entities.add(signature)
                entities.append(signature)
                label_counts[label] += 1

            ordered = sorted(entities, key=lambda item: (item[0], item[1], item[2]))
            if entities != ordered:
                _error(
                    errors,
                    split,
                    "span",
                    "entities must be in canonical offset order",
                    line_number,
                )
            for previous, current in zip(ordered, ordered[1:]):
                if previous[1] > current[0]:
                    _error(
                        errors,
                        split,
                        "span",
                        "entity spans must not overlap",
                        line_number,
                    )

        meta = record.get("meta")
        if not isinstance(meta, Mapping):
            _error(errors, split, "metadata", "meta must be an object", line_number)
            meta = {}

        expected_metadata: Tuple[Tuple[str, Any], ...] = (
            ("source", EXPECTED_SOURCE),
            ("review_status", EXPECTED_REVIEW_STATUS),
            ("license_reviewed", True),
            ("scenario_kind", EXPECTED_SCENARIO_KIND),
            ("generator_version", EXPECTED_GENERATOR_VERSION),
            ("intended_split", split),
        )
        for key, expected in expected_metadata:
            if meta.get(key) != expected:
                code = "intended_split" if key == "intended_split" else "metadata"
                _error(
                    errors,
                    split,
                    code,
                    f"meta.{key} must equal {expected!r}",
                    line_number,
                )

        source_id = meta.get("source_record_id")
        if not _nonempty_string(source_id):
            _error(
                errors,
                split,
                "metadata",
                "meta.source_record_id must be a non-empty string",
                line_number,
            )
            source_id = ""
        else:
            facts.source_ids.append(source_id)
            if (
                not _IDENTIFIER_RE.fullmatch(source_id)
                or not source_id.startswith(f"ml-v1.1-{split}-")
            ):
                _error(
                    errors,
                    split,
                    "metadata",
                    "meta.source_record_id has the wrong split namespace",
                    line_number,
                )

        family = meta.get("template_family")
        if not _nonempty_string(family) or not _IDENTIFIER_RE.fullmatch(family):
            _error(
                errors,
                split,
                "metadata",
                "meta.template_family must be a stable identifier",
                line_number,
            )
            family = ""
        else:
            facts.families.add(family)
            if not (
                family.startswith(f"ml_v1_1_{split}_") and family.endswith("_v1_1")
            ):
                _error(
                    errors,
                    split,
                    "metadata",
                    "meta.template_family has the wrong split namespace/version",
                    line_number,
                )
            if family == FORBIDDEN_PARENT_FAMILY:
                _error(
                    errors,
                    split,
                    "forbidden_parent_template",
                    "the failed ML-v1 hard-negative family is forbidden",
                    line_number,
                )

        group_id = meta.get("contrast_group_id")
        if not _nonempty_string(group_id) or not _IDENTIFIER_RE.fullmatch(group_id):
            _error(
                errors,
                split,
                "metadata",
                "meta.contrast_group_id must be a stable identifier",
                line_number,
            )
            group_id = ""
        elif not group_id.startswith(f"ml-v1.1-{split}-"):
            _error(
                errors,
                split,
                "metadata",
                "meta.contrast_group_id has the wrong split namespace",
                line_number,
            )

        text_format = meta.get("format")
        if text_format not in ALLOWED_FORMATS:
            _error(
                errors,
                split,
                "metadata",
                "meta.format is not an approved ML-v1.1 format",
                line_number,
            )
            text_format = ""
        else:
            format_counts[text_format] += 1

        surface_style = meta.get("surface_style")
        if surface_style not in ALLOWED_SURFACE_STYLES:
            _error(
                errors,
                split,
                "metadata",
                "meta.surface_style is not in the approved vocabulary",
                line_number,
            )
            surface_style = ""
        else:
            style_counts[surface_style] += 1

        category = meta.get("contrast_category")
        if category not in CATEGORY_SPECS:
            _error(
                errors,
                split,
                "metadata",
                "meta.contrast_category is not recognized",
                line_number,
            )
            category = ""
        else:
            category_counts[category] += 1
            if split != "train_addition" and category in TRAIN_ONLY_CATEGORIES:
                _error(
                    errors,
                    split,
                    "metadata",
                    "train-only contrast category appears in a challenge split",
                    line_number,
                )

        role = meta.get("context_role")
        if not _nonempty_string(role):
            _error(
                errors,
                split,
                "metadata",
                "meta.context_role must be a non-empty string",
                line_number,
            )
            role = ""
        else:
            role_counts[role] += 1
            if category and role not in CATEGORY_SPECS[category]["roles"]:
                _error(
                    errors,
                    split,
                    "metadata",
                    "meta.context_role is inconsistent with contrast_category",
                    line_number,
                )

        morphology = meta.get("morphology_class")
        if not _nonempty_string(morphology) or not _IDENTIFIER_RE.fullmatch(morphology):
            _error(
                errors,
                split,
                "metadata",
                "meta.morphology_class must be a stable identifier",
                line_number,
            )
            morphology = ""

        provenance = meta.get("entity_provenance")
        provenance_signatures: List[Tuple[int, int, str]] = []
        if not isinstance(provenance, list):
            _error(
                errors,
                split,
                "provenance",
                "meta.entity_provenance must be a list",
                line_number,
            )
        else:
            for provenance_index, entry in enumerate(provenance):
                if not isinstance(entry, Mapping):
                    _error(
                        errors,
                        split,
                        "provenance",
                        f"provenance {provenance_index} must be an object",
                        line_number,
                    )
                    continue
                start, end, label = (
                    entry.get("start"),
                    entry.get("end"),
                    entry.get("label"),
                )
                if (
                    not _strict_int(start)
                    or not _strict_int(end)
                    or start < 0
                    or end > len(text)
                    or start >= end
                    or label not in CANONICAL_ENTITIES
                ):
                    _error(
                        errors,
                        split,
                        "provenance",
                        f"provenance {provenance_index} has invalid canonical coordinates",
                        line_number,
                    )
                    continue
                provenance_signatures.append((start, end, label))
                if not _nonempty_string(entry.get("source_label")):
                    _error(
                        errors,
                        split,
                        "provenance",
                        f"provenance {provenance_index} needs source_label",
                        line_number,
                    )
                elif label == "BUSINESS_ID":
                    source_label = entry.get("source_label")
                    if source_label not in BUSINESS_ID_SUBTYPES:
                        _error(
                            errors,
                            split,
                            "provenance",
                            f"provenance {provenance_index} has an unsupported BUSINESS_ID subtype",
                            line_number,
                        )
                    if entry.get("business_id_subtype") != source_label:
                        _error(
                            errors,
                            split,
                            "provenance",
                            f"provenance {provenance_index}.business_id_subtype must match source_label",
                            line_number,
                        )
                if not _nonempty_string(entry.get("action")):
                    _error(
                        errors,
                        split,
                        "provenance",
                        f"provenance {provenance_index} needs action",
                        line_number,
                    )
                if entry.get("training_exclusion") is not False:
                    _error(
                        errors,
                        split,
                        "provenance",
                        f"provenance {provenance_index}.training_exclusion must be false",
                        line_number,
                    )
                if "text" in entry and entry.get("text") != text[start:end]:
                    _error(
                        errors,
                        split,
                        "provenance",
                        f"provenance {provenance_index}.text does not match its span",
                        line_number,
                    )

        if Counter(provenance_signatures) != Counter(entities):
            _error(
                errors,
                split,
                "provenance",
                "entities and entity_provenance must have a one-to-one canonical mapping",
                line_number,
            )

        targets = meta.get("contrast_targets")
        target_item: Optional[_GroupItem] = None
        if not isinstance(targets, list) or len(targets) != 1:
            _error(
                errors,
                split,
                "contrast_target",
                "meta.contrast_targets must contain exactly one target",
                line_number,
            )
        else:
            target = targets[0]
            if not isinstance(target, Mapping):
                _error(
                    errors,
                    split,
                    "contrast_target",
                    "the contrast target must be an object",
                    line_number,
                )
            else:
                start = target.get("start")
                end = target.get("end")
                expected_label = target.get("expected_label")
                value_hash = target.get("value_sha256")
                target_valid = True
                if (
                    not _strict_int(start)
                    or not _strict_int(end)
                    or start < 0
                    or end > len(text)
                    or start >= end
                ):
                    _error(
                        errors,
                        split,
                        "contrast_target",
                        "contrast target has invalid offsets",
                        line_number,
                    )
                    target_valid = False
                if expected_label not in set(CANONICAL_ENTITIES) | {"O"}:
                    _error(
                        errors,
                        split,
                        "contrast_target",
                        "contrast target has a non-canonical expected_label",
                        line_number,
                    )
                    target_valid = False
                if not isinstance(value_hash, str) or not _SHA256_RE.fullmatch(value_hash):
                    _error(
                        errors,
                        split,
                        "contrast_target",
                        "contrast target value_sha256 must be lowercase SHA-256",
                        line_number,
                    )
                    target_valid = False
                if not _nonempty_string(target.get("source_label")):
                    _error(
                        errors,
                        split,
                        "contrast_target",
                        "contrast target source_label must be a non-empty string",
                        line_number,
                    )
                    target_valid = False

                if target_valid:
                    value = text[start:end]
                    if _sha256_text(value) != value_hash:
                        _error(
                            errors,
                            split,
                            "contrast_target",
                            "contrast target value_sha256 does not match text[start:end]",
                            line_number,
                        )
                        target_valid = False
                    if expected_label == "O":
                        if any(start < entity_end and end > entity_start for entity_start, entity_end, _ in entities):
                            _error(
                                errors,
                                split,
                                "o_target_overlap",
                                "an O contrast target must not overlap any canonical entity",
                                line_number,
                            )
                            target_valid = False
                    elif (start, end, expected_label) not in set(entities):
                        _error(
                            errors,
                            split,
                            "contrast_target",
                            "positive contrast target must exactly match a canonical entity",
                            line_number,
                        )
                        target_valid = False

                    if category and expected_label not in CATEGORY_SPECS[category]["labels"]:
                        _error(
                            errors,
                            split,
                            "contrast_target",
                            "expected_label is inconsistent with contrast_category",
                            line_number,
                        )
                        target_valid = False

                    if expected_label == "BUSINESS_ID":
                        subtype = target.get("source_label")
                        if subtype not in BUSINESS_ID_SUBTYPES:
                            _error(
                                errors,
                                split,
                                "contrast_target",
                                "BUSINESS_ID target has an unsupported subtype",
                                line_number,
                            )
                            target_valid = False
                        elif target.get("business_id_subtype") != subtype:
                            _error(
                                errors,
                                split,
                                "contrast_target",
                                "BUSINESS_ID target subtype must match source_label",
                                line_number,
                            )
                            target_valid = False
                        else:
                            business_id_subtype_counts[subtype] += 1

                    contrast_label_counts[expected_label] += 1
                    facts.value_hashes.add(value_hash)
                    if target_valid and all(
                        (
                            family,
                            group_id,
                            category,
                            morphology,
                            role,
                            text_format,
                            surface_style,
                        )
                    ):
                        target_item = _GroupItem(
                            line=line_number,
                            family=family,
                            category=category,
                            morphology=morphology,
                            role=role,
                            expected_label=expected_label,
                            value_hash=value_hash,
                            value=value,
                            text_format=text_format,
                            surface_style=surface_style,
                        )
                        facts.groups[group_id].append(target_item)
                        facts.family_items[family].append(target_item)

        lowered_text = text.casefold()
        if all(part in lowered_text for part in FORBIDDEN_PARENT_WORDING_PARTS):
            _error(
                errors,
                split,
                "forbidden_parent_template",
                "record reuses the failed parent hard-negative wording",
                line_number,
            )

        if len(errors) > before:
            invalid_records += 1

    text_counts = Counter(facts.texts)
    id_counts = Counter(facts.source_ids)
    duplicate_text_records = sum(count - 1 for count in text_counts.values() if count > 1)
    duplicate_id_records = sum(count - 1 for count in id_counts.values() if count > 1)
    if duplicate_text_records:
        _error(
            errors,
            split,
            "duplicate_text",
            f"split has {duplicate_text_records} repeated record text values",
        )
    if duplicate_id_records:
        _error(
            errors,
            split,
            "duplicate_source_id",
            f"split has {duplicate_id_records} repeated source_record_id values",
        )

    complete_groups = 0
    for group_id, items in sorted(facts.groups.items()):
        group_ok = True
        if len(items) != 2:
            _error(
                errors,
                split,
                "pair_group",
                f"contrast group {group_id!r} must contain exactly two records",
            )
            group_ok = False
        for field_name, values in (
            ("template_family", {item.family for item in items}),
            ("contrast_category", {item.category for item in items}),
            ("morphology_class", {item.morphology for item in items}),
            ("target value", {item.value for item in items}),
            ("value_sha256", {item.value_hash for item in items}),
        ):
            if len(values) != 1:
                _error(
                    errors,
                    split,
                    "pair_group",
                    f"contrast group {group_id!r} must share one {field_name}",
                )
                group_ok = False
        if items:
            category_name = items[0].category
            spec = CATEGORY_SPECS.get(category_name)
            if spec is not None:
                actual_roles = {item.role for item in items}
                actual_labels = {item.expected_label for item in items}
                if actual_roles != spec["roles"]:
                    _error(
                        errors,
                        split,
                        "pair_group",
                        f"contrast group {group_id!r} does not contain both required roles",
                    )
                    group_ok = False
                if actual_labels != spec["labels"]:
                    _error(
                        errors,
                        split,
                        "pair_group",
                        f"contrast group {group_id!r} does not contain both required labels",
                    )
                    group_ok = False
        if group_ok:
            complete_groups += 1

    for family_name, items in sorted(facts.family_items.items()):
        for field_name, values in (
            ("contrast category", {item.category for item in items}),
            ("morphology class", {item.morphology for item in items}),
        ):
            if len(values) != 1:
                _error(
                    errors,
                    split,
                    "family_consistency",
                    f"template family {family_name!r} spans multiple {field_name} values",
                )

    category_families: Dict[str, Set[str]] = defaultdict(set)
    for family_name, items in facts.family_items.items():
        if items:
            category_families[items[0].category].add(family_name)

    code_counts = Counter(error["code"] for error in errors)
    result = {
        "passed": not errors,
        "record_count": len(records),
        "valid_record_count": len(records) - invalid_records,
        "invalid_record_count": invalid_records,
        "entity_span_count": sum(label_counts.values()),
        "label_counts": dict(sorted(label_counts.items())),
        "family_count": len(facts.families),
        "group_count": len(facts.groups),
        "complete_group_count": complete_groups,
        "categories": dict(sorted(category_counts.items())),
        "category_family_counts": {
            category_name: len(families)
            for category_name, families in sorted(category_families.items())
        },
        "formats": dict(sorted(format_counts.items())),
        "surface_styles": dict(sorted(style_counts.items())),
        "context_roles": dict(sorted(role_counts.items())),
        "contrast_labels": dict(sorted(contrast_label_counts.items())),
        "business_id_subtypes": dict(sorted(business_id_subtype_counts.items())),
        "unique_text_count": len(text_counts),
        "unique_source_id_count": len(id_counts),
        "unique_value_hash_count": len(facts.value_hashes),
        "duplicate_text_records": duplicate_text_records,
        "duplicate_source_id_records": duplicate_id_records,
        "error_counts": dict(sorted(code_counts.items())),
        "errors": errors,
    }
    return result, facts


def validate_split_records(
    records: Sequence[Mapping[str, Any]], split: str
) -> Dict[str, Any]:
    """Validate one in-memory new split and return JSON-serializable results."""

    result, _ = _validate_split(records, split)
    return result


def normalize_template_skeleton(template: str) -> str:
    """Normalize explicit generator placeholders, whitespace, and case."""

    normalized = template
    for pattern in _PLACEHOLDER_PATTERNS:
        normalized = pattern.sub("<placeholder>", normalized)
    return " ".join(normalized.casefold().split())


def _literal_token_trigrams(template: str) -> Set[Tuple[str, str, str]]:
    skeleton = normalize_template_skeleton(template).replace("<placeholder>", " ")
    tokens = _TOKEN_RE.findall(skeleton)
    return {
        (tokens[index], tokens[index + 1], tokens[index + 2])
        for index in range(max(0, len(tokens) - 2))
    }


def _trigram_jaccard(left: str, right: str) -> float:
    left_grams = _literal_token_trigrams(left)
    right_grams = _literal_token_trigrams(right)
    union = left_grams | right_grams
    if not union:
        return 1.0 if left_grams == right_grams else 0.0
    return len(left_grams & right_grams) / len(union)


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _field(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _flatten_catalog(
    catalog: Any,
    *,
    inherited_partition: Optional[str] = None,
    inherited_family: Optional[str] = None,
) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if isinstance(catalog, Mapping):
        direct_template = _field(catalog, "template", "text_template")
        if isinstance(direct_template, str):
            family = _field(catalog, "family", "template_family", "family_id")
            partition = _field(catalog, "partition", "intended_split", "split")
            entries.append(
                {
                    "family": str(family or inherited_family or "UNKNOWN"),
                    "partition": str(partition or inherited_partition or "UNKNOWN"),
                    "template": direct_template,
                }
            )
            return entries
        for key, value in catalog.items():
            partition = inherited_partition
            if key in NEW_SPLITS:
                partition = key
            entries.extend(
                _flatten_catalog(
                    value,
                    inherited_partition=partition,
                    inherited_family=inherited_family,
                )
            )
        return entries

    if isinstance(catalog, (list, tuple, set)):
        for item in catalog:
            entries.extend(
                _flatten_catalog(
                    item,
                    inherited_partition=inherited_partition,
                    inherited_family=inherited_family,
                )
            )
        return entries

    template = _field(catalog, "template", "text_template")
    family = _field(catalog, "family", "template_family", "family_id")
    partition = _field(catalog, "partition", "intended_split", "split")
    if isinstance(template, str):
        entries.append(
            {
                "family": str(family or inherited_family or "UNKNOWN"),
                "partition": str(partition or inherited_partition or "UNKNOWN"),
                "template": template,
            }
        )
        return entries

    child_family = str(family or inherited_family or "UNKNOWN")
    for child_name in ("templates", "variants", "contexts", "records", "roles"):
        children = _field(catalog, child_name)
        if children is not None:
            entries.extend(
                _flatten_catalog(
                    children,
                    inherited_partition=str(partition or inherited_partition or "UNKNOWN"),
                    inherited_family=child_family,
                )
            )
    return entries


def _max_similarity(
    left_entries: Sequence[Mapping[str, str]],
    right_entries: Sequence[Mapping[str, str]],
    *,
    exclude_same_object: bool = False,
) -> Dict[str, Any]:
    max_jaccard: Dict[str, Any] = {"score": 0.0, "left_family": None, "right_family": None}
    max_sequence: Dict[str, Any] = {"score": 0.0, "left_family": None, "right_family": None}
    for left_index, left in enumerate(left_entries):
        for right_index, right in enumerate(right_entries):
            if exclude_same_object and left_entries is right_entries and left_index >= right_index:
                continue
            if (
                exclude_same_object
                and left.get("partition") == right.get("partition")
            ):
                continue
            jaccard = _trigram_jaccard(left["template"], right["template"])
            sequence = SequenceMatcher(
                None,
                normalize_template_skeleton(left["template"]),
                normalize_template_skeleton(right["template"]),
                autojunk=False,
            ).ratio()
            if jaccard > max_jaccard["score"]:
                max_jaccard = {
                    "score": round(jaccard, 6),
                    "left_family": left.get("family"),
                    "right_family": right.get("family"),
                }
            if sequence > max_sequence["score"]:
                max_sequence = {
                    "score": round(sequence, 6),
                    "left_family": left.get("family"),
                    "right_family": right.get("family"),
                }
    return {
        "max_literal_token_3gram_jaccard": max_jaccard,
        "max_sequence_matcher_ratio": max_sequence,
    }


def audit_template_catalogs(
    new_generator_path: str | Path,
    parent_generator_path: str | Path,
) -> Dict[str, Any]:
    """Audit normalized generator templates without claiming semantic proof."""

    errors: List[str] = []
    new_entries: List[Dict[str, str]] = []
    parent_entries: List[Dict[str, str]] = []
    try:
        new_module = _load_module(Path(new_generator_path), "_securelogx_v11_generator_audit")
        if not callable(getattr(new_module, "catalog", None)):
            raise AttributeError("new generator must expose catalog()")
        new_entries = _flatten_catalog(new_module.catalog())
        if not new_entries:
            raise ValueError("new catalog did not expose any template strings")
    except (OSError, ImportError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"could not load new generator catalog: {exc}")

    try:
        parent_module = _load_module(
            Path(parent_generator_path), "_securelogx_parent_generator_audit"
        )
        generator_class = getattr(parent_module, "SecureLogXDataGenerator", None)
        if generator_class is None:
            raise AttributeError("parent generator is missing SecureLogXDataGenerator")
        parent_entries = _flatten_catalog(generator_class(seed=42).template_catalog())
        if not parent_entries:
            raise ValueError("parent catalog did not expose any template strings")
    except (OSError, ImportError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"could not load parent generator catalog: {exc}")

    skeleton_locations: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for entry in new_entries:
        skeleton_locations[normalize_template_skeleton(entry["template"])].append(
            (entry["partition"], entry["family"])
        )
    cross_split_overlaps: List[Dict[str, Any]] = []
    for locations in skeleton_locations.values():
        partitions = {partition for partition, _ in locations}
        if len(partitions) > 1:
            cross_split_overlaps.append(
                {
                    "partitions": sorted(partitions),
                    "families": sorted({family for _, family in locations}),
                }
            )

    parent_skeletons: Dict[str, Set[str]] = defaultdict(set)
    for entry in parent_entries:
        parent_skeletons[normalize_template_skeleton(entry["template"])].add(
            entry["family"]
        )
    parent_overlaps: List[Dict[str, Any]] = []
    for entry in new_entries:
        parent_families = parent_skeletons.get(
            normalize_template_skeleton(entry["template"]), set()
        )
        if parent_families:
            parent_overlaps.append(
                {
                    "new_family": entry["family"],
                    "parent_families": sorted(parent_families),
                }
            )

    forbidden_entries = [
        entry["family"]
        for entry in new_entries
        if entry["family"] == FORBIDDEN_PARENT_FAMILY
        or all(
            part in normalize_template_skeleton(entry["template"])
            for part in FORBIDDEN_PARENT_WORDING_PARTS
        )
    ]

    readability = not errors
    gates = {
        "catalogs_readable": _gate(readability, len(errors), 0, errors),
        "cross_split_normalized_skeleton_overlap": _gate(
            readability and not cross_split_overlaps,
            len(cross_split_overlaps),
            0,
            cross_split_overlaps,
        ),
        "parent_normalized_skeleton_overlap": _gate(
            readability and not parent_overlaps,
            len(parent_overlaps),
            0,
            parent_overlaps,
        ),
        "forbidden_parent_template": _gate(
            readability and not forbidden_entries,
            len(forbidden_entries),
            0,
            sorted(set(forbidden_entries)),
        ),
    }
    return {
        "passed": all(gate["passed"] for gate in gates.values()),
        "new_template_count": len(new_entries),
        "parent_template_count": len(parent_entries),
        "gates": gates,
        "cross_split_similarity": _max_similarity(
            new_entries, new_entries, exclude_same_object=True
        ),
        "parent_similarity": _max_similarity(new_entries, parent_entries),
        "interpretation": (
            "Similarity ratios are descriptive lexical diagnostics only; they do "
            "not establish semantic independence."
        ),
        "errors": errors,
    }


def _resolve_paths(
    repo_root: Path, overrides: Optional[Mapping[str, str | Path]]
) -> Dict[str, Path]:
    unknown = set(overrides or {}) - set(DEFAULT_RELATIVE_PATHS)
    if unknown:
        raise ValueError(f"unknown path override keys: {', '.join(sorted(unknown))}")
    resolved: Dict[str, Path] = {}
    for key, relative in DEFAULT_RELATIVE_PATHS.items():
        value = (overrides or {}).get(key, relative)
        path = Path(value)
        resolved[key] = path if path.is_absolute() else repo_root / path
    return resolved


def _collect_parent_facts(
    parent_records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    texts: Set[str] = set()
    source_ids: Set[str] = set()
    families: Set[str] = set()
    counts: Dict[str, Dict[str, Any]] = {}
    labels_by_split: Dict[str, Counter[str]] = {}
    for split, records in parent_records.items():
        split_texts: Set[str] = set()
        split_ids: Set[str] = set()
        split_families: Set[str] = set()
        split_labels: Counter[str] = Counter()
        for record in records:
            text = record.get("text") if isinstance(record, Mapping) else None
            meta = record.get("meta") if isinstance(record, Mapping) else None
            if isinstance(text, str):
                texts.add(text)
                split_texts.add(text)
            if isinstance(meta, Mapping):
                source_id = meta.get("source_record_id")
                family = meta.get("template_family")
                if isinstance(source_id, str):
                    source_ids.add(source_id)
                    split_ids.add(source_id)
                if isinstance(family, str):
                    families.add(family)
                    split_families.add(family)
            entities = record.get("entities") if isinstance(record, Mapping) else None
            if isinstance(entities, list):
                for entity in entities:
                    if (
                        isinstance(entity, (list, tuple))
                        and len(entity) == 3
                        and entity[2] in CANONICAL_ENTITIES
                    ):
                        split_labels[str(entity[2])] += 1
        labels_by_split[split] = split_labels
        counts[split] = {
            "records": len(records),
            "unique_texts": len(split_texts),
            "unique_source_ids": len(split_ids),
            "template_families": len(split_families),
            "label_counts": dict(sorted(split_labels.items())),
        }
    return {
        "texts": texts,
        "source_ids": source_ids,
        "families": families,
        "counts": counts,
        "labels_by_split": labels_by_split,
    }


def _pairwise_intersections(
    facts: Mapping[str, _SplitFacts], attribute: str
) -> Dict[str, Any]:
    pair_counts: Dict[str, int] = {}
    total = 0
    for left_index, left in enumerate(NEW_SPLITS):
        left_values = set(getattr(facts[left], attribute))
        for right in NEW_SPLITS[left_index + 1 :]:
            overlap = left_values & set(getattr(facts[right], attribute))
            pair_counts[f"{left}__{right}"] = len(overlap)
            total += len(overlap)
    return {"total": total, "pairs": pair_counts}


def validate_dataset(
    repo_root: str | Path = REPO_ROOT,
    paths: Optional[Mapping[str, str | Path]] = None,
    *,
    include_template_audit: bool = True,
) -> Dict[str, Any]:
    """Validate all v1.1 data gates and return a JSON-serializable result.

    ``paths`` is an optional mapping using keys from
    :data:`DEFAULT_RELATIVE_PATHS`; relative overrides resolve beneath
    ``repo_root``.  The function is read-only.
    """

    root = Path(repo_root).resolve()
    resolved = _resolve_paths(root, paths)
    top_errors: List[Dict[str, Any]] = []
    ontology = validate_ontology(resolved["labels"])

    loaded: Dict[str, List[Dict[str, Any]]] = {}
    file_stats: Dict[str, Dict[str, Any]] = {}
    for key in (*NEW_SPLITS, *PARENT_SPLITS):
        path = resolved[key]
        try:
            records = load_jsonl(path)
            loaded[key] = records
            file_stats[key] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "records": len(records),
                "readable": True,
            }
        except (OSError, ValueError) as exc:
            loaded[key] = []
            file_stats[key] = {
                "path": str(path),
                "sha256": None,
                "records": 0,
                "readable": False,
                "error": str(exc),
            }
            _error(top_errors, key, "input_read", f"could not read JSONL: {exc}")

    split_results: Dict[str, Dict[str, Any]] = {}
    split_facts: Dict[str, _SplitFacts] = {}
    for split in NEW_SPLITS:
        split_result, facts = _validate_split(loaded[split], split)
        split_results[split] = split_result
        split_facts[split] = facts

    parent_facts = _collect_parent_facts(
        {split: loaded[split] for split in PARENT_SPLITS}
    )

    text_leakage = _pairwise_intersections(split_facts, "texts")
    id_leakage = _pairwise_intersections(split_facts, "source_ids")
    family_leakage = _pairwise_intersections(split_facts, "families")
    value_hash_leakage = _pairwise_intersections(split_facts, "value_hashes")

    all_new_texts = set().union(*(set(facts.texts) for facts in split_facts.values()))
    all_new_ids = set().union(*(set(facts.source_ids) for facts in split_facts.values()))
    all_new_families = set().union(
        *(set(facts.families) for facts in split_facts.values())
    )
    parent_text_overlap = all_new_texts & parent_facts["texts"]
    parent_id_overlap = all_new_ids & parent_facts["source_ids"]
    parent_family_overlap = all_new_families & parent_facts["families"]

    all_errors = top_errors + [
        error
        for split in NEW_SPLITS
        for error in split_results[split]["errors"]
    ]
    code_counts = Counter(error["code"] for error in all_errors)

    dev_family_counts = split_results["dev_challenge"]["category_family_counts"]
    dev_bid_ssn = int(dev_family_counts.get("business_id_vs_ssn", 0))
    dev_card_account = int(
        dev_family_counts.get("business_id_vs_credit_card", 0)
        + dev_family_counts.get("business_id_vs_bank_account", 0)
    )
    dev_network = int(
        dev_family_counts.get("ip_address_vs_technical_reference", 0)
    )

    sealed = split_results["sealed_challenge"]
    sealed_categories = set(sealed["category_family_counts"])
    sealed_formats = set(sealed["formats"])
    train_styles = set(split_results["train_addition"]["surface_styles"])
    train_labels = split_results["train_addition"]["label_counts"]
    parent_train_labels: Counter[str] = parent_facts["labels_by_split"].get(
        "parent_train", Counter()
    )
    business_id_minimum = math.ceil(
        parent_train_labels["BUSINESS_ID"] * MIN_PARENT_BASELINE_FRACTION
    )
    ssn_minimum = math.ceil(parent_train_labels["SSN"] * MIN_PARENT_BASELINE_FRACTION)
    train_subtypes = split_results["train_addition"]["business_id_subtypes"]
    subtype_shortfalls = {
        subtype: int(train_subtypes.get(subtype, 0))
        for subtype in BUSINESS_ID_SUBTYPES
        if int(train_subtypes.get(subtype, 0)) < MIN_TRAIN_BUSINESS_SUBTYPE_RECORDS
    }

    gates: Dict[str, Dict[str, Any]] = {
        "inputs_readable": _gate(
            not top_errors,
            sum(not stats["readable"] for stats in file_stats.values()),
            0,
        ),
        "ontology_exact_25_entities_51_bio": _gate(
            ontology["passed"],
            {
                "entities": ontology["entity_count"],
                "bio_labels": ontology["bio_label_count"],
            },
            {"entities": 25, "bio_labels": 51},
            ontology["errors"],
        ),
        "canonical_spans": _gate(
            code_counts["span"] == 0,
            code_counts["span"],
            0,
        ),
        "entity_provenance_bijection": _gate(
            code_counts["provenance"] == 0,
            code_counts["provenance"],
            0,
        ),
        "v1_1_metadata": _gate(
            code_counts["metadata"] == 0,
            code_counts["metadata"],
            0,
        ),
        "intended_split": _gate(
            code_counts["intended_split"] == 0,
            code_counts["intended_split"],
            0,
        ),
        "contrast_targets": _gate(
            code_counts["contrast_target"] == 0,
            code_counts["contrast_target"],
            0,
        ),
        "o_target_non_overlap": _gate(
            code_counts["o_target_overlap"] == 0,
            code_counts["o_target_overlap"],
            0,
        ),
        "complete_paired_contrast_groups": _gate(
            code_counts["pair_group"] == 0,
            code_counts["pair_group"],
            0,
        ),
        "family_metadata_consistency": _gate(
            code_counts["family_consistency"] == 0,
            code_counts["family_consistency"],
            0,
        ),
        "unique_record_texts": _gate(
            code_counts["duplicate_text"] == 0 and text_leakage["total"] == 0,
            code_counts["duplicate_text"] + text_leakage["total"],
            0,
            text_leakage["pairs"],
        ),
        "unique_source_record_ids": _gate(
            code_counts["duplicate_source_id"] == 0 and id_leakage["total"] == 0,
            code_counts["duplicate_source_id"] + id_leakage["total"],
            0,
            id_leakage["pairs"],
        ),
        "cross_split_exact_text_leakage": _gate(
            text_leakage["total"] == 0,
            text_leakage["total"],
            0,
            text_leakage["pairs"],
        ),
        "cross_split_source_id_leakage": _gate(
            id_leakage["total"] == 0,
            id_leakage["total"],
            0,
            id_leakage["pairs"],
        ),
        "cross_split_template_family_leakage": _gate(
            family_leakage["total"] == 0,
            family_leakage["total"],
            0,
            family_leakage["pairs"],
        ),
        "cross_split_value_hash_leakage": _gate(
            value_hash_leakage["total"] == 0,
            value_hash_leakage["total"],
            0,
            value_hash_leakage["pairs"],
        ),
        "parent_exact_text_contamination": _gate(
            not parent_text_overlap, len(parent_text_overlap), 0
        ),
        "parent_source_id_contamination": _gate(
            not parent_id_overlap, len(parent_id_overlap), 0
        ),
        "parent_template_family_contamination": _gate(
            not parent_family_overlap, len(parent_family_overlap), 0
        ),
        "forbidden_parent_hard_negative": _gate(
            code_counts["forbidden_parent_template"] == 0,
            code_counts["forbidden_parent_template"],
            0,
        ),
        "dev_business_id_ssn_family_minimum": _gate(dev_bid_ssn >= 5, dev_bid_ssn, ">=5"),
        "dev_card_account_family_minimum": _gate(
            dev_card_account >= 3, dev_card_account, ">=3"
        ),
        "dev_network_family_minimum": _gate(dev_network >= 2, dev_network, ">=2"),
        "sealed_family_count_8_to_12": _gate(
            8 <= sealed["family_count"] <= 12,
            sealed["family_count"],
            "8..12",
        ),
        "sealed_required_categories": _gate(
            CORE_CHALLENGE_CATEGORIES <= sealed_categories,
            sorted(sealed_categories),
            sorted(CORE_CHALLENGE_CATEGORIES),
            {"missing": sorted(CORE_CHALLENGE_CATEGORIES - sealed_categories)},
        ),
        "sealed_required_formats": _gate(
            REQUIRED_SEALED_FORMATS <= sealed_formats,
            sorted(sealed_formats),
            sorted(REQUIRED_SEALED_FORMATS),
            {"missing": sorted(REQUIRED_SEALED_FORMATS - sealed_formats)},
        ),
        "train_surface_style_diversity": _gate(
            len(train_styles) >= 8,
            len(train_styles),
            ">=8",
            {"styles": sorted(train_styles)},
        ),
        "train_business_id_subtype_coverage": _gate(
            not subtype_shortfalls,
            {
                subtype: int(train_subtypes.get(subtype, 0))
                for subtype in BUSINESS_ID_SUBTYPES
            },
            f">={MIN_TRAIN_BUSINESS_SUBTYPE_RECORDS} each",
            {"below_minimum": subtype_shortfalls},
        ),
        "train_business_id_baseline_fraction": _gate(
            int(train_labels.get("BUSINESS_ID", 0)) >= business_id_minimum,
            int(train_labels.get("BUSINESS_ID", 0)),
            {
                "minimum": business_id_minimum,
                "fraction": MIN_PARENT_BASELINE_FRACTION,
                "parent_train_baseline": parent_train_labels["BUSINESS_ID"],
            },
        ),
        "train_ssn_baseline_fraction": _gate(
            int(train_labels.get("SSN", 0)) >= ssn_minimum,
            int(train_labels.get("SSN", 0)),
            {
                "minimum": ssn_minimum,
                "fraction": MIN_PARENT_BASELINE_FRACTION,
                "parent_train_baseline": parent_train_labels["SSN"],
            },
        ),
        "record_shape": _gate(
            code_counts["record_shape"] == 0,
            code_counts["record_shape"],
            0,
        ),
    }

    for split, expected_counts in EXPECTED_SPLIT_COUNTS.items():
        split_result = split_results[split]
        gates[f"{split}_record_count"] = _gate(
            split_result["record_count"] == expected_counts["records"],
            split_result["record_count"],
            expected_counts["records"],
        )
        gates[f"{split}_family_count"] = _gate(
            split_result["family_count"] == expected_counts["families"],
            split_result["family_count"],
            expected_counts["families"],
        )

    template_audit: Optional[Dict[str, Any]] = None
    if include_template_audit:
        template_audit = audit_template_catalogs(
            resolved["new_generator"], resolved["parent_generator"]
        )
        for name, gate in template_audit["gates"].items():
            gates[f"template_catalog_{name}"] = gate

    passed = all(gate["passed"] for gate in gates.values())
    result: Dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": "SecureLogX ML-v1.1",
        "passed": passed,
        "ready": passed,
        "recommendation": (
            "READY FOR ML-v1.1 RETRAINING"
            if passed
            else "NOT READY FOR ML-v1.1 RETRAINING"
        ),
        "repo_root": str(root),
        "files": file_stats,
        "ontology": ontology,
        "splits": split_results,
        "parents": parent_facts["counts"],
        "leakage": {
            "cross_split_exact_text": text_leakage,
            "cross_split_source_id": id_leakage,
            "cross_split_template_family": family_leakage,
            "cross_split_value_hash": value_hash_leakage,
            "parent_exact_text_count": len(parent_text_overlap),
            "parent_source_id_count": len(parent_id_overlap),
            "parent_template_family_count": len(parent_family_overlap),
        },
        "coverage": {
            "dev_challenge": {
                "business_id_vs_ssn_families": dev_bid_ssn,
                "business_id_vs_card_or_account_families": dev_card_account,
                "network_conflict_families": dev_network,
            },
            "sealed_challenge": {
                "families": sealed["family_count"],
                "categories": sorted(sealed_categories),
                "formats": sorted(sealed_formats),
            },
            "train_surface_styles": sorted(train_styles),
            "train_business_id_subtypes": {
                subtype: int(train_subtypes.get(subtype, 0))
                for subtype in BUSINESS_ID_SUBTYPES
            },
            "train_parent_baseline_fraction": {
                "fraction": MIN_PARENT_BASELINE_FRACTION,
                "business_id": {
                    "actual": int(train_labels.get("BUSINESS_ID", 0)),
                    "minimum": business_id_minimum,
                    "parent_train": parent_train_labels["BUSINESS_ID"],
                },
                "ssn": {
                    "actual": int(train_labels.get("SSN", 0)),
                    "minimum": ssn_minimum,
                    "parent_train": parent_train_labels["SSN"],
                },
            },
        },
        "gates": gates,
        "error_count": len(all_errors),
        "errors": all_errors,
    }
    if template_audit is not None:
        result["template_audit"] = template_audit
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate SecureLogX ML-v1.1 context-contrast JSONL data"
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--labels")
    parser.add_argument("--train-additions")
    parser.add_argument("--dev-challenge")
    parser.add_argument("--sealed-challenge")
    parser.add_argument("--parent-train")
    parser.add_argument("--parent-dev")
    parser.add_argument("--parent-test")
    parser.add_argument("--new-generator")
    parser.add_argument("--parent-generator")
    parser.add_argument(
        "--skip-template-audit",
        action="store_true",
        help="Skip catalog imports (data gates still run); intended only for isolated tests",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    option_to_key = {
        "labels": "labels",
        "train_additions": "train_addition",
        "dev_challenge": "dev_challenge",
        "sealed_challenge": "sealed_challenge",
        "parent_train": "parent_train",
        "parent_dev": "parent_dev",
        "parent_test": "parent_test",
        "new_generator": "new_generator",
        "parent_generator": "parent_generator",
    }
    overrides = {
        key: getattr(args, option)
        for option, key in option_to_key.items()
        if getattr(args, option) is not None
    }
    try:
        result = validate_dataset(
            args.repo_root,
            overrides,
            include_template_audit=not args.skip_template_audit,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}), file=sys.stderr)
        return 2

    if args.compact:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
