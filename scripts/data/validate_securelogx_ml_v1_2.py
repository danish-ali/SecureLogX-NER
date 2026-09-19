#!/usr/bin/env python3
"""Read-only validation for the SecureLogX ML-v1.2 counterbalance data.

The validator is independent from data generation.  It validates the two new
v1.2 partitions as one dataset and checks them against every frozen parent
surface: the ML-v1 train/dev/test splits, the ML-v1.1 training addition, the
ML-v1.1 dev challenge, and the sealed final challenge.

The sealed challenge file is NEVER opened.  ML-v1.1 partitions are compared
through deterministic in-memory regeneration by the frozen v1.1 generator
(seed 42); the canonical serialization of every regenerated partition must
reproduce the byte SHA-256 recorded in ``configs/ml_v1_2_parent_freeze.json``
before any comparison is trusted.  When a v1.1 file is readable its on-disk
hash is used instead; the two paths are equivalent because the hashes match.

Public entry points: :func:`validate_ontology` (re-exported),
:func:`validate_split_records`, :func:`audit_template_catalogs`, and
:func:`validate_dataset`.  The module never writes datasets, reports, or
manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import validate_securelogx_ml_v1_1 as v1_1_validator  # noqa: E402
from validate_securelogx_ml_v1_1 import (  # noqa: E402
    CANONICAL_ENTITIES,
    load_jsonl,
    normalize_template_skeleton,
    validate_ontology,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

NEW_SPLITS: Tuple[str, ...] = ("train_addition", "dev_challenge")
PARENT_SPLITS: Tuple[str, ...] = ("parent_train", "parent_dev", "parent_test")
V1_1_SPLITS: Tuple[str, ...] = (
    "v1_1_train_addition",
    "v1_1_dev_challenge",
    "v1_1_sealed_challenge",
)
# The sealed challenge is compared only through hash-verified regeneration.
V1_1_FILE_FORBIDDEN: Set[str] = {"v1_1_sealed_challenge"}

DEFAULT_RELATIVE_PATHS: Mapping[str, str] = {
    "labels": "configs/securelogx_labels.json",
    "parent_freeze": "configs/ml_v1_2_parent_freeze.json",
    "train_addition": "data/ml_v1_2/counterbalance/train_additions.jsonl",
    "dev_challenge": "data/ml_v1_2/counterbalance/dev_challenge.jsonl",
    "parent_train": "data/split/train.jsonl",
    "parent_dev": "data/split/dev.jsonl",
    "parent_test": "data/split/test.jsonl",
    "new_generator": "scripts/data/generate_securelogx_ml_v1_2_counterbalance.py",
    "v1_1_generator": "scripts/data/generate_securelogx_ml_v1_1_context_contrast.py",
    "parent_generator": "scripts/data/generate_securelogx_synthetic_logs.py",
}

V1_1_PARTITION_PATHS: Mapping[str, str] = {
    "v1_1_train_addition": "data/ml_v1_1/context_contrast/train_additions.jsonl",
    "v1_1_dev_challenge": "data/ml_v1_1/context_contrast/dev_challenge.jsonl",
    "v1_1_sealed_challenge": "data/ml_v1_1/challenge/sealed_test.jsonl",
}
V1_1_GENERATOR_PARTITIONS: Mapping[str, str] = {
    "v1_1_train_addition": "train_addition",
    "v1_1_dev_challenge": "dev_challenge",
    "v1_1_sealed_challenge": "sealed_challenge",
}

EXPECTED_SOURCE = "securelogx_counterbalance_v1_2"
EXPECTED_REVIEW_STATUS = "synthetic_generated"
EXPECTED_SCENARIO_KIND = "counterbalance_context_contrast"
EXPECTED_GENERATOR_VERSION = "securelogx-ml-v1.2-counterbalance-v1"
EXPECTED_OBJECTIVE = "identifier_shape_does_not_override_semantic_context"
EXPECTED_INDEPENDENCE_SCOPE = "new_ml_v1_2_family_and_value_namespace"

ALLOWED_FORMATS: Set[str] = {"key_value", "json", "text", "csv"}
ALLOWED_SURFACE_STYLES: Set[str] = {
    "countercheck_kv",
    "countercheck_json",
    "countercheck_nested_json",
    "countercheck_audit",
    "countercheck_syslog",
    "countercheck_bracketed",
    "countercheck_query",
    "countercheck_csv",
}

_CONTRAST_SENSITIVE_LABELS: Mapping[str, str] = {
    "business_id_vs_ssn": "SSN",
    "business_id_vs_credit_card": "CREDIT_CARD_NUMBER",
    "business_id_vs_bank_account": "BANK_ACCOUNT_NUMBER",
    "business_id_vs_routing_number": "ROUTING_NUMBER",
    "business_id_vs_itin": "ITIN",
    "business_id_vs_tax_id": "TAX_ID",
    "business_id_vs_passport": "PASSPORT_NUMBER",
    "business_id_vs_driver_license": "DRIVER_LICENSE",
    "business_id_vs_api_key": "API_KEY",
    "business_id_vs_auth_token": "AUTH_TOKEN",
    "business_id_vs_ip_address": "IP_ADDRESS",
}

CATEGORY_SPECS: Mapping[str, Mapping[str, Set[str]]] = {
    **{
        category: {
            "roles": {"business_identifier", f"canonical_{label.casefold()}"},
            "labels": {"BUSINESS_ID", label},
        }
        for category, label in _CONTRAST_SENSITIVE_LABELS.items()
    },
    "ip_address_vs_technical_reference": {
        "roles": {"network_address", "technical_reference"},
        "labels": {"IP_ADDRESS", "O"},
    },
}

EXPECTED_SPLIT_COUNTS: Mapping[str, Mapping[str, int]] = {
    "train_addition": {"records": 1600, "families": 40, "records_per_family": 40},
    "dev_challenge": {"records": 540, "families": 15, "records_per_family": 36},
}
RECORD_COUNT_RANGES: Mapping[str, Tuple[int, int]] = {
    "train_addition": (1200, 2000),
    "dev_challenge": (400, 600),
}
EXPECTED_CATEGORY_FAMILY_COUNTS: Mapping[str, Mapping[str, int]] = {
    "train_addition": {
        "business_id_vs_ssn": 10,
        "business_id_vs_credit_card": 4,
        "business_id_vs_bank_account": 3,
        "business_id_vs_routing_number": 3,
        "business_id_vs_itin": 2,
        "business_id_vs_tax_id": 2,
        "business_id_vs_passport": 2,
        "business_id_vs_driver_license": 2,
        "business_id_vs_api_key": 2,
        "business_id_vs_auth_token": 2,
        "business_id_vs_ip_address": 2,
        "ip_address_vs_technical_reference": 6,
    },
    "dev_challenge": {
        "business_id_vs_ssn": 2,
        "business_id_vs_credit_card": 1,
        "business_id_vs_bank_account": 1,
        "business_id_vs_routing_number": 1,
        "business_id_vs_itin": 1,
        "business_id_vs_tax_id": 1,
        "business_id_vs_passport": 1,
        "business_id_vs_driver_license": 1,
        "business_id_vs_api_key": 1,
        "business_id_vs_auth_token": 1,
        "business_id_vs_ip_address": 1,
        "ip_address_vs_technical_reference": 3,
    },
}

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

MIN_TRAIN_IP_TECHNICAL_FAMILIES = 6
MIN_DEV_IP_TECHNICAL_FAMILIES = 3
MIN_TRAIN_CUSTOMER_ID_FAMILIES = 4
MIN_TRAIN_REQUEST_ID_FAMILIES = 4
MIN_MULTI_ENTITY_RATIO = 0.7
MIN_TRAIN_SSN_SPANS = 300
MIN_TRAIN_SENSITIVE_SPANS_EACH = 40

FORBIDDEN_PARENT_FAMILY = v1_1_validator.FORBIDDEN_PARENT_FAMILY
FORBIDDEN_PARENT_WORDING_PARTS = v1_1_validator.FORBIDDEN_PARENT_WORDING_PARTS

_IDENTIFIER_RE = v1_1_validator._IDENTIFIER_RE
_SHA256_RE = v1_1_validator._SHA256_RE

_strict_int = v1_1_validator._strict_int
_nonempty_string = v1_1_validator._nonempty_string
_sha256_text = v1_1_validator._sha256_text
_sha256_file = v1_1_validator._sha256_file
_error = v1_1_validator._error
_gate = v1_1_validator._gate


def canonical_jsonl_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash records exactly as the frozen generators serialize them."""

    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


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


@dataclass
class _SplitFacts:
    texts: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    families: Set[str] = field(default_factory=set)
    focus_value_hashes: Set[str] = field(default_factory=set)
    groups: Dict[str, List[_GroupItem]] = field(
        default_factory=lambda: defaultdict(list)
    )
    family_items: Dict[str, List[_GroupItem]] = field(
        default_factory=lambda: defaultdict(list)
    )
    family_records: Counter = field(default_factory=Counter)
    family_subtypes: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    multi_entity_records: int = 0


def _validate_target(
    errors: List[Dict[str, Any]],
    split: str,
    line_number: int,
    text: str,
    entities: Sequence[Tuple[int, int, str]],
    target: Mapping[str, Any],
    target_role: str,
    category: str,
) -> Optional[Dict[str, Any]]:
    """Validate one contrast target; return canonical facts or ``None``."""

    start = target.get("start")
    end = target.get("end")
    expected_label = target.get("expected_label")
    value_hash = target.get("value_sha256")
    valid = True
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
            f"{target_role} target has invalid offsets",
            line_number,
        )
        return None
    if expected_label not in set(CANONICAL_ENTITIES) | {"O"}:
        _error(
            errors,
            split,
            "contrast_target",
            f"{target_role} target has a non-canonical expected_label",
            line_number,
        )
        valid = False
    if not isinstance(value_hash, str) or not _SHA256_RE.fullmatch(value_hash):
        _error(
            errors,
            split,
            "contrast_target",
            f"{target_role} target value_sha256 must be lowercase SHA-256",
            line_number,
        )
        valid = False
    if not _nonempty_string(target.get("source_label")):
        _error(
            errors,
            split,
            "contrast_target",
            f"{target_role} target source_label must be a non-empty string",
            line_number,
        )
        valid = False
    if not valid:
        return None

    value = text[start:end]
    if _sha256_text(value) != value_hash:
        _error(
            errors,
            split,
            "contrast_target",
            f"{target_role} target value_sha256 does not match text[start:end]",
            line_number,
        )
        return None
    if expected_label == "O":
        if any(
            start < entity_end and end > entity_start
            for entity_start, entity_end, _ in entities
        ):
            _error(
                errors,
                split,
                "o_target_overlap",
                "an O contrast target must not overlap any canonical entity",
                line_number,
            )
            return None
    elif (start, end, expected_label) not in set(entities):
        _error(
            errors,
            split,
            "contrast_target",
            f"{target_role} target must exactly match a canonical entity",
            line_number,
        )
        return None

    if target_role == "focus" and category in CATEGORY_SPECS:
        if expected_label not in CATEGORY_SPECS[category]["labels"]:
            _error(
                errors,
                split,
                "contrast_target",
                "focus expected_label is inconsistent with contrast_category",
                line_number,
            )
            return None
    if target_role == "companion" and expected_label == "O":
        _error(
            errors,
            split,
            "contrast_target",
            "companion target must be a canonical entity, not O",
            line_number,
        )
        return None

    if expected_label == "BUSINESS_ID":
        subtype = target.get("business_id_subtype")
        if subtype not in BUSINESS_ID_SUBTYPES or target.get("source_label") != subtype:
            _error(
                errors,
                split,
                "contrast_target",
                f"{target_role} BUSINESS_ID target has an inconsistent subtype",
                line_number,
            )
            return None

    return {
        "value": value,
        "value_hash": value_hash,
        "expected_label": str(expected_label),
        "business_id_subtype": target.get("business_id_subtype"),
    }


def _validate_split(
    records: Sequence[Mapping[str, Any]], split: str
) -> Tuple[Dict[str, Any], _SplitFacts]:
    if split not in NEW_SPLITS:
        raise ValueError(f"unsupported ML-v1.2 split: {split!r}")

    errors: List[Dict[str, Any]] = []
    facts = _SplitFacts()
    label_counts: Counter = Counter()
    category_counts: Counter = Counter()
    format_counts: Counter = Counter()
    style_counts: Counter = Counter()
    role_counts: Counter = Counter()
    focus_label_counts: Counter = Counter()
    business_id_subtype_counts: Counter = Counter()
    invalid_records = 0
    signature_prefix = f"cb12-{split.replace('_', '-')}-"

    for line_number, record in enumerate(records, 1):
        before = len(errors)
        if not isinstance(record, Mapping):
            _error(errors, split, "record_shape", "record must be an object", line_number)
            invalid_records += 1
            continue

        text = record.get("text")
        if not _nonempty_string(text):
            _error(
                errors, split, "record_shape", "text must be a non-empty string", line_number
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
                        errors, split, "span", "entity spans must not overlap", line_number
                    )

        meta = record.get("meta")
        if not isinstance(meta, Mapping):
            _error(errors, split, "metadata", "meta must be an object", line_number)
            meta = {}

        for key, expected in (
            ("source", EXPECTED_SOURCE),
            ("review_status", EXPECTED_REVIEW_STATUS),
            ("license_reviewed", True),
            ("scenario_kind", EXPECTED_SCENARIO_KIND),
            ("generator_version", EXPECTED_GENERATOR_VERSION),
            ("intended_split", split),
            ("counterbalance_objective", EXPECTED_OBJECTIVE),
            ("independence_scope", EXPECTED_INDEPENDENCE_SCOPE),
        ):
            if meta.get(key) != expected:
                code = "intended_split" if key == "intended_split" else "metadata"
                _error(
                    errors, split, code, f"meta.{key} must equal {expected!r}", line_number
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
            if not _IDENTIFIER_RE.fullmatch(source_id) or not source_id.startswith(
                f"ml-v1.2-{split}-"
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
            facts.family_records[family] += 1
            if not (
                family.startswith(f"ml_v1_2_{split}_") and family.endswith("_v1_2")
            ):
                _error(
                    errors,
                    split,
                    "metadata",
                    "meta.template_family has the wrong split namespace/version",
                    line_number,
                )
            if family == FORBIDDEN_PARENT_FAMILY or "ml_v1_1" in family:
                _error(
                    errors,
                    split,
                    "forbidden_parent_template",
                    "template_family reuses a forbidden parent namespace",
                    line_number,
                )

        semantic_signature = meta.get("semantic_signature")
        if not _nonempty_string(semantic_signature) or not str(
            semantic_signature
        ).startswith(signature_prefix):
            _error(
                errors,
                split,
                "metadata",
                "meta.semantic_signature has the wrong v1.2 namespace",
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
        elif not group_id.startswith(f"ml-v1.2-{split}-"):
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
                "meta.format is not an approved ML-v1.2 format",
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
                "meta.surface_style is not in the approved v1.2 vocabulary",
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

        expected_worst_flag = category == "ip_address_vs_technical_reference"
        if meta.get("actual_worst_category_target") is not expected_worst_flag:
            _error(
                errors,
                split,
                "metadata",
                "meta.actual_worst_category_target must match the contrast_category",
                line_number,
            )
        if meta.get("multi_entity") is not (len(entities) > 1):
            _error(
                errors,
                split,
                "metadata",
                "meta.multi_entity must match the canonical entity count",
                line_number,
            )
        if len(entities) > 1:
            facts.multi_entity_records += 1
        if not _nonempty_string(meta.get("annotation_rationale")):
            _error(
                errors,
                split,
                "metadata",
                "meta.annotation_rationale must be a non-empty string",
                line_number,
            )

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
                source_label = entry.get("source_label")
                if not _nonempty_string(source_label):
                    _error(
                        errors,
                        split,
                        "provenance",
                        f"provenance {provenance_index} needs source_label",
                        line_number,
                    )
                expected_action = "normalized" if label == "BUSINESS_ID" else "kept"
                if entry.get("action") != expected_action:
                    _error(
                        errors,
                        split,
                        "provenance",
                        f"provenance {provenance_index}.action must be {expected_action!r}",
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
                if label == "BUSINESS_ID":
                    if source_label not in BUSINESS_ID_SUBTYPES:
                        _error(
                            errors,
                            split,
                            "provenance",
                            f"provenance {provenance_index} has an unsupported BUSINESS_ID subtype",
                            line_number,
                        )
                    elif entry.get("business_id_subtype") != source_label:
                        _error(
                            errors,
                            split,
                            "provenance",
                            f"provenance {provenance_index}.business_id_subtype must match source_label",
                            line_number,
                        )
                    else:
                        business_id_subtype_counts[str(source_label)] += 1
                        if family:
                            facts.family_subtypes[family].add(str(source_label))

        if Counter(provenance_signatures) != Counter(entities):
            _error(
                errors,
                split,
                "provenance",
                "entities and entity_provenance must have a one-to-one canonical mapping",
                line_number,
            )

        targets = meta.get("contrast_targets")
        focus_fact: Optional[Dict[str, Any]] = None
        if not isinstance(targets, list) or len(targets) != 2:
            _error(
                errors,
                split,
                "contrast_target",
                "meta.contrast_targets must contain exactly two targets",
                line_number,
            )
        else:
            by_role: Dict[str, Mapping[str, Any]] = {}
            for target in targets:
                if not isinstance(target, Mapping):
                    _error(
                        errors,
                        split,
                        "contrast_target",
                        "each contrast target must be an object",
                        line_number,
                    )
                    continue
                by_role[str(target.get("target_role"))] = target
            if set(by_role) != {"focus", "companion"}:
                _error(
                    errors,
                    split,
                    "contrast_target",
                    "contrast_targets must contain one focus and one companion",
                    line_number,
                )
            else:
                focus_fact = _validate_target(
                    errors,
                    split,
                    line_number,
                    text,
                    entities,
                    by_role["focus"],
                    "focus",
                    category,
                )
                companion_fact = _validate_target(
                    errors,
                    split,
                    line_number,
                    text,
                    entities,
                    by_role["companion"],
                    "companion",
                    category,
                )
                if focus_fact is not None:
                    focus_label_counts[focus_fact["expected_label"]] += 1
                    facts.focus_value_hashes.add(focus_fact["value_hash"])
                    if meta.get("primary_entity") != focus_fact["expected_label"]:
                        _error(
                            errors,
                            split,
                            "metadata",
                            "meta.primary_entity must equal the focus expected_label",
                            line_number,
                        )
                if (
                    focus_fact is not None
                    and companion_fact is not None
                    and all(
                        (family, group_id, category, morphology, role)
                    )
                ):
                    item = _GroupItem(
                        line=line_number,
                        family=family,
                        category=category,
                        morphology=morphology,
                        role=role,
                        expected_label=focus_fact["expected_label"],
                        value_hash=focus_fact["value_hash"],
                        value=focus_fact["value"],
                    )
                    facts.groups[group_id].append(item)
                    facts.family_items[family].append(item)

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
    duplicate_text_records = sum(
        count - 1 for count in text_counts.values() if count > 1
    )
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
            ("focus value", {item.value for item in items}),
            ("focus value_sha256", {item.value_hash for item in items}),
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
    expected_per_family = EXPECTED_SPLIT_COUNTS[split]["records_per_family"]
    for family_name, count in sorted(facts.family_records.items()):
        if count != expected_per_family:
            _error(
                errors,
                split,
                "family_consistency",
                f"template family {family_name!r} has {count} records; "
                f"expected {expected_per_family}",
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
        "focus_labels": dict(sorted(focus_label_counts.items())),
        "business_id_subtypes": dict(sorted(business_id_subtype_counts.items())),
        "family_subtypes": {
            family_name: sorted(subtypes)
            for family_name, subtypes in sorted(facts.family_subtypes.items())
        },
        "multi_entity_records": facts.multi_entity_records,
        "multi_entity_ratio": (
            facts.multi_entity_records / len(records) if records else 0.0
        ),
        "unique_text_count": len(text_counts),
        "unique_source_id_count": len(id_counts),
        "unique_focus_value_hash_count": len(facts.focus_value_hashes),
        "duplicate_text_records": duplicate_text_records,
        "duplicate_source_id_records": duplicate_id_records,
        "error_counts": dict(sorted(code_counts.items())),
        "errors": errors,
    }
    return result, facts


def validate_split_records(
    records: Sequence[Mapping[str, Any]], split: str
) -> Dict[str, Any]:
    """Validate one in-memory v1.2 split and return JSON-serializable results."""

    result, _ = _validate_split(records, split)
    return result


def _load_v1_1_partitions(
    root: Path, v1_1_generator_path: Path, expected_hashes: Mapping[str, str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """Load frozen v1.1 partitions with hash verification.

    Readable non-sealed files are loaded from disk and hash-checked.  The
    sealed challenge (always) and unreadable files (as a fallback) come from
    deterministic in-memory regeneration; the canonical serialization hash of
    each regenerated partition must equal the frozen expected hash.
    """

    module = v1_1_validator._load_module(
        v1_1_generator_path, "_securelogx_v1_1_generator_for_v1_2"
    )
    regenerated = module.generate_all(seed=42)

    partitions: Dict[str, List[Dict[str, Any]]] = {}
    evidence: Dict[str, Dict[str, Any]] = {}
    for key, relative in V1_1_PARTITION_PATHS.items():
        partition = V1_1_GENERATOR_PARTITIONS[key]
        expected = expected_hashes[key]
        path = root / relative
        records: Optional[List[Dict[str, Any]]] = None
        method = "regenerated_in_memory"
        if key not in V1_1_FILE_FORBIDDEN:
            try:
                records = load_jsonl(path)
                actual = _sha256_file(path)
                method = "file"
            except PermissionError:
                records = None
        if records is None:
            records = regenerated[partition]
            actual = canonical_jsonl_sha256(records)
        if actual != expected:
            raise RuntimeError(
                f"STOP: frozen ML-v1.1 partition {relative} hash mismatch; "
                f"expected {expected}, got {actual} (method: {method})"
            )
        partitions[key] = records
        evidence[key] = {
            "path": relative,
            "records": len(records),
            "sha256": actual,
            "expected_sha256": expected,
            "method": method,
            "file_opened": method == "file",
        }
    return partitions, evidence


def _collect_reference_facts(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    facts: Dict[str, Dict[str, Set[Any]]] = {}
    for split, records in records_by_split.items():
        texts: Set[str] = set()
        source_ids: Set[str] = set()
        families: Set[str] = set()
        value_hashes: Set[str] = set()
        skeletons: Set[str] = set()
        for record in records:
            if not isinstance(record, Mapping):
                continue
            text = record.get("text")
            if isinstance(text, str):
                texts.add(text)
            meta = record.get("meta")
            if isinstance(meta, Mapping):
                source_id = meta.get("source_record_id")
                if isinstance(source_id, str):
                    source_ids.add(source_id)
                family = meta.get("template_family")
                if isinstance(family, str):
                    families.add(family)
                targets = meta.get("contrast_targets")
                if isinstance(targets, list):
                    for target in targets:
                        if isinstance(target, Mapping):
                            value_hash = target.get("value_sha256")
                            if isinstance(value_hash, str):
                                value_hashes.add(value_hash)
            if isinstance(text, str):
                skeletons.add(normalize_template_skeleton(text))
        facts[split] = {
            "texts": texts,
            "source_ids": source_ids,
            "families": families,
            "value_hashes": value_hashes,
            "skeletons": skeletons,
        }
    return facts


def audit_template_catalogs(
    new_generator_path: str | Path,
    v1_1_generator_path: str | Path,
    parent_generator_path: str | Path,
) -> Dict[str, Any]:
    """Audit normalized v1.2 catalog templates against every parent catalog."""

    errors: List[str] = []
    new_entries: List[Dict[str, str]] = []
    v1_1_entries: List[Dict[str, str]] = []
    parent_entries: List[Dict[str, str]] = []
    try:
        new_module = v1_1_validator._load_module(
            Path(new_generator_path), "_securelogx_v1_2_generator_audit"
        )
        if not callable(getattr(new_module, "catalog", None)):
            raise AttributeError("v1.2 generator must expose catalog()")
        new_entries = v1_1_validator._flatten_catalog(new_module.catalog())
        if not new_entries:
            raise ValueError("v1.2 catalog did not expose any template strings")
    except (OSError, ImportError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"could not load v1.2 generator catalog: {exc}")

    try:
        v1_1_module = v1_1_validator._load_module(
            Path(v1_1_generator_path), "_securelogx_v1_1_generator_audit_for_v1_2"
        )
        if not callable(getattr(v1_1_module, "catalog", None)):
            raise AttributeError("v1.1 generator must expose catalog()")
        v1_1_entries = v1_1_validator._flatten_catalog(v1_1_module.catalog())
        if not v1_1_entries:
            raise ValueError("v1.1 catalog did not expose any template strings")
    except (OSError, ImportError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"could not load v1.1 generator catalog: {exc}")

    try:
        parent_module = v1_1_validator._load_module(
            Path(parent_generator_path), "_securelogx_parent_generator_audit_for_v1_2"
        )
        generator_class = getattr(parent_module, "SecureLogXDataGenerator", None)
        if generator_class is None:
            raise AttributeError("parent generator is missing SecureLogXDataGenerator")
        parent_entries = v1_1_validator._flatten_catalog(
            generator_class(seed=42).template_catalog()
        )
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

    def _skeleton_overlap(
        reference_entries: Sequence[Mapping[str, str]]
    ) -> List[Dict[str, Any]]:
        reference_skeletons: Dict[str, Set[str]] = defaultdict(set)
        for entry in reference_entries:
            reference_skeletons[normalize_template_skeleton(entry["template"])].add(
                entry["family"]
            )
        overlaps: List[Dict[str, Any]] = []
        for entry in new_entries:
            reference_families = reference_skeletons.get(
                normalize_template_skeleton(entry["template"]), set()
            )
            if reference_families:
                overlaps.append(
                    {
                        "new_family": entry["family"],
                        "reference_families": sorted(reference_families),
                    }
                )
        return overlaps

    v1_1_overlaps = _skeleton_overlap(v1_1_entries)
    parent_overlaps = _skeleton_overlap(parent_entries)

    new_families = {entry["family"] for entry in new_entries}
    v1_1_families = {entry["family"] for entry in v1_1_entries}
    family_name_overlap = sorted(new_families & v1_1_families)

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
        "v1_1_normalized_skeleton_overlap": _gate(
            readability and not v1_1_overlaps,
            len(v1_1_overlaps),
            0,
            v1_1_overlaps,
        ),
        "v1_1_family_name_overlap": _gate(
            readability and not family_name_overlap,
            len(family_name_overlap),
            0,
            family_name_overlap,
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
        "v1_1_template_count": len(v1_1_entries),
        "parent_template_count": len(parent_entries),
        "gates": gates,
        "cross_split_similarity": v1_1_validator._max_similarity(
            new_entries, new_entries, exclude_same_object=True
        ),
        "v1_1_similarity": v1_1_validator._max_similarity(new_entries, v1_1_entries),
        "parent_similarity": v1_1_validator._max_similarity(
            new_entries, parent_entries
        ),
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
    """Validate all v1.2 data gates and return a JSON-serializable result.

    ``paths`` may override keys from :data:`DEFAULT_RELATIVE_PATHS`; relative
    overrides resolve beneath ``repo_root``.  The function is read-only and
    never opens the sealed challenge file.
    """

    root = Path(repo_root).resolve()
    resolved = _resolve_paths(root, paths)
    top_errors: List[Dict[str, Any]] = []
    ontology = validate_ontology(resolved["labels"])

    parent_freeze = json.loads(
        Path(resolved["parent_freeze"]).read_text(encoding="utf-8-sig")
    )
    freeze_files = parent_freeze["groups"]["ml_v1_1_datasets"]["files"]
    expected_v1_1_hashes = {
        key: freeze_files[relative]["sha256"]
        for key, relative in V1_1_PARTITION_PATHS.items()
    }
    sealed_expected = parent_freeze["sealed_challenge"]["expected_sha256"]
    if expected_v1_1_hashes["v1_1_sealed_challenge"] != sealed_expected:
        raise RuntimeError(
            "STOP: parent freeze is internally inconsistent about the sealed hash"
        )

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

    v1_1_partitions, v1_1_evidence = _load_v1_1_partitions(
        root, Path(resolved["v1_1_generator"]), expected_v1_1_hashes
    )

    split_results: Dict[str, Dict[str, Any]] = {}
    split_facts: Dict[str, _SplitFacts] = {}
    for split in NEW_SPLITS:
        split_result, facts = _validate_split(loaded[split], split)
        split_results[split] = split_result
        split_facts[split] = facts

    parent_facts = v1_1_validator._collect_parent_facts(
        {split: loaded[split] for split in PARENT_SPLITS}
    )
    v1_1_facts = _collect_reference_facts(v1_1_partitions)

    text_leakage = _pairwise_intersections(split_facts, "texts")
    id_leakage = _pairwise_intersections(split_facts, "source_ids")
    family_leakage = _pairwise_intersections(split_facts, "families")
    value_hash_leakage = _pairwise_intersections(split_facts, "focus_value_hashes")

    all_new_texts = set().union(*(set(facts.texts) for facts in split_facts.values()))
    all_new_ids = set().union(
        *(set(facts.source_ids) for facts in split_facts.values())
    )
    all_new_families = set().union(
        *(set(facts.families) for facts in split_facts.values())
    )
    all_new_focus_hashes = set().union(
        *(set(facts.focus_value_hashes) for facts in split_facts.values())
    )

    parent_text_overlap = all_new_texts & parent_facts["texts"]
    parent_id_overlap = all_new_ids & parent_facts["source_ids"]
    parent_family_overlap = all_new_families & parent_facts["families"]
    parent_test_text_overlap = all_new_texts & {
        record.get("text")
        for record in loaded["parent_test"]
        if isinstance(record, Mapping)
    }

    v1_1_overlap: Dict[str, Dict[str, int]] = {}
    for key in V1_1_SPLITS:
        reference = v1_1_facts[key]
        v1_1_overlap[key] = {
            "texts": len(all_new_texts & reference["texts"]),
            "source_ids": len(all_new_ids & reference["source_ids"]),
            "families": len(all_new_families & reference["families"]),
            "focus_value_hashes": len(
                all_new_focus_hashes & reference["value_hashes"]
            ),
            "normalized_text_skeletons": len(
                {normalize_template_skeleton(text) for text in all_new_texts}
                & reference["skeletons"]
            ),
        }
    v1_1_overlap_total = sum(
        count for overlap in v1_1_overlap.values() for count in overlap.values()
    )

    all_errors = top_errors + [
        error
        for split in NEW_SPLITS
        for error in split_results[split]["errors"]
    ]
    code_counts = Counter(error["code"] for error in all_errors)

    coverage_gates: Dict[str, Dict[str, Any]] = {}
    for split in NEW_SPLITS:
        family_counts = split_results[split]["category_family_counts"]
        expected_family_counts = EXPECTED_CATEGORY_FAMILY_COUNTS[split]
        coverage_gates[f"{split}_category_family_allocation"] = _gate(
            {key: int(value) for key, value in family_counts.items()}
            == dict(expected_family_counts),
            dict(family_counts),
            dict(expected_family_counts),
        )
        low, high = RECORD_COUNT_RANGES[split]
        coverage_gates[f"{split}_record_count_range"] = _gate(
            low <= split_results[split]["record_count"] <= high,
            split_results[split]["record_count"],
            f"{low}..{high}",
        )
        coverage_gates[f"{split}_multi_entity_ratio"] = _gate(
            split_results[split]["multi_entity_ratio"] >= MIN_MULTI_ENTITY_RATIO,
            round(split_results[split]["multi_entity_ratio"], 6),
            f">={MIN_MULTI_ENTITY_RATIO}",
        )

    train_result = split_results["train_addition"]
    train_family_subtypes = train_result["family_subtypes"]
    train_subtype_families: Counter = Counter()
    for subtypes in train_family_subtypes.values():
        for subtype in subtypes:
            train_subtype_families[subtype] += 1
    train_labels = train_result["label_counts"]
    sensitive_shortfalls = {
        label: int(train_labels.get(label, 0))
        for label in sorted(set(_CONTRAST_SENSITIVE_LABELS.values()))
        if int(train_labels.get(label, 0)) < MIN_TRAIN_SENSITIVE_SPANS_EACH
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
        "sealed_challenge_hash_unchanged": _gate(
            v1_1_evidence["v1_1_sealed_challenge"]["sha256"] == sealed_expected,
            v1_1_evidence["v1_1_sealed_challenge"]["sha256"],
            sealed_expected,
        ),
        "sealed_challenge_file_never_opened": _gate(
            not v1_1_evidence["v1_1_sealed_challenge"]["file_opened"],
            v1_1_evidence["v1_1_sealed_challenge"]["method"],
            "regenerated_in_memory",
        ),
        "canonical_spans": _gate(code_counts["span"] == 0, code_counts["span"], 0),
        "entity_provenance_bijection": _gate(
            code_counts["provenance"] == 0, code_counts["provenance"], 0
        ),
        "v1_2_metadata": _gate(
            code_counts["metadata"] == 0, code_counts["metadata"], 0
        ),
        "intended_split": _gate(
            code_counts["intended_split"] == 0, code_counts["intended_split"], 0
        ),
        "contrast_targets": _gate(
            code_counts["contrast_target"] == 0, code_counts["contrast_target"], 0
        ),
        "o_target_non_overlap": _gate(
            code_counts["o_target_overlap"] == 0, code_counts["o_target_overlap"], 0
        ),
        "complete_paired_contrast_groups": _gate(
            code_counts["pair_group"] == 0, code_counts["pair_group"], 0
        ),
        "family_metadata_consistency": _gate(
            code_counts["family_consistency"] == 0,
            code_counts["family_consistency"],
            0,
        ),
        "record_shape": _gate(
            code_counts["record_shape"] == 0, code_counts["record_shape"], 0
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
            text_leakage["total"] == 0, text_leakage["total"], 0, text_leakage["pairs"]
        ),
        "cross_split_source_id_leakage": _gate(
            id_leakage["total"] == 0, id_leakage["total"], 0, id_leakage["pairs"]
        ),
        "cross_split_template_family_leakage": _gate(
            family_leakage["total"] == 0,
            family_leakage["total"],
            0,
            family_leakage["pairs"],
        ),
        "cross_split_focus_value_hash_leakage": _gate(
            value_hash_leakage["total"] == 0,
            value_hash_leakage["total"],
            0,
            value_hash_leakage["pairs"],
        ),
        "parent_exact_text_contamination": _gate(
            not parent_text_overlap, len(parent_text_overlap), 0
        ),
        "parent_test_exact_text_contamination": _gate(
            not parent_test_text_overlap, len(parent_test_text_overlap), 0
        ),
        "parent_source_id_contamination": _gate(
            not parent_id_overlap, len(parent_id_overlap), 0
        ),
        "parent_template_family_contamination": _gate(
            not parent_family_overlap, len(parent_family_overlap), 0
        ),
        "v1_1_contamination_zero": _gate(
            v1_1_overlap_total == 0, v1_1_overlap_total, 0, v1_1_overlap
        ),
        "forbidden_parent_hard_negative": _gate(
            code_counts["forbidden_parent_template"] == 0,
            code_counts["forbidden_parent_template"],
            0,
        ),
        "train_ip_technical_family_minimum": _gate(
            int(
                train_result["category_family_counts"].get(
                    "ip_address_vs_technical_reference", 0
                )
            )
            >= MIN_TRAIN_IP_TECHNICAL_FAMILIES,
            int(
                train_result["category_family_counts"].get(
                    "ip_address_vs_technical_reference", 0
                )
            ),
            f">={MIN_TRAIN_IP_TECHNICAL_FAMILIES}",
        ),
        "dev_ip_technical_family_minimum": _gate(
            int(
                split_results["dev_challenge"]["category_family_counts"].get(
                    "ip_address_vs_technical_reference", 0
                )
            )
            >= MIN_DEV_IP_TECHNICAL_FAMILIES,
            int(
                split_results["dev_challenge"]["category_family_counts"].get(
                    "ip_address_vs_technical_reference", 0
                )
            ),
            f">={MIN_DEV_IP_TECHNICAL_FAMILIES}",
        ),
        "train_customer_id_family_minimum": _gate(
            train_subtype_families.get("CUSTOMER_ID", 0)
            >= MIN_TRAIN_CUSTOMER_ID_FAMILIES,
            train_subtype_families.get("CUSTOMER_ID", 0),
            f">={MIN_TRAIN_CUSTOMER_ID_FAMILIES}",
        ),
        "train_request_id_family_minimum": _gate(
            train_subtype_families.get("REQUEST_ID", 0)
            >= MIN_TRAIN_REQUEST_ID_FAMILIES,
            train_subtype_families.get("REQUEST_ID", 0),
            f">={MIN_TRAIN_REQUEST_ID_FAMILIES}",
        ),
        "train_all_business_id_subtypes_retained": _gate(
            set(train_result["business_id_subtypes"]) == set(BUSINESS_ID_SUBTYPES),
            sorted(train_result["business_id_subtypes"]),
            sorted(BUSINESS_ID_SUBTYPES),
        ),
        "train_ssn_span_minimum": _gate(
            int(train_labels.get("SSN", 0)) >= MIN_TRAIN_SSN_SPANS,
            int(train_labels.get("SSN", 0)),
            f">={MIN_TRAIN_SSN_SPANS}",
        ),
        "train_sensitive_counterexample_minimum": _gate(
            not sensitive_shortfalls,
            {
                label: int(train_labels.get(label, 0))
                for label in sorted(set(_CONTRAST_SENSITIVE_LABELS.values()))
            },
            f">={MIN_TRAIN_SENSITIVE_SPANS_EACH} each",
            {"below_minimum": sensitive_shortfalls},
        ),
        **coverage_gates,
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
            resolved["new_generator"],
            resolved["v1_1_generator"],
            resolved["parent_generator"],
        )
        for name, gate in template_audit["gates"].items():
            gates[f"template_catalog_{name}"] = gate

    passed = all(gate["passed"] for gate in gates.values())
    result: Dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": "SecureLogX ML-v1.2",
        "passed": passed,
        "ready": passed,
        "recommendation": (
            "READY FOR ML-v1.2 RETRAINING"
            if passed
            else "NOT READY FOR ML-v1.2 RETRAINING"
        ),
        "repo_root": str(root),
        "files": file_stats,
        "ontology": ontology,
        "splits": split_results,
        "parents": parent_facts["counts"],
        "v1_1_partitions": v1_1_evidence,
        "leakage": {
            "cross_split_exact_text": text_leakage,
            "cross_split_source_id": id_leakage,
            "cross_split_template_family": family_leakage,
            "cross_split_focus_value_hash": value_hash_leakage,
            "parent_exact_text_count": len(parent_text_overlap),
            "parent_test_exact_text_count": len(parent_test_text_overlap),
            "parent_source_id_count": len(parent_id_overlap),
            "parent_template_family_count": len(parent_family_overlap),
            "v1_1_overlap": v1_1_overlap,
            "v1_1_overlap_total": v1_1_overlap_total,
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
        description="Validate SecureLogX ML-v1.2 counterbalance JSONL data"
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--train-additions")
    parser.add_argument("--dev-challenge")
    parser.add_argument(
        "--skip-template-audit",
        action="store_true",
        help="Skip catalog imports (data gates still run); intended only for isolated tests",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides: Dict[str, str] = {}
    if args.train_additions is not None:
        overrides["train_addition"] = args.train_additions
    if args.dev_challenge is not None:
        overrides["dev_challenge"] = args.dev_challenge
    try:
        result = validate_dataset(
            args.repo_root,
            overrides,
            include_template_audit=not args.skip_template_audit,
        )
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}), file=sys.stderr)
        return 2

    if args.compact:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
