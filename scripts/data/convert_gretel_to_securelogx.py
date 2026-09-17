#!/usr/bin/env python3
"""Convert Gretel Finance PII records to the SecureLogX ML-v1 JSONL schema.

The converter intentionally keeps the source annotation layer auditable:

* canonical, non-overlapping training spans are written to ``entities``;
* every retained source span has an ``entity_provenance`` entry;
* deferred, healthcare, benign-unmapped, invalid, and overlap-conflict spans
  are retained in ``excluded_spans`` with an explicit disposition; and
* a record-level ``training_exclusion`` safety gate prevents partial false-O
  examples from entering the ML-v1 training merge.

Dataset loading is lazy so schema/unit tests do not require Hugging Face or a
network connection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml


DEFAULT_DATASET = "gretelai/synthetic_pii_finance_multilingual"
DEFAULT_SOURCE_ID = "gretel_finance_pii"
DEFAULT_SOURCE_URL_PREFIX = "https://huggingface.co/datasets/"

TEXT_FIELD_CANDIDATES = (
    "generated_text",
    "text",
    "document",
    "content",
    "message",
    "sample",
)
ENTITY_FIELD_CANDIDATES = (
    "pii_spans",
    "entities",
    "annotations",
    "pii",
    "entity_list",
    "labels",
)
LANGUAGE_FIELD_CANDIDATES = ("language", "lang", "locale")
START_FIELD_CANDIDATES = ("start", "start_idx", "start_index", "begin", "offset_start")
END_FIELD_CANDIDATES = ("end", "end_idx", "end_index", "stop", "offset_end")
LABEL_FIELD_CANDIDATES = ("label", "type", "entity", "entity_type", "pii_type", "category")
VALUE_FIELD_CANDIDATES = ("text", "value", "span_text", "entity_text")
SPAN_COLLECTION_KEYS = ("spans", "entities", "annotations", "pii", "items")

DEFAULT_DEFERRED_LABELS = frozenset(
    {
        "PASSWORD",
        "PIN",
        "OTP",
        "SECURITY_ANSWER",
        "JWT",
        "SESSION_ID",
        "COOKIE",
        "CVV",
        "CARD_EXPIRY",
        "MAC_ADDRESS",
        "GEOLOCATION",
        "BIOMETRIC_ID",
    }
)
DEFAULT_HEALTHCARE_LABELS = frozenset(
    {
        "PROVIDER_NPI",
        "MRN",
        "PATIENT_ID",
        "HEALTH_PLAN_ID",
        "CLAIM_ID",
        "PRESCRIPTION_ID",
        "DIAGNOSIS_CODE",
        "PROCEDURE_CODE",
        "LAB_ORDER_ID",
    }
)
DEFAULT_BENIGN_LABELS = frozenset(
    {"DATE", "TIME", "DATE_TIME", "COMPANY", "ORGANIZATION", "ORGANIZATION_NAME", "JOB"}
)

_LABEL_SEPARATORS = re.compile(r"[\s\-./]+")


def _normalise_label_key(value: str) -> str:
    """Return a case-insensitive source-label lookup key."""
    return _LABEL_SEPARATORS.sub("_", value.strip()).strip("_").casefold()


def _taxonomy_name(value: str) -> str:
    return _normalise_label_key(value).upper()


def _normalise_mapping(raw_mapping: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    for raw_key, raw_target in raw_mapping.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError("Source mapping keys must be non-empty strings")
        key = _normalise_label_key(raw_key)
        if raw_target is not None and not isinstance(raw_target, str):
            raise ValueError(f"Mapping target for {raw_key!r} must be a string or null")
        target = _taxonomy_name(raw_target) if isinstance(raw_target, str) else None
        if key in mapping and mapping[key] != target:
            raise ValueError(f"Conflicting normalized mapping for source label {raw_key!r}")
        mapping[key] = target
    return mapping


def _string_map(raw: Any, field_name: str) -> Dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    result: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{field_name} keys and values must be strings")
        result[_normalise_label_key(key)] = _taxonomy_name(value)
    return result


def _label_set(raw: Any, field_name: str, fallback: Set[str]) -> Set[str]:
    if raw is None:
        return set(fallback)
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"{field_name} must be a list of strings")
    return {_taxonomy_name(item) for item in raw}


@dataclass(frozen=True)
class LabelPolicy:
    mapping: Dict[str, Optional[str]]
    normalization_aliases: Dict[str, str]
    source_label_aliases: Dict[str, str]
    deferred_labels: Set[str]
    healthcare_labels: Set[str]
    benign_labels: Set[str]

    def source_subtype(self, source_label: str) -> str:
        key = _normalise_label_key(source_label)
        return self.source_label_aliases.get(key, _taxonomy_name(source_label))

    def canonicalize(self, label: str) -> str:
        current = _taxonomy_name(label)
        visited: Set[str] = set()
        while current in self.normalization_aliases and current not in visited:
            visited.add(current)
            current = self.normalization_aliases[current]
        return current

    def resolve(self, source_label: str, valid_labels: Set[str]) -> Tuple[Optional[str], bool]:
        key = _normalise_label_key(source_label)
        if key in self.mapping:
            target = self.mapping[key]
            return (self.canonicalize(target), True) if target is not None else (None, True)

        # Identity/normalization fallback is safe only when it resolves to a
        # frozen ML-v1 class. Everything else remains explicitly unsupported.
        candidate = self.canonicalize(self.source_subtype(source_label))
        return (candidate, False) if candidate in valid_labels else (None, False)


def load_label_policy(yaml_path: str) -> LabelPolicy:
    """Load Gretel mappings plus the ML-v1 exclusion policy."""
    with open(yaml_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("Label map YAML must contain an object at its root")

    mappings = config.get("mappings")
    if not isinstance(mappings, Mapping):
        raise ValueError("Label map YAML is missing the 'mappings' object")
    source_config = mappings.get("gretel") or mappings.get("gretel_finance")
    if not isinstance(source_config, Mapping):
        raise ValueError("Label map YAML is missing mappings.gretel.mapping")
    raw_mapping = source_config.get("mapping", source_config)
    if not isinstance(raw_mapping, Mapping):
        raise ValueError("mappings.gretel.mapping must be an object")

    normalization_aliases = {
        _taxonomy_name(key): _taxonomy_name(value)
        for key, value in _string_map(config.get("normalization_aliases"), "normalization_aliases").items()
    }
    # _string_map normalizes keys for source lookup, so rebuild taxonomy keys
    # from the original object for canonical alias resolution.
    raw_normalization = config.get("normalization_aliases") or {}
    normalization_aliases = {
        _taxonomy_name(str(key)): _taxonomy_name(str(value))
        for key, value in raw_normalization.items()
    }

    return LabelPolicy(
        mapping=_normalise_mapping(raw_mapping),
        normalization_aliases=normalization_aliases,
        source_label_aliases=_string_map(config.get("source_label_aliases"), "source_label_aliases"),
        deferred_labels=_label_set(
            config.get("deferred_labels"), "deferred_labels", set(DEFAULT_DEFERRED_LABELS)
        ),
        healthcare_labels=_label_set(
            config.get("healthcare_extension_labels"),
            "healthcare_extension_labels",
            set(DEFAULT_HEALTHCARE_LABELS),
        ),
        benign_labels=_label_set(
            config.get("benign_unmapped_labels"),
            "benign_unmapped_labels",
            set(DEFAULT_BENIGN_LABELS),
        ),
    )


def load_label_map(yaml_path: str) -> Dict[str, Optional[str]]:
    """Backward-compatible helper returning normalized Gretel mappings."""
    return load_label_policy(yaml_path).mapping


def load_label_spec(labels_json_path: str) -> Tuple[List[str], Set[str]]:
    """Load entity order and verify any supplied BIO/ID tables are stable."""
    with open(labels_json_path, encoding="utf-8") as handle:
        config = json.load(handle)
    entities = config.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("Label config must contain a non-empty 'entities' list")
    if any(not isinstance(label, str) or not label for label in entities):
        raise ValueError("Every entity label must be a non-empty string")
    if len(set(entities)) != len(entities):
        raise ValueError("Entity labels must be unique")

    expected_bio = ["O"] + [bio for entity in entities for bio in (f"B-{entity}", f"I-{entity}")]
    if "bio_labels" in config and config["bio_labels"] != expected_bio:
        raise ValueError("bio_labels is not the deterministic O/B-/I- expansion of entities")
    expected_label_to_id = {label: index for index, label in enumerate(expected_bio)}
    if "label_to_id" in config and config["label_to_id"] != expected_label_to_id:
        raise ValueError("label_to_id does not match deterministic BIO order")
    expected_id_to_label = {str(index): label for index, label in enumerate(expected_bio)}
    if "id_to_label" in config and config["id_to_label"] != expected_id_to_label:
        raise ValueError("id_to_label does not match deterministic BIO order")
    return list(entities), set(entities)


def load_valid_labels(labels_json_path: str) -> Set[str]:
    """Backward-compatible helper returning the frozen entity set."""
    return load_label_spec(labels_json_path)[1]


class _ParquetRecordDataset:
    """Small indexed/iterable adapter used for explicit offline Parquet input."""

    def __init__(self, path: str):
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("The 'pyarrow' package is required for --data-file") from exc
        self._table = parquet.read_table(path)
        self.column_names = list(self._table.column_names)

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return {
            name: self._table.column(name)[index].as_py()
            for name in self.column_names
        }

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for batch in self._table.to_batches(max_chunksize=1024):
            yield from batch.to_pylist()


def _load_dataset(dataset_name: str, split: str, data_file: Optional[str] = None) -> Any:
    if data_file:
        path = Path(data_file)
        if not path.is_file():
            raise FileNotFoundError(f"Explicit dataset file not found: {data_file}")
        if path.suffix.casefold() != ".parquet":
            raise ValueError("--data-file currently supports a local .parquet file")
        return _ParquetRecordDataset(str(path))
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required (pip install datasets)") from exc
    return load_dataset(dataset_name, split=split)


def _first_record(dataset: Any) -> Optional[Mapping[str, Any]]:
    try:
        if len(dataset) == 0:
            return None
        record = dataset[0]
    except (TypeError, KeyError, IndexError) as exc:
        raise ValueError("Dataset must support deterministic indexed access for schema inspection") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"Dataset records must be objects, found {type(record).__name__}")
    return record


def _dataset_columns(dataset: Any, sample: Optional[Mapping[str, Any]]) -> List[str]:
    columns = getattr(dataset, "column_names", None)
    if isinstance(columns, (list, tuple)):
        return [str(column) for column in columns]
    return [str(key) for key in sample.keys()] if sample else []


def _schema_types(sample: Mapping[str, Any]) -> Dict[str, str]:
    """Describe a sample without printing raw values or sensitive text."""
    summary: Dict[str, str] = {}
    for key, value in sample.items():
        if isinstance(value, (list, tuple, dict, str)):
            summary[str(key)] = f"{type(value).__name__}(length={len(value)})"
        else:
            summary[str(key)] = type(value).__name__
    return summary


def _resolve_field(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {_normalise_label_key(column): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        matched = normalized.get(_normalise_label_key(candidate))
        if matched is not None:
            return matched
    return None


def _schema_error(
    missing: str,
    columns: Sequence[str],
    sample: Mapping[str, Any],
    expected: Sequence[str],
) -> str:
    return (
        f"Schema inspection failed: missing required {missing}. "
        f"Available columns: {list(columns)}. Sample record keys: {list(sample.keys())}. "
        f"Expected one of: {list(expected)}. Suggested next action: inspect the upstream dataset "
        "schema and add an explicit field alias; do not guess offsets or labels."
    )


def inspect_gretel_dataset(
    split: str = "train",
    dataset_name: str = DEFAULT_DATASET,
    language: Optional[str] = None,
    data_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect columns and safe sample types without exposing record contents."""
    try:
        dataset = _load_dataset(dataset_name, split, data_file)
        sample = _first_record(dataset)
        if sample is None:
            return {
                "error": "Dataset split is empty",
                "dataset_id": dataset_name,
                "split": split,
                "columns": [],
                "sample_record_keys": [],
            }
        columns = _dataset_columns(dataset, sample)
        text_field = _resolve_field(columns, TEXT_FIELD_CANDIDATES)
        entities_field = _resolve_field(columns, ENTITY_FIELD_CANDIDATES)
        language_field = _resolve_field(columns, LANGUAGE_FIELD_CANDIDATES)
        requested_languages = _parse_language_filter(language)
        missing = []
        if text_field is None:
            missing.append(_schema_error("text field", columns, sample, TEXT_FIELD_CANDIDATES))
        if entities_field is None:
            missing.append(_schema_error("entity-span field", columns, sample, ENTITY_FIELD_CANDIDATES))
        if requested_languages is not None and language_field is None:
            missing.append(_schema_error("language field", columns, sample, LANGUAGE_FIELD_CANDIDATES))
        return {
            "dataset_id": dataset_name,
            "data_file": data_file,
            "split": split,
            "num_records": len(dataset),
            "columns": columns,
            "sample_record_keys": list(sample.keys()),
            "sample_schema": _schema_types(sample),
            "resolved_fields": {
                "text": text_field,
                "entities": entities_field,
                "language": language_field,
            },
            "requested_language": language,
            "errors": missing,
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "dataset_id": dataset_name,
            "split": split,
            "columns": [],
            "sample_record_keys": [],
        }


_LANGUAGE_ALIASES = {
    "*": "*",
    "all": "*",
    "any": "*",
    "en": "english",
    "eng": "english",
    "english": "english",
    "es": "spanish",
    "spa": "spanish",
    "spanish": "spanish",
    "sv": "swedish",
    "swe": "swedish",
    "swedish": "swedish",
    "de": "german",
    "deu": "german",
    "ger": "german",
    "german": "german",
    "it": "italian",
    "ita": "italian",
    "italian": "italian",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "france": "french",
    "french": "french",
    "nl": "dutch",
    "nld": "dutch",
    "dut": "dutch",
    "dutch": "dutch",
}


def _canonical_language(value: str) -> str:
    key = value.strip().casefold().replace("_", "-")
    return _LANGUAGE_ALIASES.get(key, key)


def _parse_language_filter(language: Optional[str]) -> Optional[Set[str]]:
    if language is None:
        return None
    values = [part.strip() for part in language.split(",") if part.strip()]
    if not values:
        raise ValueError("--language must name a language, a comma-separated list, or 'all'")
    normalized = {_canonical_language(value) for value in values}
    return None if "*" in normalized else normalized


def _language_matches(value: Any, requested: Optional[Set[str]]) -> bool:
    if requested is None:
        return True
    if isinstance(value, str):
        return _canonical_language(value) in requested
    if isinstance(value, (list, tuple, set)):
        return any(isinstance(item, str) and _canonical_language(item) in requested for item in value)
    return False


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def _first_present(item: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    normalized = {_normalise_label_key(str(key)): key for key in item.keys()}
    for candidate in candidates:
        key = candidate if candidate in item else normalized.get(_normalise_label_key(candidate))
        if key is not None:
            return item[key]
    return None


@dataclass(frozen=True)
class SourceSpan:
    start: Any
    end: Any
    source_label: Any
    source_span_index: int
    source_field: str
    annotated_text: Any
    source_span_metadata: Dict[str, Any]


def _span_from_item(item: Any, label_hint: Optional[str], index: int, source_field: str) -> SourceSpan:
    if isinstance(item, Mapping):
        start = _first_present(item, START_FIELD_CANDIDATES)
        end = _first_present(item, END_FIELD_CANDIDATES)
        source_label = label_hint if label_hint is not None else _first_present(item, LABEL_FIELD_CANDIDATES)
        annotated_text = _first_present(item, VALUE_FIELD_CANDIDATES)
        reserved = {
            _normalise_label_key(field)
            for field in (
                *START_FIELD_CANDIDATES,
                *END_FIELD_CANDIDATES,
                *LABEL_FIELD_CANDIDATES,
                *VALUE_FIELD_CANDIDATES,
            )
        }
        metadata = {
            str(key): _json_safe(value)
            for key, value in item.items()
            if _normalise_label_key(str(key)) not in reserved
        }
    elif isinstance(item, (list, tuple)):
        if label_hint is not None and len(item) >= 2:
            start, end, source_label = item[0], item[1], label_hint
        elif len(item) >= 3:
            start, end, source_label = item[0], item[1], item[2]
        else:
            raise ValueError(f"span {index} is not a supported coordinate tuple")
        annotated_text = None
        metadata = {}
    else:
        raise ValueError(f"span {index} must be an object or coordinate tuple")

    if start is None or end is None or source_label is None:
        raise ValueError(f"span {index} is missing start, end, or label")
    return SourceSpan(start, end, source_label, index, source_field, annotated_text, metadata)


def parse_source_spans(raw_spans: Any, source_field: str) -> List[SourceSpan]:
    """Parse supported list/dict containers without reconstructing guessed spans."""
    if isinstance(raw_spans, str):
        try:
            raw_spans = json.loads(raw_spans)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source_field} is not valid JSON: {exc.msg}") from exc

    label_items: List[Tuple[Any, Optional[str]]] = []
    if isinstance(raw_spans, list):
        label_items = [(item, None) for item in raw_spans]
    elif isinstance(raw_spans, Mapping):
        collection_key = next((key for key in SPAN_COLLECTION_KEYS if key in raw_spans), None)
        if collection_key is not None:
            collection = raw_spans[collection_key]
            if not isinstance(collection, list):
                raise ValueError(f"{source_field}.{collection_key} must be a list")
            label_items = [(item, None) for item in collection]
        elif any(_normalise_label_key(str(key)) in START_FIELD_CANDIDATES for key in raw_spans):
            label_items = [(raw_spans, None)]
        else:
            for label, spans in raw_spans.items():
                if isinstance(spans, Mapping):
                    label_items.append((spans, str(label)))
                elif isinstance(spans, (list, tuple)):
                    if len(spans) >= 2 and not isinstance(spans[0], (Mapping, list, tuple)):
                        label_items.append((spans, str(label)))
                    else:
                        label_items.extend((span, str(label)) for span in spans)
                else:
                    raise ValueError(f"{source_field}.{label} must contain span coordinates")
    else:
        raise ValueError(f"{source_field} must be a JSON list or object")

    return [
        _span_from_item(item, label_hint, index, source_field)
        for index, (item, label_hint) in enumerate(label_items)
    ]


def _provenance(span: SourceSpan, policy: LabelPolicy, canonical_label: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "start": int(span.start),
        "end": int(span.end),
        "label": canonical_label,
        "canonical_label": canonical_label,
        "source_label": span.source_label,
        "source_subtype": policy.source_subtype(span.source_label),
        "source_span_index": span.source_span_index,
        "source_field": span.source_field,
        "training_exclusion": False,
    }
    if span.source_span_metadata:
        result["source_span_metadata"] = span.source_span_metadata
    return result


def _excluded_span(
    span: SourceSpan,
    policy: LabelPolicy,
    mapped_label: Optional[str],
    category: str,
    reason: str,
    training_exclusion: bool,
    **extra: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "source_label": span.source_label if isinstance(span.source_label, str) else str(span.source_label),
        "source_subtype": (
            policy.source_subtype(span.source_label)
            if isinstance(span.source_label, str)
            else "UNKNOWN"
        ),
        "source_span_index": span.source_span_index,
        "source_field": span.source_field,
        "mapped_label": mapped_label,
        "category": category,
        "reason": reason,
        "training_exclusion": training_exclusion,
    }
    if isinstance(span.start, Integral) and not isinstance(span.start, bool):
        result["start"] = int(span.start)
    else:
        result["source_start"] = _json_safe(span.start)
    if isinstance(span.end, Integral) and not isinstance(span.end, bool):
        result["end"] = int(span.end)
    else:
        result["source_end"] = _json_safe(span.end)
    if span.source_span_metadata:
        result["source_span_metadata"] = span.source_span_metadata
    result.update({key: _json_safe(value) for key, value in extra.items()})
    return result


def _unmapped_disposition(source_subtype: str, policy: LabelPolicy, explicit: bool) -> Tuple[str, str, bool]:
    if source_subtype in policy.deferred_labels:
        return "deferred_ml_v1", "reserved_product_taxonomy_label", True
    if source_subtype in policy.healthcare_labels:
        return "healthcare_extension", "reserved_healthcare_extension_label", True
    if source_subtype in policy.benign_labels:
        return "benign_unmapped", "not_sensitive_or_not_semantically_safe_for_ml_v1", False
    if explicit:
        return "unsupported_source_label", "explicit_null_mapping_requires_review", True
    return "unsupported_source_label", "no_explicit_ml_v1_mapping", True


def _spans_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]


def resolve_mapped_overlaps(
    candidates: Sequence[Dict[str, Any]],
    label_order: Sequence[str],
) -> Tuple[List[List[Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve overlaps deterministically and retain provenance for every drop.

    Exact duplicate canonical spans are collapsed while retaining all of their
    provenance. Other conflicts prefer the longest span, then earliest start,
    canonical ontology order, lexical label, and source-span index.
    """
    grouped: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    for candidate in candidates:
        key = (candidate["start"], candidate["end"], candidate["label"])
        grouped.setdefault(key, []).append(candidate)

    representatives = [
        {
            "start": key[0],
            "end": key[1],
            "label": key[2],
            "members": sorted(value, key=lambda item: item["provenance"]["source_span_index"]),
        }
        for key, value in grouped.items()
    ]
    rank = {label: index for index, label in enumerate(label_order)}
    representatives.sort(
        key=lambda item: (
            -(item["end"] - item["start"]),
            item["start"],
            item["end"],
            rank.get(item["label"], len(rank)),
            item["label"],
            item["members"][0]["provenance"]["source_span_index"],
        )
    )

    selected: List[Dict[str, Any]] = []
    overlap_exclusions: List[Dict[str, Any]] = []
    for candidate in representatives:
        conflict = next((kept for kept in selected if _spans_overlap(candidate, kept)), None)
        if conflict is None:
            selected.append(candidate)
            continue
        for member in candidate["members"]:
            provenance = member["provenance"]
            overlap_exclusions.append(
                {
                    **provenance,
                    "mapped_label": candidate["label"],
                    "category": "overlap_conflict",
                    "reason": "deterministic_overlap_resolution",
                    "training_exclusion": True,
                    "overlaps_with": {
                        "start": conflict["start"],
                        "end": conflict["end"],
                        "label": conflict["label"],
                    },
                }
            )

    selected.sort(key=lambda item: (item["start"], item["end"], rank.get(item["label"], len(rank))))
    entities = [[item["start"], item["end"], item["label"]] for item in selected]
    provenance = [
        member["provenance"]
        for item in selected
        for member in item["members"]
    ]
    overlap_exclusions.sort(
        key=lambda item: (item["start"], item["end"], item["source_span_index"], item["source_label"])
    )
    return entities, provenance, overlap_exclusions


def _source_record_id(record: Mapping[str, Any], fallback: int) -> Any:
    for key in ("index", "id", "record_id", "level_0"):
        if key in record and record[key] is not None:
            return _json_safe(record[key])
    return fallback


def _build_metadata(
    record: Mapping[str, Any],
    record_position: int,
    dataset_name: str,
    split: str,
    text_field: str,
    entities_field: str,
    language_field: Optional[str],
    entity_provenance: List[Dict[str, Any]],
    excluded_spans: List[Dict[str, Any]],
) -> Dict[str, Any]:
    source_metadata = {
        str(key): _json_safe(value)
        for key, value in record.items()
        if key not in {text_field, entities_field}
    }
    meta: Dict[str, Any] = {
        "source": DEFAULT_SOURCE_ID,
        "source_record_id": _source_record_id(record, record_position),
        "dataset_id": dataset_name,
        "dataset": dataset_name,
        "source_url": f"{DEFAULT_SOURCE_URL_PREFIX}{dataset_name}",
        "source_split": split,
        "split": split,
        "format": "text",
        "review_status": "auto_converted",
        "license_reviewed": False,
        "source_fields": {
            "text": text_field,
            "entities": entities_field,
            "language": language_field,
        },
        "source_metadata": source_metadata,
        "entity_provenance": entity_provenance,
        "excluded_spans": excluded_spans,
    }
    for key in ("document_type", "expanded_type", "domain"):
        if key in record and record[key] is not None:
            meta[key] = _json_safe(record[key])
    if language_field is not None and record.get(language_field) is not None:
        meta["language"] = _json_safe(record[language_field])

    blocking_reasons = sorted(
        {span["reason"] for span in excluded_spans if span.get("training_exclusion") is True}
    )
    meta["training_exclusion"] = bool(blocking_reasons)
    meta["training_exclusion_reasons"] = blocking_reasons
    meta["excluded_span_counts"] = dict(
        sorted(Counter(span["category"] for span in excluded_spans).items())
    )
    return meta


def _fallback_policy(label_map: Mapping[str, Optional[str]]) -> LabelPolicy:
    return LabelPolicy(
        mapping=_normalise_mapping(label_map),
        normalization_aliases={},
        source_label_aliases={},
        deferred_labels=set(DEFAULT_DEFERRED_LABELS),
        healthcare_labels=set(DEFAULT_HEALTHCARE_LABELS),
        benign_labels=set(DEFAULT_BENIGN_LABELS),
    )


def convert_gretel_to_securelogx(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    language: Optional[str] = "English",
    label_map: Optional[Dict[str, Optional[str]]] = None,
    valid_labels: Optional[Set[str]] = None,
    output_path: str = "data/processed/gretel_finance_securelogx.jsonl",
    max_records: Optional[int] = None,
    label_policy: Optional[LabelPolicy] = None,
    label_order: Optional[Sequence[str]] = None,
    data_file: Optional[str] = None,
) -> Tuple[int, int, List[str]]:
    """Convert records, returning ``(written, matching_records, errors)``."""
    if max_records is not None and max_records <= 0:
        return 0, 0, ["max_records must be greater than zero"]
    valid_labels = set(valid_labels or set())
    if not valid_labels:
        return 0, 0, ["valid_labels must contain the frozen ML-v1 entity set"]
    policy = label_policy or _fallback_policy(label_map or {})
    label_order = list(label_order or sorted(valid_labels))

    try:
        requested_languages = _parse_language_filter(language)
        dataset = _load_dataset(dataset_name, split, data_file)
        sample = _first_record(dataset)
    except Exception as exc:
        return 0, 0, [f"Failed to load or inspect dataset {dataset_name!r} split {split!r}: {exc}"]
    if sample is None:
        return 0, 0, [f"Dataset {dataset_name!r} split {split!r} is empty"]

    columns = _dataset_columns(dataset, sample)
    print(f"Dataset: {dataset_name}; split: {split}; records: {len(dataset)}")
    print(f"Available columns: {columns}")
    print(f"Sample record keys: {list(sample.keys())}")
    print(f"Sample schema (values suppressed): {json.dumps(_schema_types(sample), sort_keys=True)}")

    text_field = _resolve_field(columns, TEXT_FIELD_CANDIDATES)
    entities_field = _resolve_field(columns, ENTITY_FIELD_CANDIDATES)
    language_field = _resolve_field(columns, LANGUAGE_FIELD_CANDIDATES)
    schema_errors: List[str] = []
    if text_field is None:
        schema_errors.append(_schema_error("text field", columns, sample, TEXT_FIELD_CANDIDATES))
    if entities_field is None:
        schema_errors.append(_schema_error("entity-span field", columns, sample, ENTITY_FIELD_CANDIDATES))
    if requested_languages is not None and language_field is None:
        schema_errors.append(_schema_error("language field", columns, sample, LANGUAGE_FIELD_CANDIDATES))
    if schema_errors:
        return 0, 0, schema_errors
    assert text_field is not None and entities_field is not None

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(f"{output_file.name}.tmp")

    errors: List[str] = []
    converted_count = 0
    matching_count = 0
    scanned_count = 0
    filtered_count = 0
    excluded_record_count = 0
    disposition_counts: Counter[str] = Counter()

    try:
        with open(temporary_file, "w", encoding="utf-8", newline="\n") as output:
            for record_position, record in enumerate(dataset):
                if max_records is not None and matching_count >= max_records:
                    break
                scanned_count += 1
                if not isinstance(record, Mapping):
                    errors.append(f"Record position {record_position}: record is not an object")
                    continue
                if requested_languages is not None:
                    if language_field not in record or not _language_matches(record.get(language_field), requested_languages):
                        filtered_count += 1
                        continue
                matching_count += 1
                source_id = _source_record_id(record, record_position)

                text = record.get(text_field)
                if not isinstance(text, str) or not text.strip():
                    errors.append(f"Source record {source_id!r}: {text_field} must be a non-empty string")
                    continue
                if entities_field not in record or record.get(entities_field) is None:
                    errors.append(f"Source record {source_id!r}: missing {entities_field}")
                    continue
                try:
                    source_spans = parse_source_spans(record[entities_field], entities_field)
                except ValueError as exc:
                    errors.append(f"Source record {source_id!r}: {exc}")
                    continue

                candidates: List[Dict[str, Any]] = []
                excluded_spans: List[Dict[str, Any]] = []
                for source_span in source_spans:
                    if not isinstance(source_span.source_label, str) or not source_span.source_label.strip():
                        excluded_spans.append(
                            _excluded_span(
                                source_span,
                                policy,
                                None,
                                "invalid_source_span",
                                "missing_or_non_string_source_label",
                                True,
                            )
                        )
                        continue

                    mapped_label, explicit_mapping = policy.resolve(source_span.source_label, valid_labels)
                    source_subtype = policy.source_subtype(source_span.source_label)
                    offsets_are_ints = (
                        isinstance(source_span.start, Integral)
                        and not isinstance(source_span.start, bool)
                        and isinstance(source_span.end, Integral)
                        and not isinstance(source_span.end, bool)
                    )
                    if not offsets_are_ints:
                        excluded_spans.append(
                            _excluded_span(
                                source_span,
                                policy,
                                mapped_label,
                                "invalid_source_span",
                                "offsets_are_not_integers",
                                True,
                            )
                        )
                        continue

                    start, end = int(source_span.start), int(source_span.end)
                    if start < 0 or start >= end or end > len(text):
                        excluded_spans.append(
                            _excluded_span(
                                source_span,
                                policy,
                                mapped_label,
                                "invalid_source_span",
                                "offsets_outside_source_text",
                                True,
                                source_text_length=len(text),
                            )
                        )
                        continue
                    if isinstance(source_span.annotated_text, str) and text[start:end] != source_span.annotated_text:
                        excluded_spans.append(
                            _excluded_span(
                                source_span,
                                policy,
                                mapped_label,
                                "invalid_source_span",
                                "annotated_text_does_not_match_offsets",
                                True,
                            )
                        )
                        continue

                    if mapped_label is None:
                        category, reason, blocks_training = _unmapped_disposition(
                            source_subtype, policy, explicit_mapping
                        )
                        excluded_spans.append(
                            _excluded_span(
                                source_span,
                                policy,
                                None,
                                category,
                                reason,
                                blocks_training,
                            )
                        )
                        continue
                    if mapped_label not in valid_labels:
                        excluded_spans.append(
                            _excluded_span(
                                source_span,
                                policy,
                                mapped_label,
                                "invalid_label_mapping",
                                "mapped_label_is_not_in_frozen_ml_v1",
                                True,
                            )
                        )
                        continue

                    candidates.append(
                        {
                            "start": start,
                            "end": end,
                            "label": mapped_label,
                            "source_span": source_span,
                            "provenance": _provenance(source_span, policy, mapped_label),
                        }
                    )

                entities, entity_provenance, overlap_exclusions = resolve_mapped_overlaps(
                    candidates, label_order
                )
                excluded_spans.extend(overlap_exclusions)
                excluded_spans.sort(
                    key=lambda item: (
                        item.get("start", sys.maxsize),
                        item.get("end", sys.maxsize),
                        item.get("source_span_index", sys.maxsize),
                        item.get("source_label", ""),
                    )
                )
                meta = _build_metadata(
                    record,
                    record_position,
                    dataset_name,
                    split,
                    text_field,
                    entities_field,
                    language_field,
                    entity_provenance,
                    excluded_spans,
                )
                if meta["training_exclusion"]:
                    excluded_record_count += 1
                disposition_counts.update(span["category"] for span in excluded_spans)
                output.write(json.dumps({"text": text, "entities": entities, "meta": meta}, ensure_ascii=False))
                output.write("\n")
                converted_count += 1

        if converted_count == 0:
            temporary_file.unlink(missing_ok=True)
            if matching_count == 0:
                errors.append(
                    f"No records matched --language {language!r}; available language values were not converted"
                )
            else:
                errors.append("No records were converted")
        else:
            temporary_file.replace(output_file)
    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise

    print(
        f"Converted {converted_count}/{matching_count} matching records; "
        f"scanned={scanned_count}, language_filtered={filtered_count}, "
        f"training_excluded={excluded_record_count}"
    )
    print(f"Excluded span dispositions: {dict(sorted(disposition_counts.items()))}")
    print(f"Output: {output_file}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    return converted_count, matching_count, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Gretel data to SecureLogX ML-v1 JSONL")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset ID")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--data-file",
        help="Optional local Parquet snapshot; bypasses network/cache resolution",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Case-insensitive language name/code, comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--label-map",
        default="configs/securelogx_label_map.yml",
        help="Source-to-ML-v1 label map YAML",
    )
    parser.add_argument(
        "--labels",
        default="configs/securelogx_labels.json",
        help="Frozen ML-v1 label JSON",
    )
    parser.add_argument(
        "--out",
        default="data/processed/gretel_finance_securelogx.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Maximum matching records")
    parser.add_argument("--inspect", action="store_true", help="Inspect safe schema details and exit")
    args = parser.parse_args()

    if args.inspect:
        inspection = inspect_gretel_dataset(
            args.split, args.dataset, args.language, args.data_file
        )
        print(json.dumps(inspection, indent=2, ensure_ascii=False, default=str))
        has_error = bool(inspection.get("error") or inspection.get("errors"))
        raise SystemExit(1 if has_error else 0)

    try:
        policy = load_label_policy(args.label_map)
        entity_order, valid_labels = load_label_spec(args.labels)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR loading configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    converted, _, errors = convert_gretel_to_securelogx(
        dataset_name=args.dataset,
        split=args.split,
        language=args.language,
        label_map=policy.mapping,
        valid_labels=valid_labels,
        output_path=args.out,
        max_records=args.max_records,
        label_policy=policy,
        label_order=entity_order,
        data_file=args.data_file,
    )
    raise SystemExit(0 if converted > 0 and not errors else 1)


if __name__ == "__main__":
    main()
