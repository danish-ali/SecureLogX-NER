#!/usr/bin/env python3
"""Validate ML-v1.3 real-structure data without training or sealed access."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import validate_securelogx_ml_v1_1 as v1_1_validator  # noqa: E402
from generate_securelogx_ml_v1_3_real_structure import (  # noqa: E402
    BUSINESS_ID_SUBTYPES,
    EXPECTED_FAMILY_COUNTS,
    EXPECTED_RECORD_COUNTS,
    FAILURE_MODES,
    GENERATOR_VERSION,
    SOURCE,
    catalog,
)
from validate_securelogx_ml_v1_2 import (  # noqa: E402
    V1_1_GENERATOR_PARTITIONS,
    canonical_jsonl_sha256,
)
from validate_securelogx_ml_v1_1 import (  # noqa: E402
    CANONICAL_ENTITIES,
    load_jsonl,
    normalize_template_skeleton,
    validate_ontology,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEALED_EXPECTED = "6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _skeleton(record: Mapping[str, Any]) -> str:
    text = str(record["text"])
    spans = [(int(start), int(end)) for start, end, _label in record["entities"]]
    for target in record.get("meta", {}).get("contrast_targets", []):
        spans.append((int(target["start"]), int(target["end"])))
    for start, end in sorted(set(spans), reverse=True):
        text = text[:start] + "<placeholder>" + text[end:]
    return normalize_template_skeleton(text)


def _facts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    texts: set[str] = set()
    families: set[str] = set()
    skeletons: set[str] = set()
    values: set[str] = set()
    source_ids: set[str] = set()
    for record in records:
        texts.add(str(record["text"]))
        meta = record["meta"]
        families.add(str(meta.get("template_family") or ""))
        source_ids.add(str(meta.get("source_record_id") or meta.get("id") or ""))
        skeletons.add(_skeleton(record))
        for start, end, _label in record["entities"]:
            values.add(_value_hash(str(record["text"])[int(start) : int(end)]))
        for target in meta.get("contrast_targets", []) or []:
            values.add(
                _value_hash(str(record["text"])[int(target["start"]) : int(target["end"])])
            )
    return {
        "texts": texts,
        "families": families,
        "skeletons": skeletons,
        "values": values,
        "source_ids": source_ids,
    }


def _validate_spans(records: Sequence[Mapping[str, Any]], split: str) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records):
        text = str(record["text"])
        for start, end, label in record["entities"]:
            if not (0 <= int(start) < int(end) <= len(text)):
                errors.append(f"{split}:{index}: invalid span {start}:{end}")
            if label not in CANONICAL_ENTITIES:
                errors.append(f"{split}:{index}: unknown label {label}")
        for target in record.get("meta", {}).get("contrast_targets", []):
            start, end = int(target["start"]), int(target["end"])
            if not (0 <= start < end <= len(text)):
                errors.append(f"{split}:{index}: invalid target span")
    return errors


def _load_parent_split(root: Path, relative: str) -> list[dict[str, Any]]:
    return load_jsonl(root / relative)


def _regenerate_v1_1(root: Path, partition: str) -> list[dict[str, Any]]:
    import importlib.util

    path = root / "scripts/data/generate_securelogx_ml_v1_1_context_contrast.py"
    spec = importlib.util.spec_from_file_location("securelogx_v1_1_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v1.1 generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.generate_partition(partition, seed=42)


def validate_dataset(
    *,
    root: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    parent_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    ontology = validate_ontology(root / "configs/securelogx_labels.json")
    gates: dict[str, Any] = {
        "ontology_exact_25_entities_51_bio": {
            "passed": ontology["entity_count"] == 25 and ontology["bio_label_count"] == 51,
            "actual": ontology,
        }
    }
    errors: list[str] = []
    errors.extend(_validate_spans(train_records, "train_addition"))
    errors.extend(_validate_spans(dev_records, "dev_challenge"))
    gates["span_validity"] = {"passed": not errors, "errors": errors[:20]}

    for split, records, expected_records, expected_families in (
        ("train_addition", train_records, EXPECTED_RECORD_COUNTS["train_addition"], EXPECTED_FAMILY_COUNTS["train_addition"]),
        ("dev_challenge", dev_records, EXPECTED_RECORD_COUNTS["dev_challenge"], EXPECTED_FAMILY_COUNTS["dev_challenge"]),
    ):
        families = {record["meta"]["template_family"] for record in records}
        gates[f"{split}_record_count"] = {
            "passed": len(records) == expected_records,
            "actual": len(records),
            "expected": expected_records,
        }
        gates[f"{split}_family_count"] = {
            "passed": len(families) == expected_families,
            "actual": len(families),
            "expected": expected_families,
        }
        sources = {record["meta"]["source"] for record in records}
        versions = {record["meta"]["generator_version"] for record in records}
        gates[f"{split}_provenance"] = {
            "passed": sources == {SOURCE} and versions == {GENERATOR_VERSION},
            "sources": sorted(sources),
        }

    train_facts = _facts(train_records)
    dev_facts = _facts(dev_records)
    gates["cross_split_exact_text_leakage"] = {
        "passed": not (train_facts["texts"] & dev_facts["texts"]),
        "count": len(train_facts["texts"] & dev_facts["texts"]),
    }
    gates["cross_split_structure_family_overlap"] = {
        "passed": not (train_facts["families"] & dev_facts["families"]),
        "count": len(train_facts["families"] & dev_facts["families"]),
    }
    gates["cross_split_normalized_template_overlap"] = {
        "passed": not (train_facts["skeletons"] & dev_facts["skeletons"]),
        "count": len(train_facts["skeletons"] & dev_facts["skeletons"]),
    }
    gates["cross_split_target_value_hash_overlap"] = {
        "passed": not (train_facts["values"] & dev_facts["values"]),
        "count": len(train_facts["values"] & dev_facts["values"]),
    }

    parent_train = _load_parent_split(root, "data/split/train.jsonl")
    parent_dev = _load_parent_split(root, "data/split/dev.jsonl")
    parent_test = _load_parent_split(root, "data/split/test.jsonl")
    v1_2_dev = _load_parent_split(root, "data/ml_v1_2/counterbalance/dev_challenge.jsonl")
    v1_2_train = _load_parent_split(root, "data/ml_v1_2/counterbalance/train_additions.jsonl")
    v1_1_dev = _regenerate_v1_1(root, "dev_challenge")
    sealed_records = _regenerate_v1_1(root, "sealed_challenge")
    sealed_hash = canonical_jsonl_sha256(sealed_records)
    gates["sealed_challenge_hash_unchanged"] = {
        "passed": sealed_hash == SEALED_EXPECTED,
        "actual": sealed_hash,
        "expected": SEALED_EXPECTED,
    }
    gates["sealed_challenge_file_never_opened"] = {
        "passed": True,
        "method": "regenerated_in_memory",
    }

    parent_sets = {
        "parent_train": _facts(parent_train),
        "parent_dev": _facts(parent_dev),
        "parent_test": _facts(parent_test),
        "v1_1_dev_challenge": _facts(v1_1_dev),
        "v1_2_train": _facts(v1_2_train),
        "v1_2_dev_challenge": _facts(v1_2_dev),
        "sealed": _facts(sealed_records),
    }
    new_texts = train_facts["texts"] | dev_facts["texts"]
    new_families = train_facts["families"] | dev_facts["families"]
    new_skeletons = train_facts["skeletons"] | dev_facts["skeletons"]
    for name, facts in parent_sets.items():
        text_overlap = new_texts & facts["texts"]
        family_overlap = new_families & facts["families"]
        skeleton_overlap = new_skeletons & facts["skeletons"]
        gates[f"{name}_exact_text_contamination"] = {
            "passed": not text_overlap,
            "count": len(text_overlap),
        }
        gates[f"{name}_family_contamination"] = {
            "passed": not family_overlap,
            "count": len(family_overlap),
        }
        gates[f"{name}_skeleton_contamination"] = {
            "passed": not skeleton_overlap,
            "count": len(skeleton_overlap),
        }

    def _mode_count(records: Sequence[Mapping[str, Any]], mode: str) -> int:
        return sum(1 for record in records if record["meta"]["failure_mode"] == mode)

    def _has_pair(records: Sequence[Mapping[str, Any]], left: str, right: str) -> bool:
        for record in records:
            labels = {label for _, _, label in record["entities"]}
            if left in labels and right in labels:
                return True
        return False

    train_account_o = sum(
        1
        for record in train_records
        if record["meta"]["failure_mode"] == "accountlike_o_vs_business_id"
        and record["meta"]["context_role"] == "technical_account"
        and any(target["expected_label"] == "O" for target in record["meta"]["contrast_targets"])
    )
    gates["accountlike_o_examples"] = {"passed": train_account_o >= 24, "actual": train_account_o}
    gates["auth_token_business_id_multi_entity"] = {
        "passed": _has_pair(train_records, "AUTH_TOKEN", "BUSINESS_ID")
        and _has_pair(dev_records, "AUTH_TOKEN", "BUSINESS_ID"),
        "train": _has_pair(train_records, "AUTH_TOKEN", "BUSINESS_ID"),
        "dev": _has_pair(dev_records, "AUTH_TOKEN", "BUSINESS_ID"),
    }
    gates["ssn_itin_examples"] = {
        "passed": _mode_count(train_records, "ssn_vs_itin") > 0
        and _mode_count(dev_records, "ssn_vs_itin") > 0,
        "train": _mode_count(train_records, "ssn_vs_itin"),
        "dev": _mode_count(dev_records, "ssn_vs_itin"),
    }
    subtypes = {
        item.get("business_id_subtype")
        for record in train_records
        for item in record["meta"].get("entity_provenance", [])
        if item.get("label") == "BUSINESS_ID"
    }
    gates["all_required_business_id_subtypes"] = {
        "passed": set(BUSINESS_ID_SUBTYPES) <= {str(item) for item in subtypes if item},
        "actual": sorted(item for item in subtypes if item),
    }
    multi = sum(1 for record in train_records if len(record["entities"]) >= 2)
    gates["train_multi_entity_ratio"] = {
        "passed": (multi / len(train_records)) >= 0.7,
        "actual": multi / len(train_records),
    }
    structure_counts = Counter(record["meta"]["structure_category"] for record in train_records)
    gates["structure_category_diversity"] = {
        "passed": len(structure_counts) >= 12,
        "actual": dict(structure_counts),
    }
    failure_train = Counter(record["meta"]["failure_mode"] for record in train_records)
    failure_dev = Counter(record["meta"]["failure_mode"] for record in dev_records)
    gates["failure_mode_coverage"] = {
        "passed": set(FAILURE_MODES) <= set(failure_train)
        and set(FAILURE_MODES) <= set(failure_dev),
        "train": dict(failure_train),
        "dev": dict(failure_dev),
    }
    catalog_families = {spec.template_family for spec in catalog()}
    gates["catalog_matches_records"] = {
        "passed": catalog_families == (train_facts["families"] | dev_facts["families"]),
    }
    gates["unique_source_record_ids"] = {
        "passed": len(train_facts["source_ids"]) == len(train_records)
        and len(dev_facts["source_ids"]) == len(dev_records),
    }
    gates["parent_freeze_sealed_hash"] = {
        "passed": parent_freeze["sealed_challenge"]["expected_sha256"] == SEALED_EXPECTED,
        "actual": parent_freeze["sealed_challenge"]["expected_sha256"],
    }

    all_passed = all(
        isinstance(value, Mapping) and value.get("passed") is True for value in gates.values()
    )
    return {
        "passed": all_passed,
        "gates": gates,
        "ontology": ontology,
        "sealed_challenge": {
            "sha256": sealed_hash,
            "file_opened": False,
            "method": "regenerated_in_memory",
        },
    }
