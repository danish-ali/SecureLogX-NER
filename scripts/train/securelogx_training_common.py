"""Shared, deterministic training and span-evaluation utilities for SecureLogX ML-v1.

The frozen datasets use half-open character spans.  BERT is trained on BIO
token labels produced from fast-tokenizer offset mappings.  Long records are
handled with overlapping windows so an entity is never silently discarded by
ordinary right truncation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


EXPECTED_ENTITIES: tuple[str, ...] = (
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

HIGH_RISK_ENTITIES: frozenset[str] = frozenset(
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

IGNORE_INDEX = -100

EXPECTED_FROZEN_MANIFEST: dict[str, Any] = {
    "manifest_version": 1,
    "ontology": "SecureLogX ML-v1",
    "frozen_at": "2026-09-13",
    "files": {
        "configs/securelogx_labels.json": {
            "sha256": "60361655594004a36c154725a79d43f64896014e8f8b445552b57305b3ba24e5"
        },
        "configs/securelogx_label_map.yml": {
            "sha256": "31d8fe82f829a67642ce3c13861c74ff972f0dfd7c063948437d32cdaa67bc41"
        },
        "configs/training_profile.yml": {
            "sha256": "e1ab0f70136fd6a3cb472714a8926c01c7b854acc2b52a29e813b431b7d435dd"
        },
        "configs/ml_v1_onnx_validation_gate.json": {
            "sha256": "1d8249e1d4f052d0ba4cb64c607e6178d75f34ae160af17e0060907d02e56176"
        },
        "data/split/train.jsonl": {
            "sha256": "b36926d148c3672c463f9530aac77f78f4ccd6e864f2cd70b8c8577a766a83b5",
            "records": 23558,
            "entity_spans": 32203,
        },
        "data/split/dev.jsonl": {
            "sha256": "126d65a63e0b37bff8462b2f0f56e8cdceab2ec54ad97b868fb1981a4bc8053a",
            "records": 2994,
            "entity_spans": 4020,
        },
        "data/split/test.jsonl": {
            "sha256": "32c0dc0a259fbc4c51c4a7ce74e3258dedb18ed5dcc48898e182d0991b06fbb7",
            "records": 2989,
            "entity_spans": 4358,
        },
    },
    "totals": {"records": 29541, "entity_spans": 40581},
    "independence_gates": {
        "exact_text_leakage": 0,
        "synthetic_template_family_leakage": 0,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_canonical_labels(path: Path) -> dict[str, Any]:
    config = load_json(path)
    entities = config.get("entities")
    bio_labels = config.get("bio_labels")
    if entities != list(EXPECTED_ENTITIES):
        raise ValueError(
            "Canonical entity mismatch: expected the exact ordered SecureLogX "
            f"ML-v1 ontology, received {entities!r}"
        )
    expected_bio = ["O"] + [
        bio_label
        for entity in EXPECTED_ENTITIES
        for bio_label in (f"B-{entity}", f"I-{entity}")
    ]
    if bio_labels != expected_bio:
        raise ValueError("Canonical BIO label order does not match ML-v1")
    expected_label_to_id = {label: index for index, label in enumerate(expected_bio)}
    expected_id_to_label = {
        str(index): label for index, label in enumerate(expected_bio)
    }
    if config.get("label_to_id") != expected_label_to_id:
        raise ValueError("label_to_id is not the deterministic ML-v1 mapping")
    if config.get("id_to_label") != expected_id_to_label:
        raise ValueError("id_to_label is not the inverse ML-v1 mapping")
    if len(entities) != 25 or len(bio_labels) != 51:
        raise ValueError("ML-v1 must contain exactly 25 entities and 51 BIO labels")
    if config["label_to_id"].get("O") != 0:
        raise ValueError("O must have label ID 0")
    if config["label_to_id"].get("I-API_KEY") != 50:
        raise ValueError("I-API_KEY must have label ID 50")
    return config


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            yield record


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def validate_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest != EXPECTED_FROZEN_MANIFEST:
        raise ValueError(
            "Frozen-manifest contract mismatch: the manifest must exactly match "
            "the protected SecureLogX ML-v1 file set, hashes, counts, totals, and "
            "zero-leakage declarations"
        )
    mismatches: list[str] = []
    for relative_path, expected in manifest.get("files", {}).items():
        path = root / relative_path
        if not path.is_file():
            mismatches.append(f"missing file: {relative_path}")
            continue
        actual_hash = sha256_file(path)
        expected_hash = str(expected.get("sha256", "")).lower()
        if actual_hash.lower() != expected_hash:
            mismatches.append(
                f"SHA-256 mismatch for {relative_path}: expected {expected_hash}, "
                f"received {actual_hash}"
            )
        if "records" in expected or "entity_spans" in expected:
            record_count = 0
            span_count = 0
            for record in iter_jsonl(path):
                record_count += 1
                entities = record.get("entities")
                if not isinstance(entities, list):
                    mismatches.append(f"{relative_path}: record {record_count} has no entity list")
                    break
                span_count += len(entities)
            if record_count != expected.get("records"):
                mismatches.append(
                    f"record-count mismatch for {relative_path}: expected "
                    f"{expected.get('records')}, received {record_count}"
                )
            if span_count != expected.get("entity_spans"):
                mismatches.append(
                    f"span-count mismatch for {relative_path}: expected "
                    f"{expected.get('entity_spans')}, received {span_count}"
                )
    if mismatches:
        raise ValueError("Frozen-manifest validation failed:\n- " + "\n- ".join(mismatches))
    return manifest


def audit_records(
    records: Sequence[Mapping[str, Any]],
    split: str,
    canonical_entities: Sequence[str],
) -> dict[str, Any]:
    allowed = set(canonical_entities)
    errors: list[str] = []
    texts: set[str] = set()
    families: set[str] = set()
    external_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    span_count = 0
    negative_records = 0
    source_counts: Counter[str] = Counter()

    for index, record in enumerate(records):
        prefix = f"{split} record {index}"
        text = record.get("text")
        entities = record.get("entities")
        meta = record.get("meta")
        if not isinstance(text, str) or not text:
            errors.append(f"{prefix}: text must be non-empty")
            continue
        if text in texts:
            errors.append(f"{prefix}: duplicate text within split")
        texts.add(text)
        if not isinstance(entities, list):
            errors.append(f"{prefix}: entities must be a list")
            continue
        if not isinstance(meta, dict):
            errors.append(f"{prefix}: meta must be an object")
            continue
        if meta.get("training_exclusion") is True:
            errors.append(f"{prefix}: training-excluded record present in frozen split")
        source = str(meta.get("source", ""))
        source_counts[source] += 1
        family = meta.get("template_family")
        if source == "securelogx_custom_logs":
            if not isinstance(family, str) or not family:
                errors.append(f"{prefix}: synthetic record lacks template_family")
            else:
                families.add(family)
        else:
            source_record_id = meta.get("source_record_id")
            if source_record_id is None:
                errors.append(f"{prefix}: external record lacks source_record_id")
            else:
                stable_id = f"{source}:{source_record_id}"
                if stable_id in external_ids:
                    errors.append(f"{prefix}: duplicate external source_record_id")
                external_ids.add(stable_id)

        previous_end = -1
        seen_spans: set[tuple[int, int, str]] = set()
        for entity in entities:
            if (
                not isinstance(entity, list)
                or len(entity) != 3
                or isinstance(entity[0], bool)
                or isinstance(entity[1], bool)
                or not isinstance(entity[0], int)
                or not isinstance(entity[1], int)
                or not isinstance(entity[2], str)
            ):
                errors.append(f"{prefix}: malformed entity {entity!r}")
                continue
            start, end, label = entity
            signature = (start, end, label)
            if signature in seen_spans:
                errors.append(f"{prefix}: duplicate entity {signature!r}")
            seen_spans.add(signature)
            if not (0 <= start < end <= len(text)):
                errors.append(f"{prefix}: invalid offsets {signature!r}")
            if start < previous_end:
                errors.append(f"{prefix}: overlapping/unsorted entity {signature!r}")
            previous_end = max(previous_end, end)
            if label not in allowed:
                errors.append(f"{prefix}: noncanonical entity label {label}")
            label_counts[label] += 1
            span_count += 1
        if not entities:
            negative_records += 1

        provenance = meta.get("entity_provenance")
        if not isinstance(provenance, list):
            errors.append(f"{prefix}: entity_provenance must be a list")
        else:
            provenance_spans = {
                (entry.get("start"), entry.get("end"), entry.get("label"))
                for entry in provenance
                if isinstance(entry, dict)
            }
            if provenance_spans != seen_spans:
                errors.append(f"{prefix}: provenance does not match entities")

    return {
        "split": split,
        "records": len(records),
        "entity_spans": span_count,
        "negative_records": negative_records,
        "label_counts": dict(label_counts),
        "source_counts": dict(source_counts),
        "texts": texts,
        "template_families": families,
        "external_ids": external_ids,
        "errors": errors,
    }


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.int64), q))


def summarize_lengths(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "records": len(values),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "maximum": max(values, default=0),
        "over_128": sum(value > 128 for value in values),
        "over_256": sum(value > 256 for value in values),
        "over_384": sum(value > 384 for value in values),
        "over_512": sum(value > 512 for value in values),
    }


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def collect_environment(seed: int, base_model: str, revision: str) -> dict[str, Any]:
    import accelerate
    import datasets
    import psutil
    import tokenizers
    import torch
    import transformers

    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda_available else None
        ),
        "bf16_supported": (
            bool(torch.cuda.is_bf16_supported()) if cuda_available else False
        ),
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "base_model": base_model,
        "base_model_revision": revision,
    }


@dataclass
class AlignmentBuild:
    windows: list[dict[str, Any]]
    summary: dict[str, Any]


def _nonspace_is_covered(
    text: str, start: int, end: int, offsets: Sequence[tuple[int, int]]
) -> bool:
    for position in range(start, end):
        if text[position].isspace():
            continue
        if not any(token_start <= position < token_end for token_start, token_end in offsets):
            return False
    return True


def align_record_windows(
    record: Mapping[str, Any],
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    max_length: int,
    stride: int,
    record_index: int,
    max_boundary_adjustment_chars: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Tokenize one record into overlapping windows and align every gold span.

    A character span whose boundaries fall inside a WordPiece is expanded or
    contracted deterministically to the overlapping token boundary and reported
    as ``boundary_adjusted``.  Partial spans in a particular overflow window are
    assigned -100 in that window, preventing them from becoming false O labels.
    """

    text = str(record["text"])
    entities = [tuple(entity) for entity in record["entities"]]
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    input_windows = encoded["input_ids"]
    if input_windows and isinstance(input_windows[0], int):
        input_windows = [input_windows]
    attention_windows = encoded["attention_mask"]
    if attention_windows and isinstance(attention_windows[0], int):
        attention_windows = [attention_windows]
    offset_windows = encoded["offset_mapping"]
    if offset_windows and offset_windows[0] and isinstance(offset_windows[0][0], int):
        offset_windows = [offset_windows]
    token_type_windows = encoded.get("token_type_ids")
    if token_type_windows and isinstance(token_type_windows[0], int):
        token_type_windows = [token_type_windows]

    span_state: list[dict[str, Any]] = [
        {
            "record_index": record_index,
            "record_id": str(record.get("meta", {}).get("source_record_id", record_index)),
            "source": str(record.get("meta", {}).get("source", "")),
            "template_family": record.get("meta", {}).get("template_family"),
            "start": int(start),
            "end": int(end),
            "label": str(label),
            "aligned_windows": 0,
            "seen_overlap": False,
            "boundary_adjusted": False,
            "adjusted_boundaries": set(),
            "collision": False,
            "excessive_boundary_adjustment": False,
        }
        for start, end, label in entities
    ]
    windows: list[dict[str, Any]] = []
    supervised_span_instances = 0

    for window_index, (input_ids, attention_mask, raw_offsets) in enumerate(
        zip(input_windows, attention_windows, offset_windows)
    ):
        offsets = [(int(start), int(end)) for start, end in raw_offsets]
        labels = [
            IGNORE_INDEX if start == end else int(label_to_id["O"])
            for start, end in offsets
        ]
        overlap_by_span: dict[int, list[int]] = {}
        token_to_spans: MutableMapping[int, set[int]] = defaultdict(set)
        fully_covered: set[int] = set()

        for span_index, (start, end, _label) in enumerate(entities):
            token_indices = [
                token_index
                for token_index, (token_start, token_end) in enumerate(offsets)
                if token_start != token_end and token_start < end and token_end > start
            ]
            if not token_indices:
                continue
            span_state[span_index]["seen_overlap"] = True
            overlap_by_span[span_index] = token_indices
            relevant_offsets = [offsets[token_index] for token_index in token_indices]
            if _nonspace_is_covered(text, int(start), int(end), relevant_offsets):
                fully_covered.add(span_index)
            for token_index in token_indices:
                token_to_spans[token_index].add(span_index)

        colliding_spans: set[int] = set()
        for span_indexes in token_to_spans.values():
            if len(span_indexes) > 1:
                colliding_spans.update(span_indexes)
        for span_index in colliding_spans:
            span_state[span_index]["collision"] = True

        # Ignore partial or ambiguous entity tokens in this window so they are
        # never trained as O merely because of overflow boundaries.
        for span_index, token_indices in overlap_by_span.items():
            if span_index not in fully_covered or span_index in colliding_spans:
                for token_index in token_indices:
                    labels[token_index] = IGNORE_INDEX

        for span_index in sorted(fully_covered - colliding_spans):
            start, end, label = entities[span_index]
            token_indices = overlap_by_span[span_index]
            adjusted = (offsets[token_indices[0]][0], offsets[token_indices[-1]][1])
            adjustment_size = abs(int(start) - adjusted[0]) + abs(int(end) - adjusted[1])
            if adjustment_size > max_boundary_adjustment_chars:
                span_state[span_index]["excessive_boundary_adjustment"] = True
                span_state[span_index]["adjusted_boundaries"].add(adjusted)
                for token_index in token_indices:
                    labels[token_index] = IGNORE_INDEX
                continue
            begin_id = int(label_to_id[f"B-{label}"])
            inside_id = int(label_to_id[f"I-{label}"])
            for position, token_index in enumerate(token_indices):
                if labels[token_index] not in {int(label_to_id["O"]), IGNORE_INDEX}:
                    span_state[span_index]["collision"] = True
                    break
                labels[token_index] = begin_id if position == 0 else inside_id
            else:
                span_state[span_index]["aligned_windows"] += 1
                span_state[span_index]["adjusted_boundaries"].add(adjusted)
                if adjusted != (start, end):
                    span_state[span_index]["boundary_adjusted"] = True
                supervised_span_instances += 1

        window: dict[str, Any] = {
            "input_ids": [int(value) for value in input_ids],
            "attention_mask": [int(value) for value in attention_mask],
            "labels": labels,
            "offset_mapping": offsets,
            "record_index": record_index,
            "window_index": window_index,
        }
        if token_type_windows is not None:
            window["token_type_ids"] = [
                int(value) for value in token_type_windows[window_index]
            ]
        windows.append(window)

    diagnostics: list[dict[str, Any]] = []
    for state in span_state:
        adjusted_boundaries = sorted(state.pop("adjusted_boundaries"))
        state["adjusted_boundaries"] = [list(value) for value in adjusted_boundaries]
        if state["aligned_windows"]:
            state["status"] = (
                "boundary_adjusted" if state["boundary_adjusted"] else "aligned"
            )
            state["reason"] = (
                "character_boundary_inside_or_outside_token"
                if state["boundary_adjusted"]
                else "exact_token_boundaries"
            )
        elif state["collision"]:
            state["status"] = "failed"
            state["reason"] = "multiple_gold_spans_share_a_wordpiece"
        elif state["excessive_boundary_adjustment"]:
            state["status"] = "failed"
            state["reason"] = "token_boundary_adjustment_exceeds_three_characters"
        elif state["seen_overlap"]:
            state["status"] = "truncated"
            state["reason"] = "no_overflow_window_fully_covers_nonspace_span_text"
        else:
            state["status"] = "failed"
            state["reason"] = "tokenizer_emitted_no_token_for_span"
        diagnostics.append(state)

    return windows, diagnostics, supervised_span_instances


def build_aligned_features(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    max_length: int,
    stride: int,
) -> AlignmentBuild:
    windows: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    supervised_span_instances = 0
    for record_index, record in enumerate(records):
        record_windows, diagnostics, span_instances = align_record_windows(
            record,
            tokenizer,
            label_to_id,
            max_length,
            stride,
            record_index,
        )
        windows.extend(record_windows)
        all_diagnostics.extend(diagnostics)
        supervised_span_instances += span_instances

    status_counts = Counter(item["status"] for item in all_diagnostics)
    reason_counts = Counter(item["reason"] for item in all_diagnostics)
    affected_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for item in all_diagnostics:
        if item["status"] != "aligned":
            affected_labels[item["label"]][item["status"]] += 1
    summary = {
        "records": len(records),
        "windows": len(windows),
        "max_length": max_length,
        "stride": stride,
        "total_spans": len(all_diagnostics),
        "successfully_aligned_spans": status_counts["aligned"]
        + status_counts["boundary_adjusted"],
        "exact_boundary_spans": status_counts["aligned"],
        "boundary_adjusted_spans": status_counts["boundary_adjusted"],
        "truncated_spans": status_counts["truncated"],
        "failed_alignments": status_counts["failed"],
        "supervised_span_instances_across_windows": supervised_span_instances,
        "reason_counts": dict(reason_counts),
        "affected_labels": {
            label: dict(counts) for label, counts in sorted(affected_labels.items())
        },
        "diagnostics": [
            item for item in all_diagnostics if item["status"] != "aligned"
        ],
    }
    return AlignmentBuild(windows=windows, summary=summary)


class AlignedWindowDataset:
    def __init__(self, windows: Sequence[Mapping[str, Any]]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.windows[index]


class TokenClassificationCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            maximum = int(
                math.ceil(maximum / self.pad_to_multiple_of) * self.pad_to_multiple_of
            )
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        token_type_ids: list[list[int]] = []
        labels: list[list[int]] = []
        offsets: list[list[tuple[int, int]]] = []
        has_token_type_ids = any("token_type_ids" in feature for feature in features)

        for feature in features:
            length = len(feature["input_ids"])
            padding = maximum - length
            input_ids.append(list(feature["input_ids"]) + [pad_id] * padding)
            attention_mask.append(list(feature["attention_mask"]) + [0] * padding)
            labels.append(list(feature["labels"]) + [IGNORE_INDEX] * padding)
            offsets.append(list(feature["offset_mapping"]) + [(0, 0)] * padding)
            if has_token_type_ids:
                token_type_ids.append(
                    list(feature.get("token_type_ids", [0] * length)) + [0] * padding
                )

        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "offset_mapping": offsets,
            "record_indices": [int(feature["record_index"]) for feature in features],
            "window_indices": [int(feature["window_index"]) for feature in features],
        }
        if has_token_type_ids:
            batch["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
        return batch


def model_inputs_from_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    inputs = {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }
    if "token_type_ids" in batch:
        inputs["token_type_ids"] = batch["token_type_ids"].to(
            device, non_blocking=True
        )
    return inputs


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def decode_bio_spans(
    offsets: Sequence[tuple[int, int]],
    label_ids: Sequence[int],
    confidences: Sequence[float],
    id_to_label: Mapping[int, str],
) -> tuple[list[dict[str, Any]], int]:
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    invalid_transitions = 0

    def close_current() -> None:
        nonlocal current
        if current is not None:
            token_confidences = current.pop("token_confidences")
            current["confidence"] = float(statistics.fmean(token_confidences))
            current["minimum_token_confidence"] = float(min(token_confidences))
            spans.append(current)
            current = None

    for (start, end), label_id, confidence in zip(offsets, label_ids, confidences):
        label = id_to_label[int(label_id)]
        if label == "O":
            close_current()
            continue
        prefix, entity = label.split("-", 1)
        if prefix == "B":
            close_current()
            current = {
                "start": int(start),
                "end": int(end),
                "label": entity,
                "token_confidences": [float(confidence)],
            }
        elif prefix == "I" and current is not None and current["label"] == entity:
            current["end"] = int(end)
            current["token_confidences"].append(float(confidence))
        else:
            invalid_transitions += 1
            close_current()
            current = {
                "start": int(start),
                "end": int(end),
                "label": entity,
                "token_confidences": [float(confidence)],
            }
    close_current()
    return spans, invalid_transitions


def normalize_decoded_spans(
    text: str,
    spans: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Trim tokenizer-added surrounding whitespace from decoded entity spans.

    SentencePiece tokenizers such as DeBERTa-v3 may emit offset mappings whose
    first token for an entity includes an adjacent space. Training alignment can
    legitimately supervise that token, but entity-level evaluation and runtime
    masking should refer to the actual non-whitespace character span.

    Only surrounding whitespace is removed. Internal characters and entity
    labels/confidences are unchanged.
    """
    normalized: list[dict[str, Any]] = []
    text_length = len(text)
    for span in spans:
        item = dict(span)
        start = max(0, min(text_length, int(item["start"])))
        end = max(start, min(text_length, int(item["end"])))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        item["start"] = start
        item["end"] = end
        if start < end:
            normalized.append(item)
    return normalized


def evaluate_model(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    id_to_label: Mapping[int, str],
    batch_size: int,
    device: Any,
    use_bf16: bool,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    data_loader = DataLoader(
        AlignedWindowDataset(windows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TokenClassificationCollator(tokenizer),
        pin_memory=device.type == "cuda",
    )
    accumulated: list[dict[tuple[int, int], list[Any]]] = [
        {} for _ in records
    ]
    loss_numerator = 0.0
    loss_denominator = 0
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            inputs = model_inputs_from_batch(batch, device)
            valid_tokens = int((inputs["labels"] != IGNORE_INDEX).sum().item())
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                outputs = model(**inputs)
            if outputs.loss is not None:
                loss_numerator += float(outputs.loss.detach().float().item()) * valid_tokens
                loss_denominator += valid_tokens
            logits = outputs.logits.detach().float().cpu().numpy()
            attention = batch["attention_mask"].numpy()
            for row_index, record_index in enumerate(batch["record_indices"]):
                for token_index, (start, end) in enumerate(
                    batch["offset_mapping"][row_index]
                ):
                    if not attention[row_index, token_index] or start == end:
                        continue
                    key = (int(start), int(end))
                    existing = accumulated[record_index].get(key)
                    if existing is None:
                        accumulated[record_index][key] = [
                            logits[row_index, token_index].copy(),
                            1,
                        ]
                    else:
                        existing[0] += logits[row_index, token_index]
                        existing[1] += 1

    predictions: list[list[dict[str, Any]]] = []
    token_details: list[dict[tuple[int, int], dict[str, Any]]] = []
    invalid_transitions = 0
    for record_index, token_map in enumerate(accumulated):
        ordered_offsets = sorted(token_map)
        averaged_logits = np.stack(
            [token_map[offset][0] / token_map[offset][1] for offset in ordered_offsets]
        )
        probabilities = _softmax(averaged_logits)
        predicted_ids = probabilities.argmax(axis=1).tolist()
        predicted_confidences = probabilities.max(axis=1).tolist()
        spans, invalid = decode_bio_spans(
            ordered_offsets,
            predicted_ids,
            predicted_confidences,
            id_to_label,
        )
        invalid_transitions += invalid
        spans = normalize_decoded_spans(str(records[record_index]["text"]), spans)
        predictions.append(spans)
        token_details.append(
            {
                offset: {
                    "probabilities": probabilities[position],
                    "predicted_id": int(predicted_ids[position]),
                    "confidence": float(predicted_confidences[position]),
                }
                for position, offset in enumerate(ordered_offsets)
            }
        )

    return {
        "loss": loss_numerator / loss_denominator if loss_denominator else 0.0,
        "predictions": predictions,
        "token_details": token_details,
        "invalid_bio_transitions": invalid_transitions,
    }


def _metric_values(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> int:
    return max(0, min(int(first["end"]), int(second["end"])) - max(int(first["start"]), int(second["start"])))


def _span_signature(span: Mapping[str, Any]) -> tuple[int, int, str]:
    return (int(span["start"]), int(span["end"]), str(span["label"]))


def score_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    label_order: Sequence[str],
    split: str,
    token_details: Sequence[Mapping[tuple[int, int], Mapping[str, Any]]] | None = None,
    label_to_id: Mapping[str, int] | None = None,
    record_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    if record_indices is None:
        record_indices = list(range(len(records)))
    per_label = {
        label: {"tp": 0, "predicted": 0, "support": 0} for label in label_order
    }
    error_rows: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    exact_matches = 0
    boundary_too_short = 0
    boundary_too_long = 0
    boundary_shifted = 0
    wrong_class = 0
    confidence_groups: MutableMapping[str, list[float]] = defaultdict(list)

    for local_index, (record, predicted_spans) in enumerate(zip(records, predictions)):
        global_index = int(record_indices[local_index])
        text = str(record["text"])
        meta = record.get("meta", {})
        gold_spans = [
            {"start": int(start), "end": int(end), "label": str(label)}
            for start, end, label in record["entities"]
        ]
        predicted = [dict(span) for span in predicted_spans]
        for gold in gold_spans:
            per_label[gold["label"]]["support"] += 1
        for prediction in predicted:
            per_label[prediction["label"]]["predicted"] += 1

        unmatched_gold = set(range(len(gold_spans)))
        unmatched_pred = set(range(len(predicted)))
        pred_by_signature: MutableMapping[tuple[int, int, str], list[int]] = defaultdict(list)
        for pred_index, prediction in enumerate(predicted):
            pred_by_signature[_span_signature(prediction)].append(pred_index)
        for gold_index, gold in enumerate(gold_spans):
            candidates = pred_by_signature.get(_span_signature(gold), [])
            pred_index = next(
                (candidate for candidate in candidates if candidate in unmatched_pred),
                None,
            )
            if pred_index is None:
                continue
            unmatched_gold.remove(gold_index)
            unmatched_pred.remove(pred_index)
            per_label[gold["label"]]["tp"] += 1
            exact_matches += 1
            confusion[(gold["label"], gold["label"])] += 1
            confidence_groups["correct"].append(float(predicted[pred_index]["confidence"]))

        overlap_candidates: list[tuple[float, int, int]] = []
        for gold_index in unmatched_gold:
            gold = gold_spans[gold_index]
            for pred_index in unmatched_pred:
                prediction = predicted[pred_index]
                overlap = _overlap(gold, prediction)
                if not overlap:
                    continue
                union = (
                    max(gold["end"], prediction["end"])
                    - min(gold["start"], prediction["start"])
                )
                overlap_candidates.append((overlap / union, gold_index, pred_index))
        for _iou, gold_index, pred_index in sorted(
            overlap_candidates, reverse=True
        ):
            if gold_index not in unmatched_gold or pred_index not in unmatched_pred:
                continue
            unmatched_gold.remove(gold_index)
            unmatched_pred.remove(pred_index)
            gold = gold_spans[gold_index]
            prediction = predicted[pred_index]
            if gold["label"] != prediction["label"]:
                category = "wrong entity class"
                wrong_class += 1
                confusion[(gold["label"], prediction["label"])] += 1
                confidence_groups["wrong_class"].append(float(prediction["confidence"]))
            else:
                prediction_inside_gold = (
                    prediction["start"] >= gold["start"]
                    and prediction["end"] <= gold["end"]
                )
                gold_inside_prediction = (
                    gold["start"] >= prediction["start"]
                    and gold["end"] <= prediction["end"]
                )
                if prediction_inside_gold:
                    category = "boundary-too-short"
                    boundary_too_short += 1
                    confidence_groups["boundary_too_short"].append(
                        float(prediction["confidence"])
                    )
                elif gold_inside_prediction:
                    category = "boundary-too-long"
                    boundary_too_long += 1
                    confidence_groups["boundary_too_long"].append(
                        float(prediction["confidence"])
                    )
                else:
                    category = "boundary-shifted"
                    boundary_shifted += 1
                    confidence_groups["boundary_shifted"].append(
                        float(prediction["confidence"])
                    )
                confusion[(gold["label"], "<BOUNDARY>")] += 1
            error_rows.append(
                _error_row(
                    split,
                    global_index,
                    record,
                    category,
                    gold,
                    prediction,
                    text,
                )
            )

        for gold_index in sorted(unmatched_gold):
            gold = gold_spans[gold_index]
            confusion[(gold["label"], "<MISSED>")] += 1
            gold_confidence = None
            if token_details is not None and label_to_id is not None:
                gold_confidence = _gold_span_confidence(
                    gold,
                    token_details[local_index],
                    label_to_id,
                )
                if gold_confidence is not None:
                    confidence_groups["false_negative_gold_probability"].append(
                        gold_confidence
                    )
            row = _error_row(
                split,
                global_index,
                record,
                "false negative",
                gold,
                None,
                text,
            )
            row["gold_label_probability"] = gold_confidence
            error_rows.append(row)
        for pred_index in sorted(unmatched_pred):
            prediction = predicted[pred_index]
            confusion[("<SPURIOUS>", prediction["label"])] += 1
            confidence_groups["false_positive"].append(float(prediction["confidence"]))
            error_rows.append(
                _error_row(
                    split,
                    global_index,
                    record,
                    "false positive",
                    None,
                    prediction,
                    text,
                )
            )

    per_label_metrics: dict[str, dict[str, Any]] = {}
    total_tp = total_pred = total_gold = 0
    for label in label_order:
        values = per_label[label]
        tp = int(values["tp"])
        predicted_count = int(values["predicted"])
        support = int(values["support"])
        metrics = _metric_values(tp, predicted_count - tp, support - tp)
        metrics["support"] = support
        metrics["predicted"] = predicted_count
        per_label_metrics[label] = metrics
        total_tp += tp
        total_pred += predicted_count
        total_gold += support

    micro = _metric_values(total_tp, total_pred - total_tp, total_gold - total_tp)
    macro = {
        metric: float(
            statistics.fmean(per_label_metrics[label][metric] for label in label_order)
        )
        for metric in ("precision", "recall", "f1")
    }
    high_risk_tp = sum(per_label[label]["tp"] for label in HIGH_RISK_ENTITIES)
    high_risk_pred = sum(
        per_label[label]["predicted"] for label in HIGH_RISK_ENTITIES
    )
    high_risk_gold = sum(
        per_label[label]["support"] for label in HIGH_RISK_ENTITIES
    )
    high_risk = _metric_values(
        int(high_risk_tp),
        int(high_risk_pred - high_risk_tp),
        int(high_risk_gold - high_risk_tp),
    )
    high_risk["support"] = int(high_risk_gold)

    return {
        "split": split,
        "records": len(records),
        "gold_entities": total_gold,
        "predicted_entities": total_pred,
        "exact_span_matches": exact_matches,
        "micro": micro,
        "macro": macro,
        "per_label": per_label_metrics,
        "high_risk": high_risk,
        "error_counts": {
            "false_positives": total_pred - total_tp,
            "false_negatives": total_gold - total_tp,
            "boundary_too_short": boundary_too_short,
            "boundary_too_long": boundary_too_long,
            "boundary_shifted": boundary_shifted,
            "wrong_entity_class": wrong_class,
        },
        "errors": error_rows,
        "confusion": {
            f"{gold}\t{predicted}": count
            for (gold, predicted), count in sorted(confusion.items())
        },
        "confidence": {
            name: _describe_confidence(values)
            for name, values in sorted(confidence_groups.items())
        },
    }


def _gold_span_confidence(
    gold: Mapping[str, Any],
    token_map: Mapping[tuple[int, int], Mapping[str, Any]],
    label_to_id: Mapping[str, int],
) -> float | None:
    offsets = sorted(
        offset
        for offset in token_map
        if offset[0] < int(gold["end"]) and offset[1] > int(gold["start"])
    )
    if not offsets:
        return None
    probabilities: list[float] = []
    for index, offset in enumerate(offsets):
        prefix = "B" if index == 0 else "I"
        label_id = int(label_to_id[f"{prefix}-{gold['label']}"])
        probabilities.append(float(token_map[offset]["probabilities"][label_id]))
    return float(statistics.fmean(probabilities))


def _describe_confidence(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def _error_row(
    split: str,
    record_index: int,
    record: Mapping[str, Any],
    category: str,
    gold: Mapping[str, Any] | None,
    prediction: Mapping[str, Any] | None,
    text: str,
) -> dict[str, Any]:
    meta = record.get("meta", {})
    return {
        "split": split,
        "record_index": record_index,
        "record_id": str(meta.get("source_record_id", record_index)),
        "source": str(meta.get("source", "")),
        "template_family": meta.get("template_family") or "",
        "error_category": category,
        "text": text,
        "gold_entity": gold["label"] if gold else "",
        "predicted_entity": prediction["label"] if prediction else "",
        "gold_span": (
            f"[{gold['start']},{gold['end']})" if gold is not None else ""
        ),
        "predicted_span": (
            f"[{prediction['start']},{prediction['end']})"
            if prediction is not None
            else ""
        ),
        "gold_text": (
            text[int(gold["start"]):int(gold["end"])] if gold is not None else ""
        ),
        "predicted_text": (
            text[int(prediction["start"]):int(prediction["end"])]
            if prediction is not None
            else ""
        ),
        "prediction_confidence": (
            float(prediction["confidence"]) if prediction is not None else None
        ),
        "gold_label_probability": None,
    }


def subset_score(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    label_order: Sequence[str],
    split: str,
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    subset_records = [records[index] for index in selected_indices]
    subset_predictions = [predictions[index] for index in selected_indices]
    return score_predictions(
        subset_records,
        subset_predictions,
        label_order,
        split,
        record_indices=selected_indices,
    )


def write_error_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "record_index",
        "record_id",
        "source",
        "template_family",
        "error_category",
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(
    path: Path,
    confusion: Mapping[str, int],
    label_order: Sequence[str],
) -> None:
    rows = list(label_order) + ["<SPURIOUS>"]
    columns = list(label_order) + ["<BOUNDARY>", "<MISSED>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\predicted", *columns])
        for gold in rows:
            writer.writerow(
                [
                    gold,
                    *[
                        confusion.get(f"{gold}\t{predicted}", 0)
                        for predicted in columns
                    ],
                ]
            )
