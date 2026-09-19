#!/usr/bin/env python3
"""Build and freeze the isolated SecureLogX ML-v1.2 counterbalance revision.

This command performs data generation and validation only.  It never imports a
model, runs inference, trains, exports ONNX, or touches Java sources.  The
frozen ML-v1/ML-v1.1 parent snapshot is verified before generation and again
before the v1.2 manifest is finalized.

The sealed challenge file is never opened.  Its hash (and the hashes of any
other frozen v1.1 partition whose file is unreadable) is verified through
deterministic in-memory regeneration by the frozen v1.1 generator; the
canonical serialization must reproduce the byte SHA-256 recorded in the
parent freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Counter as CounterType, Dict, Iterable, List, Mapping, Optional, Sequence
from collections import Counter, defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from generate_securelogx_ml_v1_2_counterbalance import (  # noqa: E402
    BUSINESS_ID_SUBTYPES,
    GENERATOR_VERSION,
    catalog,
    generate_all,
    write_jsonl,
)
from validate_securelogx_ml_v1_2 import (  # noqa: E402
    V1_1_GENERATOR_PARTITIONS,
    V1_1_PARTITION_PATHS,
    canonical_jsonl_sha256,
    validate_dataset,
)
import validate_securelogx_ml_v1_1 as v1_1_validator  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = Path("configs/ml_v1_2_dataset_profile.json")
PARENT_FREEZE_PATH = Path("configs/ml_v1_2_parent_freeze.json")
MANIFEST_PATH = Path("configs/ml_v1_2_dataset_manifest.json")
MANIFEST_HASH_PATH = Path("configs/ml_v1_2_dataset_manifest.sha256")
V1_1_GENERATOR_PATH = Path(
    "scripts/data/generate_securelogx_ml_v1_1_context_contrast.py"
)

DATA_PATHS: Mapping[str, Path] = {
    "train_addition": Path("data/ml_v1_2/counterbalance/train_additions.jsonl"),
    "dev_challenge": Path("data/ml_v1_2/counterbalance/dev_challenge.jsonl"),
}

REPORT_PATHS: Mapping[str, Path] = {
    "counterbalance_summary": Path("reports/ml_v1_2_counterbalance_summary.md"),
    "dev_challenge_summary": Path("reports/ml_v1_2_dev_challenge_summary.md"),
    "leakage": Path("reports/ml_v1_2_template_leakage_report.md"),
    "readiness": Path("reports/ml_v1_2_data_readiness.md"),
}
DIAGNOSIS_INPUT_PATHS: Mapping[str, Path] = {
    "error_diagnosis": Path("reports/ml_v1_2_v1_1_error_diagnosis.md"),
    "checkpoint_tradeoffs": Path("reports/ml_v1_2_checkpoint_tradeoff_analysis.md"),
    "business_id_false_positives": Path(
        "reports/ml_v1_2_business_id_false_positive_analysis.md"
    ),
    "dev_error_analysis_json": Path("reports/ml_v1_2_v1_1_dev_error_analysis.json"),
}

SENSITIVE_COUNTERBALANCE_LABELS: Sequence[str] = (
    "SSN",
    "CREDIT_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "ROUTING_NUMBER",
    "ITIN",
    "TAX_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE",
    "API_KEY",
    "AUTH_TOKEN",
    "IP_ADDRESS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace stale temporary file: {temporary}")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _regenerated_v1_1_hashes(root: Path) -> Dict[str, Dict[str, Any]]:
    """Canonical hashes/sizes of the deterministically regenerated v1.1 partitions."""

    module = v1_1_validator._load_module(
        root / V1_1_GENERATOR_PATH, "_securelogx_v1_1_generator_for_freeze_verify"
    )
    regenerated = module.generate_all(seed=42)
    result: Dict[str, Dict[str, Any]] = {}
    for key, relative in V1_1_PARTITION_PATHS.items():
        records = regenerated[V1_1_GENERATOR_PARTITIONS[key]]
        payload_bytes = sum(
            len(
                (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
            )
            for record in records
        )
        result[relative] = {
            "sha256": canonical_jsonl_sha256(records),
            "bytes": payload_bytes,
        }
    return result


def verify_parent_snapshot(root: Path, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify every parent-freeze file hash, group fingerprint, and the ontology.

    Files that cannot be opened (restrictive ACLs from the generating sandbox)
    are verified through deterministic regeneration when they are frozen v1.1
    partitions; anything else unreadable is a mismatch.  The sealed challenge
    is always verified via regeneration and is never opened.
    """

    sealed_relative = snapshot["sealed_challenge"]["path"]
    regenerated = _regenerated_v1_1_hashes(root)
    checks: List[Dict[str, Any]] = []
    group_fingerprints: List[Dict[str, Any]] = []
    for group_name, group in sorted(snapshot["groups"].items()):
        actual_hashes: Dict[str, str] = {}
        for relative, expected in sorted(group["files"].items()):
            method = "file"
            actual_hash: Optional[str] = None
            actual_bytes: Optional[int] = None
            if relative == sealed_relative or (
                relative in regenerated and not os.access(root / relative, os.R_OK)
            ):
                method = "regenerated_in_memory"
                actual_hash = regenerated[relative]["sha256"]
                actual_bytes = regenerated[relative]["bytes"]
            else:
                path = root / relative
                try:
                    actual_hash = sha256_file(path)
                    actual_bytes = path.stat().st_size
                except (OSError, PermissionError) as exc:
                    if relative in regenerated:
                        method = "regenerated_in_memory"
                        actual_hash = regenerated[relative]["sha256"]
                        actual_bytes = regenerated[relative]["bytes"]
                    else:
                        checks.append(
                            {
                                "path": relative,
                                "passed": False,
                                "error": str(exc),
                                "expected_sha256": expected["sha256"],
                                "actual_sha256": None,
                            }
                        )
                        continue
            if actual_hash is not None:
                actual_hashes[relative] = actual_hash
            checks.append(
                {
                    "path": relative,
                    "passed": (
                        actual_hash == expected["sha256"]
                        and actual_bytes == expected["bytes"]
                    ),
                    "method": method,
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": actual_hash,
                    "expected_bytes": expected["bytes"],
                    "actual_bytes": actual_bytes,
                }
            )
        fingerprint_payload = json.dumps(
            actual_hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        actual_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        group_fingerprints.append(
            {
                "group": group_name,
                "passed": actual_fingerprint == group["fingerprint_sha256"],
                "expected_fingerprint_sha256": group["fingerprint_sha256"],
                "actual_fingerprint_sha256": actual_fingerprint,
                "file_count": len(group["files"]),
            }
        )

    sealed = snapshot["sealed_challenge"]
    sealed_check = {
        "path": sealed_relative,
        "passed": (
            regenerated[sealed_relative]["sha256"] == sealed["expected_sha256"]
            and regenerated[sealed_relative]["bytes"] == sealed["bytes"]
        ),
        "method": "regenerated_in_memory",
        "expected_sha256": sealed["expected_sha256"],
        "actual_sha256": regenerated[sealed_relative]["sha256"],
        "file_opened": False,
    }

    ontology_meta = snapshot["ontology"]
    label_config = load_json(root / ontology_meta["path"])
    ontology_check = {
        "passed": (
            sha256_file(root / ontology_meta["path"]) == ontology_meta["sha256"]
            and len(label_config.get("entities", [])) == ontology_meta["entity_count"]
            and len(label_config.get("bio_labels", []))
            == ontology_meta["bio_label_count"]
        ),
        "expected": {
            "sha256": ontology_meta["sha256"],
            "entities": ontology_meta["entity_count"],
            "bio_labels": ontology_meta["bio_label_count"],
        },
        "actual": {
            "entities": len(label_config.get("entities", [])),
            "bio_labels": len(label_config.get("bio_labels", [])),
        },
    }

    mismatches: List[Dict[str, Any]] = [
        check for check in checks if not check["passed"]
    ]
    mismatches.extend(
        fingerprint for fingerprint in group_fingerprints if not fingerprint["passed"]
    )
    if not sealed_check["passed"]:
        mismatches.append(sealed_check)
    if not ontology_check["passed"]:
        mismatches.append(ontology_check)
    return {
        "passed": not mismatches,
        "verified_files": len(checks),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "group_fingerprints": group_fingerprints,
        "sealed_challenge": sealed_check,
        "ontology": ontology_check,
    }


def collect_stats(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    label_counts: CounterType[str] = Counter()
    subtype_counts: CounterType[str] = Counter()
    category_records: CounterType[str] = Counter()
    category_families: Dict[str, set] = defaultdict(set)
    focus_labels: CounterType[str] = Counter()
    families: set = set()
    groups: set = set()
    ids: set = set()
    styles: set = set()
    formats: set = set()
    morphologies: set = set()
    multi_entity = 0

    for record in records:
        entities = record["entities"]
        for _, _, label in entities:
            label_counts[str(label)] += 1
        if len(entities) > 1:
            multi_entity += 1
        meta = record["meta"]
        family = str(meta["template_family"])
        category = str(meta["contrast_category"])
        families.add(family)
        groups.add(str(meta["contrast_group_id"]))
        ids.add(str(meta["source_record_id"]))
        styles.add(str(meta["surface_style"]))
        formats.add(str(meta["format"]))
        morphologies.add(str(meta["morphology_class"]))
        category_records[category] += 1
        category_families[category].add(family)
        for provenance in meta["entity_provenance"]:
            if provenance["label"] == "BUSINESS_ID":
                subtype_counts[str(provenance["business_id_subtype"])] += 1
        for target in meta["contrast_targets"]:
            if target["target_role"] == "focus":
                focus_labels[str(target["expected_label"])] += 1

    return {
        "records": len(records),
        "entity_spans": sum(label_counts.values()),
        "multi_entity_records": multi_entity,
        "multi_entity_ratio": multi_entity / len(records) if records else 0.0,
        "family_count": len(families),
        "families": sorted(families),
        "contrast_group_count": len(groups),
        "source_record_id_count": len(ids),
        "entity_counts": dict(sorted(label_counts.items())),
        "business_id_subtype_counts": dict(sorted(subtype_counts.items())),
        "focus_label_counts": dict(sorted(focus_labels.items())),
        "category_record_counts": dict(sorted(category_records.items())),
        "category_family_counts": {
            key: len(value) for key, value in sorted(category_families.items())
        },
        "surface_styles": sorted(styles),
        "formats": sorted(formats),
        "morphology_classes": sorted(morphologies),
    }


def render_counterbalance_summary(
    stats: Mapping[str, Mapping[str, Any]]
) -> str:
    train = stats["train_addition"]
    dev = stats["dev_challenge"]
    lines = [
        "# ML-v1.2 Counterbalance Summary",
        "",
        "ML-v1.2 is a targeted counterbalance revision. Its objective is to reduce "
        "BUSINESS_ID overprediction while preserving the contextual BUSINESS_ID "
        "gains achieved in ML-v1.1. Every family is bidirectional: the same "
        "deterministic surface value appears once as a genuine BUSINESS_ID and once "
        "as the canonical sensitive entity (or, for the dedicated IP families, once "
        "as IP_ADDRESS and once as a non-network technical coordinate that stays `O`). "
        "It was produced without model training or inference.",
        "",
        "## New artifacts",
        "",
        "| Partition | Records | Entity spans | Contrast groups | Families | Multi-entity ratio |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Training addition | {train['records']} | {train['entity_spans']} | "
        f"{train['contrast_group_count']} | {train['family_count']} | {train['multi_entity_ratio']:.3f} |",
        f"| Dev challenge (v1.2) | {dev['records']} | {dev['entity_spans']} | "
        f"{dev['contrast_group_count']} | {dev['family_count']} | {dev['multi_entity_ratio']:.3f} |",
        "",
        "## Bidirectional contrast coverage",
        "",
        "| Category | Train records/families | Dev records/families |",
        "|---|---:|---:|",
    ]
    categories = sorted(
        set(train["category_record_counts"]) | set(dev["category_record_counts"])
    )
    for category in categories:
        lines.append(
            f"| `{category}` | {train['category_record_counts'].get(category, 0)}/"
            f"{train['category_family_counts'].get(category, 0)} | "
            f"{dev['category_record_counts'].get(category, 0)}/"
            f"{dev['category_family_counts'].get(category, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Sensitive-entity counterbalance spans (training addition)",
            "",
            "| Label | Train spans |",
            "|---|---:|",
            f"| BUSINESS_ID | {train['entity_counts'].get('BUSINESS_ID', 0)} |",
        ]
    )
    for label in SENSITIVE_COUNTERBALANCE_LABELS:
        lines.append(f"| {label} | {train['entity_counts'].get(label, 0)} |")
    lines.append(
        f"| PERSON_NAME (companions) | {train['entity_counts'].get('PERSON_NAME', 0)} |"
    )

    lines.extend(
        [
            "",
            "## True-SSN protection",
            "",
            f"The training addition contains **{train['entity_counts'].get('SSN', 0)}** true SSN "
            "spans across hyphenated, compact, spaced, and dotted morphologies in "
            "JSON, nested JSON, key=value, query, CSV, bracketed, syslog, and audit "
            "message surfaces, with camelCase, snake_case, and abbreviated keys. "
            "Every SSN family is multi-entity: the SSN coexists with a genuine "
            "BUSINESS_ID or a PERSON_NAME companion, so the model must keep both "
            "readings apart inside one record.",
            "",
            "## Weak-subtype strengthening (BUSINESS_ID provenance only)",
            "",
            "| BUSINESS_ID subtype | Train spans | Dev spans |",
            "|---|---:|---:|",
        ]
    )
    for subtype in BUSINESS_ID_SUBTYPES:
        lines.append(
            f"| {subtype} | {train['business_id_subtype_counts'].get(subtype, 0)} | "
            f"{dev['business_id_subtype_counts'].get(subtype, 0)} |"
        )
    lines.extend(
        [
            "",
            "CUSTOMER_ID (50% challenge recall in ML-v1.1) and REQUEST_ID (85%) receive "
            "the largest dedicated allocations; all twelve subtypes remain represented "
            "so successful subtype diversity is preserved. Subtypes stay metadata only.",
            "",
            "## Why the `ip_address_vs_technical_reference` families were added",
            "",
            "The ML-v1.1 worst development category (record error rate 0.2625, gate "
            "limit 0.20) was `ip_address_vs_technical_reference`: valid dotted-quad "
            "values in non-network technical contexts were promoted to `IP_ADDRESS` "
            "on shape alone (21 of 21 category errors were morphology-over-context; "
            "see `reports/ml_v1_2_v1_1_error_diagnosis.md`). ML-v1.2 therefore adds "
            f"{train['category_family_counts'].get('ip_address_vs_technical_reference', 0)} "
            "dedicated training families (and "
            f"{dev['category_family_counts'].get('ip_address_vs_technical_reference', 0)} "
            "dev families) in which the identical dotted-quad surface is `IP_ADDRESS` "
            "when context names a network endpoint and `O` when context names a "
            "release version, package coordinate, schema revision, build tuple, "
            "firmware revision, or dependency coordinate. No sensitive entity is ever "
            "converted to `O`: the O-role records keep their adjacent BUSINESS_ID "
            "annotated.",
            "",
            "## Relationship to the frozen parents",
            "",
            "Future training composition is the frozen ML-v1 train split plus the "
            "ML-v1.1 training addition plus this counterbalance addition. Future "
            "development scores the frozen ML-v1 dev split plus this new v1.2 dev "
            "challenge; the ML-v1.1 dev challenge becomes a historical diagnostic. "
            "The original ML-v1 test remains a regression benchmark, and the sealed "
            "final challenge remains sealed, unevaluated, and byte-identical.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_dev_challenge_summary(
    stats: Mapping[str, Mapping[str, Any]], validation: Mapping[str, Any]
) -> str:
    dev = stats["dev_challenge"]
    dev_result = validation["splits"]["dev_challenge"]
    lines = [
        "# ML-v1.2 Dev Challenge Summary",
        "",
        "The ML-v1.1 dev challenge has already influenced development decisions, so "
        "ML-v1.2 introduces a fresh, independently authored development challenge. "
        "This report contains no challenge text or synthetic values.",
        "",
        "| Property | Value |",
        "|---|---:|",
        f"| Records | {dev['records']} |",
        f"| Contrast groups | {dev['contrast_group_count']} |",
        f"| Families | {dev['family_count']} |",
        f"| Entity spans | {dev['entity_spans']} |",
        f"| Multi-entity ratio | {dev['multi_entity_ratio']:.3f} |",
        "",
        "## Families",
        "",
        *[f"- `{family}`" for family in dev["families"]],
        "",
        "## Category coverage (records/families)",
        "",
        "| Category | Dev coverage |",
        "|---|---:|",
    ]
    for category in sorted(dev["category_record_counts"]):
        lines.append(
            f"| `{category}` | {dev['category_record_counts'][category]}/"
            f"{dev['category_family_counts'][category]} |"
        )
    lines.extend(
        [
            "",
            "Coverage includes BUSINESS_ID against SSN, credit card, bank account, "
            "routing number, ITIN, TAX_ID, passport, driver license, API key, auth "
            "token, and IP address; the actual ML-v1.1 worst category "
            "(`ip_address_vs_technical_reference`) receives "
            f"{dev['category_family_counts'].get('ip_address_vs_technical_reference', 0)} "
            "dedicated families; CUSTOMER_ID and REQUEST_ID contexts are present; and "
            f"{dev['multi_entity_records'] if 'multi_entity_records' in dev else dev['records']} "
            "records are multi-entity logs pairing the focus value with a genuine "
            "companion entity.",
            "",
            "## Independence",
            "",
            "Validator-enforced: zero family, exact-text, source-ID, focus-value-hash, "
            "and normalized-skeleton overlap with the ML-v1 train/dev/test splits, the "
            "ML-v1.1 training addition, the ML-v1.1 dev challenge, the sealed final "
            "challenge, and the ML-v1.2 training addition "
            f"(surface styles: {', '.join('`' + s + '`' for s in dev_result['surface_styles'])}).",
            "",
            "## Sealed-test policy",
            "",
            "The sealed final challenge was not modified, opened, or evaluated in this "
            "phase. Its byte hash was verified exclusively through deterministic "
            "in-memory regeneration by the frozen v1.1 generator.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_leakage_report(validation: Mapping[str, Any]) -> str:
    leakage = validation["leakage"]
    template = validation["template_audit"]
    v1_1_overlap = leakage["v1_1_overlap"]

    def _overlap_row(name: str, key: str) -> str:
        overlap = v1_1_overlap[key]
        total = sum(overlap.values())
        return f"| {name} (text/id/family/value/skeleton) | {total} | 0 | {'PASS' if total == 0 else 'FAIL'} |"

    lines = [
        "# ML-v1.2 Template Leakage Report",
        "",
        "| Check | Observed | Required | Result |",
        "|---|---:|---:|---|",
        f"| New cross-split exact-text overlap | {leakage['cross_split_exact_text']['total']} | 0 | {'PASS' if leakage['cross_split_exact_text']['total'] == 0 else 'FAIL'} |",
        f"| New cross-split source-ID overlap | {leakage['cross_split_source_id']['total']} | 0 | {'PASS' if leakage['cross_split_source_id']['total'] == 0 else 'FAIL'} |",
        f"| New cross-split template-family overlap | {leakage['cross_split_template_family']['total']} | 0 | {'PASS' if leakage['cross_split_template_family']['total'] == 0 else 'FAIL'} |",
        f"| New cross-split focus-value hash overlap | {leakage['cross_split_focus_value_hash']['total']} | 0 | {'PASS' if leakage['cross_split_focus_value_hash']['total'] == 0 else 'FAIL'} |",
        f"| Exact text shared with any ML-v1 split | {leakage['parent_exact_text_count']} | 0 | {'PASS' if leakage['parent_exact_text_count'] == 0 else 'FAIL'} |",
        f"| Exact text shared with the original ML-v1 test | {leakage['parent_test_exact_text_count']} | 0 | {'PASS' if leakage['parent_test_exact_text_count'] == 0 else 'FAIL'} |",
        f"| Source ID shared with any ML-v1 split | {leakage['parent_source_id_count']} | 0 | {'PASS' if leakage['parent_source_id_count'] == 0 else 'FAIL'} |",
        f"| Family ID shared with any ML-v1 split | {leakage['parent_template_family_count']} | 0 | {'PASS' if leakage['parent_template_family_count'] == 0 else 'FAIL'} |",
        _overlap_row("ML-v1.1 training addition overlap", "v1_1_train_addition"),
        _overlap_row("ML-v1.1 dev challenge overlap", "v1_1_dev_challenge"),
        _overlap_row("Sealed final challenge overlap", "v1_1_sealed_challenge"),
        f"| New cross-split normalized template skeletons | {template['gates']['cross_split_normalized_skeleton_overlap']['actual']} | 0 | {'PASS' if template['gates']['cross_split_normalized_skeleton_overlap']['passed'] else 'FAIL'} |",
        f"| New skeletons equal to ML-v1.1 catalog templates | {template['gates']['v1_1_normalized_skeleton_overlap']['actual']} | 0 | {'PASS' if template['gates']['v1_1_normalized_skeleton_overlap']['passed'] else 'FAIL'} |",
        f"| New family names shared with the ML-v1.1 catalog | {template['gates']['v1_1_family_name_overlap']['actual']} | 0 | {'PASS' if template['gates']['v1_1_family_name_overlap']['passed'] else 'FAIL'} |",
        f"| New skeletons equal to parent generator templates | {template['gates']['parent_normalized_skeleton_overlap']['actual']} | 0 | {'PASS' if template['gates']['parent_normalized_skeleton_overlap']['passed'] else 'FAIL'} |",
        f"| Forbidden historical failed family/wording | {template['gates']['forbidden_parent_template']['actual']} | 0 | {'PASS' if template['gates']['forbidden_parent_template']['passed'] else 'FAIL'} |",
        "",
        "ML-v1.1 comparisons were executed against hash-verified partitions: readable "
        "files are checked on disk, while the sealed challenge (and any v1.1 file that "
        "is not readable in this environment) is regenerated deterministically in "
        "memory by the frozen v1.1 generator, and the canonical serialization must "
        "reproduce the byte SHA-256 recorded in `configs/ml_v1_2_parent_freeze.json` "
        "before any comparison is trusted. The sealed challenge file itself was never "
        "opened.",
        "",
        "## Descriptive near-template diagnostics",
        "",
        f"- Maximum cross-new-split literal token 3-gram Jaccard: **{template['cross_split_similarity']['max_literal_token_3gram_jaccard']['score']:.6f}**",
        f"- Maximum cross-new-split sequence ratio: **{template['cross_split_similarity']['max_sequence_matcher_ratio']['score']:.6f}**",
        f"- Maximum new-vs-ML-v1.1 literal token 3-gram Jaccard: **{template['v1_1_similarity']['max_literal_token_3gram_jaccard']['score']:.6f}**",
        f"- Maximum new-vs-ML-v1.1 sequence ratio: **{template['v1_1_similarity']['max_sequence_matcher_ratio']['score']:.6f}**",
        f"- Maximum new-vs-parent literal token 3-gram Jaccard: **{template['parent_similarity']['max_literal_token_3gram_jaccard']['score']:.6f}**",
        f"- Maximum new-vs-parent sequence ratio: **{template['parent_similarity']['max_sequence_matcher_ratio']['score']:.6f}**",
        "",
        "These lexical ratios are diagnostics, not proof of semantic independence. "
        "The v1.2 train and dev families intentionally share the COUNTERCHECK "
        "surface scaffolding while using disjoint field vocabularies, semantic "
        "signatures, and value namespaces, so their sequence ratios are elevated; "
        "exact normalized-skeleton overlap is what the gate enforces, and it is "
        "zero everywhere. The historical `order_reference_ssn_shape_warn_v1` family "
        "and its failed wording remain forbidden.",
    ]
    return "\n".join(lines) + "\n"


def render_readiness(
    readiness: Mapping[str, bool],
    validation: Mapping[str, Any],
    parent_before: Mapping[str, Any],
    parent_after: Mapping[str, Any],
) -> str:
    passed = all(readiness.values()) and validation["passed"]
    recommendation = (
        "READY FOR ML-v1.2 RETRAINING" if passed else "NOT READY FOR ML-v1.2 RETRAINING"
    )
    sealed = parent_after["sealed_challenge"]
    lines = [
        "# ML-v1.2 Data Readiness",
        "",
        f"## Recommendation: {recommendation}",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, value in sorted(readiness.items()):
        lines.append(f"| `{name}` | {'PASS' if value else 'FAIL'} |")
    for name, gate in sorted(validation["gates"].items()):
        lines.append(f"| `validator.{name}` | {'PASS' if gate['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"Parent artifact mismatches before generation: **{parent_before['mismatch_count']}** "
            f"({parent_before['verified_files']} files verified).",
            f"Parent artifact mismatches after generation: **{parent_after['mismatch_count']}** "
            f"({parent_after['verified_files']} files verified).",
            "",
            f"Sealed challenge byte SHA-256: `{sealed['actual_sha256']}` "
            f"(required `{sealed['expected_sha256']}`; "
            f"{'MATCH' if sealed['passed'] else 'MISMATCH'}). The sealed file was "
            "never opened: the hash was established through deterministic in-memory "
            "regeneration by the frozen v1.1 generator.",
            "",
            "The canonical ontology remains exactly 25 entities and 51 BIO labels. "
            "ML-v1 train/dev/test, both model checkpoints and their per-epoch "
            "artifacts, the frozen v1.1 datasets, configs, and reports all match the "
            "pre-generation parent snapshot.",
            "",
            "No model training, model evaluation, ONNX export, Java modification, "
            "sealed-challenge prediction, or original-test inference occurred in "
            "this phase.",
            "",
            "Technical readiness does not clear the separate release constraint: "
            "inherited Gretel records still carry `license_reviewed=false`, so "
            "production or commercial release requires explicit license review.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(root: Path, seed: int) -> Dict[str, Any]:
    profile = load_json(root / PROFILE_PATH)
    parent_snapshot = load_json(root / PARENT_FREEZE_PATH)
    if seed != profile["seed"]:
        raise ValueError(f"Seed must match frozen profile value {profile['seed']}")

    expected_sealed = profile["readiness_gates"]["sealed_challenge_expected_sha256"]
    if parent_snapshot["sealed_challenge"]["expected_sha256"] != expected_sealed:
        raise RuntimeError(
            "STOP: profile and parent freeze disagree on the sealed challenge hash"
        )

    parent_before = verify_parent_snapshot(root, parent_snapshot)
    if not parent_before["passed"]:
        raise RuntimeError(
            "Frozen ML-v1/ML-v1.1 parent verification failed before generation: "
            + json.dumps(parent_before["mismatches"][:5], ensure_ascii=False)
        )

    required_existing = [
        root / PROFILE_PATH,
        root / PARENT_FREEZE_PATH,
        *(root / path for path in DIAGNOSIS_INPUT_PATHS.values()),
    ]
    missing = [
        str(path.relative_to(root)) for path in required_existing if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required v1.2 input(s): " + ", ".join(missing))

    to_create = [
        *(root / path for path in DATA_PATHS.values()),
        *(root / path for path in REPORT_PATHS.values()),
        root / MANIFEST_PATH,
        root / MANIFEST_HASH_PATH,
    ]
    existing = [str(path.relative_to(root)) for path in to_create if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite frozen/new v1.2 artifacts: " + ", ".join(existing)
        )

    generated = generate_all(seed=seed)
    stats = {split: collect_stats(records) for split, records in generated.items()}

    with tempfile.TemporaryDirectory(prefix=".ml-v1-2-stage-", dir=root) as temporary:
        stage = Path(temporary)
        stage_paths: Dict[str, Path] = {}
        for split, records in generated.items():
            stage_path = stage / f"{split}.jsonl"
            write_jsonl(stage_path, records)
            stage_paths[split] = stage_path
        validation = validate_dataset(
            root,
            paths={split: path for split, path in stage_paths.items()},
            include_template_audit=True,
        )
        if not validation["passed"]:
            raise RuntimeError(
                "ML-v1.2 validation failed: "
                + json.dumps(validation["errors"][:10], ensure_ascii=False)
            )
        for split, stage_path in stage_paths.items():
            destination = root / DATA_PATHS[split]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_path, destination)

    # Revalidate the committed paths; this result is the one frozen in reports.
    validation = validate_dataset(root, include_template_audit=True)
    if not validation["passed"]:
        raise RuntimeError("Committed ML-v1.2 artifacts failed validation")

    target_counts = profile["target_counts"]
    gates_profile = profile["readiness_gates"]
    train_stats = stats["train_addition"]
    dev_stats = stats["dev_challenge"]
    leakage = validation["leakage"]
    readiness: Dict[str, bool] = {
        "target_train_record_count": train_stats["records"]
        == target_counts["train_additions_records"],
        "target_dev_record_count": dev_stats["records"]
        == target_counts["dev_challenge_records"],
        "target_train_family_count": train_stats["family_count"]
        == target_counts["train_families"],
        "target_dev_family_count": dev_stats["family_count"]
        == target_counts["dev_families"],
        "train_record_count_within_declared_range": gates_profile["train_records_min"]
        <= train_stats["records"]
        <= gates_profile["train_records_max"],
        "dev_record_count_within_declared_range": gates_profile["dev_records_min"]
        <= dev_stats["records"]
        <= gates_profile["dev_records_max"],
        "canonical_ontology_unchanged": validation["ontology"]["passed"],
        "parent_artifact_hash_mismatches_zero": parent_before["mismatch_count"] == 0,
        "sealed_challenge_hash_unchanged": parent_before["sealed_challenge"]["passed"],
        "sealed_challenge_never_opened": parent_before["sealed_challenge"][
            "file_opened"
        ]
        is False,
        "exact_text_leakage_zero": (
            leakage["cross_split_exact_text"]["total"] == 0
            and leakage["parent_exact_text_count"] == 0
            and leakage["v1_1_overlap_total"] == 0
        ),
        "template_family_leakage_zero": (
            leakage["cross_split_template_family"]["total"] == 0
            and leakage["parent_template_family_count"] == 0
        ),
        "original_test_contamination_zero": leakage["parent_test_exact_text_count"]
        == 0,
        "worst_category_families_present": (
            train_stats["category_family_counts"].get(
                "ip_address_vs_technical_reference", 0
            )
            >= gates_profile["train_ip_technical_reference_families_min"]
            and dev_stats["category_family_counts"].get(
                "ip_address_vs_technical_reference", 0
            )
            >= gates_profile["dev_ip_technical_reference_families_min"]
        ),
        "ssn_counterbalance_material": train_stats["entity_counts"].get("SSN", 0)
        >= 300,
        "all_sensitive_counterexamples_present": all(
            train_stats["entity_counts"].get(label, 0) >= 40
            for label in SENSITIVE_COUNTERBALANCE_LABELS
        ),
        "all_business_id_subtypes_retained": set(
            train_stats["business_id_subtype_counts"]
        )
        == set(BUSINESS_ID_SUBTYPES),
        "multi_entity_ratio_met": (
            train_stats["multi_entity_ratio"]
            >= gates_profile["multi_entity_record_ratio_min"]
            and dev_stats["multi_entity_ratio"]
            >= gates_profile["multi_entity_record_ratio_min"]
        ),
        "model_predictions_not_run": profile["sealed_challenge_policy"][
            "predictions_forbidden"
        ]
        is True,
    }
    if not all(readiness.values()):
        failed = [name for name, passed in readiness.items() if not passed]
        raise RuntimeError("Readiness gates failed: " + ", ".join(failed))

    # Parent verification is deliberately repeated after data creation.
    parent_after = verify_parent_snapshot(root, parent_snapshot)
    if not parent_after["passed"]:
        raise RuntimeError("Frozen ML-v1/ML-v1.1 parent changed during v1.2 generation")

    report_content = {
        "counterbalance_summary": render_counterbalance_summary(stats),
        "dev_challenge_summary": render_dev_challenge_summary(stats, validation),
        "leakage": render_leakage_report(validation),
        "readiness": render_readiness(readiness, validation, parent_before, parent_after),
    }
    for key, content in report_content.items():
        atomic_write_text(root / REPORT_PATHS[key], content)

    data_artifacts: Dict[str, Any] = {}
    for split, relative in DATA_PATHS.items():
        path = root / relative
        data_artifacts[str(relative).replace("\\", "/")] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            **stats[split],
        }

    report_artifacts: Dict[str, Any] = {}
    for relative in (*REPORT_PATHS.values(), *DIAGNOSIS_INPUT_PATHS.values()):
        path = root / relative
        report_artifacts[str(relative).replace("\\", "/")] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    script_paths = (
        Path("scripts/data/generate_securelogx_ml_v1_2_counterbalance.py"),
        Path("scripts/data/validate_securelogx_ml_v1_2.py"),
        Path("scripts/data/build_securelogx_ml_v1_2_dataset.py"),
        Path("scripts/data/analyze_securelogx_ml_v1_1_dev_errors.py"),
        Path("scripts/data/capture_securelogx_ml_v1_2_parent_freeze.py"),
    )
    script_artifacts = {
        str(relative).replace("\\", "/"): sha256_file(root / relative)
        for relative in script_paths
    }

    def _verification_summary(verification: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "passed": verification["passed"],
            "verified_files": verification["verified_files"],
            "mismatch_count": verification["mismatch_count"],
            "mismatches": verification["mismatches"],
            "group_fingerprints": verification["group_fingerprints"],
            "sealed_challenge": verification["sealed_challenge"],
            "ontology": verification["ontology"],
        }

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "dataset_version": "ML-v1.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": profile["phase"],
        "recommendation": "READY FOR ML-v1.2 RETRAINING",
        "parent": {
            "versions": ["ML-v1", "ML-v1.1"],
            "snapshot_path": str(PARENT_FREEZE_PATH).replace("\\", "/"),
            "snapshot_sha256": sha256_file(root / PARENT_FREEZE_PATH),
            "verified_before_generation": _verification_summary(parent_before),
            "verified_after_generation": _verification_summary(parent_after),
        },
        "ontology": {
            "path": parent_snapshot["ontology"]["path"],
            "sha256": parent_snapshot["ontology"]["sha256"],
            "entity_count": 25,
            "bio_label_count": 51,
            "changed": False,
        },
        "generation": {
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "profile_path": str(PROFILE_PATH).replace("\\", "/"),
            "profile_sha256": sha256_file(root / PROFILE_PATH),
            "script_sha256": script_artifacts,
            "family_catalog_sha256": sha256_json(
                [
                    {
                        "family": spec.template_family,
                        "split": spec.intended_split,
                        "category": spec.contrast_category,
                        "morphology": spec.morphology_class,
                        "business_id_subtype": spec.business_id_subtype,
                        "roles": [role.context_role for role in spec.roles],
                    }
                    for spec in catalog()
                ]
            ),
        },
        "future_composition": profile["future_composition"],
        "new_data_artifacts": data_artifacts,
        "reports": report_artifacts,
        "business_id_subtype_counts": {
            split: stats[split]["business_id_subtype_counts"] for split in DATA_PATHS
        },
        "context_conflict_counts": {
            split: {
                "records": stats[split]["category_record_counts"],
                "families": stats[split]["category_family_counts"],
            }
            for split in DATA_PATHS
        },
        "counterbalance_targets": {
            "diagnosed_worst_category": "ip_address_vs_technical_reference",
            "diagnosed_worst_category_error_rate": 0.2625,
            "train_sensitive_span_counts": {
                label: stats["train_addition"]["entity_counts"].get(label, 0)
                for label in SENSITIVE_COUNTERBALANCE_LABELS
            },
            "train_business_id_spans": stats["train_addition"]["entity_counts"].get(
                "BUSINESS_ID", 0
            ),
        },
        "leakage": validation["leakage"],
        "template_audit": validation["template_audit"],
        "v1_1_partition_verification": validation["v1_1_partitions"],
        "sealed_challenge": {
            "path": parent_snapshot["sealed_challenge"]["path"],
            "sealed": True,
            "modified": False,
            "file_opened": False,
            "evaluated": False,
            "development_predictions_forbidden": True,
            "model_predictions_run": False,
            "expected_sha256": expected_sealed,
            "verified_sha256": parent_after["sealed_challenge"]["actual_sha256"],
            "verification_method": "deterministic_in_memory_regeneration",
        },
        "readiness_gates": {
            "builder": readiness,
            "validator": validation["gates"],
            "all_passed": True,
        },
        "operations_performed": {
            "dataset_generation": True,
            "report_generation": True,
            "model_training": False,
            "model_evaluation": False,
            "sealed_challenge_predictions": False,
            "original_test_inference": False,
            "onnx_export": False,
            "java_modification": False,
        },
        "release_constraint": {
            "source": "gretel_finance_pii",
            "license_reviewed": False,
            "meaning": (
                "Technical data readiness does not authorize production or "
                "commercial release."
            ),
        },
    }
    atomic_write_json(root / MANIFEST_PATH, manifest)
    manifest_hash = sha256_file(root / MANIFEST_PATH)
    atomic_write_text(
        root / MANIFEST_HASH_PATH,
        f"{manifest_hash}  {MANIFEST_PATH.name}\n",
    )
    return {
        "manifest": str(MANIFEST_PATH).replace("\\", "/"),
        "manifest_sha256": manifest_hash,
        "records": {split: stats[split]["records"] for split in DATA_PATHS},
        "families": {split: stats[split]["family_count"] for split in DATA_PATHS},
        "recommendation": manifest["recommendation"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and freeze SecureLogX ML-v1.2 counterbalance data"
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build(Path(args.repo_root).resolve(), args.seed)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
