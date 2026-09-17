#!/usr/bin/env python3
"""Build and freeze the isolated SecureLogX ML-v1.1 data revision.

This command performs data generation and validation only.  It never imports a
model, runs inference, trains, exports ONNX, or touches Java sources.  Frozen
ML-v1 hashes are checked before generation and again before the v1.1 manifest
is finalized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from generate_securelogx_ml_v1_1_context_contrast import (
    BUSINESS_ID_SUBTYPES,
    GENERATOR_VERSION,
    catalog,
    generate_all,
    write_jsonl,
)
from validate_securelogx_ml_v1_1 import validate_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = Path("configs/ml_v1_1_dataset_profile.json")
PARENT_FREEZE_PATH = Path("configs/ml_v1_1_parent_freeze.json")
MANIFEST_PATH = Path("configs/ml_v1_1_dataset_manifest.json")
MANIFEST_HASH_PATH = Path("configs/ml_v1_1_dataset_manifest.sha256")

DATA_PATHS: Mapping[str, Path] = {
    "train_addition": Path("data/ml_v1_1/context_contrast/train_additions.jsonl"),
    "dev_challenge": Path("data/ml_v1_1/context_contrast/dev_challenge.jsonl"),
    "sealed_challenge": Path("data/ml_v1_1/challenge/sealed_test.jsonl"),
}

REPORT_PATHS: Mapping[str, Path] = {
    "diagnosis": Path("reports/ml_v1_1_context_error_diagnosis.md"),
    "context_summary": Path("reports/ml_v1_1_context_contrast_summary.md"),
    "business_distribution": Path("reports/ml_v1_1_business_id_distribution.md"),
    "challenge_summary": Path("reports/ml_v1_1_challenge_split_summary.md"),
    "leakage": Path("reports/ml_v1_1_template_leakage_report.md"),
    "readiness": Path("reports/ml_v1_1_data_readiness.md"),
}

BASELINE_BUSINESS_ID = 1360
BASELINE_SSN = 843


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


def _verify_file(root: Path, relative: str, expected: Mapping[str, Any]) -> Dict[str, Any]:
    path = root / relative
    exists = path.is_file()
    actual_hash = sha256_file(path) if exists else None
    actual_bytes = path.stat().st_size if exists else None
    expected_hash = expected.get("sha256")
    expected_bytes = expected.get("bytes")
    passed = bool(
        exists
        and actual_hash == expected_hash
        and (expected_bytes is None or actual_bytes == expected_bytes)
    )
    return {
        "path": relative,
        "passed": passed,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
    }


def verify_parent_snapshot(root: Path, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    parent_manifest = snapshot["parent_manifest"]
    checks.append(_verify_file(root, parent_manifest["path"], parent_manifest))
    ontology = snapshot["ontology"]
    checks.append(_verify_file(root, ontology["path"], ontology))
    for section in (
        "parent_configs",
        "parent_splits",
        "evaluation_evidence",
    ):
        for relative, expected in snapshot[section].items():
            checks.append(_verify_file(root, relative, expected))

    checkpoint = snapshot["checkpoint"]
    checkpoint_root = root / checkpoint["path"]
    expected_files: Mapping[str, str] = checkpoint["files"]
    actual_names = (
        {path.name for path in checkpoint_root.iterdir() if path.is_file()}
        if checkpoint_root.is_dir()
        else set()
    )
    actual_hashes: Dict[str, str] = {}
    for filename, expected_hash in expected_files.items():
        path = checkpoint_root / filename
        actual_hash = sha256_file(path) if path.is_file() else None
        if actual_hash is not None:
            actual_hashes[filename] = actual_hash
        checks.append(
            {
                "path": f"{checkpoint['path']}/{filename}",
                "passed": actual_hash == expected_hash,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            }
        )

    # This reproduces the evaluator's historical fingerprint exactly.
    fingerprint_payload = json.dumps(
        actual_hashes, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    actual_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
    checkpoint_gate = {
        "path": checkpoint["path"],
        "passed": (
            actual_names == set(expected_files)
            and actual_fingerprint == checkpoint["fingerprint_sha256"]
        ),
        "expected_files": sorted(expected_files),
        "actual_files": sorted(actual_names),
        "expected_fingerprint_sha256": checkpoint["fingerprint_sha256"],
        "actual_fingerprint_sha256": actual_fingerprint,
    }

    label_config = load_json(root / ontology["path"])
    ontology_gate = {
        "passed": (
            len(label_config.get("entities", [])) == ontology["entity_count"]
            and len(label_config.get("bio_labels", [])) == ontology["bio_label_count"]
        ),
        "expected": {
            "entities": ontology["entity_count"],
            "bio_labels": ontology["bio_label_count"],
        },
        "actual": {
            "entities": len(label_config.get("entities", [])),
            "bio_labels": len(label_config.get("bio_labels", [])),
        },
    }
    mismatches = [check for check in checks if not check["passed"]]
    if not checkpoint_gate["passed"]:
        mismatches.append(checkpoint_gate)
    if not ontology_gate["passed"]:
        mismatches.append(ontology_gate)
    return {
        "passed": not mismatches,
        "verified_files": len(checks),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "checkpoint": checkpoint_gate,
        "ontology": ontology_gate,
    }


def collect_stats(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    label_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    category_records: Counter[str] = Counter()
    category_families: Dict[str, set[str]] = defaultdict(set)
    target_labels: Counter[str] = Counter()
    families: set[str] = set()
    groups: set[str] = set()
    ids: set[str] = set()
    styles: set[str] = set()
    formats: set[str] = set()
    morphologies: set[str] = set()

    for record in records:
        entities = record["entities"]
        for _, _, label in entities:
            label_counts[str(label)] += 1
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
                subtype_counts[str(provenance["source_label"])] += 1
        for target in meta["contrast_targets"]:
            target_labels[str(target["expected_label"])] += 1

    return {
        "records": len(records),
        "entity_spans": sum(label_counts.values()),
        "negative_records": sum(not record["entities"] for record in records),
        "family_count": len(families),
        "families": sorted(families),
        "contrast_group_count": len(groups),
        "source_record_id_count": len(ids),
        "entity_counts": dict(sorted(label_counts.items())),
        "business_id_subtype_counts": dict(sorted(subtype_counts.items())),
        "contrast_target_label_counts": dict(sorted(target_labels.items())),
        "category_record_counts": dict(sorted(category_records.items())),
        "category_family_counts": {
            key: len(value) for key, value in sorted(category_families.items())
        },
        "surface_styles": sorted(styles),
        "formats": sorted(formats),
        "morphology_classes": sorted(morphologies),
    }


def _table(rows: Iterable[Sequence[Any]]) -> List[str]:
    return ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def render_context_summary(stats: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "# ML-v1.1 Context-Contrast Summary",
        "",
        "This additive dataset revision targets semantic context versus surface morphology. It does not replace, rewrite, or resplit ML-v1, and it was produced without model training or inference.",
        "",
        "## New artifacts",
        "",
        "| Partition | New records | Entity spans | Contrast groups | Families | Negative records |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "train_addition": "Training addition",
        "dev_challenge": "Dev challenge",
        "sealed_challenge": "Sealed final challenge",
    }
    for split in DATA_PATHS:
        value = stats[split]
        lines.append(
            f"| {labels[split]} | {value['records']} | {value['entity_spans']} | "
            f"{value['contrast_group_count']} | {value['family_count']} | {value['negative_records']} |"
        )

    categories = sorted(
        set().union(*(set(value["category_record_counts"]) for value in stats.values()))
    )
    lines.extend(
        [
            "",
            "## Morphology-conflict coverage",
            "",
            "| Category | Train records/families | Dev records/families | Sealed records/families |",
            "|---|---:|---:|---:|",
        ]
    )
    for category in categories:
        cells = []
        for split in DATA_PATHS:
            cells.append(
                f"{stats[split]['category_record_counts'].get(category, 0)}/"
                f"{stats[split]['category_family_counts'].get(category, 0)}"
            )
        lines.append(f"| `{category}` | {cells[0]} | {cells[1]} | {cells[2]} |")

    train = stats["train_addition"]
    lines.extend(
        [
            "",
            "## Context and style design",
            "",
            "Each contrast group has two records that reuse the same synthetic value morphology under different semantic contexts. BUSINESS_ID subtype is retained in provenance while the neural label remains only `BUSINESS_ID`. O-role technical values are recorded as audit targets but are not invented as entity spans.",
            "",
            f"Training uses **{len(train['surface_styles'])}** surface styles: "
            + ", ".join(f"`{style}`" for style in train["surface_styles"])
            + ".",
            "",
            "True SSN contexts include explicit fields, JSON/nested JSON, natural-language references, WARN/ERROR messages, compact and standard formatting, masked-last-four examples, punctuation variation, and PERSON_NAME companions in selected families. Card numbers are Luhn-valid; IP contrasts reuse valid dotted quads in explicitly non-network technical contexts.",
            "",
            "No new external dataset was downloaded. This controlled synthetic revision increases contextual and stylistic coverage but is not claimed to equal real-world data.",
            "",
            "## Relationship to ML-v1",
            "",
            "Future training composition is the frozen ML-v1 train split plus the training addition. Future development may score the frozen ML-v1 dev split and the independent dev challenge. The original ML-v1 test remains a historical regression benchmark; the new sealed challenge is reserved for a single post-selection evaluation in a later phase.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_business_distribution(stats: Mapping[str, Mapping[str, Any]]) -> str:
    train = stats["train_addition"]
    train_business = int(train["entity_counts"].get("BUSINESS_ID", 0))
    train_ssn = int(train["entity_counts"].get("SSN", 0))
    lines = [
        "# ML-v1.1 BUSINESS_ID Distribution",
        "",
        "BUSINESS_ID subtypes are provenance only. The canonical neural ontology still exposes one `BUSINESS_ID` label.",
        "",
        "- Frozen ML-v1 train BUSINESS_ID spans: **1,360**",
        f"- New training-addition BUSINESS_ID spans: **{train_business}** ({train_business / BASELINE_BUSINESS_ID:.1%} of the parent count)",
        "- Frozen ML-v1 train SSN spans: **843**",
        f"- New training-addition SSN spans: **{train_ssn}** ({train_ssn / BASELINE_SSN:.1%} of the parent count)",
        "",
        "## Subtype provenance",
        "",
        "| BUSINESS_ID subtype | Train addition | Dev challenge | Sealed challenge | New total |",
        "|---|---:|---:|---:|---:|",
    ]
    for subtype in BUSINESS_ID_SUBTYPES:
        values = [
            int(stats[split]["business_id_subtype_counts"].get(subtype, 0))
            for split in DATA_PATHS
        ]
        lines.append(
            f"| {subtype} | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |"
        )
    lines.extend(
        [
            "",
            "The revision adds the previously absent training subtypes ORDER_ID, CASE_ID, TICKET_ID, INVOICE_ID, REFERENCE_ID, REQUEST_ID, and WORKFLOW_ID, while retaining established customer/account/application/transaction/user contexts. No subtype was added to the 25-entity ontology or 51-label BIO map.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_challenge_summary(
    stats: Mapping[str, Mapping[str, Any]], validation: Mapping[str, Any]
) -> str:
    lines = [
        "# ML-v1.1 Challenge Split Summary",
        "",
        "The dev challenge and sealed final challenge were independently authored and are family-disjoint from training and from each other. This report intentionally contains no challenge text or synthetic values.",
        "",
        "| Challenge | Records | Groups | Families | Entity spans |",
        "|---|---:|---:|---:|---:|",
    ]
    challenge_labels = (
        ("dev_challenge", "Dev-only context conflict"),
        ("sealed_challenge", "Sealed final challenge"),
    )
    for split, label in challenge_labels:
        value = stats[split]
        lines.append(
            f"| {label} | {value['records']} | {value['contrast_group_count']} | "
            f"{value['family_count']} | {value['entity_spans']} |"
        )
    for split, label in challenge_labels:
        value = stats[split]
        lines.extend(
            [
                "",
                f"### {label} families",
                "",
                *[f"- `{family}`" for family in value["families"]],
            ]
        )

    coverage = validation["coverage"]
    lines.extend(
        [
            "",
            "## Gate coverage",
            "",
            f"- Dev BUSINESS_ID vs SSN families: **{coverage['dev_challenge']['business_id_vs_ssn_families']}** (minimum 5)",
            f"- Dev BUSINESS_ID vs card/account families: **{coverage['dev_challenge']['business_id_vs_card_or_account_families']}** (minimum 3)",
            f"- Dev network/technical families: **{coverage['dev_challenge']['network_conflict_families']}** (minimum 2)",
            f"- Sealed challenge families: **{coverage['sealed_challenge']['families']}** (required 8-12)",
            "- Sealed formats: " + ", ".join(f"`{value}`" for value in coverage["sealed_challenge"]["formats"]),
            "",
            "## Sealed-test policy",
            "",
            "The sealed challenge was generated and hash-frozen before any ML-v1.1 training. Development-time predictions are forbidden. No model was loaded and no predictions were produced in this phase. Only aggregate family/count/hash metadata is reported here.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_leakage_report(validation: Mapping[str, Any]) -> str:
    leakage = validation["leakage"]
    template = validation["template_audit"]
    lines = [
        "# ML-v1.1 Template Leakage Report",
        "",
        "| Check | Observed | Required | Result |",
        "|---|---:|---:|---|",
        f"| New cross-split exact-text overlap | {leakage['cross_split_exact_text']['total']} | 0 | PASS |",
        f"| New cross-split source-ID overlap | {leakage['cross_split_source_id']['total']} | 0 | PASS |",
        f"| New cross-split template-family overlap | {leakage['cross_split_template_family']['total']} | 0 | PASS |",
        f"| New cross-split target-value hash overlap | {leakage['cross_split_value_hash']['total']} | 0 | PASS |",
        f"| Exact text shared with any ML-v1 split | {leakage['parent_exact_text_count']} | 0 | PASS |",
        f"| Source ID shared with any ML-v1 split | {leakage['parent_source_id_count']} | 0 | PASS |",
        f"| Family ID shared with any ML-v1 split | {leakage['parent_template_family_count']} | 0 | PASS |",
        f"| New cross-split normalized template skeletons | {template['gates']['cross_split_normalized_skeleton_overlap']['actual']} | 0 | PASS |",
        f"| New skeletons equal to parent generator templates | {template['gates']['parent_normalized_skeleton_overlap']['actual']} | 0 | PASS |",
        f"| Forbidden historical failed family/wording | {template['gates']['forbidden_parent_template']['actual']} | 0 | PASS |",
        "",
        "The original ML-v1 test has no exact text, source record ID, or template family copied into the v1.1 training addition or dev challenge. Every contrast group remains wholly within one split.",
        "",
        "## Descriptive near-template diagnostics",
        "",
        f"- Maximum cross-new-split literal token 3-gram Jaccard: **{template['cross_split_similarity']['max_literal_token_3gram_jaccard']['score']:.6f}**",
        f"- Maximum cross-new-split sequence ratio: **{template['cross_split_similarity']['max_sequence_matcher_ratio']['score']:.6f}**",
        f"- Maximum new-vs-parent literal token 3-gram Jaccard: **{template['parent_similarity']['max_literal_token_3gram_jaccard']['score']:.6f}**",
        f"- Maximum new-vs-parent sequence ratio: **{template['parent_similarity']['max_sequence_matcher_ratio']['score']:.6f}**",
        "",
        "These lexical ratios are diagnostics, not proof of semantic independence. The sealed scenarios were separately authored; zero exact normalized skeleton overlap and explicit catalog review provide the enforceable evidence. The historical `order_reference_ssn_shape_warn_v1` family and its failed sentence are forbidden.",
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
        "READY FOR ML-v1.1 RETRAINING"
        if passed
        else "NOT READY FOR ML-v1.1 RETRAINING"
    )
    lines = [
        "# ML-v1.1 Data Readiness",
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
            f"Parent artifact mismatches before generation: **{parent_before['mismatch_count']}**.",
            f"Parent artifact mismatches after generation: **{parent_after['mismatch_count']}**.",
            "",
            "The canonical ontology remains exactly 25 entities and 51 BIO labels. ML-v1 train/dev/test, checkpoint, receipt, predictions, and decisive evaluation reports match the pre-generation snapshot.",
            "",
            "No model training, model evaluation, ONNX export, Java modification, or sealed-challenge prediction occurred.",
            "",
            "Technical readiness does not clear the separate release constraint: inherited Gretel records still carry `license_reviewed=false`, so production or commercial release requires explicit license review.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(root: Path, seed: int) -> Dict[str, Any]:
    profile = load_json(root / PROFILE_PATH)
    parent_snapshot = load_json(root / PARENT_FREEZE_PATH)
    if seed != profile["seed"]:
        raise ValueError(f"Seed must match frozen profile value {profile['seed']}")

    parent_before = verify_parent_snapshot(root, parent_snapshot)
    if not parent_before["passed"]:
        raise RuntimeError("Frozen ML-v1 parent verification failed before generation")

    required_existing = [root / PROFILE_PATH, root / PARENT_FREEZE_PATH, root / REPORT_PATHS["diagnosis"]]
    missing = [str(path.relative_to(root)) for path in required_existing if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required v1.1 input(s): " + ", ".join(missing))

    to_create = [
        *(root / path for path in DATA_PATHS.values()),
        *(root / path for key, path in REPORT_PATHS.items() if key != "diagnosis"),
        root / MANIFEST_PATH,
        root / MANIFEST_HASH_PATH,
    ]
    existing = [str(path.relative_to(root)) for path in to_create if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite frozen/new v1.1 artifacts: " + ", ".join(existing)
        )

    generated = generate_all(seed=seed)
    stats = {split: collect_stats(records) for split, records in generated.items()}

    with tempfile.TemporaryDirectory(prefix=".ml-v1-1-stage-", dir=root) as temporary:
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
                "ML-v1.1 validation failed: "
                + json.dumps(validation["errors"][:10], ensure_ascii=False)
            )
        for split, stage_path in stage_paths.items():
            destination = root / DATA_PATHS[split]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_path, destination)

    # Revalidate the committed paths; this result is the one frozen in reports.
    validation = validate_dataset(root, include_template_audit=True)
    if not validation["passed"]:
        raise RuntimeError("Committed ML-v1.1 artifacts failed validation")

    target_counts = profile["target_counts"]
    subtype_counts = stats["train_addition"]["business_id_subtype_counts"]
    readiness: Dict[str, bool] = {
        "target_train_record_count": stats["train_addition"]["records"] == target_counts["train_additions_records"],
        "target_dev_record_count": stats["dev_challenge"]["records"] == target_counts["dev_challenge_records"],
        "target_sealed_record_count": stats["sealed_challenge"]["records"] == target_counts["sealed_challenge_records"],
        "target_train_family_count": stats["train_addition"]["family_count"] == target_counts["train_families"],
        "target_dev_family_count": stats["dev_challenge"]["family_count"] == target_counts["dev_families"],
        "target_sealed_family_count": stats["sealed_challenge"]["family_count"] == target_counts["sealed_families"],
        "all_requested_business_id_subtypes": set(subtype_counts) == set(BUSINESS_ID_SUBTYPES),
        "each_business_id_subtype_at_least_40": all(subtype_counts.get(value, 0) >= 40 for value in BUSINESS_ID_SUBTYPES),
        "business_id_context_increase_material": stats["train_addition"]["entity_counts"].get("BUSINESS_ID", 0) / BASELINE_BUSINESS_ID >= 0.25,
        "ssn_context_increase_material": stats["train_addition"]["entity_counts"].get("SSN", 0) / BASELINE_SSN >= 0.25,
        "canonical_ontology_unchanged": validation["ontology"]["passed"],
        "exact_text_leakage_zero": validation["leakage"]["cross_split_exact_text"]["total"] == 0 and validation["leakage"]["parent_exact_text_count"] == 0,
        "template_family_leakage_zero": validation["leakage"]["cross_split_template_family"]["total"] == 0 and validation["leakage"]["parent_template_family_count"] == 0,
        "sealed_challenge_has_multiple_families": 8 <= stats["sealed_challenge"]["family_count"] <= 12,
        "model_predictions_not_run": profile["sealed_challenge_policy"]["model_predictions_run_in_this_phase"] is False,
    }
    if not all(readiness.values()):
        failed = [name for name, passed in readiness.items() if not passed]
        raise RuntimeError("Readiness gates failed: " + ", ".join(failed))

    # Parent verification is deliberately repeated after data creation.
    parent_after = verify_parent_snapshot(root, parent_snapshot)
    if not parent_after["passed"]:
        raise RuntimeError("Frozen ML-v1 parent changed during v1.1 generation")

    report_content = {
        "context_summary": render_context_summary(stats),
        "business_distribution": render_business_distribution(stats),
        "challenge_summary": render_challenge_summary(stats, validation),
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
    for relative in REPORT_PATHS.values():
        path = root / relative
        report_artifacts[str(relative).replace("\\", "/")] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    script_paths = (
        Path("scripts/data/generate_securelogx_ml_v1_1_context_contrast.py"),
        Path("scripts/data/validate_securelogx_ml_v1_1.py"),
        Path("scripts/data/build_securelogx_ml_v1_1_dataset.py"),
    )
    script_artifacts = {
        str(relative).replace("\\", "/"): sha256_file(root / relative)
        for relative in script_paths
    }
    sealed_families = stats["sealed_challenge"]["families"]
    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "dataset_version": "ML-v1.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "data_revision_and_challenge_set_preparation_only",
        "recommendation": "READY FOR ML-v1.1 RETRAINING",
        "parent": {
            "version": "ML-v1",
            "snapshot_path": str(PARENT_FREEZE_PATH).replace("\\", "/"),
            "snapshot_sha256": sha256_file(root / PARENT_FREEZE_PATH),
            "manifest_path": parent_snapshot["parent_manifest"]["path"],
            "manifest_sha256": parent_snapshot["parent_manifest"]["sha256"],
            "verified_before_generation": parent_before,
            "verified_after_generation": parent_after,
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
        "leakage": validation["leakage"],
        "template_audit": validation["template_audit"],
        "sealed_challenge": {
            "path": str(DATA_PATHS["sealed_challenge"]).replace("\\", "/"),
            "sealed": True,
            "generated_before_future_training": True,
            "development_predictions_forbidden": True,
            "model_predictions_run": False,
            "family_count": len(sealed_families),
            "families": sealed_families,
            "family_list_sha256": sha256_json(sealed_families),
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
            "onnx_export": False,
            "java_modification": False,
        },
        "release_constraint": parent_snapshot["release_constraint"],
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
        description="Build and freeze SecureLogX ML-v1.1 context-contrast data"
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
