#!/usr/bin/env python3
"""Build and freeze the ML-v1.3 real-structure dataset. Data/analysis only."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from generate_securelogx_ml_v1_3_real_structure import (  # noqa: E402
    EXPECTED_RECORD_COUNTS,
    GENERATOR_VERSION,
    PUBLIC_SOURCES,
    catalog,
    generate_all,
    write_jsonl,
)
from validate_securelogx_ml_v1_3 import SEALED_EXPECTED, validate_dataset  # noqa: E402
from validate_securelogx_ml_v1_1 import load_jsonl  # noqa: E402
from capture_securelogx_ml_v1_3_parent_freeze import sha256_file  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = Path("configs/ml_v1_3_dataset_profile.json")
PARENT_FREEZE_PATH = Path("configs/ml_v1_3_parent_freeze.json")
MANIFEST_PATH = Path("configs/ml_v1_3_dataset_manifest.json")
MANIFEST_HASH_PATH = Path("configs/ml_v1_3_dataset_manifest.sha256")
DATA_PATHS = {
    "train_addition": Path("data/ml_v1_3/real_structure/train_additions.jsonl"),
    "dev_challenge": Path("data/ml_v1_3/real_structure/dev_challenge.jsonl"),
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def verify_parent_snapshot(root: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    mismatches = []
    verified = 0
    for group in snapshot["groups"].values():
        for relative, expected in group["files"].items():
            actual = sha256_file(root / relative)
            verified += 1
            if actual != expected["sha256"]:
                mismatches.append(relative)
    sealed = snapshot["sealed_challenge"]
    return {
        "passed": not mismatches
        and sealed["expected_sha256"] == SEALED_EXPECTED
        and sealed["sha256"] == SEALED_EXPECTED,
        "verified_files": verified,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "sealed_challenge": {
            "passed": sealed["expected_sha256"] == SEALED_EXPECTED,
            "file_opened": False,
            "actual_sha256": sealed["sha256"],
        },
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = Counter(record["meta"]["template_family"] for record in records)
    categories = Counter(record["meta"]["structure_category"] for record in records)
    modes = Counter(record["meta"]["failure_mode"] for record in records)
    labels = Counter(label for record in records for _, _, label in record["entities"])
    subtypes = Counter(
        str(item.get("business_id_subtype"))
        for record in records
        for item in record["meta"].get("entity_provenance", [])
        if item.get("label") == "BUSINESS_ID" and item.get("business_id_subtype")
    )
    multi = sum(1 for record in records if len(record["entities"]) >= 2)
    return {
        "records": len(records),
        "families": len(families),
        "records_per_family": dict(sorted(families.items())),
        "structure_categories": dict(categories),
        "failure_modes": dict(modes),
        "entities": dict(labels),
        "business_id_subtypes": dict(subtypes),
        "multi_entity_ratio": multi / len(records) if records else 0.0,
    }


def _source_report() -> str:
    lines = [
        "# ML-v1.3 Real-Structure Source Report",
        "",
        "No third-party log corpus was downloaded. Structures are original skeletons",
        "inspired by public logging format specifications. Only synthetic values are stored.",
        "",
        "| Structure category | Source | URL | License | License reviewed | Raw text stored |",
        "|---|---|---|---|---|---|",
    ]
    for category, source in PUBLIC_SOURCES.items():
        lines.append(
            f"| `{category}` | {source['name']} | {source['url'] or 'n/a'} | "
            f"{source['license']} | {source['license_reviewed']} | {source['raw_source_text_stored']} |"
        )
    lines.extend(
        [
            "",
            "Local inventory classification:",
            "",
            "- Gretel finance PII: external labeled prose (`license_reviewed=false`); not used as a log-structure source.",
            "- SecureLogX ML-v1 generated logs: existing synthetic templates; not reused.",
            "- ML-v1.1 / ML-v1.2 challenge templates: generated contrast templates; not reused.",
            "- No suitable raw public log files were present locally.",
        ]
    )
    return "\n".join(lines)


def _summary(train: Mapping[str, Any], dev: Mapping[str, Any]) -> str:
    lines = [
        "# ML-v1.3 Real-Structure Summary",
        "",
        f"- Train additions: **{train['records']}** records / **{train['families']}** families",
        f"- Dev challenge: **{dev['records']}** records / **{dev['families']}** families",
        f"- Train multi-entity ratio: **{train['multi_entity_ratio']:.3f}**",
        "",
        "## Structure categories (train)",
        "",
        "| Category | Records |",
        "|---|---:|",
    ]
    for name, count in sorted(train["structure_categories"].items()):
        lines.append(f"| `{name}` | {count} |")
    lines.extend(["", "## Failure modes (train)", "", "| Mode | Records |", "|---|---:|"])
    for name, count in sorted(train["failure_modes"].items()):
        lines.append(f"| `{name}` | {count} |")
    return "\n".join(lines)


def _failure_coverage(train: Mapping[str, Any], dev: Mapping[str, Any]) -> str:
    lines = [
        "# ML-v1.3 Failure-Mode Coverage",
        "",
        "| Failure mode | Train records | Dev records |",
        "|---|---:|---:|",
    ]
    for mode in sorted(set(train["failure_modes"]) | set(dev["failure_modes"])):
        lines.append(
            f"| `{mode}` | {train['failure_modes'].get(mode, 0)} | {dev['failure_modes'].get(mode, 0)} |"
        )
    lines.extend(
        [
            "",
            "Account-like O examples use service/deploy/environment account names as gold `O`",
            "beside a true BUSINESS_ID. AUTH_TOKEN examples put Bearer tokens in Authorization",
            "context beside opaque business identifiers. SSN and ITIN appear in payroll/tax",
            "contexts rather than by class frequency alone. IP versus technical-reference keeps",
            "dotted-quad versions as `O`.",
        ]
    )
    return "\n".join(lines)


def _dev_summary(dev: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# ML-v1.3 Development Challenge Summary",
            "",
            f"- Records: **{dev['records']}**",
            f"- Independent structure families: **{dev['families']}**",
            f"- Multi-entity ratio: **{dev['multi_entity_ratio']:.3f}**",
            "",
            "Dev families are disjoint from training, ML-v1, ML-v1.1, ML-v1.2, and the sealed challenge.",
            "The sealed challenge was not modified or predicted.",
        ]
    )


def _leakage(validation: Mapping[str, Any]) -> str:
    lines = [
        "# ML-v1.3 Template Leakage Report",
        "",
        "| Check | Passed | Detail |",
        "|---|---|---|",
    ]
    for name in sorted(validation["gates"]):
        gate = validation["gates"][name]
        if "leakage" in name or "contamination" in name or "overlap" in name or "sealed" in name:
            lines.append(
                f"| `{name}` | {'PASS' if gate.get('passed') else 'FAIL'} | {gate.get('count', gate.get('actual', ''))} |"
            )
    return "\n".join(lines)


def _composition(root: Path, train_new: int) -> str:
    original = load_jsonl(root / "data/split/train.jsonl")
    sources = Counter(str(record.get("meta", {}).get("source") or "unknown") for record in original)
    synthetic = sources.get("securelogx_custom_logs", 0)
    gretel = sources.get("gretel_finance_pii", 0)
    other = len(original) - synthetic - gretel
    v1_1 = 3000
    v1_2 = 1600
    total = len(original) + v1_1 + v1_2 + train_new
    rows = [
        ("original synthetic", synthetic),
        ("Gretel/external", gretel),
        ("other original train", other),
        ("ML-v1.1 synthetic additions", v1_1),
        ("ML-v1.2 synthetic additions", v1_2),
        ("ML-v1.3 real-structure additions", train_new),
    ]
    lines = [
        "# ML-v1.3 Dataset Composition",
        "",
        f"Future combined training size: **{total}** records.",
        "",
        "| Component | Records | Percentage |",
        "|---|---:|---:|",
    ]
    for name, count in rows:
        lines.append(f"| {name} | {count} | {100.0 * count / total:.2f}% |")
    lines.extend(
        [
            "",
            "The v1.3 additions are smaller than the original synthetic generator mass on purpose.",
            "They reduce dependence on a few generator skeletons by introducing 96 independent",
            "public-format log structures instead of more COUNTERCHECK/key=value clones.",
            "Gretel records remain `license_reviewed=false` and were not used as structure donors.",
        ]
    )
    return "\n".join(lines)


def _readiness(validation: Mapping[str, Any], parent: Mapping[str, Any]) -> str:
    lines = [
        "# ML-v1.3 Data Readiness",
        "",
        f"**Recommendation: {'READY FOR ML-v1.3 MODEL COMPARISON' if validation['passed'] else 'NOT READY FOR ML-v1.3 MODEL COMPARISON'}**",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, gate in sorted(validation["gates"].items()):
        lines.append(f"| `{name}` | {'PASS' if gate.get('passed') else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"Parent freeze files verified: **{parent['verified_files']}**; mismatches: **{parent['mismatch_count']}**.",
            f"Sealed challenge SHA-256: `{validation['sealed_challenge']['sha256']}` (never opened).",
            "No BERT/DeBERTa training, sealed evaluation, original-test inference, ONNX export, or Java changes occurred.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    root = REPO_ROOT
    profile = _load_json(root / PROFILE_PATH)
    freeze_path = root / PARENT_FREEZE_PATH
    if not freeze_path.is_file():
        raise FileNotFoundError("Capture the v1.3 parent freeze before building")
    snapshot = _load_json(freeze_path)
    parent = verify_parent_snapshot(root, snapshot)
    if not parent["passed"]:
        raise RuntimeError(f"parent freeze mismatch: {parent}")
    generated = generate_all(int(profile["seed"]))
    validation = validate_dataset(
        root=root,
        train_records=generated["train_addition"],
        dev_records=generated["dev_challenge"],
        parent_freeze=snapshot,
    )
    if not validation["passed"]:
        failed = [name for name, gate in validation["gates"].items() if not gate.get("passed")]
        raise RuntimeError(f"validation failed: {failed}\n{json.dumps({k: validation['gates'][k] for k in failed}, indent=2)}")
    for partition, path in DATA_PATHS.items():
        write_jsonl(root / path, generated[partition])
    train_stats = _counts(generated["train_addition"])
    dev_stats = _counts(generated["dev_challenge"])
    new_artifacts = {
        path.as_posix(): {
            "records": len(generated[name]),
            "sha256": sha256_file(root / path),
            "bytes": (root / path).stat().st_size,
        }
        for name, path in DATA_PATHS.items()
    }
    manifest = {
        "manifest_version": 1,
        "dataset_version": "ML-v1.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "seed": profile["seed"],
        "ontology_sha256": sha256_file(root / "configs/securelogx_labels.json"),
        "parent_freeze_sha256": sha256_file(freeze_path),
        "new_data_artifacts": new_artifacts,
        "record_counts": EXPECTED_RECORD_COUNTS,
        "family_counts": {
            "train_addition": train_stats["families"],
            "dev_challenge": dev_stats["families"],
        },
        "structure_categories": train_stats["structure_categories"],
        "failure_modes": {
            "train_addition": train_stats["failure_modes"],
            "dev_challenge": dev_stats["failure_modes"],
        },
        "sealed_challenge": {
            "path": "data/ml_v1_1/challenge/sealed_test.jsonl",
            "sha256": SEALED_EXPECTED,
            "file_opened": False,
        },
        "validation_passed": True,
    }
    manifest_path = root / MANIFEST_PATH
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = sha256_file(manifest_path)
    (root / MANIFEST_HASH_PATH).write_text(
        f"{digest}  {MANIFEST_PATH.name}\n", encoding="utf-8", newline="\n"
    )
    reports = {
        "reports/ml_v1_3_real_structure_source_report.md": _source_report(),
        "reports/ml_v1_3_real_structure_summary.md": _summary(train_stats, dev_stats),
        "reports/ml_v1_3_failure_mode_coverage.md": _failure_coverage(train_stats, dev_stats),
        "reports/ml_v1_3_dev_challenge_summary.md": _dev_summary(dev_stats),
        "reports/ml_v1_3_template_leakage_report.md": _leakage(validation),
        "reports/ml_v1_3_dataset_composition.md": _composition(root, train_stats["records"]),
        "reports/ml_v1_3_data_readiness.md": _readiness(validation, parent),
    }
    for relative, text in reports.items():
        _write(root / relative, text)
    (root / "reports/ml_v1_3_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "train": train_stats["records"],
                "dev": dev_stats["records"],
                "manifest_sha256": digest,
                "passed": validation["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
