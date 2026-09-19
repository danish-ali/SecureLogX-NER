#!/usr/bin/env python3
"""Capture the immutable ML-v1/ML-v1.1/ML-v1.2 parent boundary for ML-v1.3.

The sealed challenge is hashed only as an opaque byte stream, never parsed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("configs/ml_v1_3_parent_freeze.json")
SEALED_PATH = Path("data/ml_v1_1/challenge/sealed_test.jsonl")
EXPECTED_SEALED_SHA256 = (
    "6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a"
)

FIXED_PATHS = (
    Path("configs/securelogx_labels.json"),
    Path("configs/ml_v1_2_parent_freeze.json"),
    Path("configs/ml_v1_2_dataset_profile.json"),
    Path("configs/ml_v1_2_dataset_manifest.json"),
    Path("configs/ml_v1_2_dataset_manifest.sha256"),
    Path("configs/ml_v1_2_training_gate.json"),
    Path("configs/ml_v1_2_training_gate.sha256"),
    Path("data/split/train.jsonl"),
    Path("data/split/dev.jsonl"),
    Path("data/split/test.jsonl"),
    Path("data/ml_v1_2/counterbalance/train_additions.jsonl"),
    Path("data/ml_v1_2/counterbalance/dev_challenge.jsonl"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _files_under(root: Path, relative_directory: Path) -> list[Path]:
    directory = root / relative_directory
    if not directory.is_dir():
        raise FileNotFoundError(relative_directory.as_posix())
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _unique(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()).casefold()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return sorted(result, key=lambda value: value.as_posix())


def _entries(root: Path, paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in _unique(paths):
        if not path.is_file():
            raise FileNotFoundError(_relative(root, path))
        result[_relative(root, path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def fingerprint(entries: Mapping[str, Mapping[str, Any]]) -> str:
    hashes = {path: value["sha256"] for path, value in sorted(entries.items())}
    payload = json.dumps(
        hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _report_paths(root: Path) -> list[Path]:
    reports = root / "reports"
    return sorted(
        path
        for path in reports.iterdir()
        if path.is_file() and path.name.startswith("ml_v1_2_")
    )


def capture(root: Path) -> dict[str, Any]:
    labels = json.loads((root / "configs/securelogx_labels.json").read_text(encoding="utf-8"))
    parent_v1_2 = json.loads((root / "configs/ml_v1_2_parent_freeze.json").read_text(encoding="utf-8"))
    sealed_expected = parent_v1_2["sealed_challenge"]["expected_sha256"]
    if sealed_expected != EXPECTED_SEALED_SHA256:
        raise RuntimeError("v1.2 parent freeze lost the sealed challenge hash")

    fixed_entries = _entries(root, (root / path for path in FIXED_PATHS))
    report_entries = _entries(root, _report_paths(root))
    checkpoint_entries = _entries(
        root, _files_under(root, Path("output_securelogx/ml-v1.2/bert-base-cased"))
    )
    groups = {
        "frozen_inputs": {
            "file_count": len(fixed_entries),
            "fingerprint_sha256": fingerprint(fixed_entries),
            "files": fixed_entries,
        },
        "ml_v1_2_reports": {
            "file_count": len(report_entries),
            "fingerprint_sha256": fingerprint(report_entries),
            "files": report_entries,
        },
        "ml_v1_2_checkpoints_and_dev_evidence": {
            "file_count": len(checkpoint_entries),
            "fingerprint_sha256": fingerprint(checkpoint_entries),
            "files": checkpoint_entries,
        },
    }
    return {
        "snapshot_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Immutable parent boundary for ML-v1.3 real-structure data analysis",
        "parent_versions": ["ML-v1", "ML-v1.1", "ML-v1.2"],
        "sealed_challenge": {
            "path": SEALED_PATH.as_posix(),
            "expected_sha256": EXPECTED_SEALED_SHA256,
            "sha256": parent_v1_2["sealed_challenge"]["sha256"],
            "bytes": parent_v1_2["sealed_challenge"]["bytes"],
            "handling": "opaque_hash_inherited_from_ml_v1_2_parent_freeze_never_parsed",
        },
        "ontology": {
            "path": "configs/securelogx_labels.json",
            "sha256": sha256_file(root / "configs/securelogx_labels.json"),
            "entity_count": len(labels.get("entities", [])),
            "bio_label_count": len(labels.get("bio_labels", [])),
        },
        "groups": groups,
        "totals": {
            "files": sum(group["file_count"] for group in groups.values()),
            "bytes": sum(
                entry["bytes"]
                for group in groups.values()
                for entry in group["files"].values()
            ),
        },
        "constraints": {
            "model_training": False,
            "model_evaluation": False,
            "sealed_predictions_accessed": False,
            "sealed_dataset_parsed": False,
            "ontology_changed": False,
        },
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing stale temporary file: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    if output.exists():
        print(f"ERROR: refusing to overwrite parent freeze: {output}", file=sys.stderr)
        return 1
    snapshot = capture(root)
    atomic_write_json(output, snapshot)
    print(
        json.dumps(
            {
                "output": output.relative_to(root).as_posix(),
                "files": snapshot["totals"]["files"],
                "sealed_sha256": snapshot["sealed_challenge"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
