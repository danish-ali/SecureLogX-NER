#!/usr/bin/env python3
"""Capture the immutable ML-v1/ML-v1.1 parent boundary for ML-v1.2.

The sealed challenge is handled as an opaque byte stream: this command hashes
it, but never parses or otherwise exposes its records.  No model code is
imported and no inference is performed.
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
OUTPUT_PATH = Path("configs/ml_v1_2_parent_freeze.json")
SEALED_PATH = Path("data/ml_v1_1/challenge/sealed_test.jsonl")
EXPECTED_SEALED_SHA256 = (
    "6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a"
)

ML_V1_FIXED_PATHS = (
    Path("configs/ml_v1_frozen_manifest.json"),
    Path("configs/securelogx_labels.json"),
    Path("configs/securelogx_label_map.yml"),
    Path("configs/training_profile.yml"),
    Path("configs/ml_v1_onnx_validation_gate.json"),
    Path("data/split/train.jsonl"),
    Path("data/split/dev.jsonl"),
    Path("data/split/test.jsonl"),
    Path("output_securelogx/ml-v1/bert-base-cased/test_evaluation_receipt.json"),
    Path("output_securelogx/ml-v1/bert-base-cased/test_predictions.json"),
    Path("reports/test_metrics.json"),
    Path("reports/hard_negative_model_evaluation.md"),
    Path("reports/business_id_subtype_analysis.md"),
    Path("reports/model_errors.csv"),
    Path("reports/model_error_analysis.md"),
    Path("reports/per_label_test_metrics.csv"),
    Path("reports/source_segment_evaluation.md"),
)

ML_V1_1_CONFIG_PATHS = (
    Path("configs/ml_v1_1_parent_freeze.json"),
    Path("configs/ml_v1_1_dataset_profile.json"),
    Path("configs/ml_v1_1_dataset_manifest.json"),
    Path("configs/ml_v1_1_dataset_manifest.sha256"),
    Path("configs/ml_v1_1_training_gate.json"),
    Path("configs/ml_v1_1_training_gate.sha256"),
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
        if path.is_file()
        and (path.name.startswith("ml_v1_1_") or path.name == "ml_v1_vs_v1_1_comparison.md")
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def capture(root: Path) -> dict[str, Any]:
    root = root.resolve()
    sealed = root / SEALED_PATH
    if not sealed.is_file():
        raise FileNotFoundError(SEALED_PATH.as_posix())
    sealed_hash = sha256_file(sealed)
    if sealed_hash != EXPECTED_SEALED_SHA256:
        raise RuntimeError(
            "STOP: sealed challenge byte hash mismatch; expected "
            f"{EXPECTED_SEALED_SHA256}, got {sealed_hash}"
        )

    ml_v1_paths = [root / path for path in ML_V1_FIXED_PATHS]
    ml_v1_paths.extend(
        _files_under(root, Path("output_securelogx/ml-v1/bert-base-cased"))
    )
    ml_v1_entries = _entries(root, ml_v1_paths)

    ml_v1_1_dataset_paths = _files_under(root, Path("data/ml_v1_1"))
    ml_v1_1_dataset_entries = _entries(root, ml_v1_1_dataset_paths)
    ml_v1_1_config_entries = _entries(
        root, (root / path for path in ML_V1_1_CONFIG_PATHS)
    )
    ml_v1_1_checkpoint_entries = _entries(
        root,
        _files_under(root, Path("output_securelogx/ml-v1.1/bert-base-cased")),
    )
    ml_v1_1_report_entries = _entries(root, _report_paths(root))

    labels = _read_json_object(root / "configs/securelogx_labels.json")
    manifest = _read_json_object(root / "configs/ml_v1_1_dataset_manifest.json")
    sealed_manifest = manifest.get("new_data_artifacts", {}).get(
        SEALED_PATH.as_posix(), {}
    )
    if sealed_manifest.get("sha256") != EXPECTED_SEALED_SHA256:
        raise RuntimeError("ML-v1.1 manifest does not preserve the sealed byte hash")

    groups = {
        "ml_v1_artifacts": {
            "file_count": len(ml_v1_entries),
            "fingerprint_sha256": fingerprint(ml_v1_entries),
            "files": ml_v1_entries,
        },
        "ml_v1_1_configs": {
            "file_count": len(ml_v1_1_config_entries),
            "fingerprint_sha256": fingerprint(ml_v1_1_config_entries),
            "files": ml_v1_1_config_entries,
        },
        "ml_v1_1_datasets": {
            "file_count": len(ml_v1_1_dataset_entries),
            "fingerprint_sha256": fingerprint(ml_v1_1_dataset_entries),
            "files": ml_v1_1_dataset_entries,
        },
        "ml_v1_1_checkpoints_and_dev_evidence": {
            "file_count": len(ml_v1_1_checkpoint_entries),
            "fingerprint_sha256": fingerprint(ml_v1_1_checkpoint_entries),
            "files": ml_v1_1_checkpoint_entries,
        },
        "ml_v1_1_reports": {
            "file_count": len(ml_v1_1_report_entries),
            "fingerprint_sha256": fingerprint(ml_v1_1_report_entries),
            "files": ml_v1_1_report_entries,
        },
    }
    return {
        "snapshot_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Immutable parent boundary for ML-v1.2 data analysis and counterbalancing",
        "parent_versions": ["ML-v1", "ML-v1.1"],
        "sealed_challenge": {
            "path": SEALED_PATH.as_posix(),
            "bytes": sealed.stat().st_size,
            "sha256": sealed_hash,
            "expected_sha256": EXPECTED_SEALED_SHA256,
            "records_from_frozen_manifest": sealed_manifest.get("records"),
            "handling": "opaque_byte_hash_only_not_parsed_not_predicted",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    if output.exists():
        print(f"ERROR: refusing to overwrite parent freeze: {output}", file=sys.stderr)
        return 1
    try:
        snapshot = capture(root)
        atomic_write_json(output, snapshot)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": _relative(root, output),
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
