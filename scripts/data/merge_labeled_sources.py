#!/usr/bin/env python3
"""Merge validated SecureLogX JSONL sources with auditable lineage.

Only canonical, training-eligible records are written. Records may contain no
entities (intentional hard negatives), but malformed annotations, unsupported
labels, unreviewed statuses, and training-exclusion metadata are never silently
converted into negative examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

try:  # Package import (tests / ``python -m``).
    from .validate_jsonl_spans import DEFAULT_LABELS_PATH, JSONLValidator
except ImportError:  # Direct script execution.
    from validate_jsonl_spans import DEFAULT_LABELS_PATH, JSONLValidator


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _new_source_stats() -> Dict[str, int]:
    return {
        "total": 0,
        "valid": 0,  # Retained for compatibility; means written/merged.
        "invalid": 0,
        "duplicates": 0,
        "training_excluded": 0,
        "training_excluded_canonical_spans": 0,
        "negative_records": 0,
        "entities": 0,
        "excluded_spans": 0,
        "training_exclusion_spans": 0,
        "benign_trace_spans": 0,
    }


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Optional[Any], Optional[str]]]:
    """Yield each nonblank JSONL line as (line, value, parse_error)."""

    with path.open(encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                yield line_num, json.loads(raw_line), None
            except json.JSONDecodeError as exc:
                yield line_num, None, f"Invalid JSON - {exc}"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load decodable JSON objects, preserving the original public helper API."""

    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    for line_num, record, error in _iter_jsonl(Path(path)):
        if error is not None:
            errors.append(f"Line {line_num}: {error}")
        elif isinstance(record, dict):
            records.append(record)
        else:
            errors.append(f"Line {line_num}: Record is not a JSON object")

    if errors:
        print(f"Warnings loading {path}:", file=sys.stderr)
        for message in errors[:5]:
            print(f"  {message}", file=sys.stderr)
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more", file=sys.stderr)
    return records


def _validator_for_labels(labels: Optional[Any]) -> JSONLValidator:
    if labels is None:
        return JSONLValidator(str(DEFAULT_LABELS_PATH))
    if isinstance(labels, (str, Path)):
        return JSONLValidator(str(labels))

    # Compatibility for callers that already loaded the canonical entity set.
    canonical = list(labels)
    validator = JSONLValidator.__new__(JSONLValidator)
    validator.labels_path = Path("<in-memory>")
    validator.label_order = canonical
    validator.labels = set(canonical)
    validator._reset_results()
    return validator


def _validate_with(
    validator: JSONLValidator,
    record: Any,
    line_num: int,
) -> Tuple[
    bool,
    str,
    Optional[str],
    List[Tuple[int, int, str]],
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
    List[str],
]:
    error_start = len(validator.errors)
    warning_start = len(validator.warnings)
    valid, text, entities, meta, excluded = validator.validate_record(record, line_num)
    errors = validator.errors[error_start:]
    warnings = validator.warnings[warning_start:]
    del validator.errors[error_start:]
    del validator.warnings[warning_start:]
    error_message = "; ".join(errors)
    return valid, error_message, text, entities, meta, excluded, warnings


def validate_record(
    record: Dict[str, Any],
    source: str,
    labels: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Validate one record, retaining the original two-value helper contract."""

    del source  # Source identity must come from meta.source, never a filename.
    try:
        validator = _validator_for_labels(labels)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"Label configuration error: {exc}"
    valid, error, *_ = _validate_with(validator, record, 1)
    return valid, error


def _source_for_record(record: Any, input_path: Path) -> str:
    if isinstance(record, dict):
        meta = record.get("meta")
        if isinstance(meta, dict):
            source = meta.get("source")
            if isinstance(source, str) and source.strip():
                return source
    return f"<unattributed:{input_path.name}>"


def _is_synthetic(meta: Mapping[str, Any]) -> bool:
    status = meta.get("review_status")
    source = meta.get("source")
    return (
        status in {"synthetic_generated", "generated"}
        or "template_family" in meta
        or source == "securelogx_custom_logs"
    )


def _is_deferred(entry: Mapping[str, Any]) -> bool:
    category = str(entry.get("category", "")).strip().lower()
    disposition = str(entry.get("disposition", "")).strip().lower()
    reason = str(entry.get("reason", "")).strip().lower()
    return (
        category in {"deferred", "deferred_ml_v1"}
        or disposition.startswith("defer")
        or "defer" in reason
    )


def _is_healthcare(entry: Mapping[str, Any]) -> bool:
    category = str(entry.get("category", "")).strip().lower()
    disposition = str(entry.get("disposition", "")).strip().lower()
    reason = str(entry.get("reason", "")).strip().lower()
    return (
        category in {"healthcare", "healthcare_extension"}
        or disposition.startswith("healthcare")
        or "healthcare" in reason
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_lineage_report(
    report_path: Path,
    input_files: Sequence[str],
    source_stats: Mapping[str, Mapping[str, int]],
    summary: Mapping[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Lineage Report",
        "",
        f"**Merge Date (UTC)**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{_markdown(path)}`" for path in input_files)

    lines.extend(
        [
            "",
            "## Merge Summary",
            "",
            f"- **Input Records**: {summary['input_records']}",
            f"- **Merged Records**: {summary['merged_records']}",
            f"- **Merged Canonical Entity Spans**: {summary['merged_entities']}",
            f"- **Merged Negative Records (zero entities)**: {summary['negative_records']}",
            f"- **Invalid Records Rejected**: {summary['invalid_records']}",
            f"- **Exact Duplicate Text Records Detected**: {summary['duplicate_records']}",
            f"- **Exact Duplicate Text Records Removed**: {summary['duplicates_removed']}",
            f"- **Training-Excluded Records**: {summary['training_excluded_records']}",
            "- **Canonical Spans Removed With Training-Excluded Records**: "
            f"{summary['training_excluded_canonical_spans']}",
            f"- **All Excluded/Trace Spans Observed**: {summary['excluded_spans']}",
            f"- **Training-Excluding Metadata Spans**: {summary['training_exclusion_spans']}",
            f"- **Benign Trace-Only Metadata Spans**: {summary['benign_trace_spans']}",
            f"- **Records With Deferred ML-v1 Metadata**: {summary['deferred_records']}",
            f"- **Deferred ML-v1 Metadata Spans**: {summary['deferred_spans']}",
            f"- **Records With Healthcare-Extension Metadata**: {summary['healthcare_records']}",
            f"- **Healthcare-Extension Metadata Spans**: {summary['healthcare_spans']}",
            "- **Duplicate Texts With Conflicting Annotations**: "
            f"{summary['duplicate_annotation_conflicts']}",
            "",
            "A record is omitted when `meta.training_exclusion=true` or any "
            "`meta.excluded_spans` entry has `training_exclusion=true`. This "
            "prevents an intentionally excluded sensitive value from becoming "
            "an implicit `O` label.",
            "",
            "## Source Summary",
            "",
            "| Source | Input | Merged | Negative | Entities | Invalid | "
            "Training-Excluded | Canonical Spans Removed | Duplicates | "
            "Excluded/Trace Spans | Usage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    merged_total = summary["merged_records"]
    for source in sorted(source_stats):
        stats = source_stats[source]
        usage = (stats["valid"] / merged_total * 100) if merged_total else 0.0
        lines.append(
            f"| {_markdown(source)} | {stats['total']} | {stats['valid']} | "
            f"{stats['negative_records']} | {stats['entities']} | "
            f"{stats['invalid']} | {stats['training_excluded']} | "
            f"{stats['training_excluded_canonical_spans']} | "
            f"{stats['duplicates']} | {stats['excluded_spans']} | {usage:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Accepted Source Mix",
            "",
            "| Source Kind | Records | Canonical Spans | Record Percentage |",
            "|---|---:|---:|---:|",
        ]
    )
    for kind in ("synthetic", "external"):
        record_count = summary["source_mix_records"].get(kind, 0)
        span_count = summary["source_mix_entities"].get(kind, 0)
        percentage = (record_count / merged_total * 100) if merged_total else 0.0
        lines.append(
            f"| {kind.title()} | {record_count} | {span_count} | {percentage:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Canonical Entity Distribution in Merged Data",
            "",
            "| Entity | Count | Percentage |",
            "|---|---:|---:|",
        ]
    )
    merged_entities = summary["merged_entities"]
    entity_counts: Counter[str] = summary["entity_counts"]
    if entity_counts:
        for label, count in sorted(entity_counts.items(), key=lambda item: (-item[1], item[0])):
            percentage = (count / merged_entities * 100) if merged_entities else 0.0
            lines.append(f"| {_markdown(label)} | {count} | {percentage:.2f}% |")
    else:
        lines.append("| _None_ | 0 | 0.00% |")

    lines.extend(
        [
            "",
            "## Excluded and Deferred Source Labels",
            "",
            "| Source Label | All Trace Spans | Training-Excluding | "
            "Benign Trace-Only | Deferred ML-v1 | Healthcare Extension |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    excluded_labels = sorted(summary["excluded_by_label"])
    if excluded_labels:
        for label in excluded_labels:
            all_count = summary["excluded_by_label"][label]
            training_count = summary["training_exclusion_by_label"][label]
            benign_count = all_count - training_count
            lines.append(
                f"| {_markdown(label)} | {all_count} | {training_count} | "
                f"{benign_count} | {summary['deferred_by_label'][label]} | "
                f"{summary['healthcare_by_label'][label]} |"
            )
    else:
        lines.append("| _None_ | 0 | 0 | 0 | 0 | 0 |")

    lines.extend(
        [
            "",
            "## Excluded Span Dispositions",
            "",
            "| Source | Source Label | Category | Reason | Training Exclusion | Count |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    disposition_counts: Counter[Tuple[str, str, str, str, bool]] = summary[
        "excluded_dispositions"
    ]
    if disposition_counts:
        for key, count in sorted(
            disposition_counts.items(),
            key=lambda item: (-item[1], tuple(str(value) for value in item[0])),
        ):
            source, label, category, reason, exclusion = key
            lines.append(
                f"| {_markdown(source)} | {_markdown(label)} | "
                f"{_markdown(category or 'unspecified')} | "
                f"{_markdown(reason)} | {str(exclusion).lower()} | {count} |"
            )
    else:
        lines.append("| _None_ | _None_ | _None_ | _None_ | false | 0 |")

    rejection_details: List[str] = summary["rejection_details"]
    if rejection_details:
        lines.extend(["", "## Validation Rejections", ""])
        lines.extend(f"- {_markdown(message)}" for message in rejection_details[:50])
        if len(rejection_details) > 50:
            lines.append(f"- ... and {len(rejection_details) - 50} more rejections")

    warning_details: List[str] = summary["warning_details"]
    if warning_details:
        lines.extend(["", "## Compatibility Warnings", ""])
        lines.extend(f"- {_markdown(message)}" for message in warning_details[:20])
        if len(warning_details) > 20:
            lines.append(f"- ... and {len(warning_details) - 20} more warnings")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_sources(
    input_files: List[str],
    output_file: str,
    report_file: str,
    dedupe: bool = False,
    labels_path: Optional[str] = None,
) -> Tuple[int, int, Dict[str, Dict[str, int]]]:
    """Merge valid records and return ``(merged_count, input_count, by_source)``.

    ``labels_path`` was added after the original pipeline and is optional for
    API compatibility. It defaults to the repository's canonical label config.
    """

    paths = [Path(path) for path in input_files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input JSONL file(s) not found: {', '.join(missing)}")

    validator = JSONLValidator(str(labels_path or DEFAULT_LABELS_PATH))
    merged_records: List[Dict[str, Any]] = []
    seen_texts: Dict[str, Tuple[str, int, Tuple[Tuple[int, int, str], ...]]] = {}
    source_stats: Dict[str, Dict[str, int]] = {}

    summary: Dict[str, Any] = {
        "input_records": 0,
        "merged_records": 0,
        "merged_entities": 0,
        "negative_records": 0,
        "invalid_records": 0,
        "duplicate_records": 0,
        "duplicates_removed": 0,
        "duplicate_annotation_conflicts": 0,
        "training_excluded_records": 0,
        "training_excluded_canonical_spans": 0,
        "excluded_spans": 0,
        "training_exclusion_spans": 0,
        "benign_trace_spans": 0,
        "deferred_records": 0,
        "deferred_spans": 0,
        "healthcare_records": 0,
        "healthcare_spans": 0,
        "entity_counts": Counter(),
        "excluded_by_label": Counter(),
        "training_exclusion_by_label": Counter(),
        "deferred_by_label": Counter(),
        "healthcare_by_label": Counter(),
        "excluded_dispositions": Counter(),
        "source_mix_records": Counter(),
        "source_mix_entities": Counter(),
        "rejection_details": [],
        "warning_details": [],
    }

    print(f"Merging {len(paths)} sources...")
    for input_path in paths:
        print(f"  Loading {input_path}...")
        for line_num, record, parse_error in _iter_jsonl(input_path):
            summary["input_records"] += 1
            source = _source_for_record(record, input_path)
            stats = source_stats.setdefault(source, _new_source_stats())
            stats["total"] += 1

            if parse_error is not None:
                summary["invalid_records"] += 1
                stats["invalid"] += 1
                summary["rejection_details"].append(
                    f"{input_path}:{line_num}: {parse_error}"
                )
                continue

            (
                valid,
                validation_error,
                text,
                entities,
                meta,
                excluded_entries,
                validation_warnings,
            ) = _validate_with(validator, record, line_num)

            for warning in validation_warnings:
                summary["warning_details"].append(f"{input_path}:{warning}")
            if not valid or text is None or meta is None or not isinstance(record, dict):
                summary["invalid_records"] += 1
                stats["invalid"] += 1
                summary["rejection_details"].append(
                    f"{input_path}:{line_num}: {validation_error or 'Invalid record'}"
                )
                continue

            has_deferred_span = False
            has_healthcare_span = False
            for entry in excluded_entries:
                label = entry["source_label"]
                training_exclusion = entry.get("training_exclusion") is True
                category = str(entry.get("category", entry.get("disposition", "")))
                reason = str(entry.get("reason", ""))
                summary["excluded_spans"] += 1
                stats["excluded_spans"] += 1
                summary["excluded_by_label"][label] += 1
                summary["excluded_dispositions"][
                    (source, label, category, reason, training_exclusion)
                ] += 1
                if training_exclusion:
                    summary["training_exclusion_spans"] += 1
                    stats["training_exclusion_spans"] += 1
                    summary["training_exclusion_by_label"][label] += 1
                else:
                    summary["benign_trace_spans"] += 1
                    stats["benign_trace_spans"] += 1
                if _is_deferred(entry):
                    has_deferred_span = True
                    summary["deferred_spans"] += 1
                    summary["deferred_by_label"][label] += 1
                if _is_healthcare(entry):
                    has_healthcare_span = True
                    summary["healthcare_spans"] += 1
                    summary["healthcare_by_label"][label] += 1

            if has_deferred_span:
                summary["deferred_records"] += 1
            if has_healthcare_span:
                summary["healthcare_records"] += 1

            must_exclude = meta.get("training_exclusion") is True or any(
                entry.get("training_exclusion") is True for entry in excluded_entries
            )
            if must_exclude:
                removed_spans = len(entities)
                summary["training_excluded_records"] += 1
                summary["training_excluded_canonical_spans"] += removed_spans
                stats["training_excluded"] += 1
                stats["training_excluded_canonical_spans"] += removed_spans
                continue

            entity_signature = tuple(sorted(entities))
            if text in seen_texts:
                first_file, first_line, first_entities = seen_texts[text]
                summary["duplicate_records"] += 1
                stats["duplicates"] += 1
                if first_entities != entity_signature:
                    summary["duplicate_annotation_conflicts"] += 1
                    summary["warning_details"].append(
                        f"{input_path}:{line_num}: duplicate text has different "
                        f"annotations than {first_file}:{first_line}"
                    )
                if dedupe:
                    summary["duplicates_removed"] += 1
                    continue
            else:
                seen_texts[text] = (str(input_path), line_num, entity_signature)

            merged_records.append(record)
            stats["valid"] += 1
            stats["entities"] += len(entities)
            summary["merged_records"] += 1
            summary["merged_entities"] += len(entities)
            if not entities:
                stats["negative_records"] += 1
                summary["negative_records"] += 1
            for _, _, label in entities:
                summary["entity_counts"][label] += 1

            source_kind = "synthetic" if _is_synthetic(meta) else "external"
            summary["source_mix_records"][source_kind] += 1
            summary["source_mix_entities"][source_kind] += len(entities)

    _write_jsonl(Path(output_file), merged_records)
    _write_lineage_report(
        Path(report_file),
        input_files,
        source_stats,
        summary,
    )
    merge_sources.last_summary = summary

    print(f"Merged {summary['merged_records']} of {summary['input_records']} records")
    print(
        f"Invalid: {summary['invalid_records']}, "
        f"training-excluded: {summary['training_excluded_records']}, "
        f"duplicates removed: {summary['duplicates_removed']}"
    )
    print(
        f"Excluded spans: {summary['excluded_spans']} total, "
        f"{summary['training_exclusion_spans']} training-excluding"
    )
    print(f"Output: {output_file}")
    print(f"Report: {report_file}")
    return (
        summary["merged_records"],
        summary["input_records"],
        {source: dict(stats) for source, stats in source_stats.items()},
    )


merge_sources.last_summary = {}  # type: ignore[attr-defined]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge validated SecureLogX canonical JSONL sources"
    )
    parser.add_argument(
        "--in", dest="input_files", nargs="+", required=True, help="Input JSONL files"
    )
    parser.add_argument("--out", required=True, help="Output merged JSONL")
    parser.add_argument("--report", required=True, help="Output data lineage report")
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Canonical label config (default: configs/securelogx_labels.json)",
    )
    parser.add_argument(
        "--dedupe", action="store_true", help="Remove subsequent exact duplicate texts"
    )
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Return non-zero if any malformed/unsupported record was rejected",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        merged, total, _ = merge_sources(
            input_files=args.input_files,
            output_file=args.out,
            report_file=args.report,
            dedupe=args.dedupe,
            labels_path=args.labels,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 2

    summary = merge_sources.last_summary  # type: ignore[attr-defined]
    print(f"\nMerge Summary: {merged}/{total} records")
    if merged == 0:
        return 1
    if args.fail_on_invalid and summary.get("invalid_records", 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
