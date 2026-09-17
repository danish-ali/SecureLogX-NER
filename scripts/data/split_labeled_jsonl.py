#!/usr/bin/env python3
"""Create deterministic group-aware SecureLogX train/dev/test splits.

Synthetic template families are indivisible. External records, which have no
template-family concept, remain independently assignable and are called out as
a documented limitation. Reports are calculated from final split contents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SPLIT_NAMES: Tuple[str, ...] = ("train", "dev", "test")
SYNTHETIC_SOURCE = "securelogx_custom_logs"
DEFERRED_LABELS = {
    "PASSWORD", "PIN", "OTP", "SECURITY_ANSWER", "JWT", "SESSION_ID",
    "COOKIE", "CVV", "CARD_EXPIRY", "MAC_ADDRESS", "GEOLOCATION",
    "BIOMETRIC_ID",
}
HEALTHCARE_LABELS = {
    "PROVIDER_NPI", "MRN", "PATIENT_ID", "HEALTH_PLAN_ID", "CLAIM_ID",
    "PRESCRIPTION_ID", "DIAGNOSIS_CODE", "PROCEDURE_CODE", "LAB_ORDER_ID",
}


def load_jsonl(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def load_labels(path: str) -> List[str]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    labels = config.get("entities")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"{path}: 'entities' must be a non-empty string list")
    return labels


def record_labels(record: Mapping[str, object]) -> List[str]:
    entities = record.get("entities", [])
    if not isinstance(entities, list):
        return []
    return [str(entity[2]) for entity in entities if isinstance(entity, list) and len(entity) == 3]


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256((str(seed) + "\0" + "\0".join(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def allocation_counts(size: int, ratios: Sequence[float]) -> List[int]:
    raw = [size * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(len(ratios)), key=lambda item: raw[item] - counts[item], reverse=True)[: size - sum(counts)]:
        counts[index] += 1

    # Template coverage in all active splits is more useful than a perfect
    # ratio when a stratum has only a handful of families.
    if size >= sum(ratio > 0 for ratio in ratios):
        for index, ratio in enumerate(ratios):
            if ratio <= 0 or counts[index] > 0:
                continue
            donor = max((item for item in range(len(counts)) if counts[item] > 1), key=lambda item: counts[item], default=None)
            if donor is not None:
                counts[donor] -= 1
                counts[index] += 1
    return counts


def build_groups(records: Sequence[Dict[str, object]], valid_labels: Sequence[str]) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, Tuple[str, str]], int]:
    valid = set(valid_labels)
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    strata: Dict[str, Tuple[str, str]] = {}
    external_fallbacks = 0

    for index, record in enumerate(records):
        meta = record.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"record {index}: meta must be an object")
        source = str(meta.get("source", ""))
        labels = record_labels(record)
        declared_primary = str(meta.get("primary_entity", ""))
        primary = declared_primary if declared_primary in valid or declared_primary == "O" else (labels[0] if labels else "O")

        if source == SYNTHETIC_SOURCE:
            family = meta.get("template_family")
            if not isinstance(family, str) or not family.strip():
                raise ValueError(f"record {index}: synthetic record missing non-empty meta.template_family")
            group_id = f"synthetic:{family}"
            scenario_kind = str(meta.get("scenario_kind", "positive"))
            if scenario_kind == "hard_negative":
                # Keep the hard-negative evaluation pool independent from its
                # incidental positive label so at least one complete family is
                # held out in dev and test.
                stratum = ("synthetic_hard_negative", "ALL")
            elif scenario_kind == "negative":
                stratum = ("synthetic_negative", "O")
            else:
                stratum = ("synthetic", primary)
        else:
            source_id = meta.get("source_record_id")
            if source_id is None or str(source_id).strip() == "":
                text = str(record.get("text", ""))
                source_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                external_fallbacks += 1
            group_id = f"external:{source}:{source_id}"
            stratum = ("external", primary)

        if group_id in strata and strata[group_id] != stratum:
            raise ValueError(f"group {group_id!r} has inconsistent primary-entity metadata")
        groups[group_id].append(record)
        strata[group_id] = stratum

    return dict(groups), strata, external_fallbacks


def assign_groups(
    groups: Mapping[str, Sequence[Dict[str, object]]],
    strata: Mapping[str, Tuple[str, str]],
    ratios: Sequence[float],
    seed: int,
) -> Dict[str, str]:
    by_stratum: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for group_id, stratum in strata.items():
        by_stratum[stratum].append(group_id)

    assignment: Dict[str, str] = {}
    for stratum in sorted(by_stratum):
        group_ids = sorted(by_stratum[stratum])
        random.Random(stable_seed(seed, *stratum)).shuffle(group_ids)
        counts = allocation_counts(len(group_ids), ratios)
        cursor = 0
        for split_name, count in zip(SPLIT_NAMES, counts):
            for group_id in group_ids[cursor:cursor + count]:
                assignment[group_id] = split_name
            cursor += count
        if cursor != len(group_ids):
            raise AssertionError(f"failed to assign every group in stratum {stratum}")
    return assignment


def materialize_splits(
    groups: Mapping[str, Sequence[Dict[str, object]]],
    assignment: Mapping[str, str],
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    splits: Dict[str, List[Dict[str, object]]] = {name: [] for name in SPLIT_NAMES}
    for group_id, records in groups.items():
        splits[assignment[group_id]].extend(records)
    for name, records in splits.items():
        random.Random(stable_seed(seed, "records", name)).shuffle(records)
    return splits


def span_counts(records: Iterable[Mapping[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record_labels(record))
    return counts


def source_counts(records: Iterable[Mapping[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        meta = record.get("meta", {})
        source = meta.get("source", "UNKNOWN") if isinstance(meta, dict) else "UNKNOWN"
        counts[str(source)] += 1
    return counts


def excluded_counts(records: Iterable[Mapping[str, object]]) -> Tuple[Counter[str], Counter[str], int]:
    by_label: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    training_exclusions = 0
    for record in records:
        meta = record.get("meta", {})
        excluded = meta.get("excluded_spans", []) if isinstance(meta, dict) else []
        if not isinstance(excluded, list):
            continue
        for span in excluded:
            if not isinstance(span, dict):
                continue
            label = str(span.get("source_label") or span.get("mapped_label") or "UNKNOWN")
            category = str(span.get("category") or "unsupported")
            by_label[label] += 1
            by_category[category] += 1
            training_exclusions += int(span.get("training_exclusion") is True)
    return by_label, by_category, training_exclusions


def leakage_metrics(splits: Mapping[str, Sequence[Mapping[str, object]]]) -> Dict[str, object]:
    text_sets = {name: {str(record.get("text", "")) for record in records} for name, records in splits.items()}
    exact_pairs = {
        "train_dev": len(text_sets["train"] & text_sets["dev"]),
        "train_test": len(text_sets["train"] & text_sets["test"]),
        "dev_test": len(text_sets["dev"] & text_sets["test"]),
    }
    family_splits: Dict[str, set[str]] = defaultdict(set)
    missing_family = 0
    for split_name, records in splits.items():
        for record in records:
            meta = record.get("meta", {})
            if not isinstance(meta, dict) or meta.get("source") != SYNTHETIC_SOURCE:
                continue
            family = meta.get("template_family")
            if not isinstance(family, str) or not family:
                missing_family += 1
            else:
                family_splits[family].add(split_name)
    leaking_families = {family: sorted(names) for family, names in family_splits.items() if len(names) > 1}
    return {
        "exact_pairs": exact_pairs,
        "exact_total": sum(exact_pairs.values()),
        "template_families": len(family_splits),
        "leaking_families": leaking_families,
        "template_leakage": len(leaking_families),
        "missing_family": missing_family,
    }


def template_diversity(splits: Mapping[str, Sequence[Mapping[str, object]]], labels: Sequence[str]) -> Dict[str, Dict[str, set[str]]]:
    diversity = {label: {name: set() for name in SPLIT_NAMES} for label in labels}
    for split_name, records in splits.items():
        for record in records:
            meta = record.get("meta", {})
            if not isinstance(meta, dict) or meta.get("source") != SYNTHETIC_SOURCE:
                continue
            family = meta.get("template_family")
            if not isinstance(family, str):
                continue
            for label in set(record_labels(record)):
                if label in diversity:
                    diversity[label][split_name].add(family)
    return diversity


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_split_summary(
    path: Path,
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    labels: Sequence[str],
    leakage: Mapping[str, object],
    external_fallbacks: int,
) -> str:
    total_records = sum(len(records) for records in splits.values())
    counts = {name: span_counts(records) for name, records in splits.items()}
    total_spans = sum(sum(counter.values()) for counter in counts.values())
    all_records = [record for records in splits.values() for record in records]
    excluded_labels, excluded_categories, training_exclusions = excluded_counts(all_records)

    lines = [
        "# SecureLogX ML-v1 Dataset Split Summary", "",
        "## Record and span counts", "",
        "| Split | Records | Record % | Entity spans |", "|---|---:|---:|---:|",
    ]
    for name in SPLIT_NAMES:
        record_count = len(splits[name])
        pct = 100 * record_count / total_records if total_records else 0
        lines.append(f"| {name} | {record_count} | {pct:.2f}% | {sum(counts[name].values())} |")
    lines.extend([f"| **Total** | **{total_records}** | **100.00%** | **{total_spans}** |", "", "## Per-label span distribution", "", "| Entity | Train | Dev | Test | Total |", "|---|---:|---:|---:|---:|"])
    for label in labels:
        values = [counts[name][label] for name in SPLIT_NAMES]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |")

    all_sources = sorted({source for records in splits.values() for source in source_counts(records)})
    lines.extend(["", "## Source distribution", "", "| Source | Train | Dev | Test | Total |", "|---|---:|---:|---:|---:|"])
    per_source = {name: source_counts(records) for name, records in splits.items()}
    for source in all_sources:
        values = [per_source[name][source] for name in SPLIT_NAMES]
        lines.append(f"| {source} | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |")

    synthetic_total = sum(per_source[name][SYNTHETIC_SOURCE] for name in SPLIT_NAMES)
    external_total = total_records - synthetic_total
    lines.extend([
        "", "## Synthetic/external ratio", "",
        f"- Synthetic records: **{synthetic_total}** ({100 * synthetic_total / total_records:.2f}%)",
        f"- External records: **{external_total}** ({100 * external_total / total_records:.2f}%)",
        "", "## Independence gates", "",
        f"- Exact duplicate text leakage: **{leakage['exact_total']}**",
        f"- Synthetic template-family leakage: **{leakage['template_leakage']}**",
        f"- Synthetic records missing template family: **{leakage['missing_family']}**",
        "", "## Unsupported/deferred metadata retained in final data", "",
        f"- Spans marked as training-excluding: **{training_exclusions}** (must be 0 in final splits)",
    ])
    if excluded_labels:
        for label, count in sorted(excluded_labels.items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- No excluded-span metadata remains in final training records.")
    if excluded_categories:
        lines.append("- Categories: " + ", ".join(f"{key}={value}" for key, value in sorted(excluded_categories.items())))
    lines.extend([
        "", "## External-source limitation", "",
        "External datasets do not provide SecureLogX template-family metadata, so they are split by stable source record ID while retaining exact-text deduplication.",
        f"Records requiring a text-hash fallback because source_record_id was absent: **{external_fallbacks}**.", "",
    ])
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def write_leakage_report(path: Path, leakage: Mapping[str, object]) -> None:
    exact_pairs = leakage["exact_pairs"]
    assert isinstance(exact_pairs, dict)
    leaking = leakage["leaking_families"]
    assert isinstance(leaking, dict)
    lines = [
        "# Template and Text Leakage Report", "", "## Gate results", "",
        "| Gate | Target | Result | Status |", "|---|---:|---:|---|",
        f"| Exact duplicate text leakage | 0 | {leakage['exact_total']} | {'PASS' if leakage['exact_total'] == 0 else 'FAIL'} |",
        f"| Synthetic template-family leakage | 0 | {leakage['template_leakage']} | {'PASS' if leakage['template_leakage'] == 0 else 'FAIL'} |",
        f"| Synthetic records missing template family | 0 | {leakage['missing_family']} | {'PASS' if leakage['missing_family'] == 0 else 'FAIL'} |",
        "", "## Exact-text pair checks", "",
        f"- Train/dev: {exact_pairs['train_dev']}", f"- Train/test: {exact_pairs['train_test']}", f"- Dev/test: {exact_pairs['dev_test']}",
        "", f"Synthetic template families checked: **{leakage['template_families']}**.", "",
        "External-source records have no template-family field and therefore cannot be evaluated by the template-family gate; they remain protected by stable-record grouping and exact-text deduplication.",
    ]
    if leaking:
        lines.extend(["", "## Leaking families", ""])
        lines.extend(f"- `{family}`: {', '.join(names)}" for family, names in sorted(leaking.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readiness_report(
    path: Path,
    splits: Mapping[str, Sequence[Mapping[str, object]]],
    labels: Sequence[str],
    train_min: int,
    dev_min: int,
    test_min: int,
    leakage: Mapping[str, object],
) -> bool:
    counts = {name: span_counts(records) for name, records in splits.items()}
    diversity = template_diversity(splits, labels)
    lines = [
        "# SecureLogX ML-v1 Label Readiness", "",
        f"Coverage gates: train >= {train_min}, dev >= {dev_min}, test >= {test_min} spans.", "",
        "| Entity | Train spans | Dev spans | Test spans | Template diversity | Readiness | Notes |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    all_ready = True
    for label in labels:
        values = [counts[name][label] for name in SPLIT_NAMES]
        failures = []
        for name, value, minimum in zip(SPLIT_NAMES, values, (train_min, dev_min, test_min)):
            if value < minimum:
                failures.append(f"{name} short by {minimum - value}")
        ready = not failures
        all_ready &= ready
        family_sets = diversity[label]
        all_families = set().union(*(family_sets[name] for name in SPLIT_NAMES))
        diversity_text = f"{len(all_families)} total ({len(family_sets['train'])}/{len(family_sets['dev'])}/{len(family_sets['test'])})"
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} | {diversity_text} | {'PASS' if ready else 'NEEDS_MORE_DATA'} | {', '.join(failures) if failures else 'Meets provisional span gates'} |")

    leakage_ready = leakage["exact_total"] == 0 and leakage["template_leakage"] == 0 and leakage["missing_family"] == 0
    recommendation = "READY FOR TRAINING" if all_ready and leakage_ready else "NOT READY FOR TRAINING"
    lines.extend([
        "", "## Recommendation", "", f"**{recommendation}**", "",
        "This is a data-coverage and split-independence recommendation only. It is not model evaluation or production-readiness evidence.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_ready and leakage_ready


def write_hard_negative_report(path: Path, splits: Mapping[str, Sequence[Mapping[str, object]]]) -> Tuple[int, int]:
    families: Dict[str, Dict[str, object]] = {}
    record_count = 0
    for split_name, records in splits.items():
        for record in records:
            meta = record.get("meta", {})
            if not isinstance(meta, dict) or meta.get("scenario_kind") != "hard_negative":
                continue
            record_count += 1
            family = str(meta.get("template_family", "MISSING"))
            entry = families.setdefault(family, {"rationales": set(), "formats": set(), "splits": Counter(), "records": 0})
            entry["rationales"].add(str(meta.get("annotation_rationale", "")))  # type: ignore[union-attr]
            entry["formats"].add(str(meta.get("format", "unknown")))  # type: ignore[union-attr]
            entry["splits"][split_name] += 1  # type: ignore[index]
            entry["records"] += 1  # type: ignore[operator]

    lines = [
        "# Hard-Negative Scenario Summary", "",
        f"- Hard-negative template families: **{len(families)}**",
        f"- Hard-negative records: **{record_count}**", "",
        "| Template family | Format | Assigned split | Records | Expected annotation rationale |",
        "|---|---|---|---:|---|",
    ]
    for family, entry in sorted(families.items()):
        formats = ", ".join(sorted(entry["formats"]))  # type: ignore[arg-type]
        split_counts: Counter[str] = entry["splits"]  # type: ignore[assignment]
        assigned = ", ".join(f"{name}={split_counts[name]}" for name in SPLIT_NAMES if split_counts[name])
        rationale = " ".join(sorted(entry["rationales"]))  # type: ignore[arg-type]
        lines.append(f"| `{family}` | {formats} | {assigned} | {entry['records']} | {rationale} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(families), record_count


def split_dataset(
    input_file: str,
    output_dir: str,
    labels_path: str,
    report_path: str,
    readiness_report: str,
    leakage_report: str,
    hard_negative_report: str,
    train_ratio: float = 0.80,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    train_min: int = 300,
    dev_min: int = 40,
    test_min: int = 40,
) -> Tuple[int, int, int, bool]:
    ratios = (train_ratio, dev_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be non-negative and sum to 1.0, got {sum(ratios)}")
    labels = load_labels(labels_path)
    records = load_jsonl(input_file)
    duplicate_count = len(records) - len({str(record.get("text", "")) for record in records})
    if duplicate_count:
        raise ValueError(f"input contains {duplicate_count} duplicate text records; merge with --dedupe first")

    groups, strata, external_fallbacks = build_groups(records, labels)
    assignment = assign_groups(groups, strata, ratios, seed)
    splits = materialize_splits(groups, assignment, seed)
    leakage = leakage_metrics(splits)

    output_path = Path(output_dir)
    for name in SPLIT_NAMES:
        write_jsonl(output_path / f"{name}.jsonl", splits[name])
    summary_text = write_split_summary(Path(report_path), splits, labels, leakage, external_fallbacks)
    legacy_report = output_path / "split_summary.md"
    if legacy_report.resolve() != Path(report_path).resolve():
        legacy_report.write_text(summary_text, encoding="utf-8")
    write_leakage_report(Path(leakage_report), leakage)
    ready = write_readiness_report(Path(readiness_report), splits, labels, train_min, dev_min, test_min, leakage)
    hard_families, hard_records = write_hard_negative_report(Path(hard_negative_report), splits)

    if leakage["exact_total"] or leakage["template_leakage"] or leakage["missing_family"]:
        raise RuntimeError(
            f"split independence failed: exact={leakage['exact_total']}, "
            f"template={leakage['template_leakage']}, missing_family={leakage['missing_family']}"
        )

    print(f"Train/dev/test records: {len(splits['train'])}/{len(splits['dev'])}/{len(splits['test'])}")
    print(f"Groups: {len(groups)}; synthetic template families: {leakage['template_families']}")
    print(f"Leakage: exact={leakage['exact_total']}, template_family={leakage['template_leakage']}")
    print(f"Hard negatives: {hard_records} records across {hard_families} families")
    print(f"Coverage readiness: {'PASS' if ready else 'NEEDS_MORE_DATA'}")
    return len(splits["train"]), len(splits["dev"]), len(splits["test"]), ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Split SecureLogX ML-v1 JSONL without template leakage")
    parser.add_argument("--in", dest="input_file", required=True, help="Input merged JSONL")
    parser.add_argument("--out", required=True, help="Output directory for train/dev/test")
    parser.add_argument("--labels", default="configs/securelogx_labels.json", help="Canonical labels JSON")
    parser.add_argument("--report", default="reports/split_summary.md", help="Split summary Markdown")
    parser.add_argument("--readiness-report", default="reports/ml_v1_label_readiness.md")
    parser.add_argument("--leakage-report", default="reports/template_leakage_report.md")
    parser.add_argument("--hard-negative-report", default="reports/hard_negative_summary.md")
    parser.add_argument("--train", type=float, default=0.80, help="Train ratio")
    parser.add_argument("--dev", type=float, default=0.10, help="Dev ratio")
    parser.add_argument("--test", type=float, default=0.10, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-min", type=int, default=300)
    parser.add_argument("--dev-min", type=int, default=40)
    parser.add_argument("--test-min", type=int, default=40)
    args = parser.parse_args()
    try:
        split_dataset(
            args.input_file, args.out, args.labels, args.report,
            args.readiness_report, args.leakage_report, args.hard_negative_report,
            args.train, args.dev, args.test, args.seed,
            args.train_min, args.dev_min, args.test_min,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
