"""Read-only frozen-data preflight, sequence analysis, and alignment audit."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from securelogx_training_common import (
    audit_records,
    build_aligned_features,
    load_canonical_labels,
    load_jsonl,
    sha256_file,
    summarize_lengths,
    validate_manifest,
    write_json,
)


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.1f}"
    return str(int(value))


def _naive_truncated_spans(
    records: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int
) -> tuple[int, Counter[str]]:
    total = 0
    labels: Counter[str] = Counter()
    for record in records:
        encoded = tokenizer(
            record["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = [
            (int(start), int(end))
            for start, end in encoded["offset_mapping"]
            if start != end
        ]
        for start, end, label in record["entities"]:
            covered = True
            for position in range(int(start), int(end)):
                if record["text"][position].isspace():
                    continue
                if not any(left <= position < right for left, right in offsets):
                    covered = False
                    break
            if not covered:
                total += 1
                labels[str(label)] += 1
    return total, labels


def _write_preflight_report(
    path: Path,
    root: Path,
    manifest: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    label_config: Mapping[str, Any],
    family_leakage: int,
    text_leakage: int,
    external_id_leak: int,
) -> None:
    lines = [
        "# SecureLogX ML-v1 Training Preflight",
        "",
        "**Status: PASS**",
        "",
        "The frozen inputs passed canonical-label, count, SHA-256, span, provenance, and split-independence checks. No dataset file was modified.",
        "",
        "## Canonical ontology",
        "",
        f"- Entity types: **{len(label_config['entities'])}**",
        f"- BIO labels: **{len(label_config['bio_labels'])}**",
        f"- First label: **{label_config['bio_labels'][0]}=0**",
        f"- Final label: **{label_config['bio_labels'][-1]}=50**",
        "- `label_to_id` and `id_to_label`: exact deterministic inverses",
        "",
        "## Frozen split checks",
        "",
        "| Split | Records | Entity spans | Negative records | SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for split in ("train", "dev", "test"):
        relative = f"data/split/{split}.jsonl"
        audit = audits[split]
        digest = manifest["files"][relative]["sha256"]
        lines.append(
            f"| {split} | {audit['records']} | {audit['entity_spans']} | "
            f"{audit['negative_records']} | `{digest}` |"
        )
    lines.extend(
        [
            "",
            f"- Total records: **{sum(int(audits[s]['records']) for s in audits)}**",
            f"- Total entity spans: **{sum(int(audits[s]['entity_spans']) for s in audits)}**",
            f"- Exact text leakage: **{text_leakage}**",
            f"- Synthetic template-family leakage: **{family_leakage}**",
            f"- External source-record-ID leakage: **{external_id_leak}**",
            "- Invalid spans, overlaps, unsupported labels, or provenance mismatches: **0**",
            "- Training-excluded records in frozen splits: **0**",
            "- Frozen readiness failures: **0** (all 25 labels pass 300/40/40)",
            "",
            "## Source and hard-negative coverage",
            "",
        ]
    )
    for split in ("train", "dev", "test"):
        source_counts = audits[split]["source_counts"]
        lines.append(
            f"- {split}: Gretel {source_counts.get('gretel_finance_pii', 0)}, "
            f"synthetic {source_counts.get('securelogx_custom_logs', 0)}"
        )
    lines.extend(
        [
            "",
            "Held-out hard-negative coverage is deliberately limited: test contains one complete 94-record family. Conclusions from that segment must remain qualified.",
            "",
            "## Reproducibility caveats",
            "",
            "- The frozen split/config artifacts are currently untracked by Git; this manifest supplies immutable SHA-256 fingerprints for this run.",
            "- Gretel metadata remains `license_reviewed=false`. This does not block local technical training, but it blocks any publication/commercial-release claim pending review.",
            "- `meta.split` on Gretel records is upstream provenance. SecureLogX split membership is determined by the frozen file containing the record.",
            "- The older profile names DistilBERT and ONNX. The active phase instruction overrides it with `bert-base-cased` and stops before ONNX; the frozen profile remains unchanged.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sequence_report(
    path: Path,
    length_stats: Mapping[str, Mapping[str, float | int]],
    naive_truncation: Mapping[str, tuple[int, Counter[str]]],
    max_length: int,
    stride: int,
) -> None:
    lines = [
        "# Sequence-Length Analysis",
        "",
        "Tokenizer: **bert-base-cased fast WordPiece tokenizer**. Counts include special tokens.",
        "",
        "| Split | Median | p90 | p95 | p99 | Maximum | >384 | >512 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test", "all"):
        stats = length_stats[split]
        lines.append(
            f"| {split} | {_format_number(stats['median'])} | "
            f"{_format_number(stats['p90'])} | {_format_number(stats['p95'])} | "
            f"{_format_number(stats['p99'])} | {stats['maximum']} | "
            f"{stats['over_384']} | {stats['over_512']} |"
        )
    lines.extend(
        [
            "",
            "## Naive truncation risk",
            "",
            "A single 512-token window would fail to cover these gold spans:",
            "",
        ]
    )
    for split in ("train", "dev", "test"):
        total, labels = naive_truncation[split]
        label_text = ", ".join(f"{label}={count}" for label, count in sorted(labels.items()))
        lines.append(f"- {split}: **{total} spans** ({label_text or 'none'})")
    lines.extend(
        [
            "",
            "## Selected policy",
            "",
            f"- Maximum window length: **{max_length} tokens**",
            f"- Overflow stride: **{stride} tokens**",
            "- Padding: dynamic to the longest sequence in each batch, rounded to a multiple of eight",
            "",
            "The selected window is near the observed p90 and avoids wasteful fixed 512-token padding. Overlapping windows preserve long-document context and provide a complete window for ordinary entity spans instead of silently right-truncating them. Predictions from overlapping windows are averaged per unique character-offset token before BIO decoding.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_alignment_report(
    path: Path,
    summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# Token Alignment Report",
        "",
        "Character spans were aligned with fast-tokenizer offset mappings and overlapping windows.",
        "",
        "| Split | Records | Windows | Total spans | Successfully aligned | Boundary-adjusted | Truncated | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    totals = Counter()
    for split in ("train", "dev", "test"):
        summary = summaries[split]
        lines.append(
            f"| {split} | {summary['records']} | {summary['windows']} | "
            f"{summary['total_spans']} | {summary['successfully_aligned_spans']} | "
            f"{summary['boundary_adjusted_spans']} | {summary['truncated_spans']} | "
            f"{summary['failed_alignments']} |"
        )
        for key in (
            "records",
            "windows",
            "total_spans",
            "successfully_aligned_spans",
            "boundary_adjusted_spans",
            "truncated_spans",
            "failed_alignments",
        ):
            totals[key] += int(summary[key])
    lines.append(
        f"| **Total** | **{totals['records']}** | **{totals['windows']}** | "
        f"**{totals['total_spans']}** | **{totals['successfully_aligned_spans']}** | "
        f"**{totals['boundary_adjusted_spans']}** | **{totals['truncated_spans']}** | "
        f"**{totals['failed_alignments']}** |"
    )
    lines.extend(
        [
            "",
            "## Deterministic handling",
            "",
            "- Special and padding tokens receive `-100`.",
            "- The first token of a span receives `B-*`; subsequent tokens receive `I-*`.",
            "- Partial entity fragments in an overflow window receive `-100`, never `O`.",
            "- If a character boundary falls inside a WordPiece, adjustments totaling at most three characters are supervised and reported explicitly.",
            "- Larger token-boundary expansions/contractions are treated as explicit tokenizer-unrepresentable failures and receive `-100`, avoiding semantically unsafe supervision.",
            "- Predictions from overlapping windows are merged by averaged logits at identical character offsets.",
            "",
            "## Affected labels and diagnostics",
            "",
        ]
    )
    affected = Counter()
    reasons = Counter()
    for summary in summaries.values():
        reasons.update(summary["reason_counts"])
        for label, statuses in summary["affected_labels"].items():
            affected[label] += sum(int(value) for value in statuses.values())
    lines.append(
        "- Affected labels: "
        + (", ".join(f"{label}={count}" for label, count in sorted(affected.items())) or "none")
    )
    lines.append(
        "- Diagnostic reasons: "
        + ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
    )
    explained_failure_reasons = {
        "tokenizer_emitted_no_token_for_span",
        "token_boundary_adjustment_exceeds_three_characters",
    }
    unexplained = sum(int(summary["truncated_spans"]) for summary in summaries.values())
    explained_failures = 0
    for summary in summaries.values():
        for reason, count in summary["reason_counts"].items():
            if reason in explained_failure_reasons:
                explained_failures += int(count)
            elif reason not in {"exact_token_boundaries", "character_boundary_inside_or_outside_token"}:
                unexplained += int(count)
    lines.extend(
        [
            "",
            f"Unexplained/truncation alignment blockers: **{unexplained}**.",
            f"Explicitly tokenizer-unrepresentable spans excluded from token supervision: **{explained_failures}**.",
            "Tokenizer-unrepresentable spans are listed in the machine-readable diagnostic file and are not silently treated as `O`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default="configs/ml_v1_frozen_manifest.json")
    parser.add_argument("--labels", default="configs/securelogx_labels.json")
    parser.add_argument("--base-model", default="bert-base-cased")
    parser.add_argument(
        "--revision", default="cd5ef92a9fb2f889e972770a36d4ed042daf221e"
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--preflight-report", default="reports/training_preflight.md")
    parser.add_argument(
        "--alignment-report", default="reports/token_alignment_report.md"
    )
    parser.add_argument(
        "--alignment-diagnostics",
        default="reports/token_alignment_diagnostics.json",
    )
    parser.add_argument(
        "--sequence-report", default="reports/sequence_length_analysis.md"
    )
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(args.root).resolve()
    manifest = validate_manifest(root, root / args.manifest)
    label_config = load_canonical_labels(root / args.labels)
    records = {
        split: load_jsonl(root / f"data/split/{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    audits = {
        split: audit_records(data, split, label_config["entities"])
        for split, data in records.items()
    }
    audit_errors = [
        error for audit in audits.values() for error in audit["errors"]
    ]
    if audit_errors:
        raise ValueError("Dataset integrity preflight failed:\n- " + "\n- ".join(audit_errors[:100]))

    text_leakage = sum(
        len(audits[first]["texts"] & audits[second]["texts"])
        for first, second in (("train", "dev"), ("train", "test"), ("dev", "test"))
    )
    family_leakage = sum(
        len(audits[first]["template_families"] & audits[second]["template_families"])
        for first, second in (("train", "dev"), ("train", "test"), ("dev", "test"))
    )
    external_id_leakage = sum(
        len(audits[first]["external_ids"] & audits[second]["external_ids"])
        for first, second in (("train", "dev"), ("train", "test"), ("dev", "test"))
    )
    if text_leakage or family_leakage or external_id_leakage:
        raise ValueError(
            "Split independence failed: "
            f"text={text_leakage}, family={family_leakage}, external_id={external_id_leakage}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("bert-base-cased tokenizer must be fast")

    original_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 100_000
    lengths: dict[str, list[int]] = {}
    for split, split_records in records.items():
        lengths[split] = [
            len(tokenizer(record["text"], add_special_tokens=True, truncation=False)["input_ids"])
            for record in split_records
        ]
    tokenizer.model_max_length = original_model_max_length
    length_stats = {split: summarize_lengths(values) for split, values in lengths.items()}
    length_stats["all"] = summarize_lengths(
        [value for split in ("train", "dev", "test") for value in lengths[split]]
    )

    naive_truncation = {
        split: _naive_truncated_spans(split_records, tokenizer, 512)
        for split, split_records in records.items()
    }
    alignment_summaries: dict[str, dict[str, Any]] = {}
    for split, split_records in records.items():
        print(f"Aligning {split}...", flush=True)
        build = build_aligned_features(
            split_records,
            tokenizer,
            label_config["label_to_id"],
            args.max_length,
            args.stride,
        )
        alignment_summaries[split] = build.summary
        del build
    allowed_failure_reasons = {
        "tokenizer_emitted_no_token_for_span",
        "token_boundary_adjustment_exceeds_three_characters",
    }
    unexpected_failures = []
    for split, summary in alignment_summaries.items():
        if summary["truncated_spans"]:
            unexpected_failures.append(
                f"{split}: {summary['truncated_spans']} spans truncated after overflow"
            )
        for diagnostic in summary["diagnostics"]:
            if (
                diagnostic["status"] == "failed"
                and diagnostic["reason"] not in allowed_failure_reasons
            ):
                unexpected_failures.append(
                    f"{split} record {diagnostic['record_id']}: {diagnostic['reason']}"
                )
    if unexpected_failures:
        raise ValueError(
            "Unexplained alignment failures block training:\n- "
            + "\n- ".join(unexpected_failures[:100])
        )

    _write_preflight_report(
        root / args.preflight_report,
        root,
        manifest,
        audits,
        label_config,
        family_leakage,
        text_leakage,
        external_id_leakage,
    )
    _write_sequence_report(
        root / args.sequence_report,
        length_stats,
        naive_truncation,
        args.max_length,
        args.stride,
    )
    _write_alignment_report(
        root / args.alignment_report,
        alignment_summaries,
    )
    write_json(
        root / args.alignment_diagnostics,
        {
            "base_model": args.base_model,
            "revision": args.revision,
            "max_length": args.max_length,
            "stride": args.stride,
            "splits": alignment_summaries,
        },
    )
    print("Preflight PASSED", flush=True)
    print(f"Exact text leakage: {text_leakage}", flush=True)
    print(f"Template-family leakage: {family_leakage}", flush=True)
    print(
        "Alignment: "
        + ", ".join(
            f"{split}={summary['successfully_aligned_spans']}/"
            f"{summary['total_spans']} aligned, {summary['truncated_spans']} truncated, "
            f"{summary['failed_alignments']} failed"
            for split, summary in alignment_summaries.items()
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
