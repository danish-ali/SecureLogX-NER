"""Train the controlled SecureLogX ML-v1.1 bert-base-cased experiment.

The command consumes only the frozen ML-v1 train split, the frozen ML-v1.1
training addition, the original development split, and the independent
development challenge.  The sealed challenge is never parsed or evaluated by
this program; only its bytes are hashed as part of the frozen-evidence audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from securelogx_ml_v1_1_metrics import (
    context_diagnostics,
    evaluate_named_gates,
    harmonic_mean,
    supported_macro,
)
from securelogx_training_common import (
    AlignedWindowDataset,
    IGNORE_INDEX,
    TokenClassificationCollator,
    build_aligned_features,
    collect_environment,
    evaluate_model,
    load_canonical_labels,
    load_jsonl,
    model_inputs_from_batch,
    score_predictions,
    set_reproducible_seed,
    sha256_file,
    validate_manifest,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ALIGNMENT_FAILURES = {
    "tokenizer_emitted_no_token_for_span",
    "token_boundary_adjustment_exceeds_three_characters",
}
FULL_PATHS = {
    "manifest": "configs/ml_v1_1_dataset_manifest.json",
    "parent_freeze": "configs/ml_v1_1_parent_freeze.json",
    "gate": "configs/ml_v1_1_training_gate.json",
    "labels": "configs/securelogx_labels.json",
    "parent_train": "data/split/train.jsonl",
    "train_additions": "data/ml_v1_1/context_contrast/train_additions.jsonl",
    "standard_dev": "data/split/dev.jsonl",
    "challenge_dev": "data/ml_v1_1/context_contrast/dev_challenge.jsonl",
    "output": "output_securelogx/ml-v1.1/bert-base-cased",
    "preflight_report": "reports/ml_v1_1_training_preflight.md",
    "training_report": "reports/ml_v1_1_training_run_summary.md",
    "standard_metrics": "reports/ml_v1_1_dev_standard_metrics.json",
    "challenge_metrics": "reports/ml_v1_1_dev_challenge_metrics.json",
    "selection_report": "reports/ml_v1_1_checkpoint_selection.md",
}
FULL_SETTINGS = {
    "base_model": "bert-base-cased",
    "revision": "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
    "max_length": 384,
    "stride": 128,
    "epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "train_batch_size": 4,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "seed": 42,
    "allow_cpu": False,
}
EXPECTED_COUNTS = {
    "parent_train": 23558,
    "train_additions": 3000,
    "training": 26558,
    "standard_dev": 2994,
    "challenge_dev": 480,
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _mapping_fingerprint(values: Mapping[str, str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _detached_hash(path: Path) -> str:
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 1 or len(parts[0]) != 64:
        raise ValueError(f"Malformed detached SHA-256 file: {path}")
    return parts[0].lower()


def _verify_expected_file(root: Path, relative: str, expected: str) -> dict[str, Any]:
    path = root / relative
    actual = sha256_file(path) if path.is_file() else None
    return {
        "path": relative,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "passed": actual == expected,
    }


def verify_frozen_evidence(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Hash every frozen parent/new artifact without parsing the sealed records."""

    manifest_path = (root / args.manifest).resolve()
    parent_freeze_path = (root / args.parent_freeze).resolve()
    gate_path = (root / args.gate).resolve()
    gate = _load_json(gate_path)
    manifest = _load_json(manifest_path)
    parent = _load_json(parent_freeze_path)
    checks: list[dict[str, Any]] = []

    manifest_expected = str(
        gate["frozen_inputs"]["ml_v1_1_dataset_manifest"]["sha256"]
    )
    checks.append(
        _verify_expected_file(root, args.manifest, manifest_expected)
    )
    manifest_detached = _detached_hash(
        manifest_path.with_name("ml_v1_1_dataset_manifest.sha256")
    )
    checks.append(
        {
            "path": "configs/ml_v1_1_dataset_manifest.sha256",
            "expected_sha256": manifest_expected,
            "actual_sha256": manifest_detached,
            "passed": manifest_detached == manifest_expected,
        }
    )
    gate_actual = sha256_file(gate_path)
    gate_detached = _detached_hash(gate_path.with_suffix(".sha256"))
    checks.append(
        {
            "path": args.gate,
            "expected_sha256": gate_detached,
            "actual_sha256": gate_actual,
            "passed": gate_actual == gate_detached and gate.get("declared_before_training") is True,
        }
    )

    entries: list[tuple[str, str]] = [
        (parent["parent_manifest"]["path"], parent["parent_manifest"]["sha256"]),
        (parent["ontology"]["path"], parent["ontology"]["sha256"]),
    ]
    for section in ("parent_configs", "parent_splits", "evaluation_evidence"):
        entries.extend(
            (relative, values["sha256"])
            for relative, values in parent[section].items()
        )
    for relative, expected in entries:
        checks.append(_verify_expected_file(root, relative, expected))
    for relative, values in manifest["new_data_artifacts"].items():
        checks.append(_verify_expected_file(root, relative, values["sha256"]))

    checkpoint = parent["checkpoint"]
    checkpoint_root = root / checkpoint["path"]
    actual_names = sorted(path.name for path in checkpoint_root.iterdir() if path.is_file())
    checkpoint_hashes: dict[str, str] = {}
    for name, expected in sorted(checkpoint["files"].items()):
        relative = f"{checkpoint['path']}/{name}"
        check = _verify_expected_file(root, relative, expected)
        checks.append(check)
        if check["actual_sha256"] is not None:
            checkpoint_hashes[name] = str(check["actual_sha256"])
    checkpoint_fingerprint = _mapping_fingerprint(checkpoint_hashes)
    checkpoint_passed = (
        actual_names == sorted(checkpoint["files"])
        and checkpoint_fingerprint == checkpoint["fingerprint_sha256"]
    )

    validate_manifest(root, root / parent["parent_manifest"]["path"])
    label_config = load_canonical_labels(root / args.labels)
    ontology_passed = (
        len(label_config["entities"]) == 25
        and len(label_config["bio_labels"]) == 51
        and sha256_file(root / args.labels) == parent["ontology"]["sha256"]
    )
    failures = [check for check in checks if not check["passed"]]
    if failures or not checkpoint_passed or not ontology_passed:
        raise ValueError(
            "Frozen evidence mismatch; refusing ML-v1.1 training: "
            + json.dumps(
                {
                    "file_failures": failures,
                    "checkpoint_passed": checkpoint_passed,
                    "ontology_passed": ontology_passed,
                },
                sort_keys=True,
            )
        )
    return {
        "passed": True,
        "verified_file_checks": len(checks),
        "manifest_sha256": sha256_file(manifest_path),
        "training_gate_sha256": gate_actual,
        "parent_checkpoint_fingerprint_sha256": checkpoint_fingerprint,
        "ontology": {"entities": 25, "bio_labels": 51},
        "sealed_challenge_handling": "bytes hashed only; records not parsed",
    }


def _assert_canonical_full_run(args: argparse.Namespace, root: Path) -> None:
    problems: list[str] = []
    if root != REPOSITORY_ROOT:
        problems.append(f"root={root} (expected {REPOSITORY_ROOT})")
    for argument, expected in FULL_PATHS.items():
        actual_path = (root / str(getattr(args, argument))).resolve()
        expected_path = (root / expected).resolve()
        if actual_path != expected_path:
            problems.append(f"{argument}={actual_path} (expected {expected_path})")
    for argument, expected in FULL_SETTINGS.items():
        if getattr(args, argument) != expected:
            problems.append(f"{argument}={getattr(args, argument)!r} (expected {expected!r})")
    for argument in ("max_parent_train_records", "max_addition_records", "max_dev_records", "max_challenge_records", "max_optimizer_steps"):
        if getattr(args, argument) is not None:
            problems.append(f"{argument} is permitted only with --smoke-test")
    if os.environ.get("PYTHONHASHSEED") != "42":
        problems.append("PYTHONHASHSEED must be exported as 42 before starting Python")
    if problems:
        raise ValueError("Non-smoke ML-v1.1 run is not canonical:\n- " + "\n- ".join(problems))


def _alignment_problems(summary: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if summary["truncated_spans"]:
        problems.append(f"{summary['truncated_spans']} truncated spans")
    for diagnostic in summary["diagnostics"]:
        if diagnostic["status"] == "failed" and diagnostic["reason"] not in ALLOWED_ALIGNMENT_FAILURES:
            problems.append(f"record {diagnostic['record_id']}: {diagnostic['reason']}")
    return problems


def _assert_alignment(summary: Mapping[str, Any], split: str) -> None:
    problems = _alignment_problems(summary)
    if problems:
        raise ValueError(f"{split} has invalid alignment:\n- " + "\n- ".join(problems[:100]))


def _optimizer(model: Any, learning_rate: float, weight_decay: float) -> AdamW:
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"errors", "confusion"}}


def _score_view(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    label_to_id: Mapping[str, int],
    id_to_label: Mapping[int, str],
    split: str,
    batch_size: int,
    device: Any,
    use_bf16: bool,
    context: bool,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    evaluation = evaluate_model(
        model, tokenizer, records, windows, id_to_label, batch_size, device, use_bf16
    )
    metrics = score_predictions(
        records,
        evaluation["predictions"],
        labels,
        split,
        token_details=evaluation["token_details"],
        label_to_id=label_to_id,
    )
    metrics["loss"] = evaluation["loss"]
    metrics["invalid_bio_transitions"] = evaluation["invalid_bio_transitions"]
    metrics["supported_macro"] = supported_macro(metrics)
    if context:
        metrics["context_diagnostics"] = context_diagnostics(
            records, evaluation["predictions"], labels, split
        )
    return metrics, evaluation["predictions"]


def _write_preflight(
    path: Path,
    evidence: Mapping[str, Any],
    counts: Mapping[str, int],
    alignments: Mapping[str, Mapping[str, Any]],
    gate_hash: str,
    smoke: bool,
) -> None:
    lines = [
        "# SecureLogX ML-v1.1 Training Preflight",
        "",
        f"**Status: {'SMOKE PREFLIGHT PASSED' if smoke else 'FULL PREFLIGHT PASSED'}**",
        "",
        "The sealed challenge was not parsed or evaluated. Its frozen bytes were SHA-256 checked only.",
        "",
        "## Frozen evidence",
        "",
        f"- Verified file/hash checks: **{evidence['verified_file_checks']}**",
        f"- ML-v1.1 manifest SHA-256: `{evidence['manifest_sha256']}`",
        f"- Predeclared training-gate SHA-256: `{gate_hash}`",
        f"- Frozen ML-v1 checkpoint fingerprint: `{evidence['parent_checkpoint_fingerprint_sha256']}`",
        "- Ontology: **25 entities / 51 BIO labels**",
        "",
        "## Dataset composition",
        "",
        f"- Original ML-v1 train records: **{counts['parent_train']}**",
        f"- ML-v1.1 context additions: **{counts['train_additions']}**",
        f"- Combined in-memory training records: **{counts['training']}**",
        f"- Original standard-dev records: **{counts['standard_dev']}**",
        f"- Independent dev-challenge records: **{counts['challenge_dev']}**",
        "- No derived JSONL was needed; concatenation is in memory and every source record retains its original `meta` provenance.",
        "",
        "## Alignment",
        "",
        "| View | Spans | Aligned | Boundary adjusted | Truncated | Explicitly unrepresentable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in alignments.items():
        lines.append(
            f"| {name} | {value['total_spans']} | {value['successfully_aligned_spans']} | "
            f"{value['boundary_adjusted_spans']} | {value['truncated_spans']} | {value['failed_alignments']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dev_gate_actual(
    standard: Mapping[str, Any], challenge: Mapping[str, Any], alignments: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, float | int]:
    challenge_diag = challenge["context_diagnostics"]
    historical = gate["historical_reference"]
    return {
        "original_dev_micro_f1_minimum": standard["micro"]["f1"],
        "original_dev_supported_macro_f1_minimum": standard["supported_macro"]["f1"],
        "original_dev_high_risk_recall_minimum": standard["high_risk"]["recall"],
        "dev_challenge_micro_f1_minimum": challenge["micro"]["f1"],
        "dev_challenge_supported_macro_f1_minimum": challenge["supported_macro"]["f1"],
        "dev_challenge_business_id_recall_minimum": challenge["per_label"]["BUSINESS_ID"]["recall"],
        "dev_challenge_business_id_recall_improvement_over_ml_v1_minimum": challenge["per_label"]["BUSINESS_ID"]["recall"] - historical["ml_v1_test_business_id_recall"],
        "dev_challenge_ssn_recall_minimum": challenge["per_label"]["SSN"]["recall"],
        "dev_challenge_record_error_rate_maximum": challenge_diag["record_error_rate"],
        "dev_challenge_context_target_error_rate_maximum": challenge_diag["context_target_error_rate"],
        "dev_challenge_morphology_over_context_error_rate_maximum": challenge_diag["morphology_over_context_error_rate"],
        "dev_challenge_maximum_category_error_rate": challenge_diag["maximum_category_error_rate"],
        "unexplained_alignment_failures_maximum": sum(len(_alignment_problems(value)) for value in alignments.values()),
        "truncated_entity_spans_maximum": sum(int(value["truncated_spans"]) for value in alignments.values()),
    }


def _write_selection_report(
    path: Path,
    history: Sequence[Mapping[str, Any]],
    selected_epoch: int,
    selected_path: Path,
    gate_result: Mapping[str, Any],
) -> None:
    lines = [
        "# SecureLogX ML-v1.1 Checkpoint Selection",
        "",
        "Selection used development data only. The sealed challenge was not parsed or inferred.",
        "",
        "Score = harmonic mean of original-dev supported-entity macro F1 and dev-challenge supported-entity macro F1. Scores within 0.001 are treated as tied and the earlier epoch wins.",
        "",
        "| Epoch | Original dev macro F1 | Challenge dev macro F1 | Harmonic score | Original micro F1 | Challenge micro F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['standard_supported_macro_f1']:.6f} | {row['challenge_supported_macro_f1']:.6f} | "
            f"{row['selection_score']:.6f} | {row['standard_micro_f1']:.6f} | {row['challenge_micro_f1']:.6f} |"
        )
    selected = next(row for row in history if row["epoch"] == selected_epoch)
    lines.extend(
        [
            "",
            f"Selected epoch **{selected_epoch}** at `{selected_path.as_posix()}` with harmonic score **{selected['selection_score']:.6f}**.",
            "",
            f"Development unlock decision: **{gate_result['decision']}**.",
            "",
            "| Development gate | Threshold | Actual | Status |",
            "|---|---:|---:|---|",
        ]
    )
    for name, result in gate_result["gates"].items():
        lines.append(
            f"| `{name}` | {result['threshold']:.6f} | {result['actual']:.6f} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_training_summary(
    path: Path,
    arguments: Mapping[str, Any],
    environment: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    best_epoch: int,
    best_checkpoint: Path,
    elapsed: float,
    smoke: bool,
    smoke_loss: Mapping[str, Any] | None,
) -> None:
    lines = [
        "# SecureLogX ML-v1.1 Training Run Summary",
        "",
        f"**Status: {'SMOKE TEST PASSED' if smoke else 'TRAINING COMPLETE'}**",
        "",
        "## Controlled configuration",
        "",
        f"- Architecture/base/revision: BERT token classification / `{arguments['base_model']}` / `{arguments['revision']}`",
        f"- Epochs / learning rate / weight decay / warmup: {arguments['epochs']} / {arguments['learning_rate']} / {arguments['weight_decay']} / {arguments['warmup_ratio']}",
        f"- Max length / stride / dynamic padding: {arguments['max_length']} / {arguments['stride']} / yes",
        f"- Physical train batch / accumulation / effective batch / eval batch: {arguments['train_batch_size']} / {arguments['gradient_accumulation_steps']} / {arguments['effective_batch_size']} / {arguments['eval_batch_size']}",
        f"- Gradient clipping / BF16 / seed: {arguments['max_grad_norm']} / {arguments['use_bf16']} / {arguments['seed']}",
        f"- Training records: {arguments['training_records']} ({arguments['parent_train_records']} parent + {arguments['addition_records']} additions)",
        f"- Training gate SHA-256: `{arguments['training_gate_sha256']}`",
        "- Architecture, tokenizer, learning rate, epochs, and sequence length are unchanged from ML-v1.",
        "",
        "## Epoch development results",
        "",
        "| Epoch | Train loss | Standard loss | Standard micro F1 | Standard macro F1 | Challenge micro F1 | Challenge macro F1 | BID recall | SSN recall | Challenge record error | Selection score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['standard_loss']:.6f} | {row['standard_micro_f1']:.6f} | "
            f"{row['standard_supported_macro_f1']:.6f} | {row['challenge_micro_f1']:.6f} | {row['challenge_supported_macro_f1']:.6f} | "
            f"{row['challenge_business_id_recall']:.6f} | {row['challenge_ssn_recall']:.6f} | {row['challenge_record_error_rate']:.6f} | {row['selection_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Selected epoch: **{best_epoch}**",
            f"- Best checkpoint: `{best_checkpoint.as_posix()}`",
            f"- Elapsed train plus development evaluation time: **{elapsed / 60:.2f} minutes**",
            "- Checkpoint selection used no test or sealed-challenge prediction.",
            "",
            "## Environment",
            "",
            f"- Python / PyTorch / Transformers: `{environment['python']}` / `{environment['torch']}` / `{environment['transformers']}`",
            f"- CUDA / GPU: `{environment['cuda_available']}` / `{environment['gpu_name']}`",
        ]
    )
    if smoke_loss is not None:
        lines.extend(
            [
                "",
                "## Smoke assertions",
                "",
                f"- First optimization-group mean loss: **{smoke_loss['first_group_mean']:.6f}**",
                f"- Last optimization-group mean loss: **{smoke_loss['last_group_mean']:.6f}**",
                f"- Loss decreased: **{smoke_loss['decreased']}**",
                "- Finite loss, decoded predictions, and serialized checkpoint reload: **PASS**",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_epoch(
    path: Path,
    model: Any,
    tokenizer: Any,
    epoch_metrics: Mapping[str, Any],
    training_arguments: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    write_json(path / "epoch_metrics.json", epoch_metrics)
    write_json(path / "training_args.json", training_arguments)
    write_json(path / "environment.json", environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default=FULL_PATHS["manifest"])
    parser.add_argument("--parent-freeze", default=FULL_PATHS["parent_freeze"])
    parser.add_argument("--gate", default=FULL_PATHS["gate"])
    parser.add_argument("--labels", default=FULL_PATHS["labels"])
    parser.add_argument("--parent-train", default=FULL_PATHS["parent_train"])
    parser.add_argument("--train-additions", default=FULL_PATHS["train_additions"])
    parser.add_argument("--standard-dev", default=FULL_PATHS["standard_dev"])
    parser.add_argument("--challenge-dev", default=FULL_PATHS["challenge_dev"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-report", default=FULL_PATHS["preflight_report"])
    parser.add_argument("--training-report", default=FULL_PATHS["training_report"])
    parser.add_argument("--standard-metrics", default=FULL_PATHS["standard_metrics"])
    parser.add_argument("--challenge-metrics", default=FULL_PATHS["challenge_metrics"])
    parser.add_argument("--selection-report", default=FULL_PATHS["selection_report"])
    parser.add_argument("--base-model", default=FULL_SETTINGS["base_model"])
    parser.add_argument("--revision", default=FULL_SETTINGS["revision"])
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parent-train-records", type=int)
    parser.add_argument("--max-addition-records", type=int)
    parser.add_argument("--max-dev-records", type=int)
    parser.add_argument("--max-challenge-records", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(args.root).resolve()
    if not args.smoke_test:
        _assert_canonical_full_run(args, root)
    if args.base_model != "bert-base-cased":
        raise ValueError("This controlled experiment permits only bert-base-cased")
    output_root = (root / args.output).resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    evidence = verify_frozen_evidence(root, args)
    gate = _load_json(root / args.gate)
    gate_hash = sha256_file(root / args.gate)
    label_config = load_canonical_labels(root / args.labels)
    label_to_id = {str(key): int(value) for key, value in label_config["label_to_id"].items()}
    id_to_label = {int(key): str(value) for key, value in label_config["id_to_label"].items()}
    if len(label_config["bio_labels"]) != 51:
        raise ValueError("Frozen BIO map must contain exactly 51 labels")

    set_reproducible_seed(args.seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; refusing without --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    if not args.smoke_test and not use_bf16:
        raise RuntimeError("Canonical full training requires CUDA BF16 support")
    environment = collect_environment(args.seed, args.base_model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision, use_fast=True, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Training requires the pinned fast tokenizer")

    parent_records = load_jsonl(root / args.parent_train, args.max_parent_train_records)
    addition_records = load_jsonl(root / args.train_additions, args.max_addition_records)
    standard_records = load_jsonl(root / args.standard_dev, args.max_dev_records)
    challenge_records = load_jsonl(root / args.challenge_dev, args.max_challenge_records)
    train_records = [*parent_records, *addition_records]
    counts = {
        "parent_train": len(parent_records),
        "train_additions": len(addition_records),
        "training": len(train_records),
        "standard_dev": len(standard_records),
        "challenge_dev": len(challenge_records),
    }
    if not args.smoke_test and counts != EXPECTED_COUNTS:
        raise ValueError(f"Canonical record counts changed: {counts} != {EXPECTED_COUNTS}")
    if not all(record.get("meta", {}).get("source") == "securelogx_context_contrast_v1_1" for record in addition_records):
        raise ValueError("Training additions lost ML-v1.1 provenance")

    print(
        f"Building aligned windows: train={len(train_records)}, standard_dev={len(standard_records)}, challenge_dev={len(challenge_records)}",
        flush=True,
    )
    train_build = build_aligned_features(train_records, tokenizer, label_to_id, args.max_length, args.stride)
    standard_build = build_aligned_features(standard_records, tokenizer, label_to_id, args.max_length, args.stride)
    challenge_build = build_aligned_features(challenge_records, tokenizer, label_to_id, args.max_length, args.stride)
    alignments = {
        "train": train_build.summary,
        "standard_dev": standard_build.summary,
        "challenge_dev": challenge_build.summary,
    }
    for split, summary in alignments.items():
        _assert_alignment(summary, split)
    _write_preflight(root / args.preflight_report, evidence, counts, alignments, gate_hash, args.smoke_test)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        AlignedWindowDataset(train_build.windows),
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=TokenClassificationCollator(tokenizer),
        pin_memory=device.type == "cuda",
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        revision=args.revision,
        local_files_only=True,
        num_labels=51,
        id2label=id_to_label,
        label2id=label_to_id,
    )
    if model.config.id2label != id_to_label or model.config.label2id != label_to_id:
        raise ValueError("Model label mappings do not match the frozen map")
    model.to(device)
    optimizer = _optimizer(model, args.learning_rate, args.weight_decay)
    batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    planned_steps = steps_per_epoch * args.epochs
    if args.max_optimizer_steps is not None:
        planned_steps = min(planned_steps, args.max_optimizer_steps)
    warmup_steps = int(round(planned_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=planned_steps
    )
    training_arguments = {
        "base_model": args.base_model,
        "revision": args.revision,
        "max_length": args.max_length,
        "stride": args.stride,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "dynamic_padding": True,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "seed": args.seed,
        "use_bf16": use_bf16,
        "device": str(device),
        "planned_optimizer_steps": planned_steps,
        "parent_train_records": len(parent_records),
        "addition_records": len(addition_records),
        "training_records": len(train_records),
        "standard_dev_records": len(standard_records),
        "challenge_dev_records": len(challenge_records),
        "training_gate_sha256": gate_hash,
        "dataset_manifest_sha256": evidence["manifest_sha256"],
        "smoke_test": args.smoke_test,
    }

    history: list[dict[str, Any]] = []
    checkpoint_root = output_root / "checkpoints"
    best_epoch = 0
    best_score = -1.0
    best_source: Path | None = None
    best_standard: dict[str, Any] | None = None
    best_challenge: dict[str, Any] | None = None
    best_standard_predictions: list[list[dict[str, Any]]] | None = None
    best_challenge_predictions: list[list[dict[str, Any]]] | None = None
    global_steps = 0
    batch_losses: list[float] = []
    group_losses: list[float] = []
    run_start = time.perf_counter()
    stop_training = False
    tie_tolerance = float(gate["checkpoint_selection"]["effective_tie_absolute"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_numerator = 0.0
        epoch_denominator = 0
        epoch_steps = 0
        current_group: list[float] = []
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, batches_per_epoch - group_start)
            inputs = model_inputs_from_batch(batch, device)
            valid_tokens = int((inputs["labels"] != IGNORE_INDEX).sum().item())
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(**inputs)
                loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            loss_value = float(loss.detach().float().item())
            batch_losses.append(loss_value)
            current_group.append(loss_value)
            epoch_numerator += loss_value * valid_tokens
            epoch_denominator += valid_tokens
            (loss / group_size).backward()
            should_step = (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == batches_per_epoch
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            group_losses.append(float(statistics.fmean(current_group)))
            current_group.clear()
            global_steps += 1
            epoch_steps += 1
            if global_steps == 1 or global_steps % 100 == 0:
                elapsed = time.perf_counter() - run_start
                rate = global_steps / elapsed if elapsed else 0.0
                eta = (planned_steps - global_steps) / rate / 60 if rate else 0.0
                memory = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
                print(f"epoch={epoch} step={global_steps}/{planned_steps} loss={loss_value:.6f} gpu_peak_gib={memory:.2f} eta_min={eta:.1f}", flush=True)
            if args.max_optimizer_steps is not None and global_steps >= args.max_optimizer_steps:
                stop_training = True
                break

        train_loss = epoch_numerator / epoch_denominator if epoch_denominator else 0.0
        print(f"Evaluating both development views after epoch {epoch}...", flush=True)
        standard_metrics, standard_predictions = _score_view(
            model, tokenizer, standard_records, standard_build.windows,
            label_config["entities"], label_to_id, id_to_label, "ml_v1_1_dev_standard",
            args.eval_batch_size, device, use_bf16, False,
        )
        challenge_metrics, challenge_predictions = _score_view(
            model, tokenizer, challenge_records, challenge_build.windows,
            label_config["entities"], label_to_id, id_to_label, "ml_v1_1_dev_challenge",
            args.eval_batch_size, device, use_bf16, True,
        )
        selection_score = harmonic_mean(
            float(standard_metrics["supported_macro"]["f1"]),
            float(challenge_metrics["supported_macro"]["f1"]),
        )
        diag = challenge_metrics["context_diagnostics"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "standard_loss": standard_metrics["loss"],
            "standard_micro_f1": standard_metrics["micro"]["f1"],
            "standard_supported_macro_f1": standard_metrics["supported_macro"]["f1"],
            "standard_high_risk_recall": standard_metrics["high_risk"]["recall"],
            "standard_business_id_recall": standard_metrics["per_label"]["BUSINESS_ID"]["recall"],
            "standard_ssn_recall": standard_metrics["per_label"]["SSN"]["recall"],
            "challenge_loss": challenge_metrics["loss"],
            "challenge_micro_f1": challenge_metrics["micro"]["f1"],
            "challenge_supported_macro_f1": challenge_metrics["supported_macro"]["f1"],
            "challenge_business_id_recall": challenge_metrics["per_label"]["BUSINESS_ID"]["recall"],
            "challenge_ssn_recall": challenge_metrics["per_label"]["SSN"]["recall"],
            "challenge_record_error_rate": diag["record_error_rate"],
            "challenge_context_target_error_rate": diag["context_target_error_rate"],
            "challenge_morphology_over_context_error_rate": diag["morphology_over_context_error_rate"],
            "selection_score": selection_score,
            "optimizer_steps": epoch_steps,
            "global_optimizer_steps": global_steps,
            "epoch_minutes": (time.perf_counter() - epoch_start) / 60,
        }
        history.append(row)
        epoch_path = checkpoint_root / f"epoch-{epoch}"
        _save_epoch(
            epoch_path,
            model,
            tokenizer,
            {
                "history": row,
                "standard_dev_metrics": _compact_metrics(standard_metrics),
                "challenge_dev_metrics": _compact_metrics(challenge_metrics),
            },
            training_arguments,
            environment,
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} standard_micro={row['standard_micro_f1']:.6f} "
            f"standard_macro={row['standard_supported_macro_f1']:.6f} challenge_micro={row['challenge_micro_f1']:.6f} "
            f"challenge_macro={row['challenge_supported_macro_f1']:.6f} selection={selection_score:.6f}",
            flush=True,
        )
        if selection_score > best_score + tie_tolerance:
            best_epoch = epoch
            best_score = selection_score
            best_source = epoch_path
            best_standard = _compact_metrics(standard_metrics)
            best_challenge = _compact_metrics(challenge_metrics)
            best_standard_predictions = standard_predictions
            best_challenge_predictions = challenge_predictions
        del standard_metrics, challenge_metrics
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if stop_training:
            break

    if (
        best_source is None
        or best_standard is None
        or best_challenge is None
        or best_standard_predictions is None
        or best_challenge_predictions is None
    ):
        raise RuntimeError("Training produced no selectable checkpoint")
    if args.smoke_test:
        if len(group_losses) < 2:
            raise RuntimeError("Smoke test needs at least two optimization groups")
        smoke_loss = {
            "first_group_mean": group_losses[0],
            "last_group_mean": group_losses[-1],
            "decreased": group_losses[-1] < group_losses[0],
            "finite_batches": all(math.isfinite(value) for value in batch_losses),
        }
        if not smoke_loss["decreased"] or not smoke_loss["finite_batches"]:
            raise RuntimeError(f"Smoke loss assertion failed: {smoke_loss}")
    else:
        smoke_loss = None

    best_checkpoint = output_root / "best-checkpoint"
    shutil.copytree(best_source, best_checkpoint)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(best_checkpoint.iterdir())
        if path.is_file()
    }
    selected_metadata = {
        "selected_epoch": best_epoch,
        "selection_metric": "harmonic mean of original-dev and challenge-dev supported entity macro F1",
        "selection_value": best_score,
        "tie_tolerance": tie_tolerance,
        "tie_break": "earlier epoch",
        "source_checkpoint": best_source.relative_to(root).as_posix(),
        "model_sha256": artifact_hashes["model.safetensors"],
        "artifact_sha256": artifact_hashes,
        "checkpoint_fingerprint_sha256": _mapping_fingerprint(artifact_hashes),
    }
    write_json(best_checkpoint / "selected_checkpoint.json", selected_metadata)
    write_json(output_root / "training_history.json", history)
    write_json(output_root / "alignment_summary.json", alignments)
    gate_actual = _dev_gate_actual(best_standard, best_challenge, alignments, gate)
    evaluated_gate = evaluate_named_gates(
        gate_actual, gate["sealed_challenge_unlock"]["all_required"]
    )
    gate_result = {
        "schema_version": 1,
        "status": "PASS" if evaluated_gate["all_passed"] else "FAIL",
        "decision": (
        "READY FOR SEALED CHALLENGE EVALUATION"
        if evaluated_gate["all_passed"]
        else gate["sealed_challenge_unlock"]["decision_on_failure"]
        ),
        "gate_sha256": gate_hash,
        "dataset_manifest_sha256": evidence["manifest_sha256"],
        "checkpoint_fingerprint_sha256": selected_metadata["checkpoint_fingerprint_sha256"],
        "selected_epoch": best_epoch,
        "actual": gate_actual,
        "passed": evaluated_gate["all_passed"],
        "gates": evaluated_gate["gates"],
    }
    write_json(output_root / "dev_gate_decision.json", gate_result)
    write_json(output_root / "dev_standard_predictions.json", {
        "split": "ml_v1_1_dev_standard",
        "input_sha256": sha256_file(root / args.standard_dev),
        "selected_epoch": best_epoch,
        "checkpoint_fingerprint_sha256": selected_metadata["checkpoint_fingerprint_sha256"],
        "records": len(standard_records),
        "predictions": best_standard_predictions,
    })
    write_json(output_root / "dev_challenge_predictions.json", {
        "split": "ml_v1_1_dev_challenge",
        "input_sha256": sha256_file(root / args.challenge_dev),
        "selected_epoch": best_epoch,
        "checkpoint_fingerprint_sha256": selected_metadata["checkpoint_fingerprint_sha256"],
        "records": len(challenge_records),
        "predictions": best_challenge_predictions,
    })

    if args.smoke_test:
        write_json(output_root / "smoke_result.json", {
            "status": "SMOKE TEST PASSED",
            "loss": smoke_loss,
            "decoded_standard_records": len(standard_records),
            "decoded_challenge_records": len(challenge_records),
            "checkpoint": selected_metadata,
        })
    else:
        write_json(root / args.standard_metrics, best_standard)
        write_json(root / args.challenge_metrics, best_challenge)
    _write_selection_report(root / args.selection_report, history, best_epoch, best_checkpoint.relative_to(root), gate_result)
    elapsed = time.perf_counter() - run_start
    _write_training_summary(
        root / args.training_report,
        training_arguments,
        environment,
        history,
        best_epoch,
        best_checkpoint.relative_to(root),
        elapsed,
        args.smoke_test,
        smoke_loss,
    )

    reloaded = AutoModelForTokenClassification.from_pretrained(best_checkpoint, local_files_only=True)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(best_checkpoint, local_files_only=True, use_fast=True)
    saved_id2label = {int(key): value for key, value in reloaded.config.id2label.items()}
    if saved_id2label != id_to_label or reloaded.config.label2id != label_to_id or not reloaded_tokenizer.is_fast:
        raise ValueError("Serialized checkpoint failed the label/tokenizer reload check")
    del reloaded, reloaded_tokenizer

    verify_frozen_evidence(root, args)
    if sha256_file(root / args.gate) != gate_hash:
        raise ValueError("Predeclared training gate changed during training")
    print(f"Selected epoch {best_epoch}: {best_checkpoint}", flush=True)
    print(gate_result["decision"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
