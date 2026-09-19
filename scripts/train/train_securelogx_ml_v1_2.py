"""Train and evaluate the development-only SecureLogX ML-v1.2 experiment.

The controlled variable is data: this retains the ML-v1.1 architecture,
tokenizer, optimizer, schedule, and checkpoint-selection policy. It reads only
the original train/dev sets and the v1.1/v1.2 training additions plus the new
v1.2 development challenge. The sealed challenge and original test are never
opened or inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from securelogx_ml_v1_1_metrics import (
    context_diagnostics,
    evaluate_named_gates,
    harmonic_mean,
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
    write_json,
)
from train_securelogx_ml_v1_1 import (
    _alignment_problems,
    _compact_metrics,
    _optimizer,
    _save_epoch,
    _score_view,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_SCRIPTS = ROOT / "scripts" / "data"
if str(DATA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPTS))

from build_securelogx_ml_v1_2_dataset import verify_parent_snapshot  # noqa: E402


PATHS = {
    "manifest": "configs/ml_v1_2_dataset_manifest.json",
    "manifest_hash": "configs/ml_v1_2_dataset_manifest.sha256",
    "parent_freeze": "configs/ml_v1_2_parent_freeze.json",
    "gate": "configs/ml_v1_2_training_gate.json",
    "gate_hash": "configs/ml_v1_2_training_gate.sha256",
    "labels": "configs/securelogx_labels.json",
    "parent_train": "data/split/train.jsonl",
    "v1_1_additions": "data/ml_v1_1/context_contrast/train_additions.jsonl",
    "v1_2_additions": "data/ml_v1_2/counterbalance/train_additions.jsonl",
    "standard_dev": "data/split/dev.jsonl",
    "challenge_dev": "data/ml_v1_2/counterbalance/dev_challenge.jsonl",
    "output": "output_securelogx/ml-v1.2/bert-base-cased",
    "preflight_report": "reports/ml_v1_2_training_preflight.md",
    "training_report": "reports/ml_v1_2_training_run_summary.md",
    "standard_metrics": "reports/ml_v1_2_dev_standard_metrics.json",
    "challenge_metrics": "reports/ml_v1_2_dev_challenge_metrics.json",
    "targeted_analysis": "reports/ml_v1_2_targeted_development_analysis.json",
    "selection_report": "reports/ml_v1_2_checkpoint_selection.md",
    "decision_report": "reports/ml_v1_2_development_gate.md",
}
SETTINGS = {
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
}
COUNTS = {
    "parent_train": 23_558,
    "v1_1_additions": 3_000,
    "v1_2_additions": 1_600,
    "training": 28_158,
    "standard_dev": 2_994,
    "challenge_dev": 540,
}
ALLOWED_ALIGNMENT_FAILURES = {
    "tokenizer_emitted_no_token_for_span",
    "token_boundary_adjustment_exceeds_three_characters",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _detached_hash(path: Path, expected_name: str) -> str:
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or len(parts[0]) != 64 or parts[1] != expected_name:
        raise ValueError(f"Malformed detached SHA-256 declaration: {path}")
    int(parts[0], 16)
    return parts[0].lower()


def _mapping_fingerprint(values: Mapping[str, str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_jsonl_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _regenerate_v1_1_train(root: Path, expected_hash: str) -> list[dict[str, Any]]:
    generator_path = (
        root / "scripts/data/generate_securelogx_ml_v1_1_context_contrast.py"
    )
    spec = importlib.util.spec_from_file_location("securelogx_v1_1_generator", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen ML-v1.1 generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    records = module.generate_partition("train_addition", seed=42)
    actual = _canonical_jsonl_sha256(records)
    if actual != expected_hash:
        raise ValueError(
            "Regenerated ML-v1.1 training addition hash mismatch: "
            f"{actual} != {expected_hash}"
        )
    return records


def _load_v1_1_train(root: Path, relative: str, expected_hash: str) -> tuple[list[dict[str, Any]], str]:
    path = root / relative
    try:
        records = load_jsonl(path)
        actual = sha256_file(path)
        method = "file"
    except (OSError, PermissionError):
        records = _regenerate_v1_1_train(root, expected_hash)
        actual = _canonical_jsonl_sha256(records)
        method = "deterministic_in_memory_regeneration"
    if actual != expected_hash:
        raise ValueError(f"ML-v1.1 training addition hash mismatch: {actual}")
    return records, method


def verify_preflight(root: Path) -> dict[str, Any]:
    manifest_path = root / PATHS["manifest"]
    manifest_hash = sha256_file(manifest_path)
    declared_manifest = _detached_hash(
        root / PATHS["manifest_hash"], Path(PATHS["manifest"]).name
    )
    gate_path = root / PATHS["gate"]
    gate_hash = sha256_file(gate_path)
    declared_gate = _detached_hash(
        root / PATHS["gate_hash"], Path(PATHS["gate"]).name
    )
    manifest = _load_json(manifest_path)
    gate = _load_json(gate_path)
    parent = _load_json(root / PATHS["parent_freeze"])
    failures: list[str] = []
    if manifest_hash != declared_manifest:
        failures.append("dataset manifest detached digest mismatch")
    if gate_hash != declared_gate or gate.get("declared_before_training") is not True:
        failures.append("training gate digest/declaration mismatch")
    frozen = gate["frozen_inputs"]
    if frozen["ml_v1_2_dataset_manifest"]["sha256"] != manifest_hash:
        failures.append("training gate is not bound to dataset manifest")
    if sha256_file(root / PATHS["parent_freeze"]) != frozen["ml_v1_2_parent_freeze"]["sha256"]:
        failures.append("training gate parent-freeze hash mismatch")
    labels_hash = sha256_file(root / PATHS["labels"])
    if labels_hash != frozen["ontology"]["sha256"]:
        failures.append("ontology hash mismatch")
    labels = load_canonical_labels(root / PATHS["labels"])
    if len(labels["entities"]) != 25 or len(labels["bio_labels"]) != 51:
        failures.append("ontology is not 25 entities / 51 BIO labels")
    for relative, details in manifest["new_data_artifacts"].items():
        if sha256_file(root / relative) != details["sha256"]:
            failures.append(f"new data artifact hash mismatch: {relative}")
    parent_result = verify_parent_snapshot(root, parent)
    if not parent_result["passed"]:
        failures.append(f"parent snapshot mismatch: {parent_result['mismatches']}")
    sealed = parent_result["sealed_challenge"]
    if (
        not sealed["passed"]
        or sealed["file_opened"] is not False
        or sealed["actual_sha256"]
        != "6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a"
    ):
        failures.append("sealed challenge integrity/handling check failed")
    if failures:
        raise ValueError("ML-v1.2 preflight failed:\n- " + "\n- ".join(failures))
    return {
        "passed": True,
        "dataset_manifest_sha256": manifest_hash,
        "training_gate_sha256": gate_hash,
        "parent_freeze_sha256": sha256_file(root / PATHS["parent_freeze"]),
        "parent_verified_files": parent_result["verified_files"],
        "parent_mismatches": parent_result["mismatch_count"],
        "sealed_challenge": sealed,
        "ontology": {"entities": 25, "bio_labels": 51, "sha256": labels_hash},
        "manifest": manifest,
        "gate": gate,
    }


def _overlap(start: int, end: int, span: Mapping[str, Any]) -> bool:
    return int(span["start"]) < end and int(span["end"]) > start


def _direction(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    gold_label: str,
    predicted_label: str,
    category: str | None = None,
) -> dict[str, Any]:
    support = 0
    confused = 0
    record_indices: set[int] = set()
    for index, (record, guesses) in enumerate(zip(records, predictions)):
        if category and record.get("meta", {}).get("contrast_category") != category:
            continue
        for start, end, label in record["entities"]:
            if label != gold_label:
                continue
            support += 1
            if any(
                guess["label"] == predicted_label
                and _overlap(int(start), int(end), guess)
                for guess in guesses
            ):
                confused += 1
                record_indices.add(index)
    return {
        "gold_support": support,
        "confused_spans": confused,
        "records": len(record_indices),
        "rate": confused / support if support else 0.0,
    }


def _o_target_direction(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    predicted_label: str,
    *,
    category: str,
    context_role: str | None = None,
) -> dict[str, Any]:
    support = errors = 0
    for record, guesses in zip(records, predictions):
        meta = record.get("meta", {})
        if meta.get("contrast_category") != category:
            continue
        if context_role and meta.get("context_role") != context_role:
            continue
        targets = [
            target
            for target in meta.get("contrast_targets", [])
            if target.get("expected_label") == "O"
        ]
        if not targets:
            continue
        support += 1
        if any(
            guess["label"] == predicted_label
            and any(
                _overlap(int(target["start"]), int(target["end"]), guess)
                for target in targets
            )
            for guess in guesses
        ):
            errors += 1
    return {
        "records": support,
        "records_with_error": errors,
        "rate": errors / support if support else 0.0,
    }


def _accountlike_o_business_id(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    count = 0
    record_indices: set[int] = set()
    family_records = 0
    for index, (record, guesses) in enumerate(zip(records, predictions)):
        if record.get("meta", {}).get("template_family") != "key_value_accountlike_batch_v1":
            continue
        family_records += 1
        gold = [
            (int(start), int(end))
            for start, end, _ in record["entities"]
        ]
        for guess in guesses:
            if guess["label"] != "BUSINESS_ID":
                continue
            if not any(_overlap(start, end, guess) for start, end in gold):
                count += 1
                record_indices.add(index)
    return {
        "family_records": family_records,
        "false_positive_spans": count,
        "records_with_false_positive": len(record_indices),
    }


def _business_subtype_recall(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    support: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    for record, guesses in zip(records, predictions):
        signatures = {
            (int(guess["start"]), int(guess["end"]), str(guess["label"]))
            for guess in guesses
        }
        for item in record.get("meta", {}).get("entity_provenance", []):
            if item.get("label") != "BUSINESS_ID":
                continue
            subtype = str(item.get("business_id_subtype") or "BUSINESS_ID").upper()
            signature = (int(item["start"]), int(item["end"]), "BUSINESS_ID")
            support[subtype] += 1
            true_positive[subtype] += signature in signatures
    return {
        subtype: {
            "support": support[subtype],
            "true_positives": true_positive[subtype],
            "recall": true_positive[subtype] / support[subtype] if support[subtype] else None,
        }
        for subtype in sorted(support)
    }


def targeted_analysis(
    standard_records: Sequence[Mapping[str, Any]],
    standard_predictions: Sequence[Sequence[Mapping[str, Any]]],
    challenge_records: Sequence[Mapping[str, Any]],
    challenge_predictions: Sequence[Sequence[Mapping[str, Any]]],
    challenge_metrics: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> dict[str, Any]:
    categories = challenge_metrics["context_diagnostics"]["by_category"]
    worst_name, worst = max(
        categories.items(), key=lambda item: float(item[1]["record_error_rate"])
    )
    subtypes = _business_subtype_recall(challenge_records, challenge_predictions)
    return {
        "business_id_false_positives_on_gold_o_accountlike": _accountlike_o_business_id(
            standard_records, standard_predictions
        ),
        "business_id_vs_credit_card": {
            "credit_card_to_business_id": _direction(
                challenge_records,
                challenge_predictions,
                "CREDIT_CARD_NUMBER",
                "BUSINESS_ID",
                "business_id_vs_credit_card",
            ),
            "business_id_to_credit_card": _direction(
                challenge_records,
                challenge_predictions,
                "BUSINESS_ID",
                "CREDIT_CARD_NUMBER",
                "business_id_vs_credit_card",
            ),
        },
        "business_id_vs_ssn": {
            "ssn_to_business_id": _direction(
                challenge_records,
                challenge_predictions,
                "SSN",
                "BUSINESS_ID",
                "business_id_vs_ssn",
            ),
            "business_id_to_ssn": _direction(
                challenge_records,
                challenge_predictions,
                "BUSINESS_ID",
                "SSN",
                "business_id_vs_ssn",
            ),
        },
        "standard_dev_ssn_to_itin": _direction(
            standard_records, standard_predictions, "SSN", "ITIN"
        ),
        "ip_address_on_technical_reference": _o_target_direction(
            challenge_records,
            challenge_predictions,
            "IP_ADDRESS",
            category="ip_address_vs_technical_reference",
            context_role="technical_reference",
        ),
        "business_id_subtype_recall": subtypes,
        "challenge_record_error_rate": challenge_metrics["context_diagnostics"][
            "record_error_rate"
        ],
        "challenge_context_target_error_rate": challenge_metrics[
            "context_diagnostics"
        ]["context_target_error_rate"],
        "worst_conflict_category": {
            "category": worst_name,
            "records": worst["records"],
            "records_with_errors": worst["records_with_errors"],
            "record_error_rate": worst["record_error_rate"],
        },
        "ml_v1_1_historical_reference": dict(historical),
    }


def _gate_actual(
    standard: Mapping[str, Any],
    challenge: Mapping[str, Any],
    targeted: Mapping[str, Any],
    alignments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    standard_labels = standard["per_label"]
    challenge_labels = challenge["per_label"]
    diagnostics = challenge["context_diagnostics"]
    subtype = targeted["business_id_subtype_recall"]
    return {
        "standard_dev_micro_f1": standard["micro"]["f1"],
        "standard_dev_supported_macro_f1": standard["supported_macro"]["f1"],
        "standard_dev_high_risk_recall": standard["high_risk"]["recall"],
        **{
            f"standard_dev_{label.lower()}_{metric}": standard_labels[label][metric]
            for label in ("BUSINESS_ID", "SSN", "IP_ADDRESS")
            for metric in ("precision", "recall", "f1")
        },
        "challenge_dev_micro_f1": challenge["micro"]["f1"],
        "challenge_dev_supported_macro_f1": challenge["supported_macro"]["f1"],
        "challenge_dev_business_id_precision": challenge_labels["BUSINESS_ID"]["precision"],
        "challenge_dev_business_id_recall": challenge_labels["BUSINESS_ID"]["recall"],
        "challenge_dev_ssn_recall": challenge_labels["SSN"]["recall"],
        "challenge_dev_ip_address_recall": challenge_labels["IP_ADDRESS"]["recall"],
        "challenge_dev_record_error_rate": diagnostics["record_error_rate"],
        "challenge_dev_context_target_error_rate": diagnostics[
            "context_target_error_rate"
        ],
        "challenge_dev_maximum_category_error_rate": diagnostics[
            "maximum_category_error_rate"
        ],
        "standard_dev_accountlike_o_to_business_id_count": targeted[
            "business_id_false_positives_on_gold_o_accountlike"
        ]["false_positive_spans"],
        "standard_dev_ssn_to_itin_count": targeted["standard_dev_ssn_to_itin"][
            "confused_spans"
        ],
        "challenge_dev_credit_card_to_business_id_rate": targeted[
            "business_id_vs_credit_card"
        ]["credit_card_to_business_id"]["rate"],
        "challenge_dev_business_id_to_ssn_rate": targeted["business_id_vs_ssn"][
            "business_id_to_ssn"
        ]["rate"],
        "challenge_dev_ip_on_technical_reference_rate": targeted[
            "ip_address_on_technical_reference"
        ]["rate"],
        "challenge_dev_customer_id_recall": subtype.get("CUSTOMER_ID", {}).get(
            "recall", 0.0
        ),
        "challenge_dev_request_id_recall": subtype.get("REQUEST_ID", {}).get(
            "recall", 0.0
        ),
        "unexplained_alignment_failures": sum(
            len(_alignment_problems(summary)) for summary in alignments.values()
        ),
        "truncated_entity_spans": sum(
            int(summary["truncated_spans"]) for summary in alignments.values()
        ),
    }


def _write_preflight(
    path: Path,
    evidence: Mapping[str, Any],
    counts: Mapping[str, int],
    methods: Mapping[str, str],
    alignments: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# SecureLogX ML-v1.2 Training Preflight",
        "",
        "**FULL PREFLIGHT PASSED**",
        "",
        f"- Dataset manifest SHA-256: `{evidence['dataset_manifest_sha256']}`",
        f"- Predeclared training gate SHA-256: `{evidence['training_gate_sha256']}`",
        f"- Parent freeze: **{evidence['parent_verified_files']} files verified; {evidence['parent_mismatches']} mismatches**",
        f"- Sealed challenge: **not opened**, regenerated hash `{evidence['sealed_challenge']['actual_sha256']}`",
        "- Ontology: **25 entities / 51 BIO labels**",
        "",
        "## Dataset composition",
        "",
        f"- Original ML-v1 train: **{counts['parent_train']}**",
        f"- ML-v1.1 addition: **{counts['v1_1_additions']}** ({methods['v1_1_additions']})",
        f"- ML-v1.2 addition: **{counts['v1_2_additions']}**",
        f"- Combined in-memory train: **{counts['training']}**",
        f"- Standard dev / ML-v1.2 challenge dev: **{counts['standard_dev']} / {counts['challenge_dev']}**",
        "",
        "## Alignment",
        "",
        "| View | Spans | Aligned | Adjusted | Truncated | Failed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in alignments.items():
        lines.append(
            f"| {name} | {summary['total_spans']} | {summary['successfully_aligned_spans']} | "
            f"{summary['boundary_adjusted_spans']} | {summary['truncated_spans']} | {summary['failed_alignments']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_result_reports(
    root: Path,
    history: Sequence[Mapping[str, Any]],
    selected_epoch: int,
    selected_checkpoint: Path,
    gate_result: Mapping[str, Any],
    targeted: Mapping[str, Any],
    training_arguments: Mapping[str, Any],
    elapsed: float,
) -> None:
    selection = [
        "# SecureLogX ML-v1.2 Checkpoint Selection",
        "",
        "Selection used only standard dev and the ML-v1.2 challenge dev. No sealed-challenge or original-test data was opened.",
        "",
        "| Epoch | Standard micro | Standard macro | Challenge micro | Challenge macro | Selection score |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        selection.append(
            f"| {row['epoch']} | {row['standard_micro_f1']:.6f} | {row['standard_supported_macro_f1']:.6f} | "
            f"{row['challenge_micro_f1']:.6f} | {row['challenge_supported_macro_f1']:.6f} | {row['selection_score']:.6f} |"
        )
    selection.extend(
        [
            "",
            f"Selected epoch **{selected_epoch}** at `{selected_checkpoint.as_posix()}`.",
        ]
    )
    (root / PATHS["selection_report"]).write_text(
        "\n".join(selection) + "\n", encoding="utf-8"
    )

    write_json(root / PATHS["targeted_analysis"], targeted)
    worst = targeted["worst_conflict_category"]
    account = targeted["business_id_false_positives_on_gold_o_accountlike"]
    card = targeted["business_id_vs_credit_card"]["credit_card_to_business_id"]
    ssn = targeted["business_id_vs_ssn"]["business_id_to_ssn"]
    ip = targeted["ip_address_on_technical_reference"]
    subtypes = targeted["business_id_subtype_recall"]
    decision = [
        "# SecureLogX ML-v1.2 Development Gate",
        "",
        f"**{gate_result['decision']}**",
        "",
        "No threshold was changed after results were observed. The sealed challenge and original test were not opened or inferred.",
        "",
        "## Targeted diagnostics",
        "",
        f"- Account-like gold-O → BUSINESS_ID: **{account['false_positive_spans']} spans / {account['records_with_false_positive']} records**.",
        f"- CREDIT_CARD_NUMBER → BUSINESS_ID: **{card['confused_spans']}/{card['gold_support']} ({card['rate']:.6f})**.",
        f"- BUSINESS_ID → SSN: **{ssn['confused_spans']}/{ssn['gold_support']} ({ssn['rate']:.6f})**.",
        f"- Standard-dev SSN → ITIN: **{targeted['standard_dev_ssn_to_itin']['confused_spans']}**.",
        f"- IP_ADDRESS on technical references: **{ip['records_with_error']}/{ip['records']} ({ip['rate']:.6f})**.",
        f"- CUSTOMER_ID / REQUEST_ID recall: **{subtypes['CUSTOMER_ID']['recall']:.6f} / {subtypes['REQUEST_ID']['recall']:.6f}**.",
        f"- Challenge record/context-target error: **{targeted['challenge_record_error_rate']:.6f} / {targeted['challenge_context_target_error_rate']:.6f}**.",
        f"- Worst category: **{worst['category']} ({worst['record_error_rate']:.6f})**.",
        "",
        "## Gate results",
        "",
        "| Gate | Actual | Operator | Threshold | Result |",
        "|---|---:|:---:|---:|---|",
    ]
    for name, result in gate_result["gates"].items():
        decision.append(
            f"| `{name}` | {result['actual']:.6f} | {result['operator']} | {result['threshold']:.6f} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    (root / PATHS["decision_report"]).write_text(
        "\n".join(decision) + "\n", encoding="utf-8"
    )

    summary = [
        "# SecureLogX ML-v1.2 Training Run Summary",
        "",
        "**TRAINING AND DEVELOPMENT EVALUATION COMPLETE**",
        "",
        f"- Records: **{training_arguments['training_records']}** (23,558 + 3,000 + 1,600)",
        f"- Architecture/revision: `{training_arguments['base_model']}` / `{training_arguments['revision']}`",
        "- Epochs / LR / weight decay / warmup: 3 / 2e-5 / 0.01 / 0.1",
        "- Max length / stride / effective batch: 384 / 128 / 16",
        f"- BF16 / seed: {training_arguments['use_bf16']} / 42",
        f"- Selected epoch: **{selected_epoch}**",
        f"- Elapsed train plus dev evaluation: **{elapsed / 60:.2f} minutes**",
        f"- Development decision: **{gate_result['decision']}**",
    ]
    (root / PATHS["training_report"]).write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )


def _canonical_full_run(args: argparse.Namespace) -> None:
    problems: list[str] = []
    if Path(args.root).resolve() != ROOT:
        problems.append("--root is not repository root")
    if args.output != PATHS["output"]:
        problems.append("non-canonical output path")
    for name, expected in SETTINGS.items():
        if getattr(args, name) != expected:
            problems.append(f"{name}={getattr(args, name)!r}, expected {expected!r}")
    if os.environ.get("PYTHONHASHSEED") != "42":
        problems.append("PYTHONHASHSEED must be 42 before Python starts")
    if problems:
        raise ValueError("Non-canonical ML-v1.2 full run:\n- " + "\n- ".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=PATHS["output"])
    parser.add_argument("--base-model", default=SETTINGS["base_model"])
    parser.add_argument("--revision", default=SETTINGS["revision"])
    parser.add_argument("--max-length", type=int, default=SETTINGS["max_length"])
    parser.add_argument("--stride", type=int, default=SETTINGS["stride"])
    parser.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    parser.add_argument("--learning-rate", type=float, default=SETTINGS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=SETTINGS["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=SETTINGS["warmup_ratio"])
    parser.add_argument("--max-grad-norm", type=float, default=SETTINGS["max_grad_norm"])
    parser.add_argument("--train-batch-size", type=int, default=SETTINGS["train_batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=SETTINGS["eval_batch_size"])
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=SETTINGS["gradient_accumulation_steps"],
    )
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-dev-records", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(args.root).resolve()
    if not args.smoke_test:
        _canonical_full_run(args)
    output_root = (root / args.output).resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    evidence = verify_preflight(root)
    gate = evidence["gate"]
    label_config = load_canonical_labels(root / PATHS["labels"])
    label_to_id = {
        str(key): int(value) for key, value in label_config["label_to_id"].items()
    }
    id_to_label = {
        int(key): str(value) for key, value in label_config["id_to_label"].items()
    }
    set_reproducible_seed(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Canonical ML-v1.2 training requires CUDA with BF16")
    device = torch.device("cuda")
    use_bf16 = True
    environment = collect_environment(args.seed, args.base_model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Pinned fast tokenizer is required")

    manifest = evidence["manifest"]
    expected_v1_1_hash = manifest["v1_1_partition_verification"][
        "v1_1_train_addition"
    ]["sha256"]
    parent_records = load_jsonl(root / PATHS["parent_train"])
    v1_1_records, v1_1_method = _load_v1_1_train(
        root, PATHS["v1_1_additions"], expected_v1_1_hash
    )
    v1_2_records = load_jsonl(root / PATHS["v1_2_additions"])
    standard_records = load_jsonl(root / PATHS["standard_dev"])
    challenge_records = load_jsonl(root / PATHS["challenge_dev"])
    if args.smoke_test:
        limit = args.max_train_records or 32
        parent_records = parent_records[:limit]
        v1_1_records = v1_1_records[:limit]
        v1_2_records = v1_2_records[:limit]
        dev_limit = args.max_dev_records or 16
        standard_records = standard_records[:dev_limit]
        challenge_records = challenge_records[:dev_limit]
    train_records = [*parent_records, *v1_1_records, *v1_2_records]
    counts = {
        "parent_train": len(parent_records),
        "v1_1_additions": len(v1_1_records),
        "v1_2_additions": len(v1_2_records),
        "training": len(train_records),
        "standard_dev": len(standard_records),
        "challenge_dev": len(challenge_records),
    }
    if not args.smoke_test and counts != COUNTS:
        raise ValueError(f"Canonical record counts changed: {counts} != {COUNTS}")
    if not all(
        record.get("meta", {}).get("source") == "securelogx_context_contrast_v1_1"
        for record in v1_1_records
    ):
        raise ValueError("ML-v1.1 additions lost provenance")
    if not all(
        record.get("meta", {}).get("source") == "securelogx_counterbalance_v1_2"
        for record in v1_2_records
    ):
        raise ValueError("ML-v1.2 additions lost provenance")

    print(
        f"Building aligned windows: train={len(train_records)} "
        f"standard_dev={len(standard_records)} challenge_dev={len(challenge_records)}",
        flush=True,
    )
    train_build = build_aligned_features(
        train_records, tokenizer, label_to_id, args.max_length, args.stride
    )
    standard_build = build_aligned_features(
        standard_records, tokenizer, label_to_id, args.max_length, args.stride
    )
    challenge_build = build_aligned_features(
        challenge_records, tokenizer, label_to_id, args.max_length, args.stride
    )
    alignments = {
        "train": train_build.summary,
        "standard_dev": standard_build.summary,
        "challenge_dev": challenge_build.summary,
    }
    for split, summary in alignments.items():
        problems = _alignment_problems(summary)
        if problems:
            raise ValueError(f"{split} alignment failed: {problems[:20]}")
    if not args.smoke_test:
        _write_preflight(
            root / PATHS["preflight_report"],
            evidence,
            counts,
            {"v1_1_additions": v1_1_method},
            alignments,
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        AlignedWindowDataset(train_build.windows),
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=TokenClassificationCollator(tokenizer),
        pin_memory=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        revision=args.revision,
        local_files_only=True,
        num_labels=51,
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(device)
    optimizer = _optimizer(model, args.learning_rate, args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    planned_steps = steps_per_epoch * args.epochs
    if args.max_optimizer_steps is not None:
        planned_steps = min(planned_steps, args.max_optimizer_steps)
    warmup_steps = int(round(planned_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=planned_steps
    )
    training_arguments = {
        **SETTINGS,
        "dynamic_padding": True,
        "effective_batch_size": args.train_batch_size
        * args.gradient_accumulation_steps,
        "use_bf16": use_bf16,
        "device": str(device),
        "planned_optimizer_steps": planned_steps,
        "warmup_steps": warmup_steps,
        "parent_train_records": len(parent_records),
        "v1_1_addition_records": len(v1_1_records),
        "v1_2_addition_records": len(v1_2_records),
        "training_records": len(train_records),
        "standard_dev_records": len(standard_records),
        "challenge_dev_records": len(challenge_records),
        "dataset_manifest_sha256": evidence["dataset_manifest_sha256"],
        "training_gate_sha256": evidence["training_gate_sha256"],
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
    tie_tolerance = float(gate["checkpoint_selection"]["effective_tie_absolute"])
    global_steps = 0
    started = time.perf_counter()
    stop = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        numerator = 0.0
        denominator = 0
        epoch_steps = 0
        for batch_index, batch in enumerate(loader):
            group_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps, len(loader) - group_start
            )
            inputs = model_inputs_from_batch(batch, device)
            valid_tokens = int((inputs["labels"] != IGNORE_INDEX).sum().item())
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                loss = model(**inputs).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )
            loss_value = float(loss.detach().float().item())
            numerator += loss_value * valid_tokens
            denominator += valid_tokens
            (loss / group_size).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_steps += 1
            epoch_steps += 1
            if global_steps == 1 or global_steps % 100 == 0:
                elapsed = time.perf_counter() - started
                rate = global_steps / elapsed if elapsed else 0.0
                eta = (planned_steps - global_steps) / rate / 60 if rate else 0.0
                print(
                    f"epoch={epoch} step={global_steps}/{planned_steps} "
                    f"loss={loss_value:.6f} eta_min={eta:.1f}",
                    flush=True,
                )
            if (
                args.max_optimizer_steps is not None
                and global_steps >= args.max_optimizer_steps
            ):
                stop = True
                break

        train_loss = numerator / denominator if denominator else 0.0
        standard, standard_predictions = _score_view(
            model,
            tokenizer,
            standard_records,
            standard_build.windows,
            label_config["entities"],
            label_to_id,
            id_to_label,
            "ml_v1_2_dev_standard",
            args.eval_batch_size,
            device,
            use_bf16,
            False,
        )
        challenge, challenge_predictions = _score_view(
            model,
            tokenizer,
            challenge_records,
            challenge_build.windows,
            label_config["entities"],
            label_to_id,
            id_to_label,
            "ml_v1_2_dev_challenge",
            args.eval_batch_size,
            device,
            use_bf16,
            True,
        )
        score = harmonic_mean(
            standard["supported_macro"]["f1"],
            challenge["supported_macro"]["f1"],
        )
        diagnostics = challenge["context_diagnostics"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "standard_loss": standard["loss"],
            "standard_micro_f1": standard["micro"]["f1"],
            "standard_supported_macro_f1": standard["supported_macro"]["f1"],
            "standard_high_risk_recall": standard["high_risk"]["recall"],
            "challenge_loss": challenge["loss"],
            "challenge_micro_f1": challenge["micro"]["f1"],
            "challenge_supported_macro_f1": challenge["supported_macro"]["f1"],
            "challenge_record_error_rate": diagnostics["record_error_rate"],
            "challenge_context_target_error_rate": diagnostics[
                "context_target_error_rate"
            ],
            "challenge_maximum_category_error_rate": diagnostics[
                "maximum_category_error_rate"
            ],
            "selection_score": score,
            "optimizer_steps": epoch_steps,
            "global_optimizer_steps": global_steps,
        }
        history.append(row)
        epoch_path = checkpoint_root / f"epoch-{epoch}"
        _save_epoch(
            epoch_path,
            model,
            tokenizer,
            {
                "history": row,
                "standard_dev_metrics": _compact_metrics(standard),
                "challenge_dev_metrics": _compact_metrics(challenge),
            },
            training_arguments,
            environment,
        )
        print(
            f"epoch={epoch} standard_micro={row['standard_micro_f1']:.6f} "
            f"standard_macro={row['standard_supported_macro_f1']:.6f} "
            f"challenge_micro={row['challenge_micro_f1']:.6f} "
            f"challenge_macro={row['challenge_supported_macro_f1']:.6f} "
            f"selection={score:.6f}",
            flush=True,
        )
        if score > best_score + tie_tolerance:
            best_epoch = epoch
            best_score = score
            best_source = epoch_path
            best_standard = _compact_metrics(standard)
            best_challenge = _compact_metrics(challenge)
            best_standard_predictions = standard_predictions
            best_challenge_predictions = challenge_predictions
        del standard, challenge
        torch.cuda.empty_cache()
        if stop:
            break

    if any(
        value is None
        for value in (
            best_source,
            best_standard,
            best_challenge,
            best_standard_predictions,
            best_challenge_predictions,
        )
    ):
        raise RuntimeError("No checkpoint was selected")
    assert best_source is not None
    assert best_standard is not None
    assert best_challenge is not None
    assert best_standard_predictions is not None
    assert best_challenge_predictions is not None

    best_checkpoint = output_root / "best-checkpoint"
    shutil.copytree(best_source, best_checkpoint)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(best_checkpoint.iterdir())
        if path.is_file()
    }
    selected = {
        "selected_epoch": best_epoch,
        "selection_metric": "harmonic mean of standard-dev and ML-v1.2 challenge-dev supported entity macro F1",
        "selection_value": best_score,
        "tie_tolerance": tie_tolerance,
        "tie_break": "earlier epoch",
        "source_checkpoint": best_source.relative_to(root).as_posix(),
        "model_sha256": artifact_hashes["model.safetensors"],
        "artifact_sha256": artifact_hashes,
        "checkpoint_fingerprint_sha256": _mapping_fingerprint(artifact_hashes),
    }
    write_json(best_checkpoint / "selected_checkpoint.json", selected)
    write_json(output_root / "training_history.json", history)
    write_json(output_root / "alignment_summary.json", alignments)
    write_json(
        output_root / "dev_standard_predictions.json",
        {
            "split": "ml_v1_2_dev_standard",
            "input_sha256": sha256_file(root / PATHS["standard_dev"]),
            "selected_epoch": best_epoch,
            "checkpoint_fingerprint_sha256": selected[
                "checkpoint_fingerprint_sha256"
            ],
            "records": len(standard_records),
            "predictions": best_standard_predictions,
        },
    )
    write_json(
        output_root / "dev_challenge_predictions.json",
        {
            "split": "ml_v1_2_dev_challenge",
            "input_sha256": sha256_file(root / PATHS["challenge_dev"]),
            "selected_epoch": best_epoch,
            "checkpoint_fingerprint_sha256": selected[
                "checkpoint_fingerprint_sha256"
            ],
            "records": len(challenge_records),
            "predictions": best_challenge_predictions,
        },
    )

    targeted = targeted_analysis(
        standard_records,
        best_standard_predictions,
        challenge_records,
        best_challenge_predictions,
        best_challenge,
        gate["historical_reference"],
    )
    actual = _gate_actual(best_standard, best_challenge, targeted, alignments)
    evaluated = evaluate_named_gates(
        actual, gate["development_gate"]["all_required"]
    )
    decision = (
        gate["development_gate"]["decision_on_pass"]
        if evaluated["all_passed"]
        else gate["development_gate"]["decision_on_failure"]
    )
    gate_result = {
        "schema_version": 1,
        "status": "PASS" if evaluated["all_passed"] else "FAIL",
        "decision": decision,
        "passed": evaluated["all_passed"],
        "gate_sha256": evidence["training_gate_sha256"],
        "dataset_manifest_sha256": evidence["dataset_manifest_sha256"],
        "checkpoint_fingerprint_sha256": selected[
            "checkpoint_fingerprint_sha256"
        ],
        "selected_epoch": best_epoch,
        "actual": actual,
        "gates": evaluated["gates"],
        "sealed_challenge_opened": False,
        "original_test_opened": False,
    }
    write_json(output_root / "dev_gate_decision.json", gate_result)
    if args.smoke_test:
        write_json(
            output_root / "smoke_result.json",
            {
                "status": "SMOKE TEST PASSED",
                "optimizer_steps": global_steps,
                "selected": selected,
            },
        )
    else:
        write_json(root / PATHS["standard_metrics"], best_standard)
        write_json(root / PATHS["challenge_metrics"], best_challenge)
        _write_result_reports(
            root,
            history,
            best_epoch,
            best_checkpoint.relative_to(root),
            gate_result,
            targeted,
            training_arguments,
            time.perf_counter() - started,
        )

    verify_preflight(root)
    if sha256_file(root / PATHS["gate"]) != evidence["training_gate_sha256"]:
        raise ValueError("Predeclared training gate changed during training")
    print(f"Selected epoch {best_epoch}: {best_checkpoint}", flush=True)
    print(decision, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
