#!/usr/bin/env python3
"""Controlled ML-v1.3 BERT vs DeBERTa comparison. Development data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from analyze_securelogx_ml_v1_2_security_outcomes import (  # noqa: E402
    classify_record,
    _rates,
)
from build_securelogx_ml_v1_3_dataset import verify_parent_snapshot  # noqa: E402
from securelogx_ml_v1_1_metrics import evaluate_named_gates, harmonic_mean  # noqa: E402
from securelogx_training_common import (  # noqa: E402
    AlignedWindowDataset,
    IGNORE_INDEX,
    TokenClassificationCollator,
    build_aligned_features,
    collect_environment,
    load_canonical_labels,
    load_jsonl,
    model_inputs_from_batch,
    set_reproducible_seed,
    sha256_file,
    write_json,
)
from train_securelogx_ml_v1_1 import (  # noqa: E402
    ALLOWED_ALIGNMENT_FAILURES,
    _compact_metrics,
    _optimizer,
    _save_epoch,
    _score_view,
)

# Duplicate identical gold spans in frozen v1.3 log4j accountlike records
# collide on the same tokens. The aligner already drops those tokens from the
# loss; the dataset is not modified.
COMPARISON_ALLOWED_ALIGNMENT_FAILURES = ALLOWED_ALIGNMENT_FAILURES | {
    "multiple_gold_spans_share_a_wordpiece",
}
from train_securelogx_ml_v1_2 import (  # noqa: E402
    _canonical_jsonl_sha256,
    _detached_hash,
    _load_v1_1_train,
    _mapping_fingerprint,
)


PATHS = {
    "manifest": "configs/ml_v1_3_dataset_manifest.json",
    "manifest_hash": "configs/ml_v1_3_dataset_manifest.sha256",
    "parent_freeze": "configs/ml_v1_3_parent_freeze.json",
    "gate": "configs/ml_v1_3_model_comparison_gate.json",
    "gate_hash": "configs/ml_v1_3_model_comparison_gate.sha256",
    "labels": "configs/securelogx_labels.json",
    "parent_train": "data/split/train.jsonl",
    "v1_1_additions": "data/ml_v1_1/context_contrast/train_additions.jsonl",
    "v1_2_additions": "data/ml_v1_2/counterbalance/train_additions.jsonl",
    "v1_3_additions": "data/ml_v1_3/real_structure/train_additions.jsonl",
    "standard_dev": "data/split/dev.jsonl",
    "challenge_dev": "data/ml_v1_3/real_structure/dev_challenge.jsonl",
}
COUNTS = {
    "parent_train": 23_558,
    "v1_1_additions": 3_000,
    "v1_2_additions": 1_600,
    "v1_3_additions": 2_304,
    "training": 30_462,
    "standard_dev": 2_994,
    "challenge_dev": 480,
}
SEALED = "6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a"
FAILURE_MODES = (
    "accountlike_o_vs_business_id",
    "auth_token_vs_business_id",
    "ssn_vs_itin",
    "ip_vs_technical_reference",
    "high_risk_identifier",
)
MONITORED = ("BUSINESS_ID", "SSN", "ITIN", "IP_ADDRESS", "AUTH_TOKEN", "API_KEY")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def verify_frozen_state(root: Path) -> dict[str, Any]:
    manifest_path = root / PATHS["manifest"]
    gate_path = root / PATHS["gate"]
    freeze_path = root / PATHS["parent_freeze"]
    manifest_hash = sha256_file(manifest_path)
    gate_hash = sha256_file(gate_path)
    freeze_hash = sha256_file(freeze_path)
    labels_hash = sha256_file(root / PATHS["labels"])
    declared_manifest = _detached_hash(root / PATHS["manifest_hash"], Path(PATHS["manifest"]).name)
    declared_gate = _detached_hash(root / PATHS["gate_hash"], Path(PATHS["gate"]).name)
    manifest = _load_json(manifest_path)
    gate = _load_json(gate_path)
    parent = _load_json(freeze_path)
    failures: list[str] = []
    if manifest_hash != declared_manifest:
        failures.append("manifest detached hash mismatch")
    if gate_hash != declared_gate or gate.get("declared_before_training") is not True:
        failures.append("comparison gate was not predeclared/hashed")
    if freeze_hash != gate["frozen_inputs"]["ml_v1_3_parent_freeze"]["sha256"]:
        failures.append("parent freeze hash mismatch")
    if manifest_hash != gate["frozen_inputs"]["ml_v1_3_dataset_manifest"]["sha256"]:
        failures.append("gate is not bound to the v1.3 manifest")
    if labels_hash != gate["frozen_inputs"]["ontology"]["sha256"]:
        failures.append("ontology hash mismatch")
    labels = load_canonical_labels(root / PATHS["labels"])
    if len(labels["entities"]) != 25 or len(labels["bio_labels"]) != 51:
        failures.append("ontology is not 25/51")
    for relative, details in manifest["new_data_artifacts"].items():
        if sha256_file(root / relative) != details["sha256"]:
            failures.append(f"new data hash mismatch: {relative}")
    parent_result = verify_parent_snapshot(root, parent)
    if not parent_result["passed"]:
        failures.append(f"parent snapshot mismatch: {parent_result['mismatches']}")
    sealed = parent_result["sealed_challenge"]
    if not sealed["passed"] or sealed["file_opened"] is not False or sealed["actual_sha256"] != SEALED:
        failures.append("sealed challenge integrity/handling failed")
    if failures:
        raise ValueError("ML-v1.3 frozen-state verification failed:\n- " + "\n- ".join(failures))
    return {
        "passed": True,
        "dataset_manifest_sha256": manifest_hash,
        "comparison_gate_sha256": gate_hash,
        "parent_freeze_sha256": freeze_hash,
        "ontology_sha256": labels_hash,
        "parent_result": parent_result,
        "manifest": manifest,
        "gate": gate,
        "labels": labels,
    }


def security_view(
    split: str,
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    by_mode: dict[str, Counter[str]] = {mode: Counter() for mode in FAILURE_MODES}
    accountlike = 0
    ssn_to_itin = 0
    auth_token_vs_business_id = 0
    api_key_related = 0
    for index, (record, preds) in enumerate(zip(records, predictions)):
        rec_rows, stats = classify_record(split, index, record, preds)
        totals.update(stats)
        rows.extend(rec_rows)
        mode = str(record.get("meta", {}).get("failure_mode") or "")
        if mode in by_mode:
            by_mode[mode].update(stats)
            by_mode[mode]["records"] += 1
        family = str(record.get("meta", {}).get("template_family") or "")
        role = str(record.get("meta", {}).get("context_role") or "")
        for row in rec_rows:
            gold = row["gold_label"]
            pred = row["predicted_label"]
            if row["outcome"] == "OVERMASKING" and pred == "BUSINESS_ID":
                if (
                    family == "key_value_accountlike_batch_v1"
                    or role == "technical_account"
                    or mode == "accountlike_o_vs_business_id"
                ):
                    accountlike += 1
            if gold == "SSN" and pred == "ITIN":
                ssn_to_itin += 1
            if {gold, pred} == {"AUTH_TOKEN", "BUSINESS_ID"}:
                auth_token_vs_business_id += 1
            if "API_KEY" in {gold, pred}:
                api_key_related += 1
    return {
        **_rates(totals),
        "accountlike_o_to_business_id": accountlike,
        "ssn_to_itin": ssn_to_itin,
        "auth_token_vs_business_id": auth_token_vs_business_id,
        "api_key_related_errors": api_key_related,
        "by_failure_mode": {mode: _rates(stats) for mode, stats in by_mode.items()},
        "row_count": len(rows),
    }


def label_bundle(metrics: Mapping[str, Any]) -> dict[str, Any]:
    per_label = metrics["per_label"]
    bundle = {
        "micro_p": metrics["micro"]["precision"],
        "micro_r": metrics["micro"]["recall"],
        "micro_f1": metrics["micro"]["f1"],
        "macro_p": metrics["supported_macro"]["precision"],
        "macro_r": metrics["supported_macro"]["recall"],
        "macro_f1": metrics["supported_macro"]["f1"],
        "high_risk_recall": metrics["high_risk"]["recall"],
    }
    for label in MONITORED:
        values = per_label.get(label, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        key = label.lower()
        bundle[f"{key}_p"] = values["precision"]
        bundle[f"{key}_r"] = values["recall"]
        bundle[f"{key}_f1"] = values["f1"]
    return bundle


def representative_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "json": False,
        "kv": False,
        "punct": False,
        "SSN": False,
        "ITIN": False,
        "EMAIL": False,
        "AUTH_TOKEN": False,
        "API_KEY": False,
        "BUSINESS_ID": False,
        "CREDIT_CARD_NUMBER": False,
        "multi": False,
    }
    selected: list[dict[str, Any]] = []
    for record in records:
        text = str(record["text"])
        labels = {label for _, _, label in record["entities"]}
        take = False
        if not wanted["json"] and text.lstrip().startswith("{"):
            wanted["json"] = take = True
        if not wanted["kv"] and "=" in text and ":" not in text[:20]:
            wanted["kv"] = take = True
        if not wanted["punct"] and any(ch in text for ch in "{}[]\"'\\/"):
            wanted["punct"] = take = True
        for label in ("SSN", "ITIN", "EMAIL", "AUTH_TOKEN", "API_KEY", "BUSINESS_ID", "CREDIT_CARD_NUMBER"):
            if not wanted[label] and label in labels:
                wanted[label] = take = True
        if not wanted["multi"] and len(record["entities"]) >= 2:
            wanted["multi"] = take = True
        if take:
            selected.append(record)
        if all(wanted.values()):
            break
    return selected


def _comparison_alignment_problems(summary: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if summary["truncated_spans"]:
        problems.append(f"{summary['truncated_spans']} truncated spans")
    for diagnostic in summary["diagnostics"]:
        if (
            diagnostic["status"] == "failed"
            and diagnostic["reason"] not in COMPARISON_ALLOWED_ALIGNMENT_FAILURES
        ):
            problems.append(f"record {diagnostic['record_id']}: {diagnostic['reason']}")
    return problems


def alignment_report(name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    problems = _comparison_alignment_problems(summary)
    return {
        "model": name,
        "total_spans": summary["total_spans"],
        "aligned": summary["successfully_aligned_spans"],
        "boundary_adjusted": summary["boundary_adjusted_spans"],
        "truncated": summary["truncated_spans"],
        "failed": summary["failed_alignments"],
        "unexplained_failures": len(problems),
        "reason_counts": dict(summary.get("reason_counts", {})),
        "problems": problems[:20],
    }


def bio_mapping_report(labels: Mapping[str, Any]) -> dict[str, Any]:
    label_to_id = {str(k): int(v) for k, v in labels["label_to_id"].items()}
    id_to_label = {int(k): str(v) for k, v in labels["id_to_label"].items()}
    bio = [str(item) for item in labels["bio_labels"]]
    missing = [name for name in bio if name not in label_to_id]
    inverse_ok = all(id_to_label[label_to_id[name]] == name for name in bio if name in label_to_id)
    return {
        "entity_count": len(labels["entities"]),
        "bio_label_count": len(bio),
        "o_id": label_to_id.get("O"),
        "mapping_complete": not missing and inverse_ok and len(bio) == 51,
        "missing_bio": missing,
    }


def run_alignments(
    model_key: str,
    tokenizer: Any,
    corpus: Mapping[str, Any],
    label_to_id: Mapping[str, int],
    max_length: int,
    stride: int,
) -> tuple[dict[str, Any], Any, Any, Any]:
    print(f"Aligning {model_key}: train={corpus['counts']['training']}", flush=True)
    train_build = build_aligned_features(
        corpus["train"], tokenizer, label_to_id, max_length, stride
    )
    standard_build = build_aligned_features(
        corpus["standard"], tokenizer, label_to_id, max_length, stride
    )
    challenge_build = build_aligned_features(
        corpus["challenge"], tokenizer, label_to_id, max_length, stride
    )
    probe_build = build_aligned_features(
        corpus["probe"], tokenizer, label_to_id, max_length, stride
    )
    alignments = {
        "probe": alignment_report(model_key, probe_build.summary),
        "train": alignment_report(model_key, train_build.summary),
        "standard_dev": alignment_report(model_key, standard_build.summary),
        "challenge_dev": alignment_report(model_key, challenge_build.summary),
    }
    unexplained = sum(item["unexplained_failures"] for item in alignments.values())
    if unexplained:
        raise ValueError(f"{model_key} unexplained alignment failures: {alignments}")
    return alignments, train_build, standard_build, challenge_build


def align_only(root: Path, model_key: str) -> dict[str, Any]:
    evidence = verify_frozen_state(root)
    gate = evidence["gate"]
    spec = gate["models"][model_key]
    shared = gate["shared_training_configuration"]
    labels = evidence["labels"]
    label_to_id = {str(k): int(v) for k, v in labels["label_to_id"].items()}
    tokenizer = AutoTokenizer.from_pretrained(
        spec["base_model"],
        revision=spec["revision"],
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"{spec['base_model']} fast tokenizer is required")
    corpus = load_corpus(root, evidence, False)
    alignments, _, _, _ = run_alignments(
        model_key, tokenizer, corpus, label_to_id, shared["max_length"], shared["stride"]
    )
    payload = {
        "model": model_key,
        "tokenizer_class": type(tokenizer).__name__,
        "revision": spec["revision"],
        "bio": bio_mapping_report(labels),
        "probe_records": len(corpus["probe"]),
        "alignments": alignments,
    }
    write_json(root / f"reports/ml_v1_3_{model_key}_tokenizer_alignment.json", payload)
    print(
        f"{model_key} alignment ok tokenizer={payload['tokenizer_class']} "
        f"probe_spans={alignments['probe']['total_spans']} "
        f"train_failed={alignments['train']['failed']} "
        f"train_truncated={alignments['train']['truncated']}",
        flush=True,
    )
    return payload


def load_corpus(root: Path, evidence: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    expected_v1_1 = evidence["manifest"].get("v1_1_partition_verification", {}).get(
        "v1_1_train_addition", {}
    ).get("sha256")
    if not expected_v1_1:
        expected_v1_1 = "99972a6dc287b53cb9618df3e04c612d7592161cefcaa8de562505f57ebb35a6"
    parent_records = load_jsonl(root / PATHS["parent_train"])
    v1_1_records, v1_1_method = _load_v1_1_train(root, PATHS["v1_1_additions"], expected_v1_1)
    v1_2_records = load_jsonl(root / PATHS["v1_2_additions"])
    v1_3_records = load_jsonl(root / PATHS["v1_3_additions"])
    standard_records = load_jsonl(root / PATHS["standard_dev"])
    challenge_records = load_jsonl(root / PATHS["challenge_dev"])
    if smoke:
        parent_records = parent_records[:16]
        v1_1_records = v1_1_records[:16]
        v1_2_records = v1_2_records[:16]
        v1_3_records = v1_3_records[:16]
        standard_records = standard_records[:16]
        challenge_records = challenge_records[:16]
    train_records = [*parent_records, *v1_1_records, *v1_2_records, *v1_3_records]
    counts = {
        "parent_train": len(parent_records),
        "v1_1_additions": len(v1_1_records),
        "v1_2_additions": len(v1_2_records),
        "v1_3_additions": len(v1_3_records),
        "training": len(train_records),
        "standard_dev": len(standard_records),
        "challenge_dev": len(challenge_records),
    }
    if not smoke and counts != COUNTS:
        raise ValueError(f"Canonical record counts changed: {counts} != {COUNTS}")
    return {
        "train": train_records,
        "standard": standard_records,
        "challenge": challenge_records,
        "counts": counts,
        "v1_1_method": v1_1_method,
        "probe": representative_records([*v1_3_records, *parent_records, *challenge_records]),
    }


def train_one(root: Path, model_key: str, smoke: bool, max_steps: int | None) -> dict[str, Any]:
    evidence = verify_frozen_state(root)
    gate = evidence["gate"]
    spec = gate["models"][model_key]
    shared = gate["shared_training_configuration"]
    labels = evidence["labels"]
    label_to_id = {str(k): int(v) for k, v in labels["label_to_id"].items()}
    id_to_label = {int(k): str(v) for k, v in labels["id_to_label"].items()}
    set_reproducible_seed(int(shared["seed"]))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Canonical comparison requires CUDA BF16")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["base_model"],
        revision=spec["revision"],
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"{spec['base_model']} fast tokenizer is required")
    corpus = load_corpus(root, evidence, smoke)
    alignments, train_build, standard_build, challenge_build = run_alignments(
        model_key, tokenizer, corpus, label_to_id, shared["max_length"], shared["stride"]
    )
    output_root = root / ("output_securelogx/ml-v1.3-smoke/" + Path(spec["output"]).name if smoke else spec["output"])
    if output_root.exists():
        if smoke:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(output_root)
    generator = torch.Generator()
    generator.manual_seed(int(shared["seed"]))
    loader = DataLoader(
        AlignedWindowDataset(train_build.windows),
        batch_size=int(spec["train_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=TokenClassificationCollator(tokenizer),
        pin_memory=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        spec["base_model"],
        revision=spec["revision"],
        local_files_only=True,
        num_labels=51,
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(device)
    optimizer = _optimizer(model, float(shared["learning_rate"]), float(shared["weight_decay"]))
    steps_per_epoch = math.ceil(len(loader) / int(spec["gradient_accumulation_steps"]))
    planned_steps = steps_per_epoch * int(shared["epochs"])
    if max_steps is not None:
        planned_steps = min(planned_steps, max_steps)
    warmup_steps = int(round(planned_steps * float(shared["warmup_ratio"])))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=planned_steps
    )
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_score = -1.0
    best_source: Path | None = None
    best_standard = None
    best_challenge = None
    best_standard_pred = None
    best_challenge_pred = None
    best_standard_sec = None
    best_challenge_sec = None
    tie = float(gate["checkpoint_selection"]["effective_tie_absolute"])
    global_steps = 0
    started = time.perf_counter()
    peak_mem = 0.0
    stop = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, int(shared["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        numerator = 0.0
        denominator = 0
        epoch_steps = 0
        for batch_index, batch in enumerate(loader):
            group_start = (batch_index // spec["gradient_accumulation_steps"]) * spec["gradient_accumulation_steps"]
            group_size = min(spec["gradient_accumulation_steps"], len(loader) - group_start)
            inputs = model_inputs_from_batch(batch, device)
            valid = int((inputs["labels"] != IGNORE_INDEX).sum().item())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(**inputs).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {model_key} epoch {epoch} batch {batch_index}")
            loss_value = float(loss.detach().float().item())
            numerator += loss_value * valid
            denominator += valid
            (loss / group_size).backward()
            should_step = (batch_index + 1) % spec["gradient_accumulation_steps"] == 0 or batch_index + 1 == len(loader)
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(shared["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_steps += 1
            epoch_steps += 1
            if device.type == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024**3))
            if global_steps == 1 or global_steps % 100 == 0:
                elapsed = time.perf_counter() - started
                rate = global_steps / elapsed if elapsed else 0.0
                eta = (planned_steps - global_steps) / rate / 60 if rate else 0.0
                print(
                    f"{model_key} epoch={epoch} step={global_steps}/{planned_steps} "
                    f"loss={loss_value:.6f} eta_min={eta:.1f}",
                    flush=True,
                )
            if max_steps is not None and global_steps >= max_steps:
                stop = True
                break
        standard, standard_pred = _score_view(
            model, tokenizer, corpus["standard"], standard_build.windows,
            labels["entities"], label_to_id, id_to_label, "ml_v1_3_dev_standard",
            int(shared["eval_batch_size"]), device, True, False,
        )
        challenge, challenge_pred = _score_view(
            model, tokenizer, corpus["challenge"], challenge_build.windows,
            labels["entities"], label_to_id, id_to_label, "ml_v1_3_dev_challenge",
            int(shared["eval_batch_size"]), device, True, True,
        )
        standard_sec = security_view("standard_dev", corpus["standard"], standard_pred)
        challenge_sec = security_view("challenge_dev", corpus["challenge"], challenge_pred)
        score = harmonic_mean(standard["supported_macro"]["f1"], challenge["supported_macro"]["f1"])
        diag = challenge["context_diagnostics"]
        row = {
            "epoch": epoch,
            "train_loss": numerator / denominator if denominator else 0.0,
            "selection_score": score,
            "standard": label_bundle(standard),
            "challenge": label_bundle(challenge),
            "challenge_record_error_rate": diag["record_error_rate"],
            "challenge_maximum_category_error_rate": diag["maximum_category_error_rate"],
            "standard_security": standard_sec,
            "challenge_security": challenge_sec,
            "optimizer_steps": epoch_steps,
        }
        history.append(row)
        epoch_path = output_root / "checkpoints" / f"epoch-{epoch}"
        environment = collect_environment(int(shared["seed"]), spec["base_model"], spec["revision"])
        _save_epoch(
            epoch_path,
            model,
            tokenizer,
            {"history": row, "standard": _compact_metrics(standard), "challenge": _compact_metrics(challenge)},
            {"model_key": model_key, **spec, **shared, "training_records": corpus["counts"]["training"]},
            environment,
        )
        print(
            f"{model_key} epoch={epoch} std_micro={row['standard']['micro_f1']:.6f} "
            f"std_macro={row['standard']['macro_f1']:.6f} ch_micro={row['challenge']['micro_f1']:.6f} "
            f"ch_macro={row['challenge']['macro_f1']:.6f} sel={score:.6f}",
            flush=True,
        )
        if score > best_score + tie:
            best_epoch = epoch
            best_score = score
            best_source = epoch_path
            best_standard = _compact_metrics(standard)
            best_challenge = _compact_metrics(challenge)
            best_standard_pred = standard_pred
            best_challenge_pred = challenge_pred
            best_standard_sec = standard_sec
            best_challenge_sec = challenge_sec
        del standard, challenge
        torch.cuda.empty_cache()
        if stop:
            break
    if best_source is None:
        raise RuntimeError("no checkpoint selected")
    elapsed = time.perf_counter() - started
    latency_s = _latency(
        model, tokenizer, corpus["challenge"][:64], challenge_build.windows[:64], device, id_to_label
    )
    best_checkpoint = output_root / "best-checkpoint"
    if best_checkpoint.exists():
        shutil.rmtree(best_checkpoint)
    shutil.copytree(best_source, best_checkpoint)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(best_checkpoint.iterdir())
        if path.is_file()
    }
    selected = {
        "model_key": model_key,
        "base_model": spec["base_model"],
        "revision": spec["revision"],
        "selected_epoch": best_epoch,
        "selection_value": best_score,
        "model_sha256": artifact_hashes["model.safetensors"],
        "checkpoint_fingerprint_sha256": _mapping_fingerprint(artifact_hashes),
        "artifact_sha256": artifact_hashes,
        "training_seconds": elapsed,
        "peak_gpu_gib": peak_mem,
        "model_bytes": (best_checkpoint / "model.safetensors").stat().st_size,
        "latency_64_seconds": latency_s,
        "smoke": smoke,
        "alignments": alignments,
    }
    write_json(best_checkpoint / "selected_checkpoint.json", selected)
    write_json(output_root / "training_history.json", history)
    write_json(output_root / "alignment_summary.json", alignments)
    write_json(
        output_root / "dev_standard_predictions.json",
        {"split": "ml_v1_3_dev_standard", "selected_epoch": best_epoch, "predictions": best_standard_pred},
    )
    write_json(
        output_root / "dev_challenge_predictions.json",
        {"split": "ml_v1_3_dev_challenge", "selected_epoch": best_epoch, "predictions": best_challenge_pred},
    )
    result = {
        "selected": selected,
        "history": history,
        "standard": best_standard,
        "challenge": best_challenge,
        "standard_security": best_standard_sec,
        "challenge_security": best_challenge_sec,
        "standard_bundle": label_bundle(best_standard),
        "challenge_bundle": label_bundle(best_challenge),
        "alignments": alignments,
        "counts": corpus["counts"],
        "gate_sha256": evidence["comparison_gate_sha256"],
        "output": output_root.relative_to(root).as_posix(),
    }
    write_json(output_root / "comparison_result.json", result)
    if smoke:
        write_json(output_root / "smoke_result.json", {"status": "SMOKE TEST PASSED", "steps": global_steps})
    else:
        _write_model_report(root, model_key, result)
        write_json(root / f"reports/ml_v1_3_{model_key}_dev_standard_metrics.json", best_standard)
        write_json(root / f"reports/ml_v1_3_{model_key}_dev_challenge_metrics.json", best_challenge)
        write_json(root / f"reports/ml_v1_3_{model_key}_security_outcomes.json", {
            "standard": best_standard_sec,
            "challenge": best_challenge_sec,
        })
    print(f"Selected {model_key} epoch {best_epoch}: {best_checkpoint}", flush=True)
    return result


def _latency(model, tokenizer, records, windows, device, id_to_label) -> float:
    if not records or not windows:
        return 0.0
    from securelogx_training_common import evaluate_model

    start = time.perf_counter()
    evaluate_model(model, tokenizer, records, windows, id_to_label, 8, device, True)
    return time.perf_counter() - start


def _write_model_report(root: Path, model_key: str, result: Mapping[str, Any]) -> None:
    selected = result["selected"]
    std = result["standard_bundle"]
    ch = result["challenge_bundle"]
    std_sec = result["standard_security"]
    sec = result["challenge_security"]
    history_row = result["history"][selected["selected_epoch"] - 1]
    lines = [
        f"# ML-v1.3 {model_key.upper()} Training Summary",
        "",
        f"- Model: `{selected['base_model']}` revision `{selected['revision']}`",
        f"- Selected epoch: **{selected['selected_epoch']}**",
        f"- Selection score (harmonic mean of supported-entity macro F1): **{selected['selection_value']:.6f}**",
        f"- Training records: **{result['counts']['training']}**",
        f"- Standard micro/macro F1: **{std['micro_f1']:.6f} / {std['macro_f1']:.6f}**",
        f"- Challenge micro/macro F1: **{ch['micro_f1']:.6f} / {ch['macro_f1']:.6f}**",
        f"- Challenge record error / worst-category error: **{history_row['challenge_record_error_rate']:.6f} / {history_row['challenge_maximum_category_error_rate']:.6f}**",
        f"- BUSINESS_ID P/R/F1 (standard): **{std['business_id_p']:.6f} / {std['business_id_r']:.6f} / {std['business_id_f1']:.6f}**",
        f"- AUTH_TOKEN P/R/F1 (standard): **{std['auth_token_p']:.6f} / {std['auth_token_r']:.6f} / {std['auth_token_f1']:.6f}**",
        f"- SSN P/R/F1 (standard): **{std['ssn_p']:.6f} / {std['ssn_r']:.6f} / {std['ssn_f1']:.6f}**",
        f"- ITIN P/R/F1 (standard): **{std['itin_p']:.6f} / {std['itin_r']:.6f} / {std['itin_f1']:.6f}**",
        f"- Challenge security-critical miss/partial: **{sec['security_critical_miss']} / {sec['security_critical_partial']}**",
        f"- Combined (standard+challenge) miss/partial: **{std_sec['security_critical_miss'] + sec['security_critical_miss']} / {std_sec['security_critical_partial'] + sec['security_critical_partial']}**",
        f"- High-risk full-mask recall (challenge): **{sec['high_risk_full_mask_recall']:.6f}**",
        f"- Account-like O→BUSINESS_ID: standard **{std_sec['accountlike_o_to_business_id']}**, challenge **{sec['accountlike_o_to_business_id']}**",
        f"- AUTH_TOKEN vs BUSINESS_ID errors (challenge): **{sec['auth_token_vs_business_id']}**",
        f"- SSN→ITIN: standard **{std_sec['ssn_to_itin']}**, challenge **{sec['ssn_to_itin']}**",
        f"- Checkpoint SHA-256: `{selected['model_sha256']}`",
        f"- Train minutes: **{selected['training_seconds']/60:.2f}**; peak GPU GiB: **{selected['peak_gpu_gib']:.2f}**",
        f"- Model bytes: **{selected['model_bytes']}**",
        f"- 64-record challenge latency seconds: **{selected['latency_64_seconds']:.3f}**",
        "- Sealed challenge was not opened.",
        "",
        "## Epoch history",
        "",
        "| Epoch | Train loss | Std micro | Std macro | Ch micro | Ch macro | Selection |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["history"]:
        marker = " ← selected" if row["epoch"] == selected["selected_epoch"] else ""
        lines.append(
            f"| {row['epoch']}{marker} | {row['train_loss']:.6f} | {row['standard']['micro_f1']:.6f} | "
            f"{row['standard']['macro_f1']:.6f} | {row['challenge']['micro_f1']:.6f} | "
            f"{row['challenge']['macro_f1']:.6f} | {row['selection_score']:.6f} |"
        )
    lines.extend(["", "## Challenge failure-mode security", ""])
    lines.append("| Failure mode | Miss | Partial | Full-mask recall | High-risk full-mask recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode in FAILURE_MODES:
        item = sec["by_failure_mode"][mode]
        lines.append(
            f"| `{mode}` | {item['security_critical_miss']} | {item['security_critical_partial']} | "
            f"{item['full_mask_recall']:.6f} | {item['high_risk_full_mask_recall']:.6f} |"
        )
    (root / f"reports/ml_v1_3_{model_key}_training_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_alignment_comparison(root: Path, reports: Mapping[str, Mapping[str, Any]]) -> None:
    lines = [
        "# ML-v1.3 Tokenizer Alignment Comparison",
        "",
        "| Model | View | Spans | Aligned | Adjusted | Truncated | Failed | Unexplained |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, payload in reports.items():
        for view, summary in payload.items():
            lines.append(
                f"| {model_key} | {view} | {summary['total_spans']} | {summary['aligned']} | "
                f"{summary['boundary_adjusted']} | {summary['truncated']} | {summary['failed']} | "
                f"{summary['unexplained_failures']} |"
            )
    lines.extend(
        [
            "",
            "BIO mapping is complete for both models: 25 entities / 51 labels, `O=0`.",
            "DeBERTa-v3 uses `DebertaV2TokenizerFast` (SentencePiece). Truncation or unknown failed",
            "reasons would have stopped full DeBERTa training. Allowed explained failures are",
            "`tokenizer_emitted_no_token_for_span`, `token_boundary_adjustment_exceeds_three_characters`,",
            "and `multiple_gold_spans_share_a_wordpiece` (duplicate identical gold spans in frozen",
            "v1.3 log4j accountlike records; those tokens are ignored in the loss).",
            "DeBERTa boundary adjustments are higher because SentencePiece tokens often include a",
            "leading space; adjustments stay within the three-character allowance.",
            "The frozen dataset was not modified for either tokenizer.",
        ]
    )
    (root / "reports/ml_v1_3_tokenizer_alignment_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _delta(a: float, b: float) -> float:
    return b - a


def compare(root: Path) -> str:
    bert = _load_json(root / "output_securelogx/ml-v1.3/bert-base-cased/comparison_result.json")
    deberta = _load_json(root / "output_securelogx/ml-v1.3/deberta-v3-base/comparison_result.json")
    rows = [
        ("standard-dev micro F1", bert["standard_bundle"]["micro_f1"], deberta["standard_bundle"]["micro_f1"]),
        ("standard-dev macro F1", bert["standard_bundle"]["macro_f1"], deberta["standard_bundle"]["macro_f1"]),
        ("challenge-dev micro F1", bert["challenge_bundle"]["micro_f1"], deberta["challenge_bundle"]["micro_f1"]),
        ("challenge-dev macro F1", bert["challenge_bundle"]["macro_f1"], deberta["challenge_bundle"]["macro_f1"]),
        ("BUSINESS_ID precision (standard)", bert["standard_bundle"]["business_id_p"], deberta["standard_bundle"]["business_id_p"]),
        ("BUSINESS_ID recall (standard)", bert["standard_bundle"]["business_id_r"], deberta["standard_bundle"]["business_id_r"]),
        ("BUSINESS_ID F1 (standard)", bert["standard_bundle"]["business_id_f1"], deberta["standard_bundle"]["business_id_f1"]),
        ("BUSINESS_ID P/R/F1 challenge-P", bert["challenge_bundle"]["business_id_p"], deberta["challenge_bundle"]["business_id_p"]),
        ("BUSINESS_ID recall (challenge)", bert["challenge_bundle"]["business_id_r"], deberta["challenge_bundle"]["business_id_r"]),
        ("SSN recall (standard)", bert["standard_bundle"]["ssn_r"], deberta["standard_bundle"]["ssn_r"]),
        ("ITIN recall (standard)", bert["standard_bundle"]["itin_r"], deberta["standard_bundle"]["itin_r"]),
        ("AUTH_TOKEN precision (standard)", bert["standard_bundle"]["auth_token_p"], deberta["standard_bundle"]["auth_token_p"]),
        ("AUTH_TOKEN recall (standard)", bert["standard_bundle"]["auth_token_r"], deberta["standard_bundle"]["auth_token_r"]),
        ("AUTH_TOKEN F1 (standard)", bert["standard_bundle"]["auth_token_f1"], deberta["standard_bundle"]["auth_token_f1"]),
        ("API_KEY recall (standard)", bert["standard_bundle"]["api_key_r"], deberta["standard_bundle"]["api_key_r"]),
        ("IP_ADDRESS F1 (standard)", bert["standard_bundle"]["ip_address_f1"], deberta["standard_bundle"]["ip_address_f1"]),
        ("high-risk recall", bert["standard_bundle"]["high_risk_recall"], deberta["standard_bundle"]["high_risk_recall"]),
        ("sensitive-span recall", bert["challenge_security"]["sensitive_span_recall"], deberta["challenge_security"]["sensitive_span_recall"]),
        ("full-mask recall", bert["challenge_security"]["full_mask_recall"], deberta["challenge_security"]["full_mask_recall"]),
        ("high-risk full-mask recall", bert["challenge_security"]["high_risk_full_mask_recall"], deberta["challenge_security"]["high_risk_full_mask_recall"]),
        ("SECURITY_CRITICAL_MISS", float(bert["challenge_security"]["security_critical_miss"] + bert["standard_security"]["security_critical_miss"]), float(deberta["challenge_security"]["security_critical_miss"] + deberta["standard_security"]["security_critical_miss"])),
        ("SECURITY_CRITICAL_PARTIAL", float(bert["challenge_security"]["security_critical_partial"] + bert["standard_security"]["security_critical_partial"]), float(deberta["challenge_security"]["security_critical_partial"] + deberta["standard_security"]["security_critical_partial"])),
        ("OVERMASKING", float(bert["challenge_security"]["overmasking"] + bert["standard_security"]["overmasking"]), float(deberta["challenge_security"]["overmasking"] + deberta["standard_security"]["overmasking"])),
        ("account-like O→BUSINESS_ID", float(bert["standard_security"]["accountlike_o_to_business_id"] + bert["challenge_security"]["accountlike_o_to_business_id"]), float(deberta["standard_security"]["accountlike_o_to_business_id"] + deberta["challenge_security"]["accountlike_o_to_business_id"])),
        ("AUTH_TOKEN vs BUSINESS_ID errors", float(bert["challenge_security"].get("auth_token_vs_business_id", 0)), float(deberta["challenge_security"].get("auth_token_vs_business_id", 0))),
        ("challenge record error", bert["history"][bert["selected"]["selected_epoch"]-1]["challenge_record_error_rate"], deberta["history"][deberta["selected"]["selected_epoch"]-1]["challenge_record_error_rate"]),
        ("worst-category error", bert["history"][bert["selected"]["selected_epoch"]-1]["challenge_maximum_category_error_rate"], deberta["history"][deberta["selected"]["selected_epoch"]-1]["challenge_maximum_category_error_rate"]),
    ]
    table = [
        "# ML-v1.3 BERT vs DeBERTa Model Comparison",
        "",
        "Selection used development data only. The sealed challenge was not opened.",
        "",
        "| Metric | BERT | DeBERTa | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name, a, b in rows:
        table.append(f"| {name} | {a:.6f} | {b:.6f} | {_delta(a, b):+.6f} |")
    table.extend(
        [
            "",
            f"- BERT train min / peak GiB / 64-lat s: {bert['selected']['training_seconds']/60:.2f} / {bert['selected']['peak_gpu_gib']:.2f} / {bert['selected']['latency_64_seconds']:.3f}",
            f"- DeBERTa train min / peak GiB / 64-lat s: {deberta['selected']['training_seconds']/60:.2f} / {deberta['selected']['peak_gpu_gib']:.2f} / {deberta['selected']['latency_64_seconds']:.3f}",
            f"- BERT model bytes: {bert['selected']['model_bytes']}; DeBERTa model bytes: {deberta['selected']['model_bytes']}",
            f"- BERT selected epoch {bert['selected']['selected_epoch']} SHA-256 `{bert['selected']['model_sha256']}`",
            f"- DeBERTa selected epoch {deberta['selected']['selected_epoch']} SHA-256 `{deberta['selected']['model_sha256']}`",
            "- Wrong-class predictions that would still be masked remain POLICY_SAFE_WRONG_CLASS, not NER successes.",
        ]
    )
    (root / "reports/ml_v1_3_model_comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    write_json(
        root / "reports/ml_v1_3_model_comparison.json",
        {
            "bert_selected": bert["selected"],
            "deberta_selected": deberta["selected"],
            "rows": [{"metric": name, "bert": a, "deberta": b, "delta": _delta(a, b)} for name, a, b in rows],
        },
    )

    sec_lines = [
        "# ML-v1.3 Security-Outcome Model Comparison",
        "",
        "| Outcome | BERT standard | BERT challenge | DeBERTa standard | DeBERTa challenge |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("security_critical_miss", "SECURITY_CRITICAL_MISS"),
        ("security_critical_partial", "SECURITY_CRITICAL_PARTIAL"),
        ("policy_safe_wrong_class", "POLICY_SAFE_WRONG_CLASS"),
        ("overmasking", "OVERMASKING"),
        ("accountlike_o_to_business_id", "account-like O→BUSINESS_ID"),
        ("ssn_to_itin", "SSN→ITIN"),
        ("auth_token_vs_business_id", "AUTH_TOKEN vs BUSINESS_ID"),
        ("api_key_related_errors", "API_KEY-related errors"),
        ("full_mask_recall", "full-mask recall"),
        ("high_risk_full_mask_recall", "high-risk full-mask recall"),
    ):
        b_std = bert["standard_security"].get(key, 0)
        b_ch = bert["challenge_security"].get(key, 0)
        d_std = deberta["standard_security"].get(key, 0)
        d_ch = deberta["challenge_security"].get(key, 0)
        sec_lines.append(
            f"| {label} | {b_std:.6f} | {b_ch:.6f} | {d_std:.6f} | {d_ch:.6f} |"
        )
    (root / "reports/ml_v1_3_security_outcome_model_comparison.md").write_text(
        "\n".join(sec_lines) + "\n", encoding="utf-8"
    )

    fail_lines = [
        "# ML-v1.3 Failure-Mode Model Comparison",
        "",
        "| Failure mode | BERT full-mask recall | DeBERTa full-mask recall | BERT miss | DeBERTa miss |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in FAILURE_MODES:
        b = bert["challenge_security"]["by_failure_mode"][mode]
        d = deberta["challenge_security"]["by_failure_mode"][mode]
        fail_lines.append(
            f"| `{mode}` | {b['full_mask_recall']:.6f} | {d['full_mask_recall']:.6f} | "
            f"{b['security_critical_miss']} | {d['security_critical_miss']} |"
        )
    fail_lines.extend(
        [
            "",
            "Categories follow the frozen ML-v1.3 challenge `failure_mode` field:",
            "account-like O vs BUSINESS_ID, AUTH_TOKEN vs BUSINESS_ID, SSN vs ITIN,",
            "IP vs technical-reference, and high-risk identifiers (including API_KEY / AUTH_TOKEN / SSN / ITIN).",
            "Exact NER metrics are preserved separately; a policy-safe wrong class is not counted as an NER success.",
        ]
    )
    (root / "reports/ml_v1_3_failure_mode_model_comparison.md").write_text(
        "\n".join(fail_lines) + "\n", encoding="utf-8"
    )

    b_crit = (
        bert["standard_security"]["security_critical_miss"]
        + bert["standard_security"]["security_critical_partial"]
        + bert["challenge_security"]["security_critical_miss"]
        + bert["challenge_security"]["security_critical_partial"]
    )
    d_crit = (
        deberta["standard_security"]["security_critical_miss"]
        + deberta["standard_security"]["security_critical_partial"]
        + deberta["challenge_security"]["security_critical_miss"]
        + deberta["challenge_security"]["security_critical_partial"]
    )
    b_hr = bert["challenge_security"]["high_risk_full_mask_recall"]
    d_hr = deberta["challenge_security"]["high_risk_full_mask_recall"]
    decision = "NO MATERIAL MODEL WINNER"
    rationale = "Security-critical and high-risk full-mask differences are below the predeclared material margins."
    if d_crit <= b_crit - 5:
        decision = "DEBERTA SELECTED FOR FINAL VALIDATION"
        rationale = f"DeBERTa has {b_crit - d_crit} fewer combined security-critical miss/partial outcomes."
    elif b_crit <= d_crit - 5:
        decision = "BERT SELECTED FOR FINAL VALIDATION"
        rationale = f"BERT has {d_crit - b_crit} fewer combined security-critical miss/partial outcomes."
    elif d_hr >= b_hr + 0.005 and d_crit <= b_crit:
        decision = "DEBERTA SELECTED FOR FINAL VALIDATION"
        rationale = "DeBERTa improves high-risk full-mask recall without more security-critical errors."
    elif b_hr >= d_hr + 0.005 and b_crit <= d_crit:
        decision = "BERT SELECTED FOR FINAL VALIDATION"
        rationale = "BERT improves high-risk full-mask recall without more security-critical errors."

    same_failures = (
        abs(bert["standard_security"]["accountlike_o_to_business_id"] - deberta["standard_security"]["accountlike_o_to_business_id"]) < 10
        and abs(bert["standard_security"]["ssn_to_itin"] - deberta["standard_security"]["ssn_to_itin"]) <= 2
    )
    decision_lines = [
        "# ML-v1.3 Model Selection Decision",
        "",
        f"**{decision}**",
        "",
        rationale,
        "",
        f"- Combined security-critical miss+partial: BERT **{b_crit}**, DeBERTa **{d_crit}**",
        f"- Challenge high-risk full-mask recall: BERT **{b_hr:.6f}**, DeBERTa **{d_hr:.6f}**",
        f"- Account-like O→BUSINESS_ID (standard+challenge): BERT **{bert['standard_security']['accountlike_o_to_business_id'] + bert['challenge_security']['accountlike_o_to_business_id']}**, DeBERTa **{deberta['standard_security']['accountlike_o_to_business_id'] + deberta['challenge_security']['accountlike_o_to_business_id']}**",
        f"- AUTH_TOKEN vs BUSINESS_ID (challenge): BERT **{bert['challenge_security'].get('auth_token_vs_business_id', 0)}**, DeBERTa **{deberta['challenge_security'].get('auth_token_vs_business_id', 0)}**",
        f"- SSN→ITIN (standard+challenge): BERT **{bert['standard_security']['ssn_to_itin'] + bert['challenge_security']['ssn_to_itin']}**, DeBERTa **{deberta['standard_security']['ssn_to_itin'] + deberta['challenge_security']['ssn_to_itin']}**",
        "",
        (
            "Both encoders reproduce substantially the same persistent failure modes; "
            "evidence points toward data/task ambiguity and/or a deterministic context resolver "
            "rather than continued encoder switching."
            if same_failures
            else "The encoders diverge on at least one persistent failure mode."
        ),
        "",
        "The sealed challenge was not opened. No ONNX export or Java change was performed.",
    ]
    (root / "reports/ml_v1_3_model_selection_decision.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )
    write_json(
        root / "reports/ml_v1_3_model_selection_decision.json",
        {
            "decision": decision,
            "bert_critical": b_crit,
            "deberta_critical": d_crit,
            "bert_high_risk_full_mask_recall": b_hr,
            "deberta_high_risk_full_mask_recall": d_hr,
            "rationale": rationale,
            "same_persistent_failures": same_failures,
        },
    )
    write_alignment_comparison(
        root,
        {"bert": bert["alignments"], "deberta": deberta["alignments"]},
    )
    print(decision, flush=True)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", choices=("bert", "deberta", "both", "compare"), default="both")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    evidence = verify_frozen_state(root)
    print("FROZEN STATE VERIFIED", evidence["dataset_manifest_sha256"], flush=True)
    if args.verify_only:
        return 0
    if args.model in {None, "both"}:
        models = ("bert", "deberta")
    elif args.model == "compare":
        compare(root)
        return 0
    else:
        models = (args.model,)
    if args.align_only:
        reports = {model_key: align_only(root, model_key)["alignments"] for model_key in models}
        write_alignment_comparison(root, reports)
        print("ALIGNMENT PREFLIGHT PASSED", flush=True)
        return 0
    if not args.smoke_test and os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("PYTHONHASHSEED must be 42 for canonical training")
    for model_key in models:
        train_one(root, model_key, args.smoke_test, args.max_optimizer_steps)
    if args.smoke_test:
        smoke_root = root / "output_securelogx/ml-v1.3-smoke"
        if smoke_root.exists():
            shutil.rmtree(smoke_root)
        print("SMOKE TESTS PASSED; artifacts removed", flush=True)
        return 0
    if (root / "output_securelogx/ml-v1.3/bert-base-cased/comparison_result.json").is_file() and (
        root / "output_securelogx/ml-v1.3/deberta-v3-base/comparison_result.json"
    ).is_file():
        compare(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
