"""Train the frozen SecureLogX ML-v1 bert-base-cased reference model.

This command consumes train/dev only.  The held-out test file is intentionally
not accepted as an argument, keeping checkpoint selection isolated from test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from securelogx_training_common import (
    AlignedWindowDataset,
    HIGH_RISK_ENTITIES,
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


ALLOWED_EXPLAINED_ALIGNMENT_FAILURES = {
    "tokenizer_emitted_no_token_for_span",
    "token_boundary_adjustment_exceeds_three_characters",
}

CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_GATE_PATH = "configs/ml_v1_onnx_validation_gate.json"
CANONICAL_FULL_RUN_PATHS = {
    "manifest": "configs/ml_v1_frozen_manifest.json",
    "labels": "configs/securelogx_labels.json",
    "train": "data/split/train.jsonl",
    "dev": "data/split/dev.jsonl",
    "output": "output_securelogx/ml-v1/bert-base-cased",
    "training_report": "reports/training_run_summary.md",
    "dev_metrics": "reports/dev_metrics.json",
}
CANONICAL_FULL_RUN_SETTINGS = {
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


def _assert_canonical_full_run(args: argparse.Namespace, root: Path) -> None:
    """Fail closed when a non-smoke invocation departs from ML-v1."""

    problems: list[str] = []
    if root != CANONICAL_REPOSITORY_ROOT:
        problems.append(
            f"root={root} (expected {CANONICAL_REPOSITORY_ROOT})"
        )
    for argument, relative_path in CANONICAL_FULL_RUN_PATHS.items():
        actual_path = (root / str(getattr(args, argument))).resolve()
        expected_path = (root / relative_path).resolve()
        if actual_path != expected_path:
            problems.append(
                f"{argument}={actual_path} (expected {expected_path})"
            )
    for argument, expected in CANONICAL_FULL_RUN_SETTINGS.items():
        actual = getattr(args, argument)
        if actual != expected:
            problems.append(f"{argument}={actual!r} (expected {expected!r})")
    for argument in (
        "max_train_records",
        "max_dev_records",
        "max_optimizer_steps",
    ):
        if getattr(args, argument) is not None:
            problems.append(f"{argument} is permitted only with --smoke-test")
    if os.environ.get("PYTHONHASHSEED") != "42":
        problems.append(
            "PYTHONHASHSEED must be exported as 42 before starting Python"
        )
    if problems:
        raise ValueError(
            "Non-smoke training must exactly match the protected ML-v1 reference "
            "configuration:\n- " + "\n- ".join(problems)
        )


def _assert_alignment_is_trainable(summary: Mapping[str, Any], split: str) -> None:
    problems: list[str] = []
    if summary["truncated_spans"]:
        problems.append(f"{summary['truncated_spans']} truncated spans")
    for diagnostic in summary["diagnostics"]:
        if (
            diagnostic["status"] == "failed"
            and diagnostic["reason"] not in ALLOWED_EXPLAINED_ALIGNMENT_FAILURES
        ):
            problems.append(
                f"record {diagnostic['record_id']}: {diagnostic['reason']}"
            )
    if problems:
        raise ValueError(
            f"{split} has unexplained alignment failures:\n- "
            + "\n- ".join(problems[:100])
        )


def _optimizer(model: Any, learning_rate: float, weight_decay: float) -> AdamW:
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or "LayerNorm.weight" in name:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    return AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def _serializable_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"errors", "confusion"}
    }


def _save_checkpoint(
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


def _write_training_summary(
    path: Path,
    arguments: Mapping[str, Any],
    environment: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    best_epoch: int,
    best_checkpoint: Path,
    elapsed_seconds: float,
    train_alignment: Mapping[str, Any],
    dev_alignment: Mapping[str, Any],
    smoke_test: bool,
) -> None:
    status = "SMOKE TEST PASSED" if smoke_test else "TRAINING COMPLETE"
    lines = [
        "# SecureLogX ML-v1 Training Run Summary",
        "",
        f"**Status: {status}**",
        "",
        "## Architecture and configuration",
        "",
        f"- Base model: `{arguments['base_model']}`",
        f"- Base revision: `{arguments['revision']}`",
        "- Task: BERT token classification with the frozen 51-label BIO head",
        f"- Maximum window length / stride: {arguments['max_length']} / {arguments['stride']}",
        f"- Epochs: {arguments['epochs']}",
        f"- Learning rate: {arguments['learning_rate']}",
        f"- Weight decay: {arguments['weight_decay']}",
        f"- Warmup ratio: {arguments['warmup_ratio']}",
        f"- Train batch / gradient accumulation / effective batch: {arguments['train_batch_size']} / {arguments['gradient_accumulation_steps']} / {arguments['effective_batch_size']}",
        f"- Eval batch: {arguments['eval_batch_size']}",
        f"- Gradient clipping: {arguments['max_grad_norm']}",
        f"- Mixed precision: {'BF16' if arguments['use_bf16'] else 'disabled'}",
        f"- Seed: {arguments['seed']}",
        f"- Frozen manifest SHA-256: `{arguments['frozen_manifest_sha256']}`",
        f"- Predeclared validation-gate SHA-256: `{arguments['validation_gate_sha256']}`",
        "- Checkpoint selection metric: dev exact-span macro F1",
        "",
        "## Alignment used for this run",
        "",
        f"- Train: {train_alignment['successfully_aligned_spans']}/{train_alignment['total_spans']} aligned; {train_alignment['boundary_adjusted_spans']} boundary-adjusted; {train_alignment['truncated_spans']} truncated; {train_alignment['failed_alignments']} explicitly unrepresentable.",
        f"- Dev: {dev_alignment['successfully_aligned_spans']}/{dev_alignment['total_spans']} aligned; {dev_alignment['boundary_adjusted_spans']} boundary-adjusted; {dev_alignment['truncated_spans']} truncated; {dev_alignment['failed_alignments']} explicitly unrepresentable.",
        "",
        "## Per-epoch dev selection",
        "",
        "| Epoch | Train loss | Dev loss | Dev micro F1 | Dev macro F1 | High-risk recall | Optimizer steps |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['dev_loss']:.6f} | "
            f"{row['dev_micro_f1']:.6f} | {row['dev_macro_f1']:.6f} | "
            f"{row['high_risk_recall']:.6f} | {row['optimizer_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Selected epoch: **{best_epoch}**",
            f"- Best checkpoint: `{best_checkpoint.as_posix()}`",
            f"- Elapsed training/evaluation time: {elapsed_seconds / 60:.2f} minutes",
            "- The held-out test set was not used for model inference, metrics, training decisions, or checkpoint selection; it was read only by the frozen-manifest integrity check.",
            "",
            "## Environment",
            "",
            f"- Python: `{environment['python']}`",
            f"- PyTorch: `{environment['torch']}`",
            f"- Transformers: `{environment['transformers']}`",
            f"- Datasets: `{environment['datasets']}`",
            f"- CUDA available/runtime: `{environment['cuda_available']}` / `{environment['torch_cuda_version']}`",
            f"- GPU: `{environment['gpu_name']}`",
        f"- CPU: `{environment['processor']}` ({environment['physical_cpu_cores']} physical / {environment['logical_cpu_cores']} logical cores)",
        f"- PYTHONHASHSEED: `{environment['python_hash_seed']}`",
        f"- CUBLAS workspace configuration: `{environment['cublas_workspace_config']}`",
        "",
            "The frozen manifest was validated immediately before and after the run.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default="configs/ml_v1_frozen_manifest.json")
    parser.add_argument("--labels", default="configs/securelogx_labels.json")
    parser.add_argument("--train", default="data/split/train.jsonl")
    parser.add_argument("--dev", default="data/split/dev.jsonl")
    parser.add_argument("--base-model", default="bert-base-cased")
    parser.add_argument(
        "--revision", default="cd5ef92a9fb2f889e972770a36d4ed042daf221e"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-report", default="reports/training_run_summary.md")
    parser.add_argument("--dev-metrics", default="reports/dev_metrics.json")
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
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-dev-records", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(args.root).resolve()
    if not args.smoke_test:
        _assert_canonical_full_run(args, root)
    output_root = (root / args.output).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. Use a new versioned path."
        )
    if args.base_model != "bert-base-cased":
        raise ValueError("This phase permits only bert-base-cased")
    if args.epochs != 3 and not args.smoke_test:
        raise ValueError("The full reference run must use exactly 3 epochs")

    manifest_path = (root / args.manifest).resolve()
    validation_gate_path = (root / CANONICAL_GATE_PATH).resolve()
    validate_manifest(root, manifest_path)
    frozen_manifest_sha256 = sha256_file(manifest_path)
    validation_gate_sha256 = sha256_file(validation_gate_path)
    label_config = load_canonical_labels(root / args.labels)
    label_to_id = {str(key): int(value) for key, value in label_config["label_to_id"].items()}
    id_to_label = {int(key): str(value) for key, value in label_config["id_to_label"].items()}
    set_reproducible_seed(args.seed)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; refusing the full BERT run without --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    environment = collect_environment(args.seed, args.base_model, args.revision)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Training requires a fast tokenizer")
    train_records = load_jsonl(root / args.train, args.max_train_records)
    dev_records = load_jsonl(root / args.dev, args.max_dev_records)
    print(
        f"Building aligned windows for {len(train_records)} train and {len(dev_records)} dev records...",
        flush=True,
    )
    train_build = build_aligned_features(
        train_records,
        tokenizer,
        label_to_id,
        args.max_length,
        args.stride,
    )
    dev_build = build_aligned_features(
        dev_records,
        tokenizer,
        label_to_id,
        args.max_length,
        args.stride,
    )
    _assert_alignment_is_trainable(train_build.summary, "train")
    _assert_alignment_is_trainable(dev_build.summary, "dev")
    print(
        f"Windows: train={len(train_build.windows)}, dev={len(dev_build.windows)}; "
        f"alignment failures: train={train_build.summary['failed_alignments']}, "
        f"dev={dev_build.summary['failed_alignments']}",
        flush=True,
    )

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
        num_labels=len(label_config["bio_labels"]),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    if model.config.id2label != id_to_label or model.config.label2id != label_to_id:
        raise ValueError("Model label mappings do not match the canonical configuration")
    model.to(device)
    optimizer = _optimizer(model, args.learning_rate, args.weight_decay)
    batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    planned_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    if args.max_optimizer_steps is not None:
        planned_optimizer_steps = min(planned_optimizer_steps, args.max_optimizer_steps)
    warmup_steps = int(round(planned_optimizer_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_optimizer_steps,
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
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.train_batch_size
        * args.gradient_accumulation_steps,
        "seed": args.seed,
        "use_bf16": use_bf16,
        "device": str(device),
        "planned_optimizer_steps": planned_optimizer_steps,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_windows": len(train_build.windows),
        "dev_windows": len(dev_build.windows),
        "smoke_test": args.smoke_test,
        "frozen_manifest_path": manifest_path.relative_to(root).as_posix(),
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "validation_gate_path": validation_gate_path.relative_to(root).as_posix(),
        "validation_gate_sha256": validation_gate_sha256,
    }

    checkpoint_root = output_root / "checkpoints"
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_epoch_path: Path | None = None
    best_dev_metrics: dict[str, Any] | None = None
    global_optimizer_steps = 0
    run_start = time.perf_counter()
    stop_training = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_numerator = 0.0
        epoch_loss_denominator = 0
        epoch_optimizer_steps = 0
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            group_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps,
                batches_per_epoch - group_start,
            )
            inputs = model_inputs_from_batch(batch, device)
            valid_tokens = int((inputs["labels"] != IGNORE_INDEX).sum().item())
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(**inputs)
                loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}: {loss.item()}"
                )
            epoch_loss_numerator += float(loss.detach().float().item()) * valid_tokens
            epoch_loss_denominator += valid_tokens
            (loss / group_size).backward()

            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == batches_per_epoch
            )
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_optimizer_steps += 1
            epoch_optimizer_steps += 1
            if global_optimizer_steps % 100 == 0 or global_optimizer_steps == 1:
                elapsed = time.perf_counter() - run_start
                rate = global_optimizer_steps / elapsed if elapsed else 0.0
                remaining = (
                    (planned_optimizer_steps - global_optimizer_steps) / rate
                    if rate
                    else 0.0
                )
                memory = (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                )
                print(
                    f"epoch={epoch} optimizer_step={global_optimizer_steps}/"
                    f"{planned_optimizer_steps} loss={loss.item():.6f} "
                    f"gpu_peak_gib={memory:.2f} eta_min={remaining / 60:.1f}",
                    flush=True,
                )
            if (
                args.max_optimizer_steps is not None
                and global_optimizer_steps >= args.max_optimizer_steps
            ):
                stop_training = True
                break

        train_loss = (
            epoch_loss_numerator / epoch_loss_denominator
            if epoch_loss_denominator
            else 0.0
        )
        print(f"Evaluating dev after epoch {epoch}...", flush=True)
        dev_evaluation = evaluate_model(
            model,
            tokenizer,
            dev_records,
            dev_build.windows,
            id_to_label,
            args.eval_batch_size,
            device,
            use_bf16,
        )
        dev_metrics = score_predictions(
            dev_records,
            dev_evaluation["predictions"],
            label_config["entities"],
            "dev",
            token_details=dev_evaluation["token_details"],
            label_to_id=label_to_id,
        )
        dev_metrics["loss"] = dev_evaluation["loss"]
        dev_metrics["invalid_bio_transitions"] = dev_evaluation[
            "invalid_bio_transitions"
        ]
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_evaluation["loss"],
            "dev_micro_f1": dev_metrics["micro"]["f1"],
            "dev_macro_f1": dev_metrics["macro"]["f1"],
            "high_risk_recall": dev_metrics["high_risk"]["recall"],
            "optimizer_steps": epoch_optimizer_steps,
            "global_optimizer_steps": global_optimizer_steps,
            "epoch_minutes": (time.perf_counter() - epoch_start) / 60,
        }
        history.append(history_row)
        epoch_path = checkpoint_root / f"epoch-{epoch}"
        _save_checkpoint(
            epoch_path,
            model,
            tokenizer,
            {
                "history": history_row,
                "dev_metrics": _serializable_metrics(dev_metrics),
            },
            training_arguments,
            environment,
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} dev_loss={dev_evaluation['loss']:.6f} "
            f"dev_micro_f1={dev_metrics['micro']['f1']:.6f} "
            f"dev_macro_f1={dev_metrics['macro']['f1']:.6f} "
            f"high_risk_recall={dev_metrics['high_risk']['recall']:.6f}",
            flush=True,
        )
        if dev_metrics["macro"]["f1"] > best_macro_f1:
            best_macro_f1 = float(dev_metrics["macro"]["f1"])
            best_epoch = epoch
            best_epoch_path = epoch_path
            best_dev_metrics = _serializable_metrics(dev_metrics)
        del dev_evaluation, dev_metrics
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if stop_training:
            break

    if best_epoch_path is None or best_dev_metrics is None:
        raise RuntimeError("Training completed without a selectable dev checkpoint")
    best_checkpoint = output_root / "best-checkpoint"
    shutil.copytree(best_epoch_path, best_checkpoint)
    write_json(best_checkpoint / "selected_checkpoint.json", {
        "selected_epoch": best_epoch,
        "selection_metric": "dev exact-span macro F1",
        "selection_value": best_macro_f1,
        "source_checkpoint": best_epoch_path.relative_to(root).as_posix(),
    })
    write_json(output_root / "training_history.json", history)
    write_json(output_root / "alignment_summary.json", {
        "train": train_build.summary,
        "dev": dev_build.summary,
    })
    write_json(root / args.dev_metrics, best_dev_metrics)
    elapsed_seconds = time.perf_counter() - run_start
    _write_training_summary(
        root / args.training_report,
        training_arguments,
        environment,
        history,
        best_epoch,
        best_checkpoint.relative_to(root),
        elapsed_seconds,
        train_build.summary,
        dev_build.summary,
        args.smoke_test,
    )
    validate_manifest(root, manifest_path)
    if sha256_file(manifest_path) != frozen_manifest_sha256:
        raise ValueError("Frozen manifest changed during training")
    if sha256_file(validation_gate_path) != validation_gate_sha256:
        raise ValueError("Predeclared validation gate changed during training")
    saved_config = json.loads((best_checkpoint / "config.json").read_text(encoding="utf-8"))
    saved_id2label = {int(key): value for key, value in saved_config["id2label"].items()}
    if saved_id2label != id_to_label or saved_config["label2id"] != label_to_id:
        raise ValueError("Saved checkpoint label mappings do not match canonical labels")
    print(f"Selected epoch {best_epoch}: {best_checkpoint}", flush=True)
    print("Frozen manifest revalidated after training", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
