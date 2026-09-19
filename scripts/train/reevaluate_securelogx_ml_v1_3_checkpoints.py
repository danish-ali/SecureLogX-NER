#!/usr/bin/env python3
"""Re-evaluate existing ML-v1.3 BERT/DeBERTa checkpoints after span normalization.

Development-only corrective evaluation. This script:
- does NOT train or update model weights
- does NOT read the sealed challenge
- does NOT read the original test set
- does NOT overwrite historical ML-v1.3 reports
- applies the common decoder's whitespace-boundary normalization
- evaluates all existing epoch checkpoints and reapplies the original
  development-only checkpoint-selection and cross-model security rule
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from securelogx_training_common import (  # noqa: E402
    build_aligned_features,
    load_canonical_labels,
    load_jsonl,
    write_json,
)
from securelogx_ml_v1_1_metrics import harmonic_mean  # noqa: E402
from train_securelogx_ml_v1_1 import _compact_metrics, _score_view  # noqa: E402
from train_securelogx_ml_v1_3_comparison import (  # noqa: E402
    PATHS,
    SEALED,
    label_bundle,
    security_view,
    verify_frozen_state,
)

OUTPUT_ROOT = "reports/ml_v1_3_corrected_evaluation"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model_epochs(
    root: Path,
    model_key: str,
    evidence: Mapping[str, Any],
    standard_records: list[dict[str, Any]],
    challenge_records: list[dict[str, Any]],
) -> dict[str, Any]:
    gate = evidence["gate"]
    shared = gate["shared_training_configuration"]
    spec = gate["models"][model_key]
    labels = evidence["labels"]
    entities = [str(x) for x in labels["entities"]]
    label_to_id = {str(k): int(v) for k, v in labels["label_to_id"].items()}
    id_to_label = {int(k): str(v) for k, v in labels["id_to_label"].items()}
    device = _device()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model_root = root / spec["output"]
    if not model_root.exists():
        raise FileNotFoundError(
            f"Missing local model output: {model_root}. "
            "Run this from the local workspace containing the trained checkpoints."
        )

    epoch_paths = []
    for epoch in range(1, int(shared["epochs"]) + 1):
        path = model_root / "checkpoints" / f"epoch-{epoch}"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        epoch_paths.append((epoch, path))

    # Tokenizer is fixed within each model, so align development sets once.
    tokenizer = AutoTokenizer.from_pretrained(
        epoch_paths[0][1],
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"{model_key} requires a fast tokenizer")

    standard_build = build_aligned_features(
        standard_records,
        tokenizer,
        label_to_id,
        int(shared["max_length"]),
        int(shared["stride"]),
    )
    challenge_build = build_aligned_features(
        challenge_records,
        tokenizer,
        label_to_id,
        int(shared["max_length"]),
        int(shared["stride"]),
    )

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    tie = float(gate["checkpoint_selection"]["effective_tie_absolute"])

    for epoch, checkpoint in epoch_paths:
        print(f"Evaluating {model_key} epoch {epoch}: {checkpoint}", flush=True)
        model = AutoModelForTokenClassification.from_pretrained(
            checkpoint,
            local_files_only=True,
        ).to(device)

        standard, standard_predictions = _score_view(
            model,
            tokenizer,
            standard_records,
            standard_build.windows,
            entities,
            label_to_id,
            id_to_label,
            f"ml_v1_3_corrected_{model_key}_standard",
            int(shared["eval_batch_size"]),
            device,
            use_bf16,
            False,
        )
        challenge, challenge_predictions = _score_view(
            model,
            tokenizer,
            challenge_records,
            challenge_build.windows,
            entities,
            label_to_id,
            id_to_label,
            f"ml_v1_3_corrected_{model_key}_challenge",
            int(shared["eval_batch_size"]),
            device,
            use_bf16,
            True,
        )

        standard_security = security_view(
            "standard_dev", standard_records, standard_predictions
        )
        challenge_security = security_view(
            "challenge_dev", challenge_records, challenge_predictions
        )
        selection_score = harmonic_mean(
            standard["supported_macro"]["f1"],
            challenge["supported_macro"]["f1"],
        )

        row = {
            "epoch": epoch,
            "selection_score": selection_score,
            "standard": label_bundle(standard),
            "challenge": label_bundle(challenge),
            "standard_security": standard_security,
            "challenge_security": challenge_security,
            "challenge_record_error_rate": challenge["context_diagnostics"][
                "record_error_rate"
            ],
            "challenge_maximum_category_error_rate": challenge[
                "context_diagnostics"
            ]["maximum_category_error_rate"],
        }
        history.append(row)

        if best is None or selection_score > best["selection_score"] + tie:
            best = {
                **row,
                "checkpoint": checkpoint.relative_to(root).as_posix(),
                "standard_metrics": _compact_metrics(standard),
                "challenge_metrics": _compact_metrics(challenge),
            }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError(f"No corrected checkpoint selected for {model_key}")

    return {
        "model_key": model_key,
        "base_model": spec["base_model"],
        "revision": spec["revision"],
        "device": str(device),
        "bf16": use_bf16,
        "history": history,
        "selected": best,
        "alignment": {
            "standard": {
                "total_spans": standard_build.summary["total_spans"],
                "boundary_adjusted_spans": standard_build.summary[
                    "boundary_adjusted_spans"
                ],
                "failed_alignments": standard_build.summary["failed_alignments"],
                "truncated_spans": standard_build.summary["truncated_spans"],
            },
            "challenge": {
                "total_spans": challenge_build.summary["total_spans"],
                "boundary_adjusted_spans": challenge_build.summary[
                    "boundary_adjusted_spans"
                ],
                "failed_alignments": challenge_build.summary["failed_alignments"],
                "truncated_spans": challenge_build.summary["truncated_spans"],
            },
        },
    }


def critical_total(result: Mapping[str, Any]) -> int:
    selected = result["selected"]
    std = selected["standard_security"]
    ch = selected["challenge_security"]
    return int(
        std["security_critical_miss"]
        + std["security_critical_partial"]
        + ch["security_critical_miss"]
        + ch["security_critical_partial"]
    )


def apply_historical_cross_model_rule(
    bert: Mapping[str, Any], deberta: Mapping[str, Any]
) -> dict[str, Any]:
    b_crit = critical_total(bert)
    d_crit = critical_total(deberta)
    b_hr = float(bert["selected"]["challenge_security"]["high_risk_full_mask_recall"])
    d_hr = float(
        deberta["selected"]["challenge_security"]["high_risk_full_mask_recall"]
    )

    decision = "NO MATERIAL MODEL WINNER"
    rationale = (
        "Corrected development security differences are below the historical "
        "predeclared material margins."
    )
    if d_crit <= b_crit - 5:
        decision = "DEBERTA SELECTED BY CORRECTED DEVELOPMENT EVALUATION"
        rationale = (
            f"DeBERTa has {b_crit - d_crit} fewer corrected combined "
            "security-critical miss/partial outcomes."
        )
    elif b_crit <= d_crit - 5:
        decision = "BERT SELECTED BY CORRECTED DEVELOPMENT EVALUATION"
        rationale = (
            f"BERT has {d_crit - b_crit} fewer corrected combined "
            "security-critical miss/partial outcomes."
        )
    elif d_hr >= b_hr + 0.005 and d_crit <= b_crit:
        decision = "DEBERTA SELECTED BY CORRECTED DEVELOPMENT EVALUATION"
        rationale = (
            "DeBERTa improves corrected challenge high-risk full-mask recall "
            "without more security-critical outcomes."
        )
    elif b_hr >= d_hr + 0.005 and b_crit <= d_crit:
        decision = "BERT SELECTED BY CORRECTED DEVELOPMENT EVALUATION"
        rationale = (
            "BERT improves corrected challenge high-risk full-mask recall "
            "without more security-critical outcomes."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "bert_critical_total": b_crit,
        "deberta_critical_total": d_crit,
        "bert_challenge_high_risk_full_mask_recall": b_hr,
        "deberta_challenge_high_risk_full_mask_recall": d_hr,
        "note": (
            "This reuses the historical cross-model rule after correcting a "
            "decoder implementation defect. Historical reports remain unchanged."
        ),
    }


def write_markdown(
    root: Path,
    bert: Mapping[str, Any],
    deberta: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    b = bert["selected"]
    d = deberta["selected"]
    lines = [
        "# ML-v1.3 Corrected Development Evaluation",
        "",
        "**Development only. No sealed challenge or original-test inference. No retraining.**",
        "",
        "The common evaluator now trims tokenizer-added surrounding whitespace from "
        "decoded entity spans before entity-level scoring/runtime span output.",
        "",
        "## Selected checkpoints after corrected evaluation",
        "",
        "| Metric | BERT | DeBERTa |",
        "|---|---:|---:|",
        f"| Selected epoch | {b['epoch']} | {d['epoch']} |",
        f"| Selection score | {b['selection_score']:.6f} | {d['selection_score']:.6f} |",
        f"| Standard micro F1 | {b['standard']['micro_f1']:.6f} | {d['standard']['micro_f1']:.6f} |",
        f"| Standard macro F1 | {b['standard']['macro_f1']:.6f} | {d['standard']['macro_f1']:.6f} |",
        f"| Challenge micro F1 | {b['challenge']['micro_f1']:.6f} | {d['challenge']['micro_f1']:.6f} |",
        f"| Challenge macro F1 | {b['challenge']['macro_f1']:.6f} | {d['challenge']['macro_f1']:.6f} |",
        f"| BUSINESS_ID standard P/R/F1 | {b['standard']['business_id_p']:.3f}/{b['standard']['business_id_r']:.3f}/{b['standard']['business_id_f1']:.3f} | {d['standard']['business_id_p']:.3f}/{d['standard']['business_id_r']:.3f}/{d['standard']['business_id_f1']:.3f} |",
        f"| SSN standard F1 | {b['standard']['ssn_f1']:.6f} | {d['standard']['ssn_f1']:.6f} |",
        f"| ITIN standard F1 | {b['standard']['itin_f1']:.6f} | {d['standard']['itin_f1']:.6f} |",
        f"| AUTH_TOKEN standard F1 | {b['standard']['auth_token_f1']:.6f} | {d['standard']['auth_token_f1']:.6f} |",
        f"| API_KEY standard F1 | {b['standard']['api_key_f1']:.6f} | {d['standard']['api_key_f1']:.6f} |",
        f"| Combined security miss+partial | {critical_total(bert)} | {critical_total(deberta)} |",
        f"| Account-like O -> BUSINESS_ID (standard) | {b['standard_security']['accountlike_o_to_business_id']} | {d['standard_security']['accountlike_o_to_business_id']} |",
        "",
        f"Corrected decision: **{decision['decision']}**",
        "",
        decision["rationale"],
        "",
        "Historical ML-v1.3 reports and the original predeclared selection result are preserved.",
        "The sealed challenge remains blocked until this corrected development result is reviewed.",
    ]
    path = root / OUTPUT_ROOT / "comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    evidence = verify_frozen_state(root)
    if (
        evidence["parent_result"]["sealed_challenge"]["actual_sha256"] != SEALED
        or evidence["parent_result"]["sealed_challenge"]["file_opened"] is not False
    ):
        raise ValueError("Sealed challenge guard failed")

    standard_records = load_jsonl(root / PATHS["standard_dev"])
    challenge_records = load_jsonl(root / PATHS["challenge_dev"])

    bert = evaluate_model_epochs(
        root, "bert", evidence, standard_records, challenge_records
    )
    deberta = evaluate_model_epochs(
        root, "deberta", evidence, standard_records, challenge_records
    )
    decision = apply_historical_cross_model_rule(bert, deberta)

    output = root / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "bert.json", bert)
    write_json(output / "deberta.json", deberta)
    write_json(output / "decision.json", decision)
    write_markdown(root, bert, deberta, decision)

    print(decision["decision"])
    print(decision["rationale"])
    print(f"Wrote corrected reports under: {OUTPUT_ROOT}")
    print("Sealed challenge remains unopened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
