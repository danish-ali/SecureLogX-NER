from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = ROOT / "scripts" / "train"
if str(TRAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPTS))

import evaluate_securelogx_model as evaluator


def canonical_args(**overrides: object) -> argparse.Namespace:
    values = {
        "root": str(ROOT),
        "manifest": "configs/ml_v1_frozen_manifest.json",
        "labels": "configs/securelogx_labels.json",
        "gates": "configs/ml_v1_onnx_validation_gate.json",
        "model": "output_securelogx/ml-v1/bert-base-cased/best-checkpoint",
        "input": "data/split/test.jsonl",
        "split_name": "test",
        "reports_dir": "reports",
        "max_length": 384,
        "stride": 128,
        "batch_size": 8,
        "seed": 42,
        "max_records": None,
        "smoke_test": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class EvaluatorGuardTests(unittest.TestCase):
    def test_final_contract_accepts_only_canonical_configuration(self) -> None:
        paths = evaluator._validate_mode_contract(canonical_args(), ROOT)
        self.assertEqual(paths["input"], (ROOT / "data/split/test.jsonl").resolve())

    def test_final_contract_rejects_partial_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_records must be omitted"):
            evaluator._validate_mode_contract(canonical_args(max_records=10), ROOT)

    def test_final_contract_rejects_alternate_checkpoint_and_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "model must be"):
            evaluator._validate_mode_contract(
                canonical_args(model="alternate", seed=7), ROOT
            )

    def test_smoke_cannot_target_test_or_canonical_reports(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen test"):
            evaluator._validate_mode_contract(
                canonical_args(smoke_test=True, reports_dir=".tmp/reports"), ROOT
            )
        with self.assertRaisesRegex(ValueError, "canonical final reports"):
            evaluator._validate_mode_contract(
                canonical_args(
                    smoke_test=True,
                    input="data/split/dev.jsonl",
                    split_name="dev",
                ),
                ROOT,
            )

    def test_receipt_creation_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            evaluator._write_exclusive_json(receipt, {"status": "STARTED"})
            with self.assertRaises(FileExistsError):
                evaluator._write_exclusive_json(receipt, {"status": "STARTED"})
            self.assertEqual(json.loads(receipt.read_text())["status"], "STARTED")

    def test_unexplained_alignment_gate_uses_actual_diagnostics(self) -> None:
        alignment = {
            "diagnostics": [
                {"status": "failed", "reason": "tokenizer_emitted_no_token_for_span"},
                {"status": "failed", "reason": "new_unexplained_reason"},
            ],
            "truncated_spans": 0,
        }
        self.assertEqual(len(evaluator._unexplained_alignment_failures(alignment)), 1)


class EvaluatorReportTests(unittest.TestCase):
    def test_business_subtype_prefers_normalized_subtype(self) -> None:
        record = {
            "text": "customer-1",
            "entities": [[0, 10, "BUSINESS_ID"]],
            "meta": {
                "entity_provenance": [
                    {
                        "start": 0,
                        "end": 10,
                        "label": "BUSINESS_ID",
                        "source_label": "customer_id",
                        "source_subtype": "CUSTOMER_ID",
                    }
                ]
            },
        }
        gold = {"start": 0, "end": 10, "label": "BUSINESS_ID"}
        self.assertEqual(evaluator._business_subtype(record, gold), "CUSTOMER_ID")

    def test_business_subtype_report_does_not_invent_subtype_precision(self) -> None:
        record = {
            "text": "customer-1",
            "entities": [[0, 10, "BUSINESS_ID"]],
            "meta": {
                "entity_provenance": [
                    {
                        "start": 0,
                        "end": 10,
                        "label": "BUSINESS_ID",
                        "source_subtype": "CUSTOMER_ID",
                    }
                ]
            },
        }
        prediction = [[{"start": 0, "end": 10, "label": "BUSINESS_ID", "confidence": 0.9}]]
        with tempfile.TemporaryDirectory() as directory:
            result = evaluator._business_id_analysis(
                Path(directory) / "business.md", [record], prediction
            )
        row = result["subtypes"][0]
        self.assertEqual(row["exact_span_recall"], 1.0)
        self.assertNotIn("context_precision", row)
        self.assertNotIn("f1", row)


if __name__ == "__main__":
    unittest.main()
