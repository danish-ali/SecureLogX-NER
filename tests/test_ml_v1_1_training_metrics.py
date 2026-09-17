from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = ROOT / "scripts" / "train"
if str(TRAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPTS))

from securelogx_ml_v1_1_metrics import (  # noqa: E402
    context_diagnostics,
    evaluate_named_gates,
    harmonic_mean,
    supported_macro,
    tag_context_errors,
)
from securelogx_training_common import EXPECTED_ENTITIES, score_predictions  # noqa: E402


LABELS = list(EXPECTED_ENTITIES)


def predicted_span(
    start: int, end: int, label: str, confidence: float = 0.9
) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "label": label,
        "confidence": confidence,
        "minimum_token_confidence": confidence,
    }


def context_record(
    *,
    record_id: str,
    gold_label: str,
    category: str,
    role: str,
    morphology: str,
    surface_style: str = "natural_language",
    data_format: str = "text",
    subtype: str = "",
) -> dict[str, Any]:
    target: dict[str, Any] = {
        "start": 4,
        "end": 13,
        "expected_label": gold_label,
    }
    if subtype:
        target["business_id_subtype"] = subtype
    entities = [] if gold_label == "O" else [[4, 13, gold_label]]
    provenance = []
    if entities:
        provenance_entry: dict[str, Any] = {
            "start": 4,
            "end": 13,
            "label": gold_label,
        }
        if subtype:
            provenance_entry["business_id_subtype"] = subtype
        provenance.append(provenance_entry)
    return {
        "text": "key 123456789 tail",
        "entities": entities,
        "meta": {
            "source": "unit_test",
            "source_record_id": record_id,
            "template_family": f"family-{record_id}",
            "contrast_category": category,
            "context_role": role,
            "morphology_class": morphology,
            "surface_style": surface_style,
            "format": data_format,
            "contrast_targets": [target],
            "entity_provenance": provenance,
        },
    }


class ScalarMetricTests(unittest.TestCase):
    def test_supported_macro_excludes_zero_support(self) -> None:
        result = supported_macro(
            {
                "per_label": {
                    "A": {
                        "support": 2,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1": 2 / 3,
                    },
                    "B": {
                        "support": 1,
                        "precision": 0.5,
                        "recall": 1.0,
                        "f1": 2 / 3,
                    },
                    "UNSUPPORTED": {
                        "support": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                    },
                }
            }
        )

        self.assertEqual(result["precision"], 0.75)
        self.assertEqual(result["recall"], 0.75)
        self.assertTrue(math.isclose(result["f1"], 2 / 3))
        self.assertEqual(
            supported_macro({"per_label": {}}),
            {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        )

    def test_harmonic_mean_and_invalid_inputs(self) -> None:
        self.assertTrue(math.isclose(harmonic_mean(0.8, 0.5), 8 / 13))
        self.assertEqual(harmonic_mean(0.0, 0.9), 0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            harmonic_mean(-0.1, 0.9)
        with self.assertRaisesRegex(ValueError, "finite"):
            harmonic_mean(float("nan"), 0.9)


class ContextDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            context_record(
                record_id="business-as-ssn",
                gold_label="BUSINESS_ID",
                category="business_id_vs_ssn",
                role="business_identifier",
                morphology="ssn_compact",
                surface_style="abbreviated_key",
                data_format="key_value",
                subtype="ORDER_ID",
            ),
            context_record(
                record_id="true-ssn",
                gold_label="SSN",
                category="business_id_vs_ssn",
                role="social_security_number",
                morphology="ssn_compact",
            ),
            context_record(
                record_id="technical-ip-shape",
                gold_label="O",
                category="ip_address_vs_technical_reference",
                role="technical_reference",
                morphology="ipv4_dotted_quad",
                surface_style="json_flat",
                data_format="json",
            ),
            context_record(
                record_id="bank-as-business",
                gold_label="BANK_ACCOUNT_NUMBER",
                category="business_id_vs_bank_account",
                role="bank_account_number",
                morphology="account_12_digits",
            ),
            context_record(
                record_id="correct-card-business-plus-spurious",
                gold_label="BUSINESS_ID",
                category="business_id_vs_credit_card",
                role="business_identifier",
                morphology="card_16_luhn",
                subtype="TICKET_ID",
            ),
        ]
        self.predictions = [
            [predicted_span(4, 13, "SSN")],
            [predicted_span(4, 13, "SSN")],
            [predicted_span(4, 13, "IP_ADDRESS")],
            [predicted_span(4, 13, "BUSINESS_ID")],
            [
                predicted_span(4, 13, "BUSINESS_ID"),
                predicted_span(14, 18, "EMAIL"),
            ],
        ]

    def test_context_diagnostics_separate_record_and_target_errors(self) -> None:
        result = context_diagnostics(
            self.records, self.predictions, LABELS, "dev_challenge"
        )

        self.assertEqual(result["records"], 5)
        self.assertEqual(result["records_with_errors"], 4)
        self.assertEqual(result["record_error_rate"], 0.8)
        self.assertEqual(result["context_targets"], 5)
        self.assertEqual(result["context_target_errors"], 3)
        self.assertEqual(result["context_target_error_rate"], 0.6)
        self.assertEqual(result["morphology_over_context_errors"], 2)
        self.assertEqual(result["morphology_over_context_error_rate"], 0.4)
        self.assertEqual(result["maximum_category_error_rate"], 1.0)

        ssn_category = result["by_category"]["business_id_vs_ssn"]
        self.assertEqual(ssn_category["records"], 2)
        self.assertEqual(ssn_category["records_with_errors"], 1)
        self.assertEqual(ssn_category["record_error_rate"], 0.5)
        self.assertEqual(ssn_category["context_target_error_rate"], 0.5)
        self.assertIn("supported_macro", ssn_category)
        self.assertEqual(result["score"]["micro"]["true_positives"], 2)

        card_category = result["by_category"]["business_id_vs_credit_card"]
        self.assertEqual(card_category["record_error_rate"], 1.0)
        self.assertEqual(card_category["context_target_error_rate"], 0.0)

    def test_context_diagnostics_reject_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            context_diagnostics(self.records, self.predictions[:-1], LABELS, "dev")

    def test_context_error_tags_are_directional_and_non_mutating(self) -> None:
        scored = score_predictions(
            self.records, self.predictions, LABELS, "dev_challenge"
        )
        original_rows = [dict(row) for row in scored["errors"]]
        tagged = tag_context_errors(scored["errors"], self.records)

        self.assertEqual(scored["errors"], original_rows)
        by_record = {row["record_id"]: row for row in tagged}
        business_tags = by_record["business-as-ssn"]["context_error_tags"]
        self.assertIn("morphology-over-context", business_tags)
        self.assertIn("ambiguous-key", business_tags)
        self.assertIn("unseen-business-subtype", business_tags)
        self.assertIn("structured-field-confusion", business_tags)
        self.assertEqual(
            by_record["business-as-ssn"]["business_id_subtype"], "ORDER_ID"
        )

        technical_tags = by_record["technical-ip-shape"]["context_error_tags"]
        self.assertIn("morphology-over-context", technical_tags)
        self.assertIn("structured-field-confusion", technical_tags)

        bank_tags = by_record["bank-as-business"]["context_error_tags"]
        self.assertIn("context-over-morphology", bank_tags)
        self.assertNotIn("morphology-over-context", bank_tags)


class GateComparisonTests(unittest.TestCase):
    def test_named_minimum_maximum_and_explicit_gates(self) -> None:
        result = evaluate_named_gates(
            {
                "macro_f1": 0.91,
                "record_error_rate": 0.08,
                "exact_revision": True,
            },
            {
                "macro_f1_minimum": 0.9,
                "record_error_rate_maximum": 0.05,
                "revision_gate": {
                    "metric": "exact_revision",
                    "operator": "==",
                    "threshold": True,
                },
            },
        )

        self.assertFalse(result["all_passed"])
        self.assertTrue(result["gates"]["macro_f1_minimum"]["passed"])
        self.assertFalse(
            result["gates"]["record_error_rate_maximum"]["passed"]
        )
        self.assertTrue(result["gates"]["revision_gate"]["passed"])
        self.assertEqual(
            result["gates"]["record_error_rate_maximum"]["operator"], "<="
        )

    def test_existing_prefix_minimum_name_is_supported(self) -> None:
        result = evaluate_named_gates(
            {"minimum_individual_label_recall": 0.72},
            {"minimum_individual_label_recall": 0.7},
        )
        self.assertTrue(result["all_passed"])

    def test_ambiguous_numeric_gate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "declare minimum/maximum"):
            evaluate_named_gates({"metric": 0.9}, {"metric": 0.8})


if __name__ == "__main__":
    unittest.main()
