from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = ROOT / "scripts" / "train"
if str(TRAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPTS))

from train_securelogx_ml_v1_2 import (  # noqa: E402
    _accountlike_o_business_id,
    _business_subtype_recall,
    _direction,
    _o_target_direction,
)


def prediction(start: int, end: int, label: str) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "label": label,
        "confidence": 0.9,
        "minimum_token_confidence": 0.9,
    }


class FrozenGateTests(unittest.TestCase):
    def test_gate_hash_and_controlled_configuration(self) -> None:
        gate_path = ROOT / "configs/ml_v1_2_training_gate.json"
        detached = (
            ROOT / "configs/ml_v1_2_training_gate.sha256"
        ).read_text(encoding="utf-8").split()
        self.assertEqual(detached[1], gate_path.name)
        self.assertEqual(
            detached[0], hashlib.sha256(gate_path.read_bytes()).hexdigest()
        )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        self.assertTrue(gate["declared_before_training"])
        self.assertEqual(gate["training_composition"]["combined_training_records"], 28158)
        self.assertEqual(gate["training_configuration"]["effective_batch_size"], 16)
        self.assertEqual(gate["training_configuration"]["revision"], "cd5ef92a9fb2f889e972770a36d4ed042daf221e")
        self.assertTrue(gate["stop_boundary"]["sealed_challenge_must_not_be_opened"])
        required = gate["development_gate"]["all_required"]
        self.assertEqual(required["challenge_dev_micro_f1_minimum"], 0.9)
        self.assertEqual(required["challenge_dev_context_target_error_rate_maximum"], 0.1)
        self.assertEqual(required["challenge_dev_maximum_category_error_rate_maximum"], 0.2)


class TargetedMetricTests(unittest.TestCase):
    def test_directional_confusions_use_overlap(self) -> None:
        records = [
            {
                "text": "card=4111111111111111",
                "entities": [[5, 21, "CREDIT_CARD_NUMBER"]],
                "meta": {"contrast_category": "business_id_vs_credit_card"},
            }
        ]
        result = _direction(
            records,
            [[prediction(5, 21, "BUSINESS_ID")]],
            "CREDIT_CARD_NUMBER",
            "BUSINESS_ID",
            "business_id_vs_credit_card",
        )
        self.assertEqual(result["gold_support"], 1)
        self.assertEqual(result["confused_spans"], 1)
        self.assertEqual(result["rate"], 1.0)

    def test_accountlike_o_false_positive_excludes_gold_overlap(self) -> None:
        records = [
            {
                "text": "account=12345",
                "entities": [],
                "meta": {"template_family": "key_value_accountlike_batch_v1"},
            },
            {
                "text": "account=12345",
                "entities": [[8, 13, "BANK_ACCOUNT_NUMBER"]],
                "meta": {"template_family": "key_value_accountlike_batch_v1"},
            },
        ]
        result = _accountlike_o_business_id(
            records,
            [
                [prediction(8, 13, "BUSINESS_ID")],
                [prediction(8, 13, "BUSINESS_ID")],
            ],
        )
        self.assertEqual(result["false_positive_spans"], 1)
        self.assertEqual(result["records_with_false_positive"], 1)

    def test_technical_reference_and_subtype_metrics(self) -> None:
        records = [
            {
                "text": "revision=1.2.3.4 id=C-1",
                "entities": [[20, 23, "BUSINESS_ID"]],
                "meta": {
                    "contrast_category": "ip_address_vs_technical_reference",
                    "context_role": "technical_reference",
                    "contrast_targets": [
                        {"start": 9, "end": 16, "expected_label": "O"}
                    ],
                    "entity_provenance": [
                        {
                            "start": 20,
                            "end": 23,
                            "label": "BUSINESS_ID",
                            "business_id_subtype": "CUSTOMER_ID",
                        }
                    ],
                },
            }
        ]
        guesses = [[prediction(9, 16, "IP_ADDRESS"), prediction(20, 23, "BUSINESS_ID")]]
        technical = _o_target_direction(
            records,
            guesses,
            "IP_ADDRESS",
            category="ip_address_vs_technical_reference",
            context_role="technical_reference",
        )
        self.assertEqual(technical["records_with_error"], 1)
        self.assertEqual(technical["rate"], 1.0)
        subtype = _business_subtype_recall(records, guesses)
        self.assertEqual(subtype["CUSTOMER_ID"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
