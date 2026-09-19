from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ComparisonGateTests(unittest.TestCase):
    def test_gate_is_predeclared_and_hashed(self) -> None:
        gate_path = ROOT / "configs/ml_v1_3_model_comparison_gate.json"
        detached = (ROOT / "configs/ml_v1_3_model_comparison_gate.sha256").read_text(
            encoding="utf-8"
        ).split()
        self.assertEqual(detached[1], gate_path.name)
        self.assertEqual(detached[0], hashlib.sha256(gate_path.read_bytes()).hexdigest())
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        self.assertTrue(gate["declared_before_training"])
        self.assertEqual(gate["training_composition"]["combined_training_records"], 30462)
        self.assertEqual(gate["models"]["bert"]["revision"], "cd5ef92a9fb2f889e972770a36d4ed042daf221e")
        self.assertEqual(gate["models"]["deberta"]["revision"], "8ccc9b6f36199bec6961081d44eb72fb3f7353f3")
        self.assertEqual(gate["shared_training_configuration"]["effective_batch_size"], 16)
        self.assertTrue(gate["stop_boundary"]["sealed_challenge_must_not_be_opened"])
        self.assertEqual(
            gate["frozen_inputs"]["ml_v1_3_dataset_manifest"]["sha256"],
            "8125063016e4e519e3b1a5db4b2ebc2cb32d5d5712a7072c769c52ed17eef9c9",
        )


if __name__ == "__main__":
    unittest.main()
