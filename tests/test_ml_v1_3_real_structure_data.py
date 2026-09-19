from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_SCRIPTS = ROOT / "scripts" / "data"
if str(DATA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPTS))

import generate_securelogx_ml_v1_3_real_structure as generator  # noqa: E402
import validate_securelogx_ml_v1_3 as validator  # noqa: E402


SEALED_EXPECTED = (
    "6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MLv13RealStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated = generator.generate_all(seed=42)
        cls.labels = json.loads(
            (ROOT / "configs/securelogx_labels.json").read_text(encoding="utf-8")
        )
        cls.parent_freeze = json.loads(
            (ROOT / "configs/ml_v1_3_parent_freeze.json").read_text(encoding="utf-8")
        )
        cls.validation = validator.validate_dataset(
            root=ROOT,
            train_records=cls.generated["train_addition"],
            dev_records=cls.generated["dev_challenge"],
            parent_freeze=cls.parent_freeze,
        )

    def test_parent_hash_protection(self) -> None:
        mismatches = []
        for group in self.parent_freeze["groups"].values():
            for relative, expected in group["files"].items():
                actual = _sha256_file(ROOT / relative)
                if actual != expected["sha256"]:
                    mismatches.append(relative)
        self.assertEqual(mismatches, [])

    def test_sealed_hash_protection(self) -> None:
        self.assertEqual(
            self.parent_freeze["sealed_challenge"]["expected_sha256"],
            SEALED_EXPECTED,
        )
        self.assertEqual(
            self.validation["sealed_challenge"]["sha256"], SEALED_EXPECTED
        )
        self.assertFalse(self.validation["sealed_challenge"]["file_opened"])

    def test_ontology_immutability(self) -> None:
        self.assertEqual(len(self.labels["entities"]), 25)
        self.assertEqual(len(self.labels["bio_labels"]), 51)
        self.assertEqual(
            _sha256_file(ROOT / "configs/securelogx_labels.json"),
            self.parent_freeze["ontology"]["sha256"],
        )

    def test_span_validity_and_provenance(self) -> None:
        self.assertTrue(self.validation["gates"]["span_validity"]["passed"])
        for record in self.generated["train_addition"][:20]:
            self.assertEqual(record["meta"]["source"], generator.SOURCE)
            self.assertTrue(record["meta"]["synthetic_value_injection"])
            self.assertTrue(record["meta"]["license_reviewed"])
            self.assertFalse(record["meta"]["raw_source_text_stored"])
            self.assertIn("structure_family", record["meta"])
            self.assertIn("public_source_identifier", record["meta"])

    def test_structure_family_separation_and_contamination(self) -> None:
        self.assertTrue(
            self.validation["gates"]["cross_split_structure_family_overlap"]["passed"]
        )
        self.assertTrue(
            self.validation["gates"]["cross_split_exact_text_leakage"]["passed"]
        )
        self.assertTrue(
            self.validation["gates"]["parent_test_exact_text_contamination"]["passed"]
        )
        self.assertTrue(
            self.validation["gates"]["v1_1_dev_challenge_exact_text_contamination"]["passed"]
        )
        self.assertTrue(
            self.validation["gates"]["v1_2_dev_challenge_exact_text_contamination"]["passed"]
        )
        self.assertTrue(
            self.validation["gates"]["sealed_exact_text_contamination"]["passed"]
        )

    def test_accountlike_o_examples(self) -> None:
        self.assertTrue(self.validation["gates"]["accountlike_o_examples"]["passed"])
        found = False
        for record in self.generated["train_addition"]:
            if record["meta"]["context_role"] != "technical_account":
                continue
            if any(target["expected_label"] == "O" for target in record["meta"]["contrast_targets"]):
                found = True
                self.assertTrue(
                    any(label == "BUSINESS_ID" for _, _, label in record["entities"])
                )
                break
        self.assertTrue(found)

    def test_auth_token_business_id_multi_entity(self) -> None:
        self.assertTrue(
            self.validation["gates"]["auth_token_business_id_multi_entity"]["passed"]
        )

    def test_ssn_itin_examples(self) -> None:
        self.assertTrue(self.validation["gates"]["ssn_itin_examples"]["passed"])

    def test_validation_passes(self) -> None:
        self.assertTrue(self.validation["passed"], msg=str(
            [name for name, gate in self.validation["gates"].items() if not gate.get("passed")]
        ))


if __name__ == "__main__":
    unittest.main()
