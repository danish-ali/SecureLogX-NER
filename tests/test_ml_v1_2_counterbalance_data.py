from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_SCRIPTS = ROOT / "scripts" / "data"
if str(DATA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPTS))

import generate_securelogx_ml_v1_2_counterbalance as generator  # noqa: E402
import validate_securelogx_ml_v1_2 as validator  # noqa: E402


SEALED_EXPECTED_SHA256 = (
    "6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a"
)
EXPECTED_RECORD_COUNTS = {"train_addition": 1_600, "dev_challenge": 540}
EXPECTED_FAMILY_COUNTS = {"train_addition": 40, "dev_challenge": 15}
SENSITIVE_COUNTERBALANCE_LABELS = (
    "SSN",
    "CREDIT_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "ROUTING_NUMBER",
    "ITIN",
    "TAX_ID",
    "PASSPORT_NUMBER",
    "DRIVER_LICENSE",
    "API_KEY",
    "AUTH_TOKEN",
    "IP_ADDRESS",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels_in_record(record: dict[str, object]) -> set[str]:
    return {str(label) for _, _, label in record["entities"]}  # type: ignore[misc]


class MLv12CounterbalanceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated = generator.generate_all(seed=42)
        cls.labels = json.loads(
            (ROOT / "configs" / "securelogx_labels.json").read_text(encoding="utf-8")
        )
        cls.parent_freeze = json.loads(
            (ROOT / "configs" / "ml_v1_2_parent_freeze.json").read_text(
                encoding="utf-8"
            )
        )
        cls.profile = json.loads(
            (ROOT / "configs" / "ml_v1_2_dataset_profile.json").read_text(
                encoding="utf-8"
            )
        )
        manifest_path = ROOT / "configs" / "ml_v1_2_dataset_manifest.json"
        cls.manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        )
        # One full validation of the committed dataset, shared by tests below.
        cls.full_validation = validator.validate_dataset(
            ROOT, include_template_audit=True
        )

    # ------------------------------------------------------------------
    # Parent freeze and sealed challenge protection
    # ------------------------------------------------------------------

    def test_parent_freeze_pins_sealed_hash_and_ontology(self) -> None:
        sealed = self.parent_freeze["sealed_challenge"]
        self.assertEqual(sealed["expected_sha256"], SEALED_EXPECTED_SHA256)
        self.assertEqual(sealed["sha256"], SEALED_EXPECTED_SHA256)
        self.assertEqual(
            sealed["handling"], "opaque_byte_hash_only_not_parsed_not_predicted"
        )
        ontology = self.parent_freeze["ontology"]
        self.assertEqual(ontology["entity_count"], 25)
        self.assertEqual(ontology["bio_label_count"], 51)
        self.assertEqual(
            ontology["sha256"],
            _sha256_file(ROOT / "configs" / "securelogx_labels.json"),
        )

    def test_parent_freeze_hashes_verify_for_readable_configs(self) -> None:
        for group in ("ml_v1_1_configs", "ml_v1_artifacts"):
            files = self.parent_freeze["groups"][group]["files"]
            for relative, expected in files.items():
                path = ROOT / relative
                if not relative.startswith("configs/"):
                    continue
                with self.subTest(path=relative):
                    self.assertTrue(path.is_file(), relative)
                    self.assertEqual(_sha256_file(path), expected["sha256"])
                    self.assertEqual(path.stat().st_size, expected["bytes"])

    def test_sealed_challenge_regenerates_to_frozen_byte_hash(self) -> None:
        """Sealed hash protection without ever opening the sealed file."""

        module_path = DATA_SCRIPTS / "generate_securelogx_ml_v1_1_context_contrast.py"
        import validate_securelogx_ml_v1_1 as v1_1_validator

        module = v1_1_validator._load_module(module_path, "_v1_1_gen_sealed_test")
        sealed_records = module.generate_all(seed=42)["sealed_challenge"]
        self.assertEqual(
            validator.canonical_jsonl_sha256(sealed_records), SEALED_EXPECTED_SHA256
        )
        evidence = self.full_validation["v1_1_partitions"]["v1_1_sealed_challenge"]
        self.assertEqual(evidence["sha256"], SEALED_EXPECTED_SHA256)
        self.assertFalse(evidence["file_opened"])

    def test_v1_2_outputs_do_not_touch_frozen_parent_paths(self) -> None:
        outputs = set(self.profile["outputs"].values())
        frozen = {
            "data/ml_v1_1/challenge/sealed_test.jsonl",
            "data/ml_v1_1/context_contrast/dev_challenge.jsonl",
            "data/ml_v1_1/context_contrast/train_additions.jsonl",
            "data/split/train.jsonl",
            "data/split/dev.jsonl",
            "data/split/test.jsonl",
        }
        self.assertFalse(outputs & frozen)
        for output in outputs:
            self.assertFalse(output.startswith("data/ml_v1_1/"))
            self.assertFalse(output.startswith("output_securelogx/"))

    # ------------------------------------------------------------------
    # Ontology immutability
    # ------------------------------------------------------------------

    def test_frozen_25_entity_51_bio_ontology_is_unchanged(self) -> None:
        ontology = validator.validate_ontology(
            ROOT / "configs" / "securelogx_labels.json"
        )
        self.assertTrue(ontology["passed"], ontology["errors"])
        self.assertEqual(ontology["entity_count"], 25)
        self.assertEqual(ontology["bio_label_count"], 51)
        self.assertEqual(len(self.labels["entities"]), 25)
        self.assertEqual(len(self.labels["bio_labels"]), 51)
        self.assertEqual(self.labels["bio_labels"][0], "O")
        self.assertEqual(self.labels["bio_labels"][50], "I-API_KEY")
        self.assertEqual(self.labels["label_to_id"]["I-API_KEY"], 50)
        self.assertEqual(ontology["sha256"], self.parent_freeze["ontology"]["sha256"])

    # ------------------------------------------------------------------
    # Generation determinism and structure
    # ------------------------------------------------------------------

    def test_generation_is_deterministic(self) -> None:
        again = generator.generate_all(seed=42)
        self.assertEqual(again, self.generated)

    def test_exact_record_and_family_targets(self) -> None:
        self.assertEqual(set(self.generated), set(EXPECTED_RECORD_COUNTS))
        self.assertEqual(
            self.profile["target_counts"]["train_additions_records"], 1_600
        )
        self.assertEqual(self.profile["target_counts"]["dev_challenge_records"], 540)
        for split, records in self.generated.items():
            with self.subTest(split=split):
                self.assertEqual(len(records), EXPECTED_RECORD_COUNTS[split])
                families = {record["meta"]["template_family"] for record in records}
                self.assertEqual(len(families), EXPECTED_FAMILY_COUNTS[split])
                validation = validator.validate_split_records(records, split)
                self.assertTrue(validation["passed"], validation["errors"][:5])
                self.assertEqual(
                    validation["complete_group_count"], len(records) // 2
                )

    def test_committed_files_match_generated_records_and_manifest(self) -> None:
        self.assertIsNotNone(self.manifest, "ML-v1.2 manifest missing")
        data_paths = {
            "train_addition": "data/ml_v1_2/counterbalance/train_additions.jsonl",
            "dev_challenge": "data/ml_v1_2/counterbalance/dev_challenge.jsonl",
        }
        for split, relative in data_paths.items():
            with self.subTest(split=split):
                path = ROOT / relative
                self.assertTrue(path.is_file(), relative)
                actual = _sha256_file(path)
                self.assertEqual(
                    actual,
                    validator.canonical_jsonl_sha256(self.generated[split]),
                    "committed file is not the deterministic seed-42 output",
                )
                self.assertEqual(
                    actual, self.manifest["new_data_artifacts"][relative]["sha256"]
                )
        manifest_path = ROOT / "configs" / "ml_v1_2_dataset_manifest.json"
        detached = (
            (ROOT / "configs" / "ml_v1_2_dataset_manifest.sha256")
            .read_text(encoding="utf-8")
            .split()[0]
        )
        self.assertEqual(detached, _sha256_file(manifest_path))

    # ------------------------------------------------------------------
    # Span validation and duplicate detection (negative tests)
    # ------------------------------------------------------------------

    def test_span_corruption_is_detected(self) -> None:
        records = copy.deepcopy(self.generated["dev_challenge"][:10])
        start, end, label = records[0]["entities"][0]
        records[0]["entities"][0] = [start, end + 3, label]
        result = validator.validate_split_records(records, "dev_challenge")
        self.assertFalse(result["passed"])
        self.assertTrue(
            any(
                error["code"] in {"span", "provenance", "contrast_target"}
                for error in result["errors"]
            ),
            result["errors"][:5],
        )

    def test_non_canonical_label_is_detected(self) -> None:
        records = copy.deepcopy(self.generated["train_addition"][:10])
        start, end, _ = records[0]["entities"][0]
        records[0]["entities"][0] = [start, end, "CUSTOMER_ID"]
        result = validator.validate_split_records(records, "train_addition")
        self.assertFalse(result["passed"])
        self.assertGreater(result["error_counts"].get("span", 0), 0)

    def test_duplicate_text_is_detected(self) -> None:
        records = copy.deepcopy(self.generated["train_addition"][:10])
        records.append(copy.deepcopy(records[0]))
        result = validator.validate_split_records(records, "train_addition")
        self.assertFalse(result["passed"])
        self.assertGreater(result["duplicate_text_records"], 0)
        self.assertGreater(result["duplicate_source_id_records"], 0)

    def test_value_hash_tampering_is_detected(self) -> None:
        records = copy.deepcopy(self.generated["dev_challenge"][:10])
        records[0]["meta"]["contrast_targets"][0]["value_sha256"] = "0" * 64
        result = validator.validate_split_records(records, "dev_challenge")
        self.assertFalse(result["passed"])
        self.assertGreater(result["error_counts"].get("contrast_target", 0), 0)

    def test_forbidden_parent_wording_is_detected(self) -> None:
        records = copy.deepcopy(self.generated["train_addition"][:10])
        records[0]["text"] = (
            "order reference X9 was replayed after a warehouse timeout"
        )
        result = validator.validate_split_records(records, "train_addition")
        self.assertFalse(result["passed"])
        self.assertGreater(
            result["error_counts"].get("forbidden_parent_template", 0), 0
        )

    # ------------------------------------------------------------------
    # Contamination, template independence, and full-dataset gates
    # ------------------------------------------------------------------

    def test_full_dataset_validation_passes_with_zero_leakage(self) -> None:
        result = self.full_validation
        self.assertTrue(result["passed"], result["errors"][:10])
        self.assertTrue(result["ready"])
        self.assertEqual(result["recommendation"], "READY FOR ML-v1.2 RETRAINING")
        self.assertEqual(result["error_count"], 0)
        failed_gates = {
            name: gate for name, gate in result["gates"].items() if not gate["passed"]
        }
        self.assertFalse(failed_gates, failed_gates)

        leakage = result["leakage"]
        self.assertEqual(leakage["cross_split_exact_text"]["total"], 0)
        self.assertEqual(leakage["cross_split_template_family"]["total"], 0)
        self.assertEqual(leakage["cross_split_focus_value_hash"]["total"], 0)
        self.assertEqual(leakage["parent_exact_text_count"], 0)
        self.assertEqual(leakage["parent_test_exact_text_count"], 0)
        self.assertEqual(leakage["parent_template_family_count"], 0)
        self.assertEqual(leakage["v1_1_overlap_total"], 0)
        for key in (
            "v1_1_train_addition",
            "v1_1_dev_challenge",
            "v1_1_sealed_challenge",
        ):
            for metric, count in leakage["v1_1_overlap"][key].items():
                self.assertEqual(count, 0, f"{key}.{metric}")

    def test_template_catalog_independence(self) -> None:
        audit = self.full_validation["template_audit"]
        self.assertTrue(audit["passed"], audit["errors"])
        gates = audit["gates"]
        self.assertEqual(gates["cross_split_normalized_skeleton_overlap"]["actual"], 0)
        self.assertEqual(gates["v1_1_normalized_skeleton_overlap"]["actual"], 0)
        self.assertEqual(gates["v1_1_family_name_overlap"]["actual"], 0)
        self.assertEqual(gates["parent_normalized_skeleton_overlap"]["actual"], 0)
        self.assertEqual(gates["forbidden_parent_template"]["actual"], 0)
        # New namespaces are structurally distinct from every parent.
        for records in self.generated.values():
            for record in records:
                family = record["meta"]["template_family"]
                self.assertTrue(family.startswith("ml_v1_2_"))
                self.assertTrue(family.endswith("_v1_2"))
                self.assertNotIn("ml_v1_1", family)

    # ------------------------------------------------------------------
    # Counterbalance content requirements
    # ------------------------------------------------------------------

    def test_business_id_ssn_multi_entity_examples(self) -> None:
        train = self.generated["train_addition"]
        both = [
            record
            for record in train
            if {"SSN", "BUSINESS_ID"} <= _labels_in_record(record)
        ]
        self.assertGreaterEqual(len(both), 100, "need many SSN+BUSINESS_ID records")
        for record in both[:20]:
            labels = defaultdict(list)
            for start, end, label in record["entities"]:
                labels[label].append((start, end))
            self.assertEqual(len(labels["SSN"]), 1)
            self.assertEqual(len(labels["BUSINESS_ID"]), 1)
            self.assertTrue(record["meta"]["multi_entity"])

    def test_business_id_card_and_bank_account_examples(self) -> None:
        train = self.generated["train_addition"]
        card = [
            record
            for record in train
            if {"CREDIT_CARD_NUMBER", "BUSINESS_ID"} <= _labels_in_record(record)
        ]
        bank = [
            record
            for record in train
            if {"BANK_ACCOUNT_NUMBER", "BUSINESS_ID"} <= _labels_in_record(record)
        ]
        self.assertGreaterEqual(len(card), 80)
        self.assertGreaterEqual(len(bank), 60)
        # Both directions exist: card-shaped focus as BUSINESS_ID and as card.
        card_focus_labels = {
            target["expected_label"]
            for record in train
            if record["meta"]["contrast_category"] == "business_id_vs_credit_card"
            for target in record["meta"]["contrast_targets"]
            if target["target_role"] == "focus"
        }
        self.assertEqual(card_focus_labels, {"BUSINESS_ID", "CREDIT_CARD_NUMBER"})

    def test_bidirectional_pairs_share_the_same_surface_value(self) -> None:
        for split, records in self.generated.items():
            groups: dict[str, list[dict[str, object]]] = defaultdict(list)
            for record in records:
                groups[record["meta"]["contrast_group_id"]].append(record)
            self.assertEqual(len(groups), len(records) // 2, split)
            for group_id, pair in list(groups.items())[:50]:
                with self.subTest(split=split, group_id=group_id):
                    self.assertEqual(len(pair), 2)
                    focus = [
                        target
                        for record in pair
                        for target in record["meta"]["contrast_targets"]
                        if target["target_role"] == "focus"
                    ]
                    values = {
                        record["text"][target["start"] : target["end"]]
                        for record, target in zip(pair, focus)
                    }
                    self.assertEqual(len(values), 1)
                    self.assertEqual(
                        len({target["expected_label"] for target in focus}), 2
                    )

    def test_worst_category_and_weak_subtype_coverage(self) -> None:
        train_result = self.full_validation["splits"]["train_addition"]
        dev_result = self.full_validation["splits"]["dev_challenge"]
        self.assertGreaterEqual(
            train_result["category_family_counts"][
                "ip_address_vs_technical_reference"
            ],
            6,
        )
        self.assertGreaterEqual(
            dev_result["category_family_counts"]["ip_address_vs_technical_reference"],
            3,
        )
        train_labels = train_result["label_counts"]
        self.assertGreaterEqual(train_labels["SSN"], 300)
        for label in SENSITIVE_COUNTERBALANCE_LABELS:
            self.assertGreaterEqual(train_labels.get(label, 0), 40, label)
        subtypes = train_result["business_id_subtypes"]
        self.assertGreaterEqual(subtypes["CUSTOMER_ID"], 150)
        self.assertGreaterEqual(subtypes["REQUEST_ID"], 150)
        self.assertEqual(len(subtypes), 12)
        # O-role technical records never convert a sensitive entity to O:
        # every technical-reference record still annotates its BUSINESS_ID.
        for record in self.generated["train_addition"]:
            if record["meta"].get("context_role") == "technical_reference":
                self.assertIn("BUSINESS_ID", _labels_in_record(record))

    # ------------------------------------------------------------------
    # Phase-boundary hygiene
    # ------------------------------------------------------------------

    def test_data_phase_has_no_model_or_onnx_dependencies_or_outputs(self) -> None:
        forbidden_import_roots = {
            "datasets",
            "onnx",
            "onnxruntime",
            "safetensors",
            "tensorflow",
            "torch",
            "transformers",
        }
        for path in (
            DATA_SCRIPTS / "generate_securelogx_ml_v1_2_counterbalance.py",
            DATA_SCRIPTS / "validate_securelogx_ml_v1_2.py",
            DATA_SCRIPTS / "build_securelogx_ml_v1_2_dataset.py",
            DATA_SCRIPTS / "analyze_securelogx_ml_v1_1_dev_errors.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_roots: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(
                        alias.name.partition(".")[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.partition(".")[0])
            self.assertFalse(
                imported_roots & forbidden_import_roots,
                f"{path.name} imports prohibited ML/ONNX dependencies",
            )

        output_paths = tuple(self.profile["outputs"].values())
        self.assertTrue(
            all(
                "output_securelogx" not in path.casefold()
                and "onnx" not in path.casefold()
                and not path.casefold().endswith((".pt", ".bin", ".safetensors"))
                for path in output_paths
            )
        )

    def test_generator_main_writes_nothing_without_explicit_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = generator.main([])
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(exit_code, 0)
            self.assertEqual(list(Path(directory).rglob("*")), [])


if __name__ == "__main__":
    unittest.main()
