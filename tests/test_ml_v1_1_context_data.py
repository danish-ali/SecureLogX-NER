from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DATA_SCRIPTS = ROOT / "scripts" / "data"
if str(DATA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPTS))

import generate_securelogx_ml_v1_1_context_contrast as generator  # noqa: E402
import validate_securelogx_ml_v1_1 as validator  # noqa: E402


EXPECTED_RECORD_COUNTS = {
    "train_addition": 3_000,
    "dev_challenge": 480,
    "sealed_challenge": 480,
}
EXPECTED_FAMILY_COUNTS = {
    "train_addition": 30,
    "dev_challenge": 12,
    "sealed_challenge": 12,
}
REQUIRED_BUSINESS_ID_SUBTYPES = {
    "CUSTOMER_ID",
    "ACCOUNT_ID",
    "APPLICATION_ID",
    "TRANSACTION_ID",
    "USER_ID",
    "ORDER_ID",
    "CASE_ID",
    "TICKET_ID",
    "INVOICE_ID",
    "REFERENCE_ID",
    "REQUEST_ID",
    "WORKFLOW_ID",
}
CORE_CHALLENGE_CATEGORIES = {
    "business_id_vs_ssn",
    "business_id_vs_credit_card",
    "business_id_vs_bank_account",
    "ip_address_vs_technical_reference",
}


def _target(record: dict[str, object]) -> dict[str, object]:
    targets = record["meta"]["contrast_targets"]  # type: ignore[index]
    if not isinstance(targets, list) or len(targets) != 1:
        raise AssertionError("Each context-contrast record must have one target")
    return targets[0]


def _meta_values(
    records: list[dict[str, object]], key: str
) -> set[object]:
    return {record["meta"][key] for record in records}  # type: ignore[index]


class MLv11ContextDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated = generator.generate_all(seed=42)
        cls.labels = json.loads(
            (ROOT / "configs" / "securelogx_labels.json").read_text(
                encoding="utf-8"
            )
        )

    def test_generation_is_deterministic_and_catalog_order_independent(self) -> None:
        first = generator.generate_all(seed=42)
        self.assertEqual(first, self.generated)

        original_catalog = generator.catalog()
        with mock.patch.object(
            generator, "catalog", return_value=tuple(reversed(original_catalog))
        ):
            reordered = generator.generate_all(seed=42)
        self.assertEqual(reordered, first)

    def test_exact_record_and_family_targets(self) -> None:
        self.assertEqual(set(self.generated), set(EXPECTED_RECORD_COUNTS))
        profile = json.loads(
            (ROOT / "configs" / "ml_v1_1_dataset_profile.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            profile["target_counts"],
            {
                "train_additions_records": 3_000,
                "dev_challenge_records": 480,
                "sealed_challenge_records": 480,
                "train_families": 30,
                "dev_families": 12,
                "sealed_families": 12,
            },
        )

        for split, records in self.generated.items():
            with self.subTest(split=split):
                self.assertEqual(len(records), EXPECTED_RECORD_COUNTS[split])
                self.assertEqual(
                    len(_meta_values(records, "template_family")),
                    EXPECTED_FAMILY_COUNTS[split],
                )
                validation = validator.validate_split_records(records, split)
                self.assertTrue(validation["passed"], validation["errors"][:5])
                self.assertEqual(
                    validation["complete_group_count"], len(records) // 2
                )

    def test_contrast_groups_are_complete_pairs_with_one_shared_value(self) -> None:
        for split, records in self.generated.items():
            groups: dict[str, list[dict[str, object]]] = defaultdict(list)
            for record in records:
                groups[record["meta"]["contrast_group_id"]].append(record)  # type: ignore[index]

            self.assertEqual(len(groups), len(records) // 2, split)
            for group_id, pair in groups.items():
                with self.subTest(split=split, group_id=group_id):
                    self.assertEqual(len(pair), 2)
                    targets = [_target(record) for record in pair]
                    self.assertEqual(
                        len({target["value_sha256"] for target in targets}), 1
                    )
                    values = {
                        record["text"][target["start"] : target["end"]]  # type: ignore[index]
                        for record, target in zip(pair, targets)
                    }
                    self.assertEqual(len(values), 1)
                    self.assertEqual(
                        len(
                            {
                                record["meta"]["context_role"]  # type: ignore[index]
                                for record in pair
                            }
                        ),
                        2,
                    )
                    self.assertEqual(
                        len({target["expected_label"] for target in targets}), 2
                    )

    def test_spans_targets_and_provenance_are_canonical(self) -> None:
        canonical_entities = set(self.labels["entities"])
        observed_labels: set[str] = set()
        saw_ssn_with_person = False

        for split, records in self.generated.items():
            for record in records:
                text = record["text"]
                entities = record["entities"]
                meta = record["meta"]
                provenance = meta["entity_provenance"]
                target = _target(record)

                self.assertEqual(
                    entities,
                    sorted(entities, key=lambda item: (item[0], item[1], item[2])),
                )
                signatures = []
                previous_end = 0
                for start, end, label in entities:
                    self.assertIs(type(start), int)
                    self.assertIs(type(end), int)
                    self.assertLessEqual(previous_end, start)
                    self.assertLess(start, end)
                    self.assertLessEqual(end, len(text))
                    self.assertIn(label, canonical_entities)
                    self.assertTrue(text[start:end])
                    observed_labels.add(label)
                    signatures.append((start, end, label))
                    previous_end = end

                provenance_signatures = []
                for entry in provenance:
                    signature = (entry["start"], entry["end"], entry["label"])
                    provenance_signatures.append(signature)
                    self.assertTrue(entry["source_label"])
                    self.assertIn(entry["action"], {"kept", "normalized"})
                    self.assertIs(entry["training_exclusion"], False)
                self.assertEqual(Counter(provenance_signatures), Counter(signatures))

                start = target["start"]
                end = target["end"]
                value = text[start:end]
                self.assertEqual(
                    target["value_sha256"],
                    hashlib.sha256(value.encode("utf-8")).hexdigest(),
                )
                expected_label = target["expected_label"]
                if expected_label == "O":
                    self.assertFalse(
                        any(start < entity_end and end > entity_start for entity_start, entity_end, _ in signatures)
                    )
                else:
                    self.assertIn((start, end, expected_label), signatures)

                if expected_label == "BUSINESS_ID":
                    subtype = target["business_id_subtype"]
                    self.assertIn(subtype, REQUIRED_BUSINESS_ID_SUBTYPES)
                    matching = [
                        entry
                        for entry in provenance
                        if (entry["start"], entry["end"], entry["label"])
                        == (start, end, "BUSINESS_ID")
                    ]
                    self.assertEqual(len(matching), 1)
                    self.assertEqual(matching[0]["business_id_subtype"], subtype)

                labels_in_record = {entity[2] for entity in entities}
                saw_ssn_with_person |= {"SSN", "PERSON_NAME"} <= labels_in_record

                self.assertEqual(meta["intended_split"], split)
                self.assertEqual(meta["scenario_kind"], "context_contrast")
                self.assertEqual(meta["review_status"], "synthetic_generated")
                self.assertIs(meta["license_reviewed"], True)

        self.assertTrue(saw_ssn_with_person)
        self.assertLessEqual(observed_labels, canonical_entities)

    def test_train_business_id_subtypes_and_material_additions(self) -> None:
        train = self.generated["train_addition"]
        subtype_counts: Counter[str] = Counter()
        entity_counts: Counter[str] = Counter()

        for record in train:
            for start, end, label in record["entities"]:
                entity_counts[label] += 1
                if label == "BUSINESS_ID":
                    matches = [
                        entry
                        for entry in record["meta"]["entity_provenance"]
                        if (entry["start"], entry["end"], entry["label"])
                        == (start, end, label)
                    ]
                    self.assertEqual(len(matches), 1)
                    subtype_counts[matches[0]["business_id_subtype"]] += 1

        self.assertEqual(set(subtype_counts), REQUIRED_BUSINESS_ID_SUBTYPES)
        self.assertTrue(
            all(count >= 40 for count in subtype_counts.values()), subtype_counts
        )
        self.assertGreaterEqual(
            entity_counts["BUSINESS_ID"], math.ceil(1_360 * 0.25)
        )
        self.assertGreaterEqual(entity_counts["SSN"], math.ceil(843 * 0.25))

    def test_dev_and_sealed_challenge_coverage(self) -> None:
        dev_families: dict[str, set[str]] = defaultdict(set)
        for record in self.generated["dev_challenge"]:
            meta = record["meta"]
            dev_families[meta["contrast_category"]].add(meta["template_family"])
        self.assertGreaterEqual(len(dev_families["business_id_vs_ssn"]), 5)
        self.assertGreaterEqual(
            len(dev_families["business_id_vs_credit_card"])
            + len(dev_families["business_id_vs_bank_account"]),
            3,
        )
        self.assertGreaterEqual(
            len(dev_families["ip_address_vs_technical_reference"]), 2
        )

        sealed = self.generated["sealed_challenge"]
        sealed_families = _meta_values(sealed, "template_family")
        sealed_categories = _meta_values(sealed, "contrast_category")
        sealed_formats = _meta_values(sealed, "format")
        self.assertGreaterEqual(len(sealed_families), 8)
        self.assertLessEqual(len(sealed_families), 12)
        self.assertLessEqual(CORE_CHALLENGE_CATEGORIES, sealed_categories)
        self.assertLessEqual({"text", "json", "key_value"}, sealed_formats)

    def test_cross_split_text_id_family_and_value_reuse_is_zero(self) -> None:
        values: dict[str, dict[str, set[object]]] = {}
        for split, records in self.generated.items():
            split_values = {
                "text": {record["text"] for record in records},
                "source_record_id": _meta_values(records, "source_record_id"),
                "template_family": _meta_values(records, "template_family"),
                "value_sha256": {
                    _target(record)["value_sha256"] for record in records
                },
            }
            self.assertEqual(len(split_values["text"]), len(records))
            self.assertEqual(len(split_values["source_record_id"]), len(records))
            values[split] = split_values

        for left, right in combinations(EXPECTED_RECORD_COUNTS, 2):
            for field in values[left]:
                with self.subTest(left=left, right=right, field=field):
                    self.assertFalse(values[left][field] & values[right][field])

    def test_no_parent_test_or_parent_family_contamination(self) -> None:
        parent_test_texts: set[str] = set()
        parent_test_ids: set[str] = set()
        parent_families: set[str] = set()
        for split in ("train", "dev", "test"):
            with (ROOT / "data" / "split" / f"{split}.jsonl").open(
                encoding="utf-8"
            ) as handle:
                for raw_line in handle:
                    record = json.loads(raw_line)
                    meta = record.get("meta", {})
                    family = meta.get("template_family")
                    if isinstance(family, str):
                        parent_families.add(family)
                    if split == "test":
                        parent_test_texts.add(record["text"])
                        source_id = meta.get("source_record_id")
                        if isinstance(source_id, str):
                            parent_test_ids.add(source_id)

        new_train_dev = (
            self.generated["train_addition"] + self.generated["dev_challenge"]
        )
        all_new = [
            record
            for split_records in self.generated.values()
            for record in split_records
        ]
        self.assertFalse(
            {record["text"] for record in new_train_dev} & parent_test_texts
        )
        self.assertFalse(
            _meta_values(new_train_dev, "source_record_id") & parent_test_ids
        )
        self.assertFalse(_meta_values(all_new, "template_family") & parent_families)
        self.assertNotIn(
            "order_reference_ssn_shape_warn_v1",
            _meta_values(all_new, "template_family"),
        )
        self.assertTrue(
            all(
                "was replayed after a warehouse timeout" not in record["text"].casefold()
                for record in all_new
            )
        )

    def test_frozen_25_entity_51_bio_ontology_is_unchanged(self) -> None:
        ontology = validator.validate_ontology(
            ROOT / "configs" / "securelogx_labels.json"
        )
        self.assertTrue(ontology["passed"], ontology["errors"])
        self.assertEqual(ontology["entity_count"], 25)
        self.assertEqual(ontology["bio_label_count"], 51)
        self.assertEqual(len(self.labels["entities"]), 25)
        self.assertEqual(len(self.labels["bio_labels"]), 51)

        freeze = json.loads(
            (ROOT / "configs" / "ml_v1_1_parent_freeze.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(ontology["sha256"], freeze["ontology"]["sha256"])

    def test_end_to_end_validator_passes_without_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            paths: dict[str, Path] = {}
            for split, records in self.generated.items():
                output = temporary / f"{split}.jsonl"
                generator.write_jsonl(output, records)
                paths[split] = output

            before = sorted(path.relative_to(temporary) for path in temporary.rglob("*"))
            result = validator.validate_dataset(
                ROOT,
                paths=paths,
                include_template_audit=True,
            )
            after = sorted(path.relative_to(temporary) for path in temporary.rglob("*"))

        self.assertEqual(after, before)
        self.assertTrue(result["passed"], result["errors"][:10])
        self.assertTrue(result["ready"])
        self.assertEqual(result["recommendation"], "READY FOR ML-v1.1 RETRAINING")
        self.assertEqual(result["error_count"], 0)
        failed_gates = {
            name: gate for name, gate in result["gates"].items() if not gate["passed"]
        }
        self.assertFalse(failed_gates, failed_gates)

    def test_data_phase_has_no_model_training_or_onnx_dependencies_or_outputs(self) -> None:
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
            DATA_SCRIPTS / "generate_securelogx_ml_v1_1_context_contrast.py",
            DATA_SCRIPTS / "validate_securelogx_ml_v1_1.py",
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

        profile = json.loads(
            (ROOT / "configs" / "ml_v1_1_dataset_profile.json").read_text(
                encoding="utf-8"
            )
        )
        output_paths = tuple(profile["outputs"].values())
        self.assertTrue(
            all(
                "output_securelogx" not in path.casefold()
                and "onnx" not in path.casefold()
                and not path.casefold().endswith((".pt", ".bin", ".safetensors"))
                for path in output_paths
            )
        )

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
