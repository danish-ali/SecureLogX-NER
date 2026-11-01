"""Model-free checks of proposed candidate-only gold annotation correction."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] /
          "scripts/data/propose_bank_account_gold_amendment.py")
spec = importlib.util.spec_from_file_location("gold_amendment", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def record():
    text = "x" * 469 + "Bank Account Number: " + "123456789" + "y" * 600
    return {
        "text": text,
        "entities": [
            [10, 18, "PERSON_NAME"],
            [30, 42, "BUSINESS_ID"],
            [129, 147, "BANK_ACCOUNT_NUMBER"],
            [180, 204, "STREET_ADDRESS"],
            [700, 710, "PERSON_NAME"],
            [970, 986, "PERSON_NAME"],
        ],
        "meta": {
            "source": "gretel_finance_pii",
            "source_record_id": 36428,
            "review_status": "auto_converted",
            "excluded_spans": [],
        },
    }


class GoldAmendmentTests(unittest.TestCase):
    def test_adds_only_missing_sensitive_span_to_independent_copy(self):
        source = record()
        updated = mod.propose_record(source)
        self.assertEqual(6, len(source["entities"]))
        self.assertEqual(7, len(updated["entities"]))
        self.assertEqual(source["text"], updated["text"])
        self.assertEqual(source["meta"], updated["meta"])
        self.assertIn([490, 499, "BANK_ACCOUNT_NUMBER"], updated["entities"])
        self.assertEqual(
            sorted(updated["entities"], key=lambda x: (x[0], x[1], x[2])),
            updated["entities"],
        )

    def test_does_not_rewrite_already_labeled_span(self):
        source = record()
        source["entities"].append([490, 499, "BANK_ACCOUNT_NUMBER"])
        with self.assertRaises(ValueError):
            mod.propose_record(source)

    def test_rejects_source_identity_mismatch(self):
        source = record()
        source["meta"]["source_record_id"] = 999
        with self.assertRaises(ValueError):
            mod.propose_record(source)

    def test_rejects_non_matching_field_label(self):
        source = record()
        source["text"] = source["text"].replace(
            "Bank Account Number: ", "Some Other Field:    ", 1)
        with self.assertRaises(ValueError):
            mod.propose_record(source)

    def test_rejects_excluded_span(self):
        source = record()
        source["meta"]["excluded_spans"] = [
            {"start": 489, "end": 501, "source_label": "ignored"}
        ]
        with self.assertRaises(ValueError):
            mod.propose_record(source)

    def test_git_blob_hash_changes_when_input_changes(self):
        self.assertNotEqual(
            mod.git_blob_sha1(b"example corpus A"),
            mod.git_blob_sha1(b"example corpus B"),
        )


if __name__ == "__main__":
    unittest.main()
