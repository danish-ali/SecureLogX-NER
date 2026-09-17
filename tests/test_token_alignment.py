from __future__ import annotations

import sys
import unittest
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from securelogx_training_common import (  # noqa: E402
    IGNORE_INDEX,
    TokenClassificationCollator,
    align_record_windows,
    load_canonical_labels,
)


class TokenAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_canonical_labels(ROOT / "configs" / "securelogx_labels.json")
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-cased",
            revision="cd5ef92a9fb2f889e972770a36d4ed042daf221e",
            use_fast=True,
            local_files_only=True,
        )
        if not cls.tokenizer.is_fast:
            raise AssertionError("Unit tests require a fast tokenizer")

    def align(self, text: str, spans: list[tuple[str, str]], max_length: int = 128):
        entities = []
        search_from = 0
        for value, label in spans:
            start = text.index(value, search_from)
            entities.append([start, start + len(value), label])
            search_from = start + len(value)
        record = {
            "text": text,
            "entities": entities,
            "meta": {
                "source": "unit_test",
                "source_record_id": "test",
                "entity_provenance": [
                    {"start": start, "end": end, "label": label}
                    for start, end, label in entities
                ],
            },
        }
        windows, diagnostics, _ = align_record_windows(
            record,
            self.tokenizer,
            self.config["label_to_id"],
            max_length=max_length,
            stride=min(32, max_length // 3),
            record_index=0,
        )
        self.assertEqual(len(diagnostics), len(entities))
        self.assertTrue(
            all(item["status"] in {"aligned", "boundary_adjusted"} for item in diagnostics),
            diagnostics,
        )
        return windows, diagnostics, entities

    def assert_bio_present(self, windows, label: str) -> None:
        begin = self.config["label_to_id"][f"B-{label}"]
        inside = self.config["label_to_id"][f"I-{label}"]
        flattened = [value for window in windows for value in window["labels"]]
        self.assertIn(begin, flattened)
        # The entity may genuinely be one token, so I is checked only by tests
        # whose values are known to split.
        self.assertTrue(all(isinstance(value, int) for value in flattened))
        self.assertIsInstance(inside, int)

    def test_single_token_entity(self) -> None:
        windows, _, _ = self.align("Customer Alice signed in.", [("Alice", "PERSON_NAME")])
        self.assert_bio_present(windows, "PERSON_NAME")

    def test_multi_token_person_name(self) -> None:
        windows, _, _ = self.align("Reviewed for John Doe today.", [("John Doe", "PERSON_NAME")])
        self.assert_bio_present(windows, "PERSON_NAME")
        self.assertIn(
            self.config["label_to_id"]["I-PERSON_NAME"],
            [value for window in windows for value in window["labels"]],
        )

    def test_punctuation_adjacent_email(self) -> None:
        value = "jane.doe+7@example.com"
        windows, _, _ = self.align(f"notify({value}), now", [(value, "EMAIL")])
        self.assert_bio_present(windows, "EMAIL")

    def test_hyphenated_ssn(self) -> None:
        value = "123-45-6789"
        windows, _, _ = self.align(f"ssn:{value};", [(value, "SSN")])
        self.assert_bio_present(windows, "SSN")

    def test_credit_card_value(self) -> None:
        value = "4111111111111111"
        windows, _, _ = self.align(f"card {value} declined", [(value, "CREDIT_CARD_NUMBER")])
        self.assert_bio_present(windows, "CREDIT_CARD_NUMBER")

    def test_iban(self) -> None:
        value = "GB82WEST12345698765432"
        windows, _, _ = self.align(f"wire={value}", [(value, "IBAN")])
        self.assert_bio_present(windows, "IBAN")

    def test_api_key(self) -> None:
        value = "sk_test_aB9CdEf0123456789"
        windows, _, _ = self.align(f"leaked {value}!", [(value, "API_KEY")])
        self.assert_bio_present(windows, "API_KEY")

    def test_bearer_auth_token_excludes_scheme(self) -> None:
        value = "tok_aB9-CdEf_0123456789"
        windows, _, entities = self.align(f"Authorization: Bearer {value}", [(value, "AUTH_TOKEN")])
        self.assert_bio_present(windows, "AUTH_TOKEN")
        self.assertEqual(entities[0][0], len("Authorization: Bearer "))

    def test_business_id(self) -> None:
        value = "TXN-AB-12345678"
        windows, _, _ = self.align(f"retry {value}", [(value, "BUSINESS_ID")])
        self.assert_bio_present(windows, "BUSINESS_ID")

    def test_multiple_entities_same_line(self) -> None:
        text = "John Doe called +1-212-555-0199 from 192.0.2.44"
        windows, diagnostics, _ = self.align(
            text,
            [
                ("John Doe", "PERSON_NAME"),
                ("+1-212-555-0199", "PHONE"),
                ("192.0.2.44", "IP_ADDRESS"),
            ],
        )
        self.assertEqual(len(diagnostics), 3)
        for label in ("PERSON_NAME", "PHONE", "IP_ADDRESS"):
            self.assert_bio_present(windows, label)

    def test_entity_at_beginning(self) -> None:
        windows, _, entities = self.align("alice@example.com connected", [("alice@example.com", "EMAIL")])
        self.assertEqual(entities[0][0], 0)
        self.assert_bio_present(windows, "EMAIL")

    def test_entity_at_end(self) -> None:
        value = "DE89370400440532013000"
        text = f"destination {value}"
        windows, _, entities = self.align(text, [(value, "IBAN")])
        self.assertEqual(entities[0][1], len(text))
        self.assert_bio_present(windows, "IBAN")

    def test_json_punctuation_adjacent(self) -> None:
        value = "203.0.113.7"
        windows, _, _ = self.align(f'{{"client_ip":"{value}"}}', [(value, "IP_ADDRESS")])
        self.assert_bio_present(windows, "IP_ADDRESS")

    def test_equals_adjacent(self) -> None:
        value = "021000021"
        windows, _, _ = self.align(f"routing={value}", [(value, "ROUTING_NUMBER")])
        self.assert_bio_present(windows, "ROUTING_NUMBER")

    def test_wordpiece_split_inside_entity(self) -> None:
        value = "tok_UncommonSubword987654321"
        windows, _, _ = self.align(f"token {value}", [(value, "AUTH_TOKEN")])
        labels = [item for window in windows for item in window["labels"]]
        self.assertIn(self.config["label_to_id"]["B-AUTH_TOKEN"], labels)
        self.assertIn(self.config["label_to_id"]["I-AUTH_TOKEN"], labels)

    def test_special_and_padding_tokens_are_ignored(self) -> None:
        short, _, _ = self.align("Alice", [("Alice", "PERSON_NAME")])
        long, _, _ = self.align("Customer John Doe signed in", [("John Doe", "PERSON_NAME")])
        self.assertEqual(short[0]["labels"][0], IGNORE_INDEX)
        self.assertEqual(short[0]["labels"][-1], IGNORE_INDEX)
        collator = TokenClassificationCollator(self.tokenizer, pad_to_multiple_of=8)
        batch = collator([short[0], long[0]])
        for row in range(2):
            for token_index, attention in enumerate(batch["attention_mask"][row].tolist()):
                if attention == 0:
                    self.assertEqual(batch["labels"][row, token_index].item(), IGNORE_INDEX)

    def test_overflow_windows_preserve_late_entity(self) -> None:
        prefix = " ".join(f"word{index}" for index in range(180))
        value = "late.person@example.com"
        windows, diagnostics, _ = self.align(
            f"{prefix} {value}",
            [(value, "EMAIL")],
            max_length=64,
        )
        self.assertGreater(len(windows), 1)
        self.assertEqual(diagnostics[0]["status"], "aligned")
        self.assert_bio_present(windows, "EMAIL")

    def test_overflow_partial_entity_tokens_are_ignored(self) -> None:
        prefix = " ".join(["alpha"] * 11)
        value = "John Ronald Reuel Tolkien"
        windows, diagnostics, entities = self.align(
            f"{prefix} {value}",
            [(value, "PERSON_NAME")],
            max_length=16,
        )
        start, end, _ = entities[0]
        partial_windows = []
        for window in windows:
            overlapping = [
                index
                for index, (token_start, token_end) in enumerate(
                    window["offset_mapping"]
                )
                if token_start != token_end and token_start < end and token_end > start
            ]
            if overlapping and not (
                window["offset_mapping"][overlapping[0]][0] <= start
                and window["offset_mapping"][overlapping[-1]][1] >= end
            ):
                partial_windows.append((window, overlapping))
        self.assertTrue(partial_windows)
        for window, overlapping in partial_windows:
            self.assertTrue(
                all(window["labels"][index] == IGNORE_INDEX for index in overlapping)
            )
        self.assertEqual(diagnostics[0]["status"], "aligned")
        self.assert_bio_present(windows, "PERSON_NAME")

    def test_excessive_wordpiece_boundary_adjustment_is_explicit_failure(self) -> None:
        record = {
            "text": "Tolkien",
            "entities": [[2, 3, "PERSON_NAME"]],
            "meta": {"source": "unit_test", "source_record_id": "bad-boundary"},
        }
        windows, diagnostics, supervised = align_record_windows(
            record,
            self.tokenizer,
            self.config["label_to_id"],
            max_length=16,
            stride=4,
            record_index=0,
        )
        self.assertEqual(supervised, 0)
        self.assertEqual(diagnostics[0]["status"], "failed")
        self.assertEqual(
            diagnostics[0]["reason"],
            "token_boundary_adjustment_exceeds_three_characters",
        )
        entity_token_labels = [
            label
            for window in windows
            for (start, end), label in zip(
                window["offset_mapping"], window["labels"]
            )
            if start < 3 and end > 2
        ]
        self.assertEqual(entity_token_labels, [IGNORE_INDEX])


if __name__ == "__main__":
    unittest.main()
