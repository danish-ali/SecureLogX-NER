from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from securelogx_training_common import normalize_decoded_spans


class SpanNormalizationTests(unittest.TestCase):
    def test_trims_leading_space_from_sentencepiece_span(self) -> None:
        text = "ssn= 123-45-6789 status=ok"
        spans = [
            {
                "start": 4,
                "end": 16,
                "label": "SSN",
                "confidence": 0.99,
                "minimum_token_confidence": 0.98,
            }
        ]
        normalized = normalize_decoded_spans(text, spans)
        self.assertEqual(normalized[0]["start"], 5)
        self.assertEqual(normalized[0]["end"], 16)
        self.assertEqual(text[normalized[0]["start"]:normalized[0]["end"]], "123-45-6789")
        self.assertEqual(normalized[0]["label"], "SSN")

    def test_trims_trailing_whitespace_only(self) -> None:
        text = "token=abc123   next=x"
        spans = [{"start": 6, "end": 15, "label": "AUTH_TOKEN"}]
        normalized = normalize_decoded_spans(text, spans)
        self.assertEqual(text[normalized[0]["start"]:normalized[0]["end"]], "abc123")

    def test_preserves_internal_whitespace(self) -> None:
        text = "name= Jane Doe "
        spans = [{"start": 5, "end": len(text), "label": "PERSON_NAME"}]
        normalized = normalize_decoded_spans(text, spans)
        self.assertEqual(text[normalized[0]["start"]:normalized[0]["end"]], "Jane Doe")

    def test_bert_exact_span_is_unchanged(self) -> None:
        text = "email=jane@example.com"
        spans = [{"start": 6, "end": 22, "label": "EMAIL"}]
        normalized = normalize_decoded_spans(text, spans)
        self.assertEqual(normalized[0]["start"], 6)
        self.assertEqual(normalized[0]["end"], 22)

    def test_empty_whitespace_span_is_dropped(self) -> None:
        text = "a   b"
        spans = [{"start": 1, "end": 4, "label": "BUSINESS_ID"}]
        normalized = normalize_decoded_spans(text, spans)
        self.assertEqual(normalized, [])


if __name__ == "__main__":
    unittest.main()
