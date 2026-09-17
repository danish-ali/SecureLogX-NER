from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from evaluate_securelogx_model import (  # noqa: E402
    _metrics_without_rows,
    _supported_macro,
)
from securelogx_training_common import (  # noqa: E402
    EXPECTED_ENTITIES,
    decode_bio_spans,
    score_predictions,
    subset_score,
    write_confusion_csv,
)


LABELS = list(EXPECTED_ENTITIES)


def make_record(
    text: str,
    entities: list[list[int | str]],
    record_id: str = "metric-fixture",
) -> dict[str, Any]:
    return {
        "text": text,
        "entities": entities,
        "meta": {
            "source": "unit_test",
            "source_record_id": record_id,
            "template_family": "metric_contract",
        },
    }


def predicted_span(
    start: int,
    end: int,
    label: str,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "label": label,
        "confidence": confidence,
        "minimum_token_confidence": confidence,
    }


class BioDecodeTest(unittest.TestCase):
    ID_TO_LABEL = {
        0: "O",
        1: "B-EMAIL",
        2: "I-EMAIL",
        3: "B-SSN",
        4: "I-SSN",
    }

    def test_valid_bio_decodes_half_open_spans_and_confidence(self) -> None:
        spans, invalid = decode_bio_spans(
            offsets=[(0, 2), (2, 4), (4, 5), (6, 8)],
            label_ids=[1, 2, 0, 3],
            confidences=[0.8, 0.6, 0.99, 0.75],
            id_to_label=self.ID_TO_LABEL,
        )

        self.assertEqual(invalid, 0)
        self.assertEqual(len(spans), 2)
        self.assertEqual(
            {key: spans[0][key] for key in ("start", "end", "label")},
            {"start": 0, "end": 4, "label": "EMAIL"},
        )
        self.assertTrue(math.isclose(spans[0]["confidence"], 0.7))
        self.assertEqual(spans[0]["minimum_token_confidence"], 0.6)
        self.assertEqual(
            {key: spans[1][key] for key in ("start", "end", "label")},
            {"start": 6, "end": 8, "label": "SSN"},
        )

    def test_orphan_and_wrong_class_inside_tags_are_recovered_and_counted(self) -> None:
        spans, invalid = decode_bio_spans(
            offsets=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            label_ids=[2, 2, 4, 1, 0],
            confidences=[0.5, 0.7, 0.8, 0.9, 0.99],
            id_to_label=self.ID_TO_LABEL,
        )

        self.assertEqual(invalid, 2)
        self.assertEqual(
            [(span["start"], span["end"], span["label"]) for span in spans],
            [(0, 2, "EMAIL"), (2, 3, "SSN"), (3, 4, "EMAIL")],
        )
        self.assertTrue(math.isclose(spans[0]["confidence"], 0.6))


class ExactSpanMetricTest(unittest.TestCase):
    def test_exact_micro_macro_and_per_label_metrics(self) -> None:
        records = [
            make_record(
                "aaa bbb",
                [[0, 3, "SSN"], [4, 7, "EMAIL"]],
                "exact-two",
            ),
            make_record("cccc", [[0, 4, "CITY"]], "missed-city"),
            make_record("ddd", [], "spurious-email"),
        ]
        predictions = [
            [
                predicted_span(0, 3, "SSN", 0.95),
                predicted_span(4, 7, "EMAIL", 0.85),
            ],
            [],
            [predicted_span(0, 3, "EMAIL", 0.75)],
        ]

        metrics = score_predictions(records, predictions, LABELS, "unit")

        self.assertEqual(metrics["gold_entities"], 3)
        self.assertEqual(metrics["predicted_entities"], 3)
        self.assertEqual(metrics["exact_span_matches"], 2)
        self.assertEqual(metrics["micro"]["true_positives"], 2)
        self.assertEqual(metrics["micro"]["false_positives"], 1)
        self.assertEqual(metrics["micro"]["false_negatives"], 1)
        for name in ("precision", "recall", "f1"):
            self.assertTrue(math.isclose(metrics["micro"][name], 2 / 3))

        self.assertEqual(metrics["per_label"]["SSN"]["support"], 1)
        self.assertEqual(metrics["per_label"]["SSN"]["predicted"], 1)
        self.assertEqual(metrics["per_label"]["SSN"]["f1"], 1.0)
        self.assertEqual(metrics["per_label"]["EMAIL"]["support"], 1)
        self.assertEqual(metrics["per_label"]["EMAIL"]["predicted"], 2)
        self.assertTrue(
            math.isclose(metrics["per_label"]["EMAIL"]["f1"], 2 / 3)
        )
        self.assertEqual(metrics["per_label"]["CITY"]["false_negatives"], 1)

        self.assertTrue(math.isclose(metrics["macro"]["precision"], 1.5 / 25))
        self.assertTrue(math.isclose(metrics["macro"]["recall"], 2 / 25))
        self.assertTrue(math.isclose(metrics["macro"]["f1"], (1 + 2 / 3) / 25))
        self.assertEqual(metrics["confusion"]["SSN\tSSN"], 1)
        self.assertEqual(metrics["confusion"]["EMAIL\tEMAIL"], 1)
        self.assertEqual(metrics["confusion"]["CITY\t<MISSED>"], 1)
        self.assertEqual(metrics["confusion"]["<SPURIOUS>\tEMAIL"], 1)

    def test_high_risk_aggregate_uses_only_frozen_high_risk_labels(self) -> None:
        records = [
            make_record(
                "aa--------bb--------cc",
                [[0, 2, "SSN"], [10, 12, "API_KEY"], [20, 22, "EMAIL"]],
            )
        ]
        predictions = [
            [
                predicted_span(0, 2, "SSN"),
                predicted_span(14, 16, "API_KEY"),
                predicted_span(20, 22, "EMAIL"),
            ]
        ]

        metrics = score_predictions(records, predictions, LABELS, "unit")
        high_risk = metrics["high_risk"]

        self.assertEqual(high_risk["support"], 2)
        self.assertEqual(high_risk["true_positives"], 1)
        self.assertEqual(high_risk["false_positives"], 1)
        self.assertEqual(high_risk["false_negatives"], 1)
        self.assertEqual(high_risk["precision"], 0.5)
        self.assertEqual(high_risk["recall"], 0.5)
        self.assertEqual(high_risk["f1"], 0.5)

    def test_false_positive_and_false_negative_rows_and_confusion(self) -> None:
        records = [
            make_record("aaaa", [[0, 4, "EMAIL"]], "missing"),
            make_record("bbbb", [], "spurious"),
        ]
        predictions = [[], [predicted_span(0, 4, "CITY", 0.61)]]

        metrics = score_predictions(records, predictions, LABELS, "unit")

        self.assertEqual(metrics["error_counts"]["false_positives"], 1)
        self.assertEqual(metrics["error_counts"]["false_negatives"], 1)
        self.assertEqual(
            [row["error_category"] for row in metrics["errors"]],
            ["false negative", "false positive"],
        )
        self.assertEqual(metrics["confusion"]["EMAIL\t<MISSED>"], 1)
        self.assertEqual(metrics["confusion"]["<SPURIOUS>\tCITY"], 1)
        self.assertEqual(metrics["confidence"]["false_positive"]["count"], 1)

    def test_wrong_class_overlap_is_one_fp_and_one_fn(self) -> None:
        records = [make_record("xxabcdefyy", [[2, 8, "EMAIL"]])]
        predictions = [[predicted_span(2, 8, "SSN", 0.72)]]

        metrics = score_predictions(records, predictions, LABELS, "unit")

        self.assertEqual(metrics["exact_span_matches"], 0)
        self.assertEqual(metrics["error_counts"]["wrong_entity_class"], 1)
        self.assertEqual(metrics["error_counts"]["false_positives"], 1)
        self.assertEqual(metrics["error_counts"]["false_negatives"], 1)
        self.assertEqual(len(metrics["errors"]), 1)
        self.assertEqual(metrics["errors"][0]["error_category"], "wrong entity class")
        self.assertEqual(metrics["confusion"]["EMAIL\tSSN"], 1)
        self.assertNotIn("EMAIL\t<MISSED>", metrics["confusion"])
        self.assertNotIn("<SPURIOUS>\tSSN", metrics["confusion"])

    def test_boundary_categories_are_mutually_exclusive_and_not_exact_confusion(self) -> None:
        cases = (
            ("boundary_too_short", "boundary-too-short", (3, 8)),
            ("boundary_too_long", "boundary-too-long", (1, 9)),
            ("boundary_shifted", "boundary-shifted", (3, 9)),
        )
        for expected_key, expected_category, (start, end) in cases:
            with self.subTest(category=expected_key):
                records = [make_record("xxabcdefyy", [[2, 8, "EMAIL"]])]
                predictions = [[predicted_span(start, end, "EMAIL")]]

                metrics = score_predictions(records, predictions, LABELS, "unit")

                self.assertEqual(metrics["exact_span_matches"], 0)
                self.assertEqual(metrics["micro"]["false_positives"], 1)
                self.assertEqual(metrics["micro"]["false_negatives"], 1)
                self.assertEqual(metrics["error_counts"][expected_key], 1)
                for key in (
                    "boundary_too_short",
                    "boundary_too_long",
                    "boundary_shifted",
                ):
                    self.assertEqual(
                        metrics["error_counts"][key],
                        int(key == expected_key),
                    )
                self.assertEqual(metrics["errors"][0]["error_category"], expected_category)
                self.assertEqual(metrics["confusion"]["EMAIL\t<BOUNDARY>"], 1)
                self.assertNotIn("EMAIL\tEMAIL", metrics["confusion"])

    def test_confusion_csv_preserves_boundary_bucket(self) -> None:
        confusion = {
            "EMAIL\tEMAIL": 2,
            "EMAIL\t<BOUNDARY>": 3,
            "EMAIL\t<MISSED>": 4,
            "<SPURIOUS>\tEMAIL": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confusion.csv"
            write_confusion_csv(path, confusion, ["EMAIL"])
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))

        header = rows[0]
        self.assertIn("<BOUNDARY>", header)
        email_row = next(row for row in rows[1:] if row[0] == "EMAIL")
        self.assertEqual(email_row[header.index("EMAIL")], "2")
        self.assertEqual(email_row[header.index("<BOUNDARY>")], "3")
        self.assertEqual(email_row[header.index("<MISSED>")], "4")
        spurious_row = next(row for row in rows[1:] if row[0] == "<SPURIOUS>")
        self.assertEqual(spurious_row[header.index("EMAIL")], "5")

    def test_subset_score_retains_global_record_indices(self) -> None:
        records = [
            make_record("aaaa", [[0, 4, "EMAIL"]], "record-zero"),
            make_record("bbbb", [], "record-one"),
            make_record("cccc", [], "record-two"),
        ]
        predictions = [[], [], [predicted_span(0, 4, "SSN")]]

        metrics = subset_score(
            records,
            predictions,
            LABELS,
            "selected",
            selected_indices=[2, 0],
        )

        self.assertEqual(metrics["records"], 2)
        self.assertEqual(
            [row["record_index"] for row in metrics["errors"]],
            [2, 0],
        )
        self.assertEqual(
            [row["record_id"] for row in metrics["errors"]],
            ["record-two", "record-zero"],
        )
        self.assertTrue(all(row["split"] == "selected" for row in metrics["errors"]))


class EvaluatorHelperTest(unittest.TestCase):
    def test_supported_macro_excludes_zero_support_labels(self) -> None:
        metrics = {
            "per_label": {
                "EMAIL": {"support": 2, "precision": 0.5, "recall": 1.0, "f1": 2 / 3},
                "CITY": {"support": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3},
                "SSN": {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            }
        }

        result = _supported_macro(metrics)

        self.assertEqual(result["precision"], 0.75)
        self.assertEqual(result["recall"], 0.75)
        self.assertTrue(math.isclose(result["f1"], 2 / 3))

    def test_metrics_without_rows_removes_only_large_diagnostic_maps(self) -> None:
        metrics = {
            "split": "test",
            "micro": {"f1": 0.8},
            "errors": [{"error_category": "false positive"}],
            "confusion": {"<SPURIOUS>\tEMAIL": 1},
        }

        compact = _metrics_without_rows(metrics)

        self.assertEqual(compact, {"split": "test", "micro": {"f1": 0.8}})


if __name__ == "__main__":
    unittest.main()
