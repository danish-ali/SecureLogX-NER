#!/usr/bin/env python3
"""Generate labeled non-sealed fixtures for SecureLogX hybrid ML-gate audit.

This script reads only:
- standard development
- v1.3 development challenge
- historical original test

It does not read the sealed challenge and does not run any model inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "standard_dev": "data/split/dev.jsonl",
    "v1_3_challenge_dev": "data/ml_v1_3/real_structure/dev_challenge.jsonl",
    "original_test_regression": "data/split/test.jsonl",
}

OUTPUT = "reports/ml_v1_3_hybrid_gate/gold_fixture.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_entities(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entity in record.get("entities", []):
        if isinstance(entity, Mapping):
            start = int(entity["start"])
            end = int(entity["end"])
            label = str(entity["label"])
        else:
            start, end, label = entity
            start = int(start)
            end = int(end)
            label = str(label)

        result.append(
            {
                "start": start,
                "end": end,
                "label": label,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    cases: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    entity_counts: dict[str, int] = {}

    for source, relative_path in SOURCES.items():
        records = load_jsonl(root / relative_path)
        source_counts[source] = len(records)

        entity_count = 0
        for index, record in enumerate(records):
            entities = normalize_entities(record)
            entity_count += len(entities)
            cases.append(
                {
                    "source": source,
                    "source_index": index,
                    "text": str(record["text"]),
                    "entities": entities,
                }
            )
        entity_counts[source] = entity_count

    payload = {
        "phase": "ML-v1.3 hybrid deterministic gate gold audit",
        "sealed_challenge_accessed": False,
        "model_inference": False,
        "case_count": len(cases),
        "source_record_counts": source_counts,
        "source_entity_counts": entity_counts,
        "cases": cases,
    }

    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HYBRID GATE GOLD FIXTURE READY")
    print("Cases: " + str(len(cases)))
    print("Gold entities: " + str(sum(entity_counts.values())))
    print("Wrote: " + OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
