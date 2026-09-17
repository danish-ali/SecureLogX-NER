You are working in the SecureLogX Python workspace.

First, read:

docs/data-acquisition-training-scenarios.md
prompts/agent-prompt-data-acquisition-training.md

Follow prompts/agent-prompt-data-acquisition-training.md as the execution prompt.

Strict scope:
- Implement only the data acquisition and dataset preparation pipeline.
- Do not train a model.
- Do not evaluate a trained model.
- Do not export ONNX.
- Do not create Java runtime code.
- Do not modify src/main/java, pom.xml, build.gradle, scripts/train, output_securelogx, or onnx.

Work in small steps:
1. Inspect the current workspace structure.
2. Create missing directories and config files.
3. Implement scripts/data/validate_jsonl_spans.py.
4. Implement scripts/data/convert_gretel_to_securelogx.py.
5. Implement scripts/data/generate_securelogx_synthetic_logs.py.
6. Implement scripts/data/merge_labeled_sources.py.
7. Implement scripts/data/split_labeled_jsonl.py.
8. Run validation after each generated or converted dataset.
9. Stop after train/dev/test JSONL and reports are created.

Required final outputs:
- configs/data_sources.yml
- configs/securelogx_label_map.yml
- configs/securelogx_labels.json
- configs/training_profile.yml
- data/training/all_labeled.jsonl
- data/split/train.jsonl
- data/split/dev.jsonl
- data/split/test.jsonl
- reports/data_lineage_report.md
- reports/dataset_quality_report.md
- reports/label_distribution.csv
- reports/split_summary.md

Before making large changes, show a concise implementation plan.
After each step, run the relevant validation command.
If a Hugging Face dataset field is different than expected, inspect the columns and update the converter defensively instead of guessing.
If any blocker occurs, stop and report the exact error and the file/line involved.