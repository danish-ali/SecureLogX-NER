# Instruction and Prompt Map

Use exactly these four files.

## 1. Data Acquisition / Dataset Preparation

Instruction file:

```text
docs/data-acquisition-training-scenarios.md
```

Agent prompt:

```text
prompts/agent-prompt-data-acquisition-training.md
```

Use this pair for:

```text
Ai4Privacy
Gretel Finance PII
GLiNER baseline or pseudo-labeling
StarPII baseline
custom synthetic logs
label mapping
canonical JSONL conversion
data validation
merge/split data
```

Expected output from this phase:

```text
data/training/all_labeled.jsonl
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl
configs/securelogx_labels.json
reports/data_lineage_report.md
reports/dataset_quality_report.md
reports/label_distribution.csv
```

---

## 2. Python Model Training / ONNX Export

Instruction file:

```text
docs/python-model-training-export.md
```

Agent prompt:

```text
prompts/agent-prompt-python-model-training-export.md
```

Use this pair for:

```text
model training
model evaluation
ONNX export
quantized ONNX export
tokenizer export
labels.json export
model metadata export
Java runtime model bundle
```

Expected output from this phase:

```text
output_securelogx/model-best/
onnx/securelogx.onnx
onnx/securelogx-quantized.onnx
onnx/tokenizer.json
onnx/labels.json
onnx/securelogx-model-metadata.json
reports/model_evaluation_report.md
reports/per_entity_metrics.csv
reports/error_analysis.jsonl
```

---

## Correct Execution Order

Run in this order:

```text
1. agent-prompt-data-acquisition-training.md
2. agent-prompt-python-model-training-export.md
```

The second prompt depends on outputs from the first prompt.

---

## Key Rule

```text
Data is not exported to ONNX.
Only a trained model is exported to ONNX.
```
