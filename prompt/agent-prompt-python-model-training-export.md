# SecureLogX Agent Prompt: Python Model Training and ONNX Export

## Purpose

Use this prompt with the instruction file:

```text
docs/python-model-training-export.md
```

This prompt is for implementing the **Python-side model factory** for SecureLogX.

This includes:

```text
1. validating prepared labeled data
2. training a token-classification model
3. evaluating the trained model
4. exporting ONNX artifacts
5. exporting tokenizer and label files
6. creating Java-compatible model bundle artifacts
```

This prompt is **not** for data acquisition or dataset conversion. That belongs to:

```text
docs/data-acquisition-training-scenarios.md
```

---

## Important Distinction

Use these two instruction files separately:

```text
docs/data-acquisition-training-scenarios.md
```

Use for:

```text
datasets
Ai4Privacy
Gretel
GLiNER baseline/pseudo-labeling
StarPII baseline
custom synthetic logs
label mapping
JSONL conversion
merge/split data
```

```text
docs/python-model-training-export.md
```

Use for:

```text
training
evaluation
ONNX export
tokenizer export
labels export
metadata export
Java runtime bundle
```

Important rule:

```text
Data is not exported to ONNX.
Only a trained model is exported to ONNX.
```

---

## Agent Role

You are an expert Python ML engineering agent.

Implement the SecureLogX Python-side model training and ONNX export pipeline exactly as described in:

```text
docs/python-model-training-export.md
```

Do not re-implement the data acquisition pipeline unless required input files are missing.

---

## Required Inputs

Before starting, verify that these files exist:

```text
data/training/all_labeled.jsonl
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl
configs/securelogx_labels.json
configs/training_profile.yml
```

Recommended supporting files:

```text
reports/data_lineage_report.md
reports/dataset_quality_report.md
reports/label_distribution.csv
configs/data_sources.yml
configs/securelogx_label_map.yml
```

If these inputs are missing, stop and report that the data acquisition prompt must be run first:

```text
prompts/agent-prompt-data-acquisition-training.md
```

---

## Required Implementation

Create or update:

```text
configs/training_profile.yml
scripts/train/train_securelogx_token_classifier.py
scripts/train/evaluate_securelogx_model.py
scripts/train/export_onnx_bundle.py
```

The final output must include:

```text
output_securelogx/model-best/
onnx/securelogx.onnx
onnx/securelogx-quantized.onnx
onnx/tokenizer.json
onnx/labels.json
onnx/securelogx-model-metadata.json
onnx/tokenizer_config.json
onnx/special_tokens_map.json
onnx/hf-config.json
reports/model_evaluation_report.md
reports/per_entity_metrics.csv
reports/error_analysis.jsonl
```

---

## Step 1: Validate Prepared Data

Run or create:

```text
scripts/data/validate_jsonl_spans.py
```

Command:

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/training/all_labeled.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/dataset_quality_report.md \
  --label-dist reports/label_distribution.csv
```

Do not train if validation fails.

---

## Step 2: Train Model

Create:

```text
scripts/train/train_securelogx_token_classifier.py
```

Expected command:

```bash
python scripts/train/train_securelogx_token_classifier.py \
  --base-model distilbert-base-uncased \
  --train data/split/train.jsonl \
  --dev data/split/dev.jsonl \
  --test data/split/test.jsonl \
  --labels configs/securelogx_labels.json \
  --out output_securelogx/model-best \
  --max-length 256 \
  --epochs 3 \
  --batch-size 16
```

Requirements:

```text
load labels from configs/securelogx_labels.json
do not hard-code label order
use tokenizer offset_mapping
align character spans to BIO token labels
use -100 for ignored/special tokens
save tokenizer from the same trained model
save id2label and label2id in model config
```

---

## Step 3: Evaluate Model

Create:

```text
scripts/train/evaluate_securelogx_model.py
```

Expected command:

```bash
python scripts/train/evaluate_securelogx_model.py \
  --model output_securelogx/model-best \
  --test data/split/test.jsonl \
  --labels configs/securelogx_labels.json \
  --out-report reports/model_evaluation_report.md \
  --out-metrics reports/per_entity_metrics.csv \
  --out-errors reports/error_analysis.jsonl
```

Required metrics:

```text
overall precision
overall recall
overall F1
per-entity precision
per-entity recall
per-entity F1
false positives
false negatives
latency per record if practical
```

---

## Step 4: Export ONNX Bundle

Create:

```text
scripts/train/export_onnx_bundle.py
```

Expected command:

```bash
python scripts/train/export_onnx_bundle.py \
  --model output_securelogx/model-best \
  --out onnx \
  --task token-classification \
  --opset 17 \
  --quantize dynamic
```

Required outputs:

```text
onnx/securelogx.onnx
onnx/securelogx-quantized.onnx
onnx/tokenizer.json
onnx/labels.json
onnx/securelogx-model-metadata.json
onnx/tokenizer_config.json
onnx/special_tokens_map.json
onnx/hf-config.json
```

The script must:

```text
export ONNX from trained Hugging Face model
optionally create dynamic quantized ONNX
copy tokenizer files
copy config.json as hf-config.json
generate labels.json
generate securelogx-model-metadata.json
run smoke inference
verify ONNX output dimension matches label count
```

---

## Step 5: Smoke Test

Use this input:

```text
SECURE Customer John Doe updated SSN 123-45-6789 and card 4111111111111111.
```

Verify:

```text
ONNX file loads
tokenizer loads
labels.json loads
model output shape is correct
predicted label indexes map to labels
```

Do not claim production readiness from smoke test alone.

---

## Required Completion Response

When done, report:

```text
1. Files created or modified
2. Exact commands run
3. Training result
4. Evaluation result
5. ONNX export status
6. Quantization status
7. Smoke test result
8. Any blockers or assumptions
9. Remaining TODOs
```

Do not claim production readiness unless training, evaluation, ONNX export, and smoke test have all passed.
