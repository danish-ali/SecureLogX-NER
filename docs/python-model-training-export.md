# SecureLogX Python-Side Model Training and ONNX Export Approach

## 1. Purpose

This document defines the Python-side workflow for SecureLogX model training, evaluation, ONNX export, and Java runtime artifact preparation.

This document should now be treated as the **main model factory instruction**.

The detailed data-source strategy is intentionally moved to the companion document:

```text
docs/data-acquisition-training-scenarios.md
```

That companion document owns:

```text
1. Pretrained / fine-tuned model usage
2. Labeled public dataset conversion
3. Raw/custom SecureLogX data generation and labeling
4. Dataset source configuration
5. Dataset label mapping
6. Dataset merge/split preparation
```

This document owns:

```text
1. Training a SecureLogX token-classification model
2. Evaluating the trained model
3. Exporting ONNX runtime artifacts
4. Exporting tokenizer and labels
5. Creating the Java integration model bundle
6. Defining quality gates before Java runtime use
```

Important rule:

```text
Data is not exported to ONNX.
Only a trained model is exported to ONNX.
```

The Java side must not depend on Python at runtime. Python is used only for data preparation, training, evaluation, and artifact export.

---

## 2. Relationship to the Data Acquisition Document

Use this document together with:

```text
docs/data-acquisition-training-scenarios.md
```

Recommended ownership split:

| Area | Owner Document |
|---|---|
| Hugging Face dataset selection | `data-acquisition-training-scenarios.md` |
| Ai4Privacy / Gretel converters | `data-acquisition-training-scenarios.md` |
| GLiNER / StarPII baseline or pseudo-labeling | `data-acquisition-training-scenarios.md` |
| SecureLogX synthetic log generation | `data-acquisition-training-scenarios.md` |
| Canonical JSONL source format | Both documents, but detailed source handling belongs to data document |
| Model training | This document |
| Model evaluation | This document |
| ONNX export | This document |
| Java artifact contract | This document |

Do not duplicate detailed dataset logic in this file. This file assumes the data pipeline has already produced:

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

## 3. Updated High-Level Pipeline

The updated SecureLogX Python-side flow is:

```text
Data acquisition and conversion
    -> handled by docs/data-acquisition-training-scenarios.md

Canonical labeled SecureLogX JSONL
    -> data/training/all_labeled.jsonl

Train/dev/test split
    -> data/split/train.jsonl
    -> data/split/dev.jsonl
    -> data/split/test.jsonl

Model training
    -> output_securelogx/model-best/

Evaluation
    -> reports/model_evaluation_report.md
    -> reports/per_entity_metrics.csv
    -> reports/error_analysis.jsonl

ONNX export
    -> onnx/securelogx.onnx
    -> onnx/securelogx-quantized.onnx

Java artifact bundle
    -> tokenizer.json
    -> labels.json
    -> securelogx-model-metadata.json
```

Simplified sequence:

```text
data document produces labeled data
    -> train
    -> evaluate
    -> export ONNX
    -> package runtime bundle
    -> Java integration
```

---

## 4. Recommended Repository Structure

Updated repository structure:

```text
securelogx-ai/
├── README.md
├── AGENTS.md
├── docs/
│   ├── securelogx-python-training-approach.md
│   ├── data-acquisition-training-scenarios.md
│   ├── securelogx-data-schema.md
│   ├── securelogx-entity-taxonomy.md
│   ├── securelogx-annotation-guide.md
│   ├── securelogx-evaluation-plan.md
│   ├── securelogx-onnx-export-guide.md
│   └── securelogx-java-integration-contract.md
├── configs/
│   ├── data_sources.yml
│   ├── securelogx_label_map.yml
│   ├── securelogx_labels.json
│   └── training_profile.yml
├── data/
│   ├── external/
│   │   ├── ai4privacy_open_pii_500k/
│   │   └── gretel_finance_pii/
│   ├── custom/
│   │   ├── raw_logs/
│   │   └── generated_logs/
│   ├── processed/
│   │   ├── ai4privacy_securelogx.jsonl
│   │   ├── gretel_finance_securelogx.jsonl
│   │   ├── securelogx_generated.jsonl
│   │   └── securelogx_pseudolabeled_reviewed.jsonl
│   ├── training/
│   │   └── all_labeled.jsonl
│   └── split/
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
├── scripts/
│   ├── data/
│   │   ├── validate_jsonl_spans.py
│   │   ├── convert_ai4privacy_to_securelogx.py
│   │   ├── convert_gretel_to_securelogx.py
│   │   ├── generate_securelogx_synthetic_logs.py
│   │   ├── pseudolabel_with_gliner.py
│   │   ├── run_starpii_baseline.py
│   │   ├── merge_labeled_sources.py
│   │   └── split_labeled_jsonl.py
│   ├── train/
│   │   ├── train_securelogx_token_classifier.py
│   │   ├── evaluate_securelogx_model.py
│   │   └── export_onnx_bundle.py
│   └── rules/
│       ├── deterministic_recognizers.py
│       └── luhn_card_validator.py
├── output_securelogx/
│   └── model-best/
├── reports/
│   ├── data_lineage_report.md
│   ├── dataset_quality_report.md
│   ├── label_distribution.csv
│   ├── model_evaluation_report.md
│   ├── per_entity_metrics.csv
│   └── error_analysis.jsonl
└── onnx/
    ├── securelogx.onnx
    ├── securelogx-quantized.onnx
    ├── tokenizer.json
    ├── labels.json
    ├── securelogx-model-metadata.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── hf-config.json
```

---

## 5. Input Assumptions for This Document

This document starts after the data acquisition pipeline has completed.

Required inputs:

```text
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl
configs/securelogx_labels.json
configs/training_profile.yml
```

Optional but recommended inputs:

```text
reports/data_lineage_report.md
reports/dataset_quality_report.md
reports/label_distribution.csv
configs/data_sources.yml
configs/securelogx_label_map.yml
```

The data must already be in canonical SecureLogX JSONL format.

---

## 6. Canonical SecureLogX JSONL Format

The preferred labeled format is one JSON object per line.

```json
{
  "text": "Customer John Doe used SSN 123-45-6789.",
  "entities": [
    [9, 17, "PERSON_NAME"],
    [27, 38, "SSN"]
  ],
  "meta": {
    "source": "gretel_finance_pii",
    "format": "text",
    "review_status": "auto_converted",
    "license_reviewed": true
  }
}
```

Required rules:

```text
text must contain the exact string used for offsets.
entities must contain character offsets in [start, end, label] format.
start is inclusive.
end is exclusive.
text[start:end] must match the entity text.
entity spans must not overlap unless the training script explicitly supports nested entities.
labels must match configs/securelogx_labels.json.
raw unlabeled records must not be used for training.
```

---

## 7. Entity Scope

The canonical label set is defined in:

```text
configs/securelogx_labels.json
```

The data-source-specific label mapping is defined in:

```text
configs/securelogx_label_map.yml
```

The first production-focused model should not try to support every possible sensitive entity. Use the production-critical v1 subset first:

```text
PERSON_NAME
FIRST_NAME
LAST_NAME
DOB
SSN
EMAIL
PHONE
ADDRESS_LINE_1
ZIP_CODE
CREDIT_CARD_NUMBER
CVV
BANK_ACCOUNT_NUMBER
ROUTING_NUMBER
CUSTOMER_ID
ACCOUNT_ID
USER_ID
PASSWORD
PIN
OTP
AUTH_TOKEN
API_KEY
JWT
SESSION_ID
COOKIE
IP_ADDRESS
DRIVER_LICENSE
PASSPORT_NUMBER
```

Recommended handling split:

| Entity Type | Preferred Handling |
|---|---|
| Names | NER model |
| Addresses | NER model + rules |
| DOB | Hybrid |
| SSN | Regex/rule + optional model confirmation |
| Email | Regex/rule |
| Phone | Regex/rule |
| Credit card | Regex + Luhn validation |
| CVV | Contextual rule, not plain numeric detection |
| Routing number | Regex/rule |
| Account/customer IDs | Hybrid |
| JWT/API key/token/cookie/session ID | Rule/entropy/pattern detector |
| IP address | Regex/rule |
| Passport/driver license | Hybrid |

The Java runtime should support both:

```text
1. ONNX model-based detection
2. deterministic recognizer/rule-based detection
```

---

## 8. Training Profile

Create or use:

```text
configs/training_profile.yml
```

Recommended initial profile:

```yaml
training:
  base_model: distilbert-base-uncased
  max_length: 256
  epochs: 3
  batch_size: 16
  learning_rate: 0.00005
  weight_decay: 0.01
  seed: 42
  evaluation_strategy: epoch
  save_strategy: epoch
  metric_for_best_model: f1
  load_best_model_at_end: true

runtime:
  target: java_onnx
  opset: 17
  quantize_dynamic: true
  max_sequence_length: 256

quality_targets:
  overall_f1_minimum: 0.90
  critical_entity_recall_minimum: 0.95
  require_per_entity_report: true
```

Recommended first base model:

```text
distilbert-base-uncased
```

Possible later alternatives:

```text
microsoft/deberta-v3-small
prajjwal1/bert-mini
sentence-transformers/all-MiniLM-L6-v2 if adapted carefully
```

Guidance:

```text
More training data does not make the Java runtime model heavier.
Runtime size depends mainly on base model architecture.
Use large and diverse data for learning.
Deploy a small optimized ONNX model in Java.
```

---

## 9. Step-by-Step Workflow

## Step 1: Confirm Data Pipeline Completion

Before training, ensure the companion data pipeline has produced:

```text
data/training/all_labeled.jsonl
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl
configs/securelogx_labels.json
reports/dataset_quality_report.md
reports/label_distribution.csv
```

Command:

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/training/all_labeled.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/dataset_quality_report.md \
  --label-dist reports/label_distribution.csv
```

Do not continue if:

```text
JSONL parsing fails.
Entity spans do not match text[start:end].
Labels are missing from configs/securelogx_labels.json.
Raw unlabeled records are mixed into training data.
There are duplicate records across train/dev/test.
```

---

## Step 2: Train SecureLogX Token-Classification Model

Script:

```text
scripts/train/train_securelogx_token_classifier.py
```

Purpose:

```text
Fine-tune a small Hugging Face token-classification model on SecureLogX JSONL.
```

Expected behavior:

```text
load train/dev/test JSONL
load labels from configs/securelogx_labels.json
tokenize text
align character spans to token-level BIO labels
train token-classification model
evaluate during training
save best model
save tokenizer
save config with id2label and label2id
save training metrics
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

Expected output:

```text
output_securelogx/model-best/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── trainer_state.json
├── training_args.bin
└── training_metrics.json
```

Training requirements:

```text
Use the same tokenizer that will later be exported.
Use id2label and label2id from configs/securelogx_labels.json.
Ignore subword tokens that do not align to entity starts, or use a consistent BIO alignment strategy.
Preserve offset mapping during preprocessing for evaluation.
Do not hard-code labels in the script.
```

---

## Step 3: Evaluate Final Model

Script:

```text
scripts/train/evaluate_securelogx_model.py
```

Purpose:

```text
Evaluate final model quality on held-out test data.
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
latency per record if possible
```

Required reports:

```text
reports/model_evaluation_report.md
reports/per_entity_metrics.csv
reports/error_analysis.jsonl
```

Minimum evaluation expectations:

```text
Critical entities must have high recall.
False negatives for SSN, card number, CVV, account number, token, and password must be reviewed.
Evaluation must include plain text, JSON, XML, key-value logs, and API-style records.
Model must be tested on SecureLogX-style generated logs, not only public dataset prose.
```

---

## Step 4: Export Final Model to ONNX

Script:

```text
scripts/train/export_onnx_bundle.py
```

Purpose:

```text
Export the trained SecureLogX model into Java-loadable ONNX artifacts.
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

Export script requirements:

```text
Export ONNX from the trained Hugging Face model.
Create dynamic quantized ONNX if enabled.
Copy tokenizer.json from the same trained model.
Copy tokenizer_config.json.
Copy special_tokens_map.json.
Copy config.json as hf-config.json.
Generate labels.json.
Generate securelogx-model-metadata.json.
Run ONNX smoke inference.
Verify ONNX output dimension matches label count.
Verify quantized ONNX smoke inference if quantized model exists.
```

Alternative direct export command if needed:

```bash
python -m optimum.exporters.onnx \
  --model output_securelogx/model-best \
  --task token-classification \
  --opset 17 \
  onnx/
```

The custom export script is preferred because it should also copy tokenizer, labels, and metadata.

---

## Step 5: Export Tokenizer

The Java runtime needs the tokenizer used during training.

Required file:

```text
onnx/tokenizer.json
```

Recommended support files:

```text
onnx/tokenizer_config.json
onnx/special_tokens_map.json
```

Rules:

```text
Do not use a different tokenizer in Java.
Do not rebuild tokenizer manually.
Do not modify tokenizer vocabulary after training.
Java token offsets must align with Python token offsets.
```

---

## Step 6: Export Labels

Required file:

```text
onnx/labels.json
```

Recommended format:

```json
{
  "id2label": {
    "0": "O",
    "1": "B-PERSON_NAME",
    "2": "I-PERSON_NAME",
    "3": "B-SSN",
    "4": "I-SSN"
  },
  "label2id": {
    "O": 0,
    "B-PERSON_NAME": 1,
    "I-PERSON_NAME": 2,
    "B-SSN": 3,
    "I-SSN": 4
  }
}
```

Rules:

```text
Java must not hard-code label order.
labels.json must match model config id2label/label2id.
ONNX output dimension must match the number of labels.
```

---

## Step 7: Export SecureLogX Model Metadata

Required file:

```text
onnx/securelogx-model-metadata.json
```

Recommended format:

```json
{
  "modelName": "securelogx-pii-ner",
  "modelVersion": "0.1.0",
  "modelType": "token-classification",
  "baseModel": "distilbert-base-uncased",
  "onnxOpset": 17,
  "maxSequenceLength": 256,
  "tokenizerFile": "tokenizer.json",
  "labelsFile": "labels.json",
  "supportedEntities": [
    "PERSON_NAME",
    "SSN",
    "EMAIL",
    "PHONE",
    "DOB",
    "ADDRESS_LINE_1",
    "CREDIT_CARD_NUMBER",
    "CVV",
    "BANK_ACCOUNT_NUMBER",
    "ROUTING_NUMBER",
    "AUTH_TOKEN",
    "API_KEY",
    "JWT",
    "SESSION_ID",
    "COOKIE",
    "IP_ADDRESS"
  ],
  "recommendedThresholds": {
    "PERSON_NAME": 0.75,
    "ADDRESS_LINE_1": 0.75,
    "DOB": 0.80,
    "SSN": 0.90,
    "CREDIT_CARD_NUMBER": 0.95,
    "CVV": 0.95,
    "API_KEY": 0.95,
    "JWT": 0.95
  },
  "maskingPolicyHints": {
    "SSN": "MASK_LAST4",
    "CREDIT_CARD_NUMBER": "MASK_LAST4",
    "CVV": "FULL_MASK",
    "PASSWORD": "FULL_MASK",
    "PIN": "FULL_MASK",
    "OTP": "FULL_MASK",
    "API_KEY": "FULL_MASK",
    "JWT": "FULL_MASK",
    "PERSON_NAME": "FULL_MASK",
    "EMAIL": "PARTIAL_MASK",
    "PHONE": "PARTIAL_MASK"
  },
  "createdBy": "SecureLogX Python Training Pipeline"
}
```

Metadata requirements:

```text
supportedEntities must match the trained entity set.
thresholds must be configurable in Java.
maskingPolicyHints are defaults only; Java policy should be authoritative.
modelVersion must be updated for every released model bundle.
```

---

## Step 8: Java Smoke Test Contract

After ONNX export, the Python side should provide at least one smoke-test input and expected output.

Example smoke test input:

```text
SECURE Customer John Doe updated SSN 123-45-6789 and card 4111111111111111.
```

Expected high-level entities:

```text
PERSON_NAME: John Doe
SSN: 123-45-6789
CREDIT_CARD_NUMBER: 4111111111111111
```

The Java side should verify:

```text
ONNX session loads.
Tokenizer loads.
labels.json loads.
Input tokenization works.
Model output shape is correct.
Predicted labels map correctly.
Entity spans are reconstructed.
Masking policy applies correctly.
Non-SECURE logs bypass model inference if configured.
```

---

## 10. Java Runtime Artifact Contract

### 10.1 Minimum Runtime Files

| File | Required for Java Runtime | Purpose |
|---|---:|---|
| `securelogx.onnx` | Yes | Neural network graph executed by ONNX Runtime Java |
| `tokenizer.json` | Yes | Tokenizer used to convert log text into token IDs |
| `labels.json` | Yes | Maps output indexes to BIO labels |
| `securelogx-model-metadata.json` | Yes | Runtime contract and compatibility metadata |

### 10.2 Strongly Recommended Production Files

| File | Purpose |
|---|---|
| `securelogx-quantized.onnx` | Smaller/faster model for CPU runtime if quality is acceptable |
| `tokenizer_config.json` | Tokenizer configuration traceability |
| `special_tokens_map.json` | Special token metadata |
| `hf-config.json` | Copy of Hugging Face config |
| `model_evaluation_report.md` | Evidence of quality |
| `per_entity_metrics.csv` | Per-entity quality evidence |
| `data_lineage_report.md` | Dataset provenance |
| `dataset_quality_report.md` | Data validation proof |

### 10.3 Recommended SecureLogX Bundle Layout

```text
securelogx-model-bundle/
├── securelogx.onnx
├── securelogx-quantized.onnx
├── tokenizer.json
├── labels.json
├── securelogx-model-metadata.json
├── tokenizer_config.json
├── special_tokens_map.json
├── hf-config.json
├── model_evaluation_report.md
├── per_entity_metrics.csv
├── data_lineage_report.md
└── dataset_quality_report.md
```

---

## 11. Quality Gates

### 11.1 Data Quality Gate

Must pass before training:

```text
JSONL files are valid.
Entity spans are valid.
Labels are consistent.
No unlabeled data is accidentally used.
No duplicate records leak across train/dev/test.
All external sources have metadata.
License review flags are present.
```

### 11.2 Model Quality Gate

Must pass before export:

```text
Overall F1 is acceptable.
Critical entities have strong recall.
False negatives are reviewed for high-risk fields.
Model performs across plain text, JSON, XML, key-value, and log-style records.
Per-entity metrics are generated.
Error-analysis file is generated.
```

### 11.3 Export Quality Gate

Must pass before Java integration:

```text
ONNX export completes successfully.
ONNX inference output shape is correct.
Quantized model inference works if quantization is enabled.
Tokenizer output matches Python tokenizer behavior.
labels.json matches model output dimension.
Metadata file is generated.
Smoke test passes.
```

### 11.4 Runtime Quality Gate

Must pass in Java:

```text
Java inference works without Python.
Masking output is deterministic.
SECURE logs invoke masking.
Non-SECURE logs bypass AI masking if configured.
showLastFour behavior works only for allowed entity types.
Rules and ONNX results are merged without duplicate/conflicting spans.
```

---

## 12. Common Mistakes to Avoid

| Mistake | Why It Is Bad |
|---|---|
| Training directly on raw unlabeled logs | The model needs verified labels |
| Exporting a dataset to ONNX | ONNX stores the trained model, not data |
| Keeping Kaggle-only assumptions | SecureLogX needs log-specific, finance-specific, and generated examples |
| Changing text after annotation | Character offsets become invalid |
| Using different tokenizer in Java | Entity spans and token predictions may break |
| Trusting pseudo-labels blindly | Incorrect labels reduce final model quality |
| Evaluating only on dev data | Inflates quality estimate |
| Exporting before test evaluation | May ship a weak model |
| Hard-coding label order in Java | Output index mapping may be wrong |
| Using GLiNER as first Java runtime model | It is useful as baseline/pseudo-labeling, but not ideal for first Java ONNX target |
| Masking every detected token without span reconstruction | Can produce broken or partial masking |
| Treating every 3-digit number as CVV | Causes major false positives |
| Treating every date as DOB | Causes major false positives |

---

## 13. Updated Recommended Command Sequence

This command sequence assumes the companion data document has already been implemented.

### 13.1 Data Acquisition and Preparation

See:

```text
docs/data-acquisition-training-scenarios.md
```

Minimum first milestone data command sequence:

```bash
# Convert Gretel Finance PII dataset
python scripts/data/convert_gretel_to_securelogx.py \
  --dataset gretelai/synthetic_pii_finance_multilingual \
  --split train \
  --language English \
  --label-map configs/securelogx_label_map.yml \
  --out data/processed/gretel_finance_securelogx.jsonl

# Generate SecureLogX synthetic logs
python scripts/data/generate_securelogx_synthetic_logs.py \
  --count 25000 \
  --out-labeled data/custom/generated_logs/securelogx_generated_labeled.jsonl \
  --out-raw data/custom/generated_logs/securelogx_generated_raw.jsonl \
  --seed 42

# Merge labeled sources
python scripts/data/merge_labeled_sources.py \
  --in \
    data/processed/gretel_finance_securelogx.jsonl \
    data/custom/generated_logs/securelogx_generated_labeled.jsonl \
  --out data/training/all_labeled.jsonl \
  --report reports/data_lineage_report.md \
  --dedupe

# Validate merged data
python scripts/data/validate_jsonl_spans.py \
  --in data/training/all_labeled.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/dataset_quality_report.md \
  --label-dist reports/label_distribution.csv

# Split labeled data
python scripts/data/split_labeled_jsonl.py \
  --in data/training/all_labeled.jsonl \
  --out data/split \
  --train 0.80 \
  --dev 0.10 \
  --test 0.10 \
  --seed 42
```

### 13.2 Train Model

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

### 13.3 Evaluate Model

```bash
python scripts/train/evaluate_securelogx_model.py \
  --model output_securelogx/model-best \
  --test data/split/test.jsonl \
  --labels configs/securelogx_labels.json \
  --out-report reports/model_evaluation_report.md \
  --out-metrics reports/per_entity_metrics.csv \
  --out-errors reports/error_analysis.jsonl
```

### 13.4 Export ONNX Bundle

```bash
python scripts/train/export_onnx_bundle.py \
  --model output_securelogx/model-best \
  --out onnx \
  --task token-classification \
  --opset 17 \
  --quantize dynamic
```

---

## 14. Current Open Questions

Confirm before freezing the Python pipeline:

```text
1. Is distilbert-base-uncased acceptable for v1, or should DeBERTa-small be tested?
2. What exact Java tokenizer library will be used?
3. Does Java need token offsets directly from tokenizer output?
4. Is the target ONNX Runtime Java version compatible with opset 17?
5. Should FIRST_NAME and LAST_NAME remain separate, or should Java mainly use PERSON_NAME?
6. Should address be modeled as one ADDRESS entity or split into ADDRESS_LINE_1, CITY, STATE, ZIP_CODE?
7. Should CVV/card/SSN always be rule-first and model-confirmed?
8. What minimum per-entity recall is required before calling the model production-ready?
9. Which external dataset licenses are approved for public/commercial use?
10. Should pseudo-labeled records be allowed in final training, or only manually reviewed records?
```

---

## 15. Summary

The Python side of SecureLogX should be treated as the **model factory**.

The data acquisition document prepares the labeled data.

This document trains, evaluates, exports, and packages the model.

The Java side should be treated as the **production runtime**. It loads the exported ONNX model, tokenizer, label mapping, and metadata, then applies SecureLogX masking policies during `SECURE` logging.

Most important rules:

```text
Do not train on raw unlabeled logs.
Do not export data to ONNX.
Do not hard-code label order in Java.
Do not use GLiNER as the first Java runtime model.
Use large datasets for training, but deploy a small optimized ONNX model.
```
