# SecureLogX Agent Prompt: Implement Data Acquisition and Dataset Preparation Pipeline

## Role

You are an expert Python ML engineering agent working on the SecureLogX project.

Your task is to implement only the **data acquisition, dataset conversion, synthetic data generation, validation, merge, split, and data-reporting pipeline** for SecureLogX based on this instruction file:

```text
docs/data-acquisition-training-scenarios.md
```

If the instruction file is not already under `docs/`, copy or move the provided Markdown instruction file into:

```text
docs/data-acquisition-training-scenarios.md
```

This prompt prepares the workspace for the later model-training prompt:

```text
prompts/agent-prompt-python-model-training-export.md
```

Important rule:

```text
Data is not exported to ONNX.
Only a trained model is exported to ONNX.
```

This prompt must stop after creating validated train/dev/test JSONL data and required data reports.

---

## Strict Scope Boundary

This prompt is only for:

```text
1. creating data/config folders
2. creating source configuration files
3. creating canonical SecureLogX BIO label files
4. converting labeled external datasets into SecureLogX JSONL
5. generating custom SecureLogX synthetic labeled logs
6. validating JSONL entity spans
7. merging labeled sources
8. splitting labeled data into train/dev/test
9. producing data lineage and dataset quality reports
```

This prompt must not implement:

```text
model training
model evaluation
ONNX export
quantized ONNX export
tokenizer export
labels.json export for model bundle
Java runtime bundle creation
ONNX smoke testing
```

Those belong only to:

```text
docs/python-model-training-export.md
prompts/agent-prompt-python-model-training-export.md
```

Do not create or modify these scripts in this prompt:

```text
scripts/train/train_securelogx_token_classifier.py
scripts/train/evaluate_securelogx_model.py
scripts/train/export_onnx_bundle.py
```

Do not create or modify Java files, Maven files, Gradle files, or Java runtime classes.

---

## Primary Goal

Implement a production-ready data preparation pipeline that can:

```text
1. Load labeled public/synthetic PII datasets.
2. Convert them into SecureLogX canonical JSONL format.
3. Generate custom SecureLogX synthetic log data.
4. Validate entity offsets and labels.
5. Merge validated labeled sources into data/training/all_labeled.jsonl.
6. Split labeled data into train/dev/test.
7. Produce data lineage, dataset quality, and label distribution reports.
8. Prepare the workspace for BERT-family Hugging Face token-classification training.
```

---

## Preferred First Milestone

Implement the first working milestone with:

```text
Gretel Finance PII dataset
+
SecureLogX generated synthetic logs
+
canonical SecureLogX JSONL
+
BIO-compatible label file
+
validated train/dev/test split
+
data quality reports
```

The first milestone is complete when the data pipeline has produced:

```text
data/training/all_labeled.jsonl
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl
configs/securelogx_labels.json
configs/securelogx_label_map.yml
configs/data_sources.yml
configs/training_profile.yml
reports/data_lineage_report.md
reports/dataset_quality_report.md
reports/label_distribution.csv
reports/split_summary.md
```

Do not train a model in this milestone.

---

## Model Compatibility Requirement

The output of this data pipeline must be compatible with a later **BERT-family Hugging Face token-classification model**.

The later training/export phase will use:

```text
default base model: distilbert-base-uncased
task: token-classification
label format: BIO
runtime target: Java ONNX Runtime
```

Therefore, this data pipeline must produce:

```text
1. character-span entity annotations
2. stable canonical entity labels
3. a BIO-compatible configs/securelogx_labels.json file
4. train/dev/test splits with no duplicate leakage
5. metadata for data lineage and auditability
```

Do not use GLiNER, StarPII, spaCy, or sentence-transformer models as the Java runtime path.

---

## Required Repository Structure

Create or update the following structure:

```text
securelogx-ai/
├── docs/
│   └── data-acquisition-training-scenarios.md
├── prompts/
│   └── agent-prompt-data-acquisition-training.md
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
│   ├── training/
│   └── split/
├── reports/
└── scripts/
    ├── data/
    └── rules/
```

Do not create `onnx/` or `output_securelogx/` as part of this prompt unless they already exist. Those are model-training/export outputs.

---

## Required Config Files

### 1. `configs/data_sources.yml`

Create this file based on the instruction document.

It must define these source types:

```text
ai4privacy_open_pii_500k = labeled_dataset
gretel_finance_pii = labeled_dataset
nvidia_gliner_pii = pretrained_model
bigcode_starpii = pretrained_model
securelogx_custom_logs = raw_or_generated_data
```

Each source must include:

```text
id
kind
provider
name or path
source_url if applicable
enabled
priority
license review flag
usage notes
allowed usage
```

Rules:

```text
Use Gretel first.
Add Ai4Privacy after the Gretel converter is stable.
Use GLiNER only later for pseudo-labeling or baseline comparison.
Use StarPII only later for code-like baseline comparison if gated access is available.
Do not mark license_reviewed as true unless explicitly confirmed.
```

---

### 2. `configs/securelogx_label_map.yml`

Create source-to-SecureLogX label mappings for:

```text
ai4privacy
gretel_finance
gliner_optional
starpii_optional
```

Unsupported or unsafe mappings must map to `null`.

Important mapping rules:

```text
Do not map generic DATE to DOB unless the source clearly means date of birth.
Do not map company or organization to PERSON_NAME.
Do not map every numeric ID to a sensitive ID.
Do not map random numbers to CVV.
Do not map generic address-like text to ADDRESS_LINE_1 unless the source field is clearly address-related.
Do not merge source labels directly without normalization.
```

---

### 3. `configs/securelogx_labels.json`

Create a stable BIO label file for token classification.

It must include:

```text
O
B-<ENTITY>
I-<ENTITY>
```

Use the production-critical v1 subset first:

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

Recommended JSON format:

```json
{
  "entities": [
    "PERSON_NAME",
    "FIRST_NAME",
    "LAST_NAME",
    "DOB",
    "SSN",
    "EMAIL",
    "PHONE",
    "ADDRESS_LINE_1",
    "ZIP_CODE",
    "CREDIT_CARD_NUMBER",
    "CVV",
    "BANK_ACCOUNT_NUMBER",
    "ROUTING_NUMBER",
    "CUSTOMER_ID",
    "ACCOUNT_ID",
    "USER_ID",
    "PASSWORD",
    "PIN",
    "OTP",
    "AUTH_TOKEN",
    "API_KEY",
    "JWT",
    "SESSION_ID",
    "COOKIE",
    "IP_ADDRESS",
    "DRIVER_LICENSE",
    "PASSPORT_NUMBER"
  ],
  "bio_labels": [
    "O",
    "B-PERSON_NAME",
    "I-PERSON_NAME"
  ]
}
```

The actual `bio_labels` list must include `B-` and `I-` labels for every entity in `entities`.

---

### 4. `configs/training_profile.yml`

Create this only as a **handoff configuration file** for the next prompt.

Do not train a model in this prompt.

Suggested content:

```yaml
training:
  model_family: bert
  model_task: token-classification
  base_model: distilbert-base-uncased
  tokenizer_source: same_as_base_model
  use_fast_tokenizer: true
  label_format: BIO
  max_length: 256
  epochs: 3
  batch_size: 16
  learning_rate: 0.00005
  metric_for_best_model: f1
  load_best_model_at_end: true

runtime:
  target: java_onnx
  onnx_task: token-classification
  opset: 17
  quantize_dynamic: true
  export_tokenizer_json: true
  export_labels_json: true
  export_metadata_json: true

data:
  train_path: data/split/train.jsonl
  dev_path: data/split/dev.jsonl
  test_path: data/split/test.jsonl
  labels_path: configs/securelogx_labels.json
```

---

## Canonical SecureLogX JSONL Format

All converted or generated labeled data must use this format:

```json
{
  "text": "Customer John Doe updated SSN 123-45-6789.",
  "entities": [
    [9, 17, "PERSON_NAME"],
    [30, 41, "SSN"]
  ],
  "meta": {
    "source": "gretel_finance_pii",
    "source_record_id": "optional-id",
    "format": "text",
    "review_status": "auto_converted",
    "license_reviewed": false
  }
}
```

Rules:

```text
start offset is inclusive.
end offset is exclusive.
text[start:end] must exactly match the entity value.
entities must not overlap unless explicitly handled.
labels must exist in configs/securelogx_labels.json.
raw unlabeled records must never be merged into training data.
every external-source record must include source metadata.
license_reviewed must exist and must be false unless explicitly confirmed.
```

---

## Implementation Order

Implement in this exact order.

---

### Step 1: Create Configuration Files

Create:

```text
configs/data_sources.yml
configs/securelogx_label_map.yml
configs/securelogx_labels.json
configs/training_profile.yml
```

Also create required directories:

```text
data/external/
data/custom/raw_logs/
data/custom/generated_logs/
data/processed/
data/training/
data/split/
reports/
scripts/data/
scripts/rules/
```

Do not create training or ONNX scripts.

---

### Step 2: Create Span Validator

Create:

```text
scripts/data/validate_jsonl_spans.py
```

It must validate:

```text
JSONL parse correctness
required keys: text, entities, meta
entity offset correctness
text[start:end] exact match where entity text is available
label existence against configs/securelogx_labels.json
overlapping spans
empty text
duplicate entities
duplicate text records
metadata presence
license_reviewed field for external sources
review_status field
```

It should output:

```text
reports/dataset_quality_report.md
reports/label_distribution.csv
```

Example command:

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/processed/gretel_finance_securelogx.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/dataset_quality_report.md \
  --label-dist reports/label_distribution.csv
```

The validator must exit non-zero if critical validation fails.

---

### Step 3: Create Gretel Converter First

Create:

```text
scripts/data/convert_gretel_to_securelogx.py
```

Input:

```text
Hugging Face dataset: gretelai/synthetic_pii_finance_multilingual
```

Expected behavior:

```text
load dataset using datasets.load_dataset
inspect available columns
extract generated text
extract PII spans or reconstruct spans safely if the dataset provides structured PII metadata
map source labels to SecureLogX labels
discard unsupported labels
validate offsets
preserve useful metadata
write SecureLogX JSONL
record source URL and dataset ID in data lineage report
```

Example command:

```bash
python scripts/data/convert_gretel_to_securelogx.py \
  --dataset gretelai/synthetic_pii_finance_multilingual \
  --split train \
  --language English \
  --label-map configs/securelogx_label_map.yml \
  --out data/processed/gretel_finance_securelogx.jsonl
```

The script must be defensive because field names may vary.

If required fields cannot be identified, fail with a clear message showing:

```text
available columns
sample record keys
which expected field was missing
suggested next action
```

Do not guess spans if safe reconstruction is not possible.

---

### Step 4: Create SecureLogX Synthetic Log Generator

Create:

```text
scripts/data/generate_securelogx_synthetic_logs.py
```

It must generate labeled records with exact character offsets.

Generate these formats:

```text
plain text logs
JSON logs
XML logs
key-value logs
SECURE log events
authentication logs
payment logs
customer profile logs
loan application logs
API request/response logs
error logs
technical-secret leakage logs
negative examples without PII
hard negatives with non-sensitive numbers
```

Entities to include in v1:

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

Output:

```text
data/custom/generated_logs/securelogx_generated_labeled.jsonl
data/custom/generated_logs/securelogx_generated_raw.jsonl
```

Example command:

```bash
python scripts/data/generate_securelogx_synthetic_logs.py \
  --count 25000 \
  --out-labeled data/custom/generated_logs/securelogx_generated_labeled.jsonl \
  --out-raw data/custom/generated_logs/securelogx_generated_raw.jsonl \
  --seed 42
```

Rules:

```text
labeled output must include exact character offsets.
raw output must not be merged into training.
generated examples must include realistic SecureLogX log contexts.
include negative examples with no PII.
include hard negative examples where numbers are not sensitive.
include examples across text, JSON, XML, and key-value formats.
```

---

### Step 5: Validate Converted and Generated Data

Run validation on each labeled source before merge.

Example commands:

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/processed/gretel_finance_securelogx.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/gretel_dataset_quality_report.md \
  --label-dist reports/gretel_label_distribution.csv
```

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/custom/generated_logs/securelogx_generated_labeled.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/generated_dataset_quality_report.md \
  --label-dist reports/generated_label_distribution.csv
```

Do not merge invalid data.

---

### Step 6: Create Merge Script

Create:

```text
scripts/data/merge_labeled_sources.py
```

It must:

```text
read multiple JSONL files
validate source metadata
remove duplicate text records if --dedupe is enabled
reject invalid spans
reject unsupported labels
preserve source metadata
write merged JSONL
produce data-lineage report
produce merged label-distribution summary
```

Example command:

```bash
python scripts/data/merge_labeled_sources.py \
  --in \
    data/processed/gretel_finance_securelogx.jsonl \
    data/custom/generated_logs/securelogx_generated_labeled.jsonl \
  --out data/training/all_labeled.jsonl \
  --report reports/data_lineage_report.md \
  --dedupe
```

Expected output:

```text
data/training/all_labeled.jsonl
reports/data_lineage_report.md
```

---

### Step 7: Validate Merged Dataset

Run:

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/training/all_labeled.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/dataset_quality_report.md \
  --label-dist reports/label_distribution.csv
```

Do not split if validation fails.

---

### Step 8: Create Split Script

Create:

```text
scripts/data/split_labeled_jsonl.py
```

It must split into:

```text
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl
```

Example command:

```bash
python scripts/data/split_labeled_jsonl.py \
  --in data/training/all_labeled.jsonl \
  --out data/split \
  --train 0.80 \
  --dev 0.10 \
  --test 0.10 \
  --seed 42
```

Requirements:

```text
prevent duplicate text leakage across splits.
preserve label distribution as much as practical.
write split statistics to reports/split_summary.md.
do not put the same text in more than one split.
```

---

### Step 9: Validate Final Splits

Run validation on each final split:

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/split/train.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/train_quality_report.md \
  --label-dist reports/train_label_distribution.csv
```

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/split/dev.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/dev_quality_report.md \
  --label-dist reports/dev_label_distribution.csv
```

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/split/test.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/test_quality_report.md \
  --label-dist reports/test_label_distribution.csv
```

Also verify no duplicate text appears across train/dev/test.

---

## Optional Later Steps

Only after the first milestone works.

### Ai4Privacy Converter

Create later:

```text
scripts/data/convert_ai4privacy_to_securelogx.py
```

Use:

```text
ai4privacy/open-pii-masking-500k-ai4privacy
```

Do not add this until Gretel + generated logs are working.

### GLiNER Pseudo-Labeling

Create later:

```text
scripts/data/pseudolabel_with_gliner.py
```

Use:

```text
nvidia/gliner-PII
```

Use only for:

```text
pseudo-labeling raw logs
baseline comparison
paper evaluation
annotation acceleration
```

Do not merge pseudo-labels into training unless they are reviewed or clearly marked.

### StarPII Baseline

Create later:

```text
scripts/data/run_starpii_baseline.py
```

Use only if gated access is available.

Use for:

```text
API keys
passwords
usernames
emails
IP addresses
technical/code-like logs
secret leakage baseline
```

Do not use StarPII as the SecureLogX Java runtime model.

---

## Required Data Quality Gates

Before merge:

```text
all source JSONL files parse correctly
all entity spans match text[start:end]
all labels exist in configs/securelogx_labels.json
all records have meta.source
all external records have license_reviewed field
raw unlabeled data is excluded
unsupported labels are filtered out
```

Before split:

```text
merged dataset validates successfully
duplicate text records are removed or reported
label distribution is reported
data lineage is reported
```

Before handing off to model training:

```text
train/dev/test files exist
train/dev/test files validate successfully
no duplicate text leakage exists across splits
configs/securelogx_labels.json exists
configs/training_profile.yml exists
reports/data_lineage_report.md exists
reports/dataset_quality_report.md exists
reports/label_distribution.csv exists
```

---

## What Not To Do

Do not:

```text
train a model
evaluate a trained model
export ONNX
export quantized ONNX
create Java runtime model bundle
run ONNX smoke tests
train on raw unlabeled logs
export datasets to ONNX
hard-code label index order
use GLiNER as the first Java ONNX runtime model
use StarPII as the first Java ONNX runtime model
mix source labels without normalization
merge pseudo-labels without metadata
map generic DATE to DOB blindly
treat all numeric IDs as sensitive IDs
ignore license review flags
```

---

## Final Expected Outputs for First Milestone

The first milestone is complete only when these files exist:

```text
configs/data_sources.yml
configs/securelogx_label_map.yml
configs/securelogx_labels.json
configs/training_profile.yml

data/processed/gretel_finance_securelogx.jsonl
data/custom/generated_logs/securelogx_generated_labeled.jsonl
data/custom/generated_logs/securelogx_generated_raw.jsonl
data/training/all_labeled.jsonl
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl

reports/data_lineage_report.md
reports/dataset_quality_report.md
reports/label_distribution.csv
reports/split_summary.md
reports/train_quality_report.md
reports/dev_quality_report.md
reports/test_quality_report.md
```

Do not require these files in this prompt:

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

Those are created by the later model-training/export prompt.

---

## Handoff to Model Training Prompt

After this prompt completes successfully, run:

```text
prompts/agent-prompt-python-model-training-export.md
```

The model-training prompt should start only after these files exist:

```text
data/training/all_labeled.jsonl
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl
configs/securelogx_labels.json
configs/training_profile.yml
reports/data_lineage_report.md
reports/dataset_quality_report.md
reports/label_distribution.csv
```

---

## Completion Response

When finished, provide:

```text
1. List of files created or modified.
2. Exact commands run.
3. Dataset sources used.
4. Number of records generated or converted.
5. Label distribution summary.
6. Data validation result.
7. Merge and split result.
8. Any assumptions made about dataset fields.
9. Any license/access blockers.
10. Any remaining TODOs.
11. Confirmation that model training and ONNX export were not performed.
```

Do not claim production readiness from data preparation alone.
