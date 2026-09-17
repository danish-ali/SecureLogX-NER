# SecureLogX Data Acquisition and Dataset Preparation Scenarios

## 1. Purpose

This document defines the **Python-side data acquisition and dataset preparation workflow** for SecureLogX.

This file is intentionally limited to the data layer. It prepares validated training-ready data for a later BERT-family Hugging Face token-classification pipeline.

This document owns:

```text
1. Dataset source configuration
2. External labeled dataset conversion
3. Custom SecureLogX synthetic data generation
4. Optional pseudo-labeling strategy
5. Canonical SecureLogX JSONL format
6. Label mapping and label normalization
7. Span validation
8. Merge and split preparation
9. Data lineage and dataset quality reporting
```

This document does **not** own:

```text
model training
model evaluation
ONNX export
quantized ONNX export
tokenizer export
Java runtime implementation
Java ONNX Runtime implementation
SecureLogX Java masking engine implementation
```

Those belong to later documents:

```text
docs/python-model-training-export.md
docs/java-runtime-integration-contract.md
```

Important rule:

```text
Data is not exported to ONNX.
Only a trained model is exported to ONNX.
```

---

## 2. Scope Boundary

The purpose of this file is to produce the following handoff artifacts:

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

The later model-training/export prompt will consume those files and produce:

```text
output_securelogx/model-best/
onnx/securelogx.onnx
onnx/tokenizer.json
onnx/labels.json
onnx/securelogx-model-metadata.json
```

Do not generate model outputs from this data-preparation phase.

---

## 3. Relationship to the Model Training Document

Use this document first.

Execution order:

```text
1. docs/data-acquisition-training-scenarios.md
2. prompts/agent-prompt-data-acquisition-training.md
3. docs/python-model-training-export.md
4. prompts/agent-prompt-python-model-training-export.md
```

The data acquisition phase prepares the dataset. The model training phase trains and exports the model.

---

## 4. Target Model Compatibility

The dataset produced by this pipeline must be compatible with a later **BERT-family Hugging Face token-classification model**.

Default later model:

```text
distilbert-base-uncased
```

Later model task:

```text
token-classification
```

Required label format:

```text
BIO
```

Required annotation format:

```text
character spans in [start, end, label] format
```

This data pipeline must not assume GLiNER, StarPII, spaCy, or sentence-transformer runtime behavior.

---

## 5. Scenario Classification

SecureLogX data can come from three types of sources.

| Scenario | Input Type | Examples | Use in This Phase | Output |
|---|---|---|---|---|
| A | Pretrained/fine-tuned models | `nvidia/gliner-PII`, `bigcode/starpii` | Optional pseudo-labeling or baseline preparation only | Reviewed JSONL or reports |
| B | Labeled datasets | Gretel Finance PII, Ai4Privacy Open PII | Convert to SecureLogX canonical JSONL | Validated labeled JSONL |
| C | Raw/custom SecureLogX data | logs, JSON, XML, API payloads | Generate, normalize, manually label, or pseudo-label + review | Validated labeled JSONL |

Recommended first milestone:

```text
Gretel Finance PII
+
SecureLogX generated synthetic logs
+
canonical JSONL
+
validated train/dev/test split
```

---

## 6. Recommended Repository Structure

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
    │   ├── check_data_source_readiness.py
    │   ├── convert_gretel_to_securelogx.py
    │   ├── convert_ai4privacy_to_securelogx.py
    │   ├── generate_securelogx_synthetic_logs.py
    │   ├── pseudolabel_with_gliner.py
    │   ├── run_starpii_baseline.py
    │   ├── validate_jsonl_spans.py
    │   ├── merge_labeled_sources.py
    │   └── split_labeled_jsonl.py
    └── rules/
        ├── deterministic_recognizers.py
        └── luhn_card_validator.py
```

Do not create or modify these in this phase:

```text
scripts/train/
onnx/
output_securelogx/
src/main/java/
pom.xml
build.gradle
```

---

## 7. Data Source Inventory

Create:

```text
configs/data_sources.yml
```

Suggested content:

```yaml
sources:
  - id: gretel_finance_pii
    kind: labeled_dataset
    provider: huggingface
    name: gretelai/synthetic_pii_finance_multilingual
    source_url: https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual
    enabled: true
    use_for_training_data: true
    use_for_baseline: false
    requires_license_review: true
    license_reviewed: false
    priority: high
    notes: "Primary first external dataset for finance/banking-style PII conversion."

  - id: ai4privacy_open_pii_500k
    kind: labeled_dataset
    provider: huggingface
    name: ai4privacy/open-pii-masking-500k-ai4privacy
    source_url: https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy
    enabled: false
    use_for_training_data: true
    use_for_baseline: false
    requires_license_review: true
    license_reviewed: false
    priority: medium
    notes: "Add after Gretel converter and generated SecureLogX logs are stable."

  - id: securelogx_custom_logs
    kind: raw_or_generated_data
    provider: local
    path: data/custom/
    enabled: true
    use_for_training_data: true
    requires_labeling: true
    license_reviewed: true
    priority: critical
    notes: "Most important source for SecureLogX originality."

  - id: nvidia_gliner_pii
    kind: pretrained_model
    provider: huggingface
    name: nvidia/gliner-PII
    source_url: https://huggingface.co/nvidia/gliner-PII
    enabled: false
    use_for_pseudolabeling: true
    use_for_baseline: true
    use_as_java_runtime_model: false
    requires_license_review: true
    license_reviewed: false
    priority: later
    notes: "Use only later for pseudo-labeling or baseline comparison."

  - id: bigcode_starpii
    kind: pretrained_model
    provider: huggingface
    name: bigcode/starpii
    source_url: https://huggingface.co/bigcode/starpii
    enabled: false
    gated: true
    use_for_pseudolabeling: false
    use_for_baseline: true
    use_as_java_runtime_model: false
    requires_license_review: true
    license_reviewed: false
    priority: later
    notes: "Use only after gated access is available. Best for code-like secret leakage baseline."
```

Rules:

```text
Record source URL and dataset/model ID in reports.
Do not mark license_reviewed as true unless explicitly approved.
Do not use disabled sources unless the user explicitly enables them.
```

---

## 8. Canonical SecureLogX Entity Set

Create:

```text
configs/securelogx_labels.json
```

Use a stable internal label taxonomy. Source labels must be mapped to this taxonomy before training.

Production-critical v1 entity set:

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

Important:

```text
Do not train every possible sensitive label in v1.
Prefer high-quality coverage for fewer critical labels.
```

---

## 9. Label Mapping File

Create:

```text
configs/securelogx_label_map.yml
```

Suggested content:

```yaml
gretel_finance:
  name: PERSON_NAME
  first_name: FIRST_NAME
  last_name: LAST_NAME
  date_of_birth: DOB
  date: null
  time: null
  email: EMAIL
  phone_number: PHONE
  street_address: ADDRESS_LINE_1
  zipcode: ZIP_CODE
  ssn: SSN
  credit_card_number: CREDIT_CARD_NUMBER
  credit_card_security_code: CVV
  bank_routing_number: ROUTING_NUMBER
  bban: BANK_ACCOUNT_NUMBER
  iban: BANK_ACCOUNT_NUMBER
  customer_id: CUSTOMER_ID
  account_id: ACCOUNT_ID
  employee_id: USER_ID
  user_name: USER_ID
  password: PASSWORD
  account_pin: PIN
  api_key: API_KEY
  ipv4: IP_ADDRESS
  ipv6: IP_ADDRESS
  driver_license_number: DRIVER_LICENSE
  passport_number: PASSPORT_NUMBER
  company: null
  job: null

ai4privacy:
  GIVENNAME: FIRST_NAME
  SURNAME: LAST_NAME
  MIDDLENAME: null
  FULLNAME: PERSON_NAME
  DATE: null
  TIME: null
  TELEPHONENUM: PHONE
  EMAIL: EMAIL
  STREET: ADDRESS_LINE_1
  CITY: null
  STATE: null
  ZIPCODE: ZIP_CODE
  CREDITCARDNUMBER: CREDIT_CARD_NUMBER
  CREDITCARDCVV: CVV
  ACCOUNTNUM: BANK_ACCOUNT_NUMBER
  SOCIALNUM: SSN
  DRIVERLICENSENUM: DRIVER_LICENSE
  IDCARDNUM: null
  PASSPORTNUM: PASSPORT_NUMBER
  USERNAME: USER_ID
  PASSWORD: PASSWORD
  IPV4: IP_ADDRESS
  IPV6: IP_ADDRESS

gliner_optional:
  person name: PERSON_NAME
  email: EMAIL
  phone number: PHONE
  date of birth: DOB
  social security number: SSN
  credit card number: CREDIT_CARD_NUMBER
  bank account number: BANK_ACCOUNT_NUMBER
  address: ADDRESS_LINE_1
  api key: API_KEY
  password: PASSWORD
  ip address: IP_ADDRESS

starpii_optional:
  EMAIL: EMAIL
  KEY: API_KEY
  PASSWORD: PASSWORD
  USERNAME: USER_ID
  IP_ADDRESS: IP_ADDRESS
```

Rules:

```text
null means do not use this source label for SecureLogX training.
Do not map generic DATE to DOB unless the source explicitly means date of birth.
Do not map company to PERSON_NAME.
Do not map all numeric values to sensitive IDs.
Do not map random 3-digit numbers to CVV unless the context is card security code.
```

---

## 10. Canonical SecureLogX JSONL Format

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

Required rules:

```text
text must contain the exact string used for offsets.
start offset is inclusive.
end offset is exclusive.
text[start:end] must match the entity value.
entities must not overlap unless explicitly supported.
labels must exist in configs/securelogx_labels.json.
raw unlabeled records must not be merged into training data.
external-source records must include source metadata.
license_reviewed must exist.
review_status must exist.
```

Recommended review statuses:

```text
auto_converted
synthetic_generated
manual_reviewed
pseudolabeled
pseudolabeled_reviewed
rejected
```

Only these should be allowed into final training data by default:

```text
auto_converted
synthetic_generated
manual_reviewed
pseudolabeled_reviewed
```

Do not merge plain `pseudolabeled` records without review unless the user explicitly approves it.

---

## 11. Scenario B: Labeled Dataset Workflow

### 11.1 Gretel Finance PII — First External Dataset

Create:

```text
scripts/data/convert_gretel_to_securelogx.py
```

Input dataset:

```text
gretelai/synthetic_pii_finance_multilingual
```

Purpose:

```text
primary first public/synthetic dataset for banking and finance-style PII
finance-domain entity coverage
first milestone dataset
```

Expected behavior:

```text
load dataset using datasets.load_dataset
inspect available columns
extract generated text
extract PII spans when available
map source labels to SecureLogX labels
discard unsupported labels
validate offsets
preserve source URL, dataset ID, split, language, and document type in metadata when available
write converted output to data/processed/gretel_finance_securelogx.jsonl
record source URL and dataset ID in reports/data_lineage_report.md
```

Suggested command:

```bash
python scripts/data/convert_gretel_to_securelogx.py \
  --dataset gretelai/synthetic_pii_finance_multilingual \
  --split train \
  --language English \
  --label-map configs/securelogx_label_map.yml \
  --out data/processed/gretel_finance_securelogx.jsonl
```

The converter must be defensive.

If required fields cannot be identified, fail with a clear message showing:

```text
available columns
sample record keys
which field was missing
suggested next action
```

Do not guess spans if safe reconstruction is not possible.

---

### 11.2 Ai4Privacy Open PII — Later Dataset

Create later only after the Gretel converter is stable:

```text
scripts/data/convert_ai4privacy_to_securelogx.py
```

Input dataset:

```text
ai4privacy/open-pii-masking-500k-ai4privacy
```

Purpose:

```text
general PII training data
additional entity coverage
label normalization testing
```

Suggested command:

```bash
python scripts/data/convert_ai4privacy_to_securelogx.py \
  --dataset ai4privacy/open-pii-masking-500k-ai4privacy \
  --split train \
  --language en \
  --label-map configs/securelogx_label_map.yml \
  --out data/processed/ai4privacy_securelogx.jsonl
```

Rules:

```text
Inspect available columns before conversion.
Prefer character-span annotations if available.
Fallback to token labels only if character spans are not available and conversion is reliable.
Map source labels to SecureLogX labels.
Do not map generic DATE to DOB.
Record source metadata.
```

---

## 12. Scenario C: Custom SecureLogX Data Workflow

This is the most important source for SecureLogX originality.

Create:

```text
scripts/data/generate_securelogx_synthetic_logs.py
```

The generator should produce both labeled and raw outputs:

```text
data/custom/generated_logs/securelogx_generated_labeled.jsonl
data/custom/generated_logs/securelogx_generated_raw.jsonl
```

Generate realistic examples across:

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
The labeled output must include exact character offsets.
The raw output must not be merged into training.
Generated examples must include realistic SecureLogX contexts.
Include both SECURE and non-SECURE style text.
Include hard negatives such as order IDs, status codes, timestamps, HTTP codes, amounts, and counts.
```

---

## 13. Scenario A: Pretrained Model Workflow

Pretrained models are optional and later-stage only in this data document.

### 13.1 GLiNER-PII

Purpose:

```text
pseudo-label custom raw SecureLogX logs
baseline comparison
annotation acceleration
```

Do not use GLiNER as the Java runtime model.

Create later:

```text
scripts/data/pseudolabel_with_gliner.py
```

Expected behavior:

```text
load nvidia/gliner-PII
run inference on raw/custom logs
map predictions to SecureLogX labels
write pseudo-labeled output
write rejects/uncertain output
mark records as review_status=pseudolabeled
do not merge into training until reviewed
```

Required metadata for pseudo-labeled records:

```json
{
  "meta": {
    "source": "securelogx_custom_logs",
    "pseudo_label_model": "nvidia/gliner-PII",
    "pseudo_label_threshold": 0.30,
    "review_status": "pseudolabeled"
  }
}
```

### 13.2 StarPII

Purpose:

```text
baseline for code-like logs and secret leakage
```

Do not use StarPII as the Java runtime model.

Create later only if gated access is available:

```text
scripts/data/run_starpii_baseline.py
```

---

## 14. Span Validator

Create:

```text
scripts/data/validate_jsonl_spans.py
```

It must validate:

```text
JSONL parse correctness
required keys: text, entities, meta
entity offset correctness
text[start:end] exact match when entity value is available
label existence against configs/securelogx_labels.json
overlapping spans
empty text
duplicate entities
duplicate text records
metadata presence
license_reviewed field for external records
review_status field
```

Example command:

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/processed/gretel_finance_securelogx.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/gretel_dataset_quality_report.md \
  --label-dist reports/gretel_label_distribution.csv
```

The validator must exit non-zero if critical validation fails.

---

## 15. Merge Workflow

Create:

```text
scripts/data/merge_labeled_sources.py
```

Expected behavior:

```text
read multiple JSONL files
validate source metadata
reject invalid spans
reject unsupported labels
remove duplicate text records when --dedupe is enabled
preserve source metadata
write merged JSONL
produce data-lineage report
produce merged label-distribution summary
```

Suggested first merge:

```bash
python scripts/data/merge_labeled_sources.py \
  --in \
    data/processed/gretel_finance_securelogx.jsonl \
    data/custom/generated_logs/securelogx_generated_labeled.jsonl \
  --out data/training/all_labeled.jsonl \
  --report reports/data_lineage_report.md \
  --dedupe
```

Do not merge:

```text
raw unlabeled files
plain pseudolabeled files without review
records with invalid offsets
records with unsupported labels
records missing source metadata
```

---

## 16. Split Workflow

Create:

```text
scripts/data/split_labeled_jsonl.py
```

Expected behavior:

```text
split data/training/all_labeled.jsonl into train/dev/test
preserve label distribution as much as practical
prevent duplicate text leakage across splits
write reports/split_summary.md
```

Suggested command:

```bash
python scripts/data/split_labeled_jsonl.py \
  --in data/training/all_labeled.jsonl \
  --out data/split \
  --train 0.80 \
  --dev 0.10 \
  --test 0.10 \
  --seed 42
```

Expected outputs:

```text
data/split/train.jsonl
data/split/dev.jsonl
data/split/test.jsonl
reports/split_summary.md
```

---

## 17. Validation Sequence

Run validation in this order.

### 17.1 Validate Gretel Converted Data

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/processed/gretel_finance_securelogx.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/gretel_dataset_quality_report.md \
  --label-dist reports/gretel_label_distribution.csv
```

### 17.2 Validate Generated SecureLogX Data

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/custom/generated_logs/securelogx_generated_labeled.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/generated_dataset_quality_report.md \
  --label-dist reports/generated_label_distribution.csv
```

### 17.3 Validate Merged Data

```bash
python scripts/data/validate_jsonl_spans.py \
  --in data/training/all_labeled.jsonl \
  --labels configs/securelogx_labels.json \
  --report reports/dataset_quality_report.md \
  --label-dist reports/label_distribution.csv
```

### 17.4 Validate Final Splits

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

---

## 18. Data Quality Gates

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

Before handoff to model training:

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

## 19. Training Profile Handoff

Create:

```text
configs/training_profile.yml
```

This is only a handoff file. It does not trigger model training.

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

## 20. Final Expected Outputs

This data phase is complete only when these exist:

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

Do not require these files from this phase:

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

## 21. Completion Response

When complete, report:

```text
1. Files created or modified
2. Exact commands run
3. Dataset sources used
4. Number of records generated or converted
5. Label distribution summary
6. Data validation result
7. Merge and split result
8. Assumptions about dataset fields
9. License/access blockers
10. Remaining TODOs
11. Confirmation that model training and ONNX export were not performed
```

Do not claim production readiness from data preparation alone.
