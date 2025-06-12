import json
from pathlib import Path
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

# Load your spaCy-style file
with open("hf_data/training_logs.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Extract label set
entity_labels = set()
for example in raw_data:
    for ent in example.get("entities", []):
        entity_labels.add(ent["label"])

# Create BIO labels
bio_labels = ["O"]
for label in sorted(entity_labels):
    bio_labels.extend([f"B-{label}", f"I-{label}"])
label2id = {label: i for i, label in enumerate(bio_labels)}

# Conversion logic
def convert_to_bio(example):
    text = example["text"]
    entity_ranges = [(e["start"], e["end"], e["label"]) for e in example.get("entities", [])]

    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])[1:-1]
    offsets = encoding["offset_mapping"][1:-1]

    tags = []
    for start, end in offsets:
        tag = "O"
        for ent_start, ent_end, label in entity_ranges:
            if start == ent_start:
                tag = f"B-{label}"
                break
            elif ent_start < start < ent_end:
                tag = f"I-{label}"
                break
        tags.append(tag)

    return {
        "tokens": tokens,
        "ner_tags": tags
    }

# Convert all
converted = [convert_to_bio(example) for example in raw_data]

# Save
Path("hf_ready").mkdir(exist_ok=True)
with open("hf_ready/train_bio.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2)

print("✅ Hugging Face-compatible training data saved to: hf_ready/train_bio.json")
