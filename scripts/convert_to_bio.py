import json
import re
import os
from pathlib import Path
import spacy

# Load spaCy tokenizer (English)
nlp = spacy.blank("en")

def char_to_bio(text, entities):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    char_spans = [(token.idx, token.idx + len(token.text)) for token in doc]

    bio_tags = ["O"] * len(tokens)

    for entity in entities:
        ent_start, ent_end, label = entity["start"], entity["end"], entity["label"]
        for i, (start, end) in enumerate(char_spans):
            if start >= ent_start and end <= ent_end:
                tag_prefix = "B-" if start == ent_start else "I-"
                bio_tags[i] = f"{tag_prefix}{label}"

    return tokens, bio_tags

def convert_logs_to_bio(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    hf_ready_data = []
    for item in data:
        text = item["text"]
        entities = item.get("entities", [])
        tokens, ner_tags = char_to_bio(text, entities)
        hf_ready_data.append({"tokens": tokens, "ner_tags": ner_tags})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(hf_ready_data, out, indent=2)

    print(f"✅ Converted {len(hf_ready_data)} examples to BIO format in {output_path}")

if __name__ == "__main__":
    input_file = "training_logs.json"            # Your SecureLogX-style annotated logs
    output_file = "hf_ready/train_bio.json"      # Hugging Face-compatible output
    convert_logs_to_bio(input_file, output_file)
