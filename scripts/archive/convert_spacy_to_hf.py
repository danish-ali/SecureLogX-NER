import spacy
from spacy.tokens import DocBin
import json
from pathlib import Path

def convert_spacy_to_hf(spacy_file_path, output_json_path):
    print(f"🔄 Loading spaCy file: {spacy_file_path}")
    nlp = spacy.blank("en")  # Dummy pipeline
    doc_bin = DocBin().from_disk(spacy_file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    hf_data = []

    for doc in docs:
        tokens = [token.text for token in doc]
        labels = ["O"] * len(doc)

        for ent in doc.ents:
            start, end = ent.start, ent.end
            labels[start] = f"B-{ent.label_}"
            for i in range(start + 1, end):
                labels[i] = f"I-{ent.label_}"

        hf_data.append({
            "tokens": tokens,
            "ner_tags": labels
        })

    print(f"✅ Writing to {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(hf_data, f, indent=2)

# Convert both train and dev sets
Path("hf_data").mkdir(exist_ok=True)
convert_spacy_to_hf("data/train.spacy", "hf_data/train.json")
convert_spacy_to_hf("data/dev.spacy", "hf_data/dev.json")
