import spacy
import json
import sys
from spacy.tokens import DocBin

input_json = sys.argv[1]
output_spacy = sys.argv[2]

nlp = spacy.blank("en")
doc_bin = DocBin()

with open(input_json, "r", encoding="utf-8") as f:
    training_data = json.load(f)

for item in training_data:
    text = item["text"]
    entities = item["entities"]
    doc = nlp.make_doc(text)
    ents = []
    for ent in entities:
        span = doc.char_span(ent["start"], ent["end"], label=ent["label"])
        if span:
            ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

doc_bin.to_disk(output_spacy)
print(f"✅ Converted {input_json} to {output_spacy}")
