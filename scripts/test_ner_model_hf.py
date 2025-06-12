import re
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Load model and tokenizer
model_dir = "securelogx-hf-ner"
model = AutoModelForTokenClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Create NER pipeline with correct aggregation
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # combine B- and I- tokens
    grouped_entities=True           # ensure merged entities
)

# Sample JSON and XML logs
json_log = '''
{
  "user": { "name": "Alice Jenkins" },
  "ssn": "123-45-6789",
  "meta": { "email": "alice.jenkins@securelogx.ai" },
  "make": "Toyota",
  "model": "Corolla",
  "education": "Software Engineer"
}
'''

xml_log = '''
<log>
  <name>Alice Jenkins</name>
  <email>alice.jenkins@securelogx.ai</email>
  <ssn>123-45-6789</ssn>
  <npi>1234567890</npi>
  <education>Software Engineer<education>
</log>
'''

def clean_input(text, label):
    if label.lower().startswith("xml"):
        return re.sub(r"</?[^>]+>", "", text)
    return text

def run_test(log_text, label):
    cleaned = clean_input(log_text, label)
    print(f"\n🔎 Detected Entities in {label}:")
    results = ner(cleaned)
    for ent in results:
        print(f"{ent['word']} → {ent['entity_group']} (score: {ent['score']:.2f})")

# Run tests
run_test(json_log, "JSON log")
run_test(xml_log, "XML log")
