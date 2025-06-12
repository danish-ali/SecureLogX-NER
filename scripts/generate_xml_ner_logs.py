import json
import re
from faker import Faker
import random

faker = Faker()

sensitive_fields = {
    "name": "PERSON",
    "email": "EMAIL",
    "ssn": "SSN",
    "npi": "NPI"
}

contrastive_fields = {
    "address": "O"  # Not labeled — helps reduce false positives
}

logs = []

for _ in range(1000):
    xml_parts = []
    entity_spans = []

    # Shuffle field order to increase generalization
    fields = list(sensitive_fields.items()) + list(contrastive_fields.items())
    random.shuffle(fields)

    # Construct XML string
    for field, label in fields:
        if field == "name":
            value = faker.name()
        elif field == "email":
            value = faker.email()
        elif field == "ssn":
            value = faker.ssn()
        elif field == "npi":
            value = str(faker.random_number(digits=10, fix_len=True))
        elif field == "address":
            value = faker.address().replace("\n", ", ")
        else:
            value = faker.word()

        xml_parts.append(f"<{field}>{value}</{field}>")

    # Join with spacing to ensure token boundaries
    xml_text = " ".join(xml_parts)

    # Create plain text (without tags)
    plain_text = re.sub(r"</[^>]+>", " ", xml_text)  # replace closing tags with space
    plain_text = re.sub(r"<[^>]+>", "", plain_text)  # remove opening tags
    plain_text = re.sub(r"\s+", " ", plain_text).strip()  # normalize whitespace

    # Extract entities again to compute offsets
    offset = 0
    for match in re.finditer(r"<([^>]+)>([^<]+)</\1>", xml_text):
        tag = match.group(1)
        value = match.group(2)
        label = sensitive_fields.get(tag)

        if label:
            start = plain_text.find(value, offset)
            end = start + len(value)
            offset = end
            entity_spans.append({
                "start": start,
                "end": end,
                "label": label
            })

    logs.append({
        "text": plain_text,
        "entities": entity_spans
    })

# Save to file
with open("xml_ner_logs_enhanced.json", "w", encoding="utf-8") as f:
    json.dump(logs, f, indent=2)

print("✅ Enhanced XML-style NER logs written to xml_ner_logs_enhanced.json")
