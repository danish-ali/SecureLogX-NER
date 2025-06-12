# generate_xml_contrast_logs.py
import json
from faker import Faker

faker = Faker()

nonsensitive_fields = ["education", "occupation", "salary", "make", "model"]
xml_contrast_logs = []

for _ in range(100):
    log_text = "<log>"

    for field in nonsensitive_fields:
        if field in ["education", "occupation"]:
            value = faker.job()
        elif field == "salary":
            value = f"${faker.random_int(min=40000, max=150000)}"
        elif field == "make":
            value = faker.company()
        elif field == "model":
            value = faker.word().capitalize()

        tag = f"<{field}>{value}</{field}>"
        log_text += tag

    log_text += "</log>"

    xml_contrast_logs.append({
        "text": log_text,
        "entities": []
    })

with open("xml_contrast_logs.json", "w", encoding="utf-8") as f:
    json.dump(xml_contrast_logs, f, indent=2)

print("✅ Saved 100 contrastive XML logs to xml_contrast_logs.json")
