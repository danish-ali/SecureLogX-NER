import json
import random
from faker import Faker

faker = Faker()

# Sensitive fields to be labeled
sensitive_fields = {
    "name": "PERSON",
    "email": "EMAIL",
    "ssn": "SSN",
    "address": "ADDRESS",
    "npi": "NPI"
}

# Non-sensitive fields (not labeled)
nonsensitive_fields = ["make", "model", "year", "education", "occupation"]

training_data = []

def random_case(word):
    style = random.choice(["lower", "upper", "camel"])
    if style == "lower":
        return word.lower()
    elif style == "upper":
        return word.upper()
    elif style == "camel":
        return word[0].upper() + word[1:].lower()
    return word

def add_entity(value, label, text, entities):
    try:
        start = text.index(value)
        end = start + len(value)
        entities.append({ "start": start, "end": end, "label": label })
    except ValueError:
        print(f"⚠️ Could not find value: {value} in text")

def extract_fields(obj):
    flat = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            flat.update(extract_fields(value))
        else:
            flat[key] = value
    return flat

for i in range(1000):
    log = {}
    entities = []

    # 👉 Inject clean noise-only logs every 20 records
    if i % 20 == 0:
        log["education"] = faker.job()
        log["occupation"] = faker.job()
        log["make"] = faker.company()
        log["model"] = faker.word().capitalize()
        log["year"] = str(faker.year())
    else:
        for field in sensitive_fields:
            if random.random() < 0.1 and field != "email":
                continue  # ensure email is more frequent

            key = random_case(field)

            if field == "name":
                log[key] = faker.name()
            elif field == "email":
                log[key] = faker.email().replace("\"", "")
            elif field == "ssn":
                log[key] = faker.ssn()
            elif field == "address":
                log[key] = faker.address().replace("\n", ", ")
            elif field == "npi":
                log[key] = str(faker.random_number(digits=10, fix_len=True))

            if field == "ssn" and random.random() < 0.3:
                alias = random_case("socialSecurityNumber")
                log[alias] = log[key]

        # Always include noise fields
        log["education"] = faker.job()
        log["occupation"] = faker.job()
        log["make"] = faker.company()
        log["model"] = faker.word().capitalize()
        log["year"] = str(faker.year())

        # Optional nesting
        if "name" in log and random.random() < 0.5:
            log = { "user": { "name": log.pop("name") }, **log }

        if "email" in log and random.random() < 0.5:
            log = { **log, "meta": { "email": log.pop("email") } }

    # Serialize to text
    json_text = json.dumps(log)
    flat_fields = extract_fields(log)

    for key, value in flat_fields.items():
        label = sensitive_fields.get(key.lower())
        if label:
            add_entity(str(value), label, json_text, entities)

    training_data.append({
        "text": json_text,
        "entities": entities
    })

# Save final file
with open("training_logs.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2)

print("✅ 1000 training logs saved to training_logs.json (with boosted EMAIL and job-title controls)")
