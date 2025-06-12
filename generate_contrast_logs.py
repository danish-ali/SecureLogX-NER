import json
from faker import Faker

faker = Faker()

job_titles = [
    "Software Engineer", "Education Administrator", "Project Manager",
    "Lecturer", "Data Analyst", "Marketing Specialist", "IT Consultant",
    "Researcher", "Medical Officer", "Support Engineer"
]

contrast_logs = []

for _ in range(100):
    name = faker.name()
    occupation = faker.random_element(elements=job_titles)

    # Build log
    log = {
        "name": name,
        "occupation": occupation
    }

    # Create training record
    json_text = json.dumps(log)
    name_start = json_text.index(name)
    name_end = name_start + len(name)

    contrast_logs.append({
        "text": json_text,
        "entities": [
            { "start": name_start, "end": name_end, "label": "PERSON" }
        ]
    })

# Save to file
with open("contrast_logs.json", "w", encoding="utf-8") as f:
    json.dump(contrast_logs, f, indent=2)

print("✅ 100 contrastive examples saved to contrast_logs.json")
