import json
from faker import Faker
import random

faker = Faker()
email_logs = []

# Add common and realistic domains
domains = [
    "securelogx.ai", "gmail.com", "yahoo.com", "outlook.com",
    "hotmail.com", "aol.com", "clinic.org", "example.com"
]

for _ in range(100):
    username = faker.user_name()
    domain = random.choice(domains)
    email = f"{username}@{domain}"

    log = { "email": email }
    json_text = json.dumps(log)

    try:
        start = json_text.index(email)
        end = start + len(email)
        email_logs.append({
            "text": json_text,
            "entities": [
                { "start": start, "end": end, "label": "EMAIL" }
            ]
        })
    except ValueError:
        print(f"❌ Could not find span for: {email}")

# Save to file
with open("email_boost_logs.json", "w", encoding="utf-8") as f:
    json.dump(email_logs, f, indent=2)

print("✅ 100 email training samples written to email_boost_logs.json")
