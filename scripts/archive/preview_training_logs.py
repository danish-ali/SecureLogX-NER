import json

with open("training_logs.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

print(f"📦 Total logs: {len(logs)}")

# Find and print first 3 XML-style logs
xml_samples = [log for log in logs if log["text"].startswith("<log>")]

print(f"📄 Found {len(xml_samples)} XML-style logs")

for i, log in enumerate(xml_samples[:3]):
    print(f"\n🔹 XML Log #{i+1}:")
    print(log["text"])
    print("🧠 Entities:", log["entities"])
