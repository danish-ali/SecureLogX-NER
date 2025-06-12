import json
import os
import random

def load_logs(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            logs = json.load(f)
        print(f"📥 Loaded {len(logs)} logs from {file_name}")
        return logs
    else:
        print(f"⚠️  File not found: {file_name} (skipping)")
        return []

# Load logs
base_logs         = load_logs("training_logs.json")
contrast_logs     = load_logs("contrast_logs.json")
email_logs        = load_logs("email_boost_logs.json")
xml_contrast_logs = load_logs("xml_contrast_logs.json")
clean_xml_logs    = load_logs("xml_ner_logs_cleanstyle.json")
enhanced_xml_logs = load_logs("xml_ner_logs_enhanced.json")

# Downsample base JSON logs to balance XML-heavy set
if len(base_logs) > 2000:
    base_logs = random.sample(base_logs, 2000)
    print(f"🔁 Downsampled base logs to 2000")

# Merge all logs
all_logs = base_logs + contrast_logs + email_logs + xml_contrast_logs + clean_xml_logs + enhanced_xml_logs

# Save merged output
with open("training_logs.json", "w", encoding="utf-8") as f:
    json.dump(all_logs, f, indent=2)

print(f"\n✅ Merged {len(all_logs)} total logs into training_logs.json (balanced)")
