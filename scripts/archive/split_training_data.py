import json
import random

with open("training_logs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)
split_index = int(len(data) * 0.8)

train_data = data[:split_index]
dev_data = data[split_index:]

with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

with open("dev.json", "w", encoding="utf-8") as f:
    json.dump(dev_data, f, indent=2)

print("✅ Split into train.json and dev.json")
