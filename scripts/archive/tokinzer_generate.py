from transformers import AutoTokenizer

model_dir = "securelogx-hf-ner"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 🔄 Save tokenizer.json explicitly
tokenizer.save_pretrained(model_dir, legacy_format=False)

print("✅ tokenizer.json generated in:", model_dir)
