import os
import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support

def main():
    # === ✅ CPU Optimization Settings (3 Cores Max) ===
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_MAX_THREADS"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # === ✅ Load Data ===
    with open("hf_ready/train_bio.json", "r") as f:
        raw_data = json.load(f)
    print(f"✅ Loaded {len(raw_data)} examples")

    dataset = Dataset.from_list(raw_data)

    # === ✅ Create Label Mappings ===
    label_list = sorted(list(set(tag for row in raw_data for tag in row["ner_tags"])))

    # Ensure both B- and I- labels are included in label set
    expanded_labels = set()
    for label in label_list:
        expanded_labels.add(label)
        if label.startswith("B-"):
            i_label = label.replace("B-", "I-")
            expanded_labels.add(i_label)

    label_list = sorted(expanded_labels)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    # === ✅ Tokenizer and Model ===
    model_name = "albert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # === ✅ Tokenization with Alignment ===
    def tokenize(example):
        tokenized_inputs = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length"
        )

        word_ids = tokenized_inputs.word_ids()
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[example["ner_tags"][word_idx]])
            else:
                current_label = example["ner_tags"][word_idx]
                if current_label.startswith("B-"):
                    current_label = current_label.replace("B-", "I-")
                label_ids.append(label2id.get(current_label, -100))
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs

    dataset = dataset.map(tokenize)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # === ✅ Data Collator ===
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # === ✅ Metrics ===
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids
        preds_flat, labels_flat = [], []

        for pred, label in zip(preds, labels):
            for p_, l_ in zip(pred, label):
                if l_ != -100:
                    preds_flat.append(p_)
                    labels_flat.append(l_)

        precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average="macro")
        print(f"📊 Metrics -> F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
        return {"precision": precision, "recall": recall, "f1": f1}

    # === ✅ Training Arguments ===
    args = TrainingArguments(
        output_dir="securelogx-hf-ner",
        per_device_train_batch_size=2,
        max_steps=600,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        disable_tqdm=False,
        save_strategy="epoch",
        evaluation_strategy="no",
        dataloader_num_workers=2,
        load_best_model_at_end=False
    )

    # === ✅ Trainer ===
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("securelogx-hf-ner")
    tokenizer.save_pretrained("securelogx-hf-ner")
    print("✅ Model training complete and saved to securelogx-hf-ner")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()