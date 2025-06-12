@echo off
echo ==================================================
echo 🚀 Starting SecureLogX NER Training Pipeline (Windows)
echo ==================================================

:: 1. Merge all logs into training_logs.json
echo [1/5] 🔄 Merging logs...
python merge_all_logs.py

:: 2. Split into train/dev JSON sets
echo [2/5] ✂️ Splitting training and dev sets...
python split_training_data.py

:: 3. Convert to spaCy binary format
echo [3/5] 📦 Converting JSON to .spacy format...
python json_to_docbin.py train.json ./data/train.spacy
python json_to_docbin.py dev.json ./data/dev.spacy

:: 4. Train the model
echo [4/5] 🧠 Training the NER model...
python -m spacy train config_trf.cfg --output ./output --paths.train ./data/train.spacy --paths.dev ./data/dev.spacy
::python -m spacy init config config_trf.cfg --lang en --pipeline transformer,ner --optimize accuracy --force --gpu

:: 5. Test with real logs
echo [5/5] 🔍 Running model on test log sample...
python test_model.py

echo ==================================================
echo ✅ Pipeline Complete! Model trained and tested.
echo ==================================================
pause
