@echo off
REM Create folders
mkdir scripts
mkdir scripts\archive

REM Move essential scripts
move "train_and_export_bert.py" scripts\
move "test_ner_model_hf.py" scripts\
move "quantize_model.py" scripts\
move "merge_all_logs.py" scripts\
move "generate_json_training_logs.py" scripts\
move "generate_xml_ner_logs.py" scripts\
move "generate_xml_contrast_logs.py" scripts\
move "sanitize_xml_training_logs.py" scripts\
move "training_logs_hf.py" scripts\
move "convert_to_bio.py" scripts\

REM Move optional/reference scripts
move "onnx_export.py" scripts\archive\
move "tokinzer_generate.py" scripts\archive\
move "train_ner_hf_logs.py" scripts\archive\
move "convert_spacy_to_hf.py" scripts\archive\
move "split_training_data.py" scripts\archive\
move "preview_training_logs.py" scripts\archive\

REM Delete unnecessary scripts
del "merge_email_logs.py"
del "merge_logs.py"
del "test_model.py"
del "preview_docbin.py"
del "save_to_docbin.py"
del "json_to_docbin.py"
del "json_log_to_training_data.py"
del "log_training_data.py"

echo ✅ Cleanup complete!
pause
