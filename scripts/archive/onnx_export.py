import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

# === ✅ Paths ===
model_dir = "securelogx-hf-ner"
onnx_output_path = Path("onnx-model")
onnx_output_path.mkdir(parents=True, exist_ok=True)

# === ✅ Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)

# === ✅ Prepare ONNX export config ===
model_kind, onnx_config_class = FeaturesManager.check_supported_model_or_raise(model, feature="token-classification")
onnx_config = onnx_config_class(config)

# === ✅ Export ===
export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=14,
    output=onnx_output_path / "securelogx-ner.onnx"
)

print("✅ ONNX export complete → onnx-model/securelogx-ner.onnx")
