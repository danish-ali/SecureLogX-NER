import numpy as np
import onnxruntime as ort

model_path = "onnx-model/securelogx-ner-quantized.onnx"

# Create dummy inputs (batch size 1, sequence length 128)
input_ids = np.zeros((1, 128), dtype=np.int64)
attention_mask = np.ones((1, 128), dtype=np.int64)

session = ort.InferenceSession(model_path)
outputs = session.run(None, {
    "input_ids": input_ids,
    "attention_mask": attention_mask
})

print("✅ Output shape:", outputs[0].shape)
