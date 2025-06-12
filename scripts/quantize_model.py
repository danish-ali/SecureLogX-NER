import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model():
    input_model = "onnx-model/securelogx-ner.onnx"
    output_model = "onnx-model/securelogx-ner-quantized.onnx"

    if not os.path.exists(input_model):
        raise FileNotFoundError(f"Model not found: {input_model}")

    print("🚀 Starting quantization...")

    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QInt8  # can try QUInt8 if needed
    )

    print(f"✅ Quantized model saved to: {output_model}")

if __name__ == "__main__":
    quantize_onnx_model()
