#!/usr/bin/env bash
# Download MobileNetV2 in all required formats
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR"

echo "=== Downloading MobileNetV2 models ==="

# --- TensorFlow Lite (float32, 224x224) ---
TFLITE_URL="https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1?lite-format=tflite"
TFLITE_PATH="$MODELS_DIR/mobilenet_v2_1.0_224.tflite"
if [ ! -f "$TFLITE_PATH" ]; then
    echo "[TFLite] Downloading..."
    curl -sSL -o "$TFLITE_PATH" "$TFLITE_URL"
    echo "[TFLite] Saved to $(basename $TFLITE_PATH)"
else
    echo "[TFLite] Already exists, skipping"
fi

# --- ONNX (MobileNetV2 from ONNX Model Zoo) ---
ONNX_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
ONNX_PATH="$MODELS_DIR/mobilenetv2-7.onnx"
if [ ! -f "$ONNX_PATH" ]; then
    echo "[ONNX] Downloading..."
    curl -sSL -o "$ONNX_PATH" "$ONNX_URL"
    echo "[ONNX] Saved to $(basename $ONNX_PATH)"
else
    echo "[ONNX] Already exists, skipping"
fi

# --- ImageNet labels ---
LABELS_URL="https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
LABELS_PATH="$MODELS_DIR/imagenet_labels.txt"
if [ ! -f "$LABELS_PATH" ]; then
    echo "[Labels] Downloading..."
    curl -sSL -o "$LABELS_PATH" "$LABELS_URL"
    echo "[Labels] Saved to $(basename $LABELS_PATH)"
else
    echo "[Labels] Already exists, skipping"
fi

# --- Sample input (a real test image, not noise) ---
# We'll generate a synthetic dog image via Python instead
echo "[Sample] Will generate synthetic test input at runtime"

echo ""
echo "=== Download complete ==="
ls -lh "$MODELS_DIR"/*.tflite "$MODELS_DIR"/*.onnx 2>/dev/null || true
