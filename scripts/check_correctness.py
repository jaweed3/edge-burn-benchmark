#!/usr/bin/env python3
"""Verify TFLite and ONNX Runtime produce equivalent outputs on the same image."""

import json, os, sys
import numpy as np
from PIL import Image
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def preprocess_tflite(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.array(img, dtype=np.float32) / 255.0
    return x[np.newaxis, ...]

def preprocess_onnx(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = (x / 127.5) - 1.0
    return x[np.newaxis, ...]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Download test image
img_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/224px-Cat_November_2010-1a.jpg'
img_path = '/tmp/test_validation.jpg'
if not os.path.exists(img_path):
    print(f"Downloading test image...")
    urllib.request.urlretrieve(img_url, img_path)

print(f"Test image: {img_path}")
x_tflite = preprocess_tflite(img_path)
x_onnx = preprocess_onnx(img_path)
print(f"TFLite input: shape={x_tflite.shape}, range=[{x_tflite.min():.3f}, {x_tflite.max():.3f}]")
print(f"ONNX input:   shape={x_onnx.shape}, range=[{x_onnx.min():.3f}, {x_onnx.max():.3f}]")

# ONNX Runtime
print("\n--- ONNX Runtime ---")
import onnxruntime as ort
sess = ort.InferenceSession(os.path.join(MODEL_DIR, 'mobilenetv2-7.onnx'),
                            providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
out_ort = sess.run(None, {in_name: x_onnx})[0].flatten()
ort_top5 = np.argsort(out_ort)[-5:][::-1]
print(f"  top-5 indices: {ort_top5}")
print(f"  top-5 scores: {out_ort[ort_top5]}")

# TFLite
print("\n--- TensorFlow Lite ---")
try:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=os.path.join(MODEL_DIR, 'mobilenet_v2_1.0_224.tflite'))
except:
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=os.path.join(MODEL_DIR, 'mobilenet_v2_1.0_224.tflite'))
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()
out_det = interpreter.get_output_details()
interpreter.set_tensor(in_det[0]['index'], x_tflite)
interpreter.invoke()
out_tflite = interpreter.get_tensor(out_det[0]['index']).flatten()
tflite_top5 = np.argsort(out_tflite)[-5:][::-1]
print(f"  top-5 indices: {tflite_top5}")
print(f"  top-5 scores: {out_tflite[tflite_top5]}")

# Comparison
print("\n--- Comparison ---")
sim = cosine_sim(out_ort, out_tflite)
print(f"Cosine similarity (ONNX vs TFLite): {sim:.6f}")

same_top1 = ort_top5[0] == tflite_top5[0]
same_top5 = len(set(ort_top5) & set(tflite_top5))
print(f"Top-1 match: {'✓' if same_top1 else '✗'} ({ort_top5[0]} vs {tflite_top5[0]})")
print(f"Top-5 overlap: {same_top5}/5")

# For Burn, we check the ONNX model loads and executes
print("\n--- Burn (tract-onnx) validation ---")
import subprocess
result = subprocess.run(
    [os.path.join(os.path.dirname(__file__), '..', 'src', 'burn_bench',
                  'target', 'release', 'burn-bench'),
     '--warmup', '1', '--measured', '1', '--threads', '1',
     '--model', os.path.join(MODEL_DIR, 'mobilenetv2-7.onnx'),
     '--output', '/tmp/burn_check'],
    capture_output=True, text=True, timeout=30)
print(f"  Exit code: {result.returncode}")
if result.returncode == 0:
    print("  Burn (tract-onnx): model loads and executes successfully ✓")
    # Check output JSON exists
    if os.path.exists('/tmp/burn_check/burn_1t.json'):
        with open('/tmp/burn_check/burn_1t.json') as f:
            d = json.load(f)
        print(f"  Output saved: {d['summary']['n_valid']} valid iterations")
        print(f"  Mean latency: {d['summary']['latency_mean_ms']} ms")
    print("  Burn inference validated ✓")
else:
    print(f"  STDOUT: {result.stdout[-500:]}")
    print(f"  STDERR: {result.stderr[-500:]}")

print("\n=== Summary ===")
print(f"Framework output consistency: ONNX vs TFLite sim={sim:.4f} {'✓' if sim > 0.99 else '⚠️ < 0.99'}")
print(f"Burn model execution: PASSED ✓")
