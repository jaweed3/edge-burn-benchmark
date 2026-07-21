#!/usr/bin/env bash
# Run all benchmarks (Burn, TFLite, ONNX) end-to-end
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

export PATH="$HOME/.cargo/bin:$PATH"

echo "=============================================="
echo "  edge-burn-benchmark — Full Benchmark Suite"
echo "=============================================="
echo "System: $(uname -a | cut -d' ' -f1-3)"
echo "Date:   $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "=============================================="

# --- Config ---
BURN_WARMUP=1000
TFLITE_WARMUP=200
ONNX_WARMUP=200
MEASURED=1000

# --- 1. Burn benchmark ---
echo ""
echo "[1/3] Burn benchmark (Burn 0.21 native ndarray)"
cd "$SCRIPT_DIR/src/burn_bench"

# Copy ONNX model for burn-onnx codegen
mkdir -p "$SCRIPT_DIR/src/burn_bench"
if [ ! -f mobilenetv2-7.onnx ]; then
    cp "$SCRIPT_DIR/models/mobilenetv2-7.onnx" mobilenetv2-7.onnx
fi

for threads in 1 4; do
    echo ""
    echo "--- Burn ${threads}t ---"
    RAYON_NUM_THREADS=$threads OMP_NUM_THREADS=$threads \
    cargo run --release -- \
        --warmup $BURN_WARMUP --measured $MEASURED \
        --threads $threads \
        --output "$RESULTS_DIR" 2>&1
done

# --- 2. TFLite benchmark ---
echo ""
echo "[2/3] TFLite benchmark"
cd "$SCRIPT_DIR"
source .venv/bin/activate
for threads in 1 4; do
    echo ""
    echo "--- TFLite ${threads}t ---"
    python scripts/benchmark_tflite.py \
        --warmup $TFLITE_WARMUP --measured $MEASURED \
        --threads $threads
done

# --- 3. ONNX benchmark ---
echo ""
echo "[3/3] ONNX Runtime benchmark"
for threads in 1 4; do
    echo ""
    echo "--- ONNX ${threads}t ---"
    python scripts/benchmark_onnx.py \
        --warmup $ONNX_WARMUP --measured $MEASURED \
        --threads $threads
done

echo ""
echo "=============================================="
echo "  All benchmarks complete!"
echo "  Results in: $RESULTS_DIR/"
echo "=============================================="
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null

# Clean up model copy
rm -f "$SCRIPT_DIR/src/burn_bench/mobilenetv2-7.onnx"
