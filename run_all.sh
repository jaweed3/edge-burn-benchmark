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
WARMUP=200
MEASURED=1000

# --- 1. Burn benchmark ---
echo ""
echo "[1/3] Burn benchmark"
cd "$SCRIPT_DIR/src/burn_bench"
for threads in 1 4; do
    echo ""
    echo "--- Burn ${threads}t ---"
    cargo run --release -- \
        --warmup $WARMUP --measured $MEASURED \
        --threads $threads \
        --output "$RESULTS_DIR" \
        --model "$SCRIPT_DIR/models/mobilenetv2-7.onnx" 2>&1
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
        --warmup $WARMUP --measured $MEASURED \
        --threads $threads
done

# --- 3. ONNX benchmark ---
echo ""
echo "[3/3] ONNX Runtime benchmark"
for threads in 1 4; do
    echo ""
    echo "--- ONNX ${threads}t ---"
    python scripts/benchmark_onnx.py \
        --warmup $WARMUP --measured $MEASURED \
        --threads $threads
done

echo ""
echo "=============================================="
echo "  All benchmarks complete!"
echo "  Results in: $RESULTS_DIR/"
echo "=============================================="
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null
