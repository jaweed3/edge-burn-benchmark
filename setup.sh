#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# edge-burn-benchmark — Full environment setup
# Tested on: Raspberry Pi 5 (Debian 13, aarch64)
# =============================================================================

echo "[1/6] System dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake pkg-config libssl-dev curl \
    python3-venv python3-dev \
    tmux htop \
    2>&1 | tail -2

echo "[2/6] Rust toolchain..."
if ! command -v rustc &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup default stable 2>&1 | tail -3
echo "  rustc $(rustc --version) / cargo $(cargo --version)"

echo "[3/6] Python virtual environment..."
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet \
    tensorflow \
    onnxruntime \
    numpy scipy psutil \
    2>&1 | tail -2

echo "[4/6] Download models..."
bash models/download_models.sh

echo "[5/6] Build Burn benchmark..."
cd src/burn_bench
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release 2>&1 | tail -3
cd ../..

echo "[6/6] Setup complete!"
echo ""
echo "  Run benchmarks:"
echo "    Burn:    ./run_burn.sh"
echo "    TFLite:  source .venv/bin/activate && python scripts/benchmark_tflite.py"
echo "    ONNX:    source .venv/bin/activate && python scripts/benchmark_onnx.py"
echo "    All:     bash run_all.sh"
