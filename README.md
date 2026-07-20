# edge-burn-benchmark

**Systematic benchmark comparing Burn, TensorFlow Lite, and ONNX Runtime for inference on ARM64 edge hardware.**

This is the official benchmark harness for the paper:

> *"Burn vs TensorFlow Lite vs ONNX Runtime: Systematic Benchmarking of Rust-Native Deep Learning on ARM64 Edge Platforms"*

---

## Hardware

| Component | Spec |
|-----------|------|
| Board | Raspberry Pi 5 Model B Rev 1.0 |
| SoC | Broadcom BCM2712 |
| CPU | ARM Cortex-A76 (4 cores) |
| RAM | 8 GB LPDDR4X |
| Storage | 32 GB microSD (or NVMe via PCIe 2.0) |
| Thermal | Passive heatsink, no active cooling |
| OS | Debian 13 "Trixie" (aarch64), kernel 6.18.34 |

> **Note:** The original paper abstract targets Raspberry Pi 4 (Cortex-A72). The actual hardware is a Pi 5 (Cortex-A76). The same harness runs on Pi 4 — swap the board and re-run.

---

## Frameworks Under Test

| Framework | Version | Language | Backend |
|-----------|---------|----------|---------|
| **Burn** | 0.13 | Rust | NdArray (CPU, NEON) |
| **TensorFlow Lite** | latest | Python | TFLite Runtime (ARM64) |
| **ONNX Runtime** | latest | Python | CPUExecutionProvider |

## Model

- **Architecture:** MobileNetV2 (depth multiplier 1.0, input 224×224)
- **Formats:**
  - `.tflite` — TensorFlow Lite float32 (NHWC)
  - `.onnx` — ONNX opset 7 (NCHW)
  - Burn imports the ONNX model at compile time via `burn-import-onnx`

---

## Setup

### One-command setup

```bash
bash setup.sh
```

This installs:
1. System packages (cmake, build-essential, etc.)
2. Rust toolchain via rustup
3. Python venv with `tflite-runtime`, `onnxruntime`, `numpy`, `scipy`, `psutil`
4. Models (MobileNetV2 in .tflite and .onnx formats)
5. Builds the Burn benchmark binary

### Manual steps

```bash
# 1. System deps
sudo apt update && sudo apt install -y cmake pkg-config libssl-dev curl python3-venv tmux

# 2. Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
rustup default stable

# 3. Python
python3 -m venv .venv
source .venv/bin/activate
pip install tflite-runtime onnxruntime numpy scipy psutil

# 4. Models
bash models/download_models.sh

# 5. Build Burn
cd src/burn_bench && cargo build --release && cd ../..
```

---

## Running

### Full suite (all frameworks, 1t + 4t)

```bash
bash run_all.sh
```

### Individual benchmarks

```bash
# Burn (Rust)
cd src/burn_bench
cargo run --release -- --threads 1 --warmup 200 --measured 1000

# TFLite (Python)
source .venv/bin/activate
python scripts/benchmark_tflite.py --threads 1 --warmup 200 --measured 1000

# ONNX Runtime (Python)
source .venv/bin/activate
python scripts/benchmark_onnx.py --threads 1 --warmup 200 --measured 1000
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--threads` | 1 | Number of threads (1 or 4) |
| `--warmup` | 200 | Warm-up iterations (discarded) |
| `--measured` | 1000 | Measured iterations |
| `--output` | `results/` | Output directory |
| `--model` | `models/...` | Model path |

---

## Metrics Collected

| Metric | Method |
|--------|--------|
| **Latency** | Per-inference wall time (monotonic clock), outlier removal via IQR |
| **Throughput** | Total inferences / total elapsed time |
| **95% CI** | Percentile bootstrap (10,000 resamples) |
| **CPU utilization** | Sampled at 50ms intervals during inference (§ /proc/stat) |
| **Memory** | Peak and mean RSS (§ /proc/self/status) |
| **Temperature** | Sampled at 50ms intervals (§ /sys/class/thermal/thermal_zone0/temp) |

### Statistical Methodology

1. **Warmup:** 200 iterations (discarded) to reach thermal steady state
2. **Measurement:** 1,000 iterations recorded
3. **Outlier removal:** IQR method (1.5× IQR below Q1 or above Q3)
4. **Confidence intervals:** Bootstrap with 10,000 resamples at 95% confidence
5. **Configurations:**
   - Single-threaded (1 thread, sequential execution)
   - Multi-threaded (4 threads, parallel execution)

---

## Output Format

Results are saved as JSON in `results/`:

```json
{
  "config": {
    "framework": "burn",
    "model": "mobilenetv2-7.onnx",
    "n_threads": 1,
    "n_warmup": 200,
    "n_measured": 1000
  },
  "summary": {
    "latency_mean_ms": 145.23,
    "latency_median_ms": 142.10,
    "latency_std_ms": 12.45,
    "latency_min_ms": 120.01,
    "latency_max_ms": 198.76,
    "throughput_fps": 6.88,
    "ci_95_lower_ms": 140.50,
    "ci_95_upper_ms": 150.10,
    "cpu_mean_pct": 85.3,
    "memory_peak_mb": 245.0,
    "memory_mean_mb": 230.0,
    "temp_mean_c": 65.2,
    "temp_peak_c": 72.1,
    "n_valid": 985,
    "n_outliers_removed": 15
  }
}
```

A `_machine.json` sidecar records system info for reproducibility.

---

## Reproducibility

This repository contains everything needed to reproduce the benchmarks:

- **Source code:** All benchmark scripts and Burn Rust source
- **Model download:** Automated via `models/download_models.sh`
- **Environment setup:** Automated via `setup.sh`
- **Results:** Raw JSON output with full metadata
- **Dockerfile:** Containerized environment (see `docker/`)

To reproduce:

```bash
git clone https://github.com/jaweed3/edge-burn-benchmark.git
cd edge-burn-benchmark
bash setup.sh
bash run_all.sh
```

---

## Project Structure

```
edge-burn-benchmark/
├── README.md
├── setup.sh                  # Full environment setup
├── run_all.sh                # Run all benchmarks
├── .gitignore
├── docker/
│   └── Dockerfile            # Containerized environment
├── models/
│   └── download_models.sh    # Download MobileNetV2 models
├── src/
│   └── burn_bench/           # Burn Rust benchmark project
│       ├── Cargo.toml
│       ├── build.rs          # ONNX-to-Burn compile-time import
│       └── src/
│           └── main.rs
├── scripts/
│   ├── benchmark_common.py   # Shared metrics/statistics library
│   ├── benchmark_tflite.py   # TensorFlow Lite benchmark
│   └── benchmark_onnx.py     # ONNX Runtime benchmark
└── results/                  # Benchmark output (gitignored)
```

---

## License

MIT
