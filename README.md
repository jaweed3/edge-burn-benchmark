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
|| **Burn** | 0.13 | Rust | tract-onnx (CPU, NEON) |
| **TensorFlow Lite** | latest | Python | TFLite Runtime (ARM64) |
| **ONNX Runtime** | latest | Python | CPUExecutionProvider |

## Model

- **Architecture:** MobileNetV2 (depth multiplier 1.0, input 224×224)
- **Formats:**
  - `.tflite` — TensorFlow Lite float32 (NHWC)
  - `.onnx` — ONNX opset 7 (NCHW)
  - Burn loads ONNX at runtime via tract-onnx

---

## Setup

### One-command setup

```bash
bash setup.sh
```

This installs:
1. System packages (cmake, build-essential, etc.)
2. Rust toolchain via rustup
3. Python venv with `tensorflow`, `onnxruntime`, `numpy`, `scipy`, `psutil`, `Pillow`
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
pip install tensorflow onnxruntime numpy scipy psutil Pillow

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

## Results

### Raspberry Pi 5 (Cortex-A76) — MobileNetV2 @ 224×224

| Framework | Threads | Latency (ms) | ±σ | Throughput | CPU | Memory | Temp |
|---|---|---|---|---|---|---|---|
| **Burn** (tract-onnx) | 1 | **103.18** | 0.46 | 9.7 fps | 21.9% | 41.8 MB | 48.7°C |
| **Burn** (tract-onnx) | 4 | 103.13 | 0.45 | 9.7 fps | 22.0% | 41.7 MB | 48.9°C |
| **TensorFlow Lite** | 1 | **22.23** | 0.04 | 45.0 fps | 25.1% | 540 MB | 46.7°C |
| **TensorFlow Lite** | 4 | **10.74** | 0.10 | 92.6 fps | 94.0% | 540 MB | 54.4°C |
| **ONNX Runtime** | 1 | **45.41** | 0.09 | 22.0 fps | 25.1% | 100 MB | 47.9°C |
| **ONNX Runtime** | 4 | **20.37** | 0.24 | 49.0 fps | 99.8% | 100 MB | 55.5°C |

### Key findings

1. **TFLite is 4–5× faster than Burn** — MobileNetV2 runs at 22 ms vs 103 ms single-threaded. TFLite's ARM NEON kernels are well-tuned for Cortex-A76
2. **Burn is memory-efficient** — 41.7 MB RSS vs 100 MB (ONNX) vs 540 MB (TFLite, includes Python overhead). Rust's lack of GC and tract's lean execution engine make it ideal for memory-constrained edge devices
3. **Burn does not scale with threads** — tract-onnx uses a single-threaded execution plan; `RAYON_NUM_THREADS` has no effect. Both TFLite and ONNX Runtime show 2–2.2× speedup going from 1→4 threads
4. **ONNX Runtime is ~2× slower than TFLite** single-threaded (45 vs 22 ms), but scales similarly with 4 threads (20 ms)
5. **Temperature stable across all frameworks** — all runs stay below 57°C with passive cooling; no thermal throttling detected at 1000-iteration workloads

> **Note:** Hardware is Raspberry Pi 5 (Cortex-A76). Pi 4 (Cortex-A72) will show proportionally higher latencies.

All raw results as structured JSON: `results/burn_*.json`, `results/tflite_*.json`, `results/onnx_*.json`.

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
    "latency_mean_ms": 103.18,
    "latency_median_ms": 103.05,
    "latency_std_ms": 0.46,
    "latency_min_ms": 102.77,
    "latency_max_ms": 108.88,
    "throughput_fps": 9.68,
    "ci_95_lower_ms": 103.15,
    "ci_95_upper_ms": 103.21,
    "cpu_mean_pct": 21.9,
    "memory_peak_mb": 41.8,
    "memory_mean_mb": 41.7,
    "temp_mean_c": 48.7,
    "temp_peak_c": 50.7,
    "n_valid": 885,
    "n_outliers_removed": 115
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
