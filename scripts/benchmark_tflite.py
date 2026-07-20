#!/usr/bin/env python3
"""
TensorFlow Lite benchmark for edge-burn-benchmark.

Loads MobileNetV2 TFLite model, runs inference, measures:
  - End-to-end latency
  - CPU utilization
  - Peak memory consumption
  - Thermal behavior
"""

import os
import sys
import argparse
import time
import json

import numpy as np
from PIL import Image

# Add parent to path for benchmark_common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_common import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, save_result,
    get_machine_info,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "mobilenet_v2_1.0_224.tflite")
INPUT_SIZE = 224
MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
STD = np.array([127.5, 127.5, 127.5], dtype=np.float32)


# ---------------------------------------------------------------------------
# Inference harness
# ---------------------------------------------------------------------------

class TFLiteInference:
    """Wraps a TFLite interpreter for benchmarking."""

    def __init__(self, model_path: str, n_threads: int = 1):
        from tflite_runtime.interpreter import Interpreter

        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=n_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Verify input shape
        expected = (1, INPUT_SIZE, INPUT_SIZE, 3)  # TFLite uses NHWC
        actual = self.input_details[0]["shape"]
        assert list(actual) == list(expected), (
            f"Model expects {actual}, benchmark expects {expected}"
        )

        print(f"  TFLite model loaded: {model_path}")
        print(f"  Input shape: {actual}")
        print(f"  Threads: {n_threads}")
        print(f"  Input dtype: {self.input_details[0]['dtype']}")

    def infer(self, input_data: np.ndarray | None) -> float:
        """Run single inference. input_data is preprocessed NHWC float32."""
        if input_data is not None:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        _ = self.interpreter.get_tensor(self.output_details[0]["index"])
        return 0.0  # timing is done externally


def preprocess_input() -> np.ndarray:
    """Create a synthetic input tensor (simulates real image preprocessing).

    Uses random noise normalized to [0,1] then standardized.
    Shape: (1, 224, 224, 3) in NHWC format for TFLite.
    """
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, size=(1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TFLite benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .tflite model")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--measured", type=int, default=1000, help="Measured iterations")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--output", default="", help="Output JSON path")
    args = parser.parse_args()

    # Check model exists
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run 'bash models/download_models.sh' first.")
        sys.exit(1)

    # Load model
    print(f"\n{'='*60}")
    print(f"  TFLite Benchmark — {args.threads} thread(s)")
    print(f"{'='*60}")
    engine = TFLiteInference(model_path, n_threads=args.threads)

    # Config
    config = BenchmarkConfig(
        framework="tflite",
        model=os.path.basename(model_path),
        input_shape=(1, 224, 224, 3),
        n_warmup=args.warmup,
        n_measured=args.measured,
        n_threads=args.threads,
    )

    # Run
    result = run_benchmark(
        config=config,
        inference_fn=engine.infer,
        preprocess_fn=preprocess_input,
        n_warmup=args.warmup,
        n_measured=args.measured,
    )

    # Print summary
    s = result  # alias
    print(f"\n  --- Results ({args.threads}t) ---")
    print(f"  Latency:     {s.latency_mean_ms:.2f} ± {s.latency_std_ms:.2f} ms "
          f"(median {s.latency_median_ms:.2f})")
    print(f"  95% CI:      [{s.ci_lower_ms:.2f}, {s.ci_upper_ms:.2f}] ms")
    print(f"  Throughput:  {s.throughput_fps:.1f} fps")
    print(f"  CPU:         {s.cpu_mean_pct:.1f}%")
    print(f"  Memory:      {s.memory_peak_mb:.0f} MB peak ({s.memory_mean_mb:.0f} MB mean)")
    print(f"  Temp:        {s.temp_mean_c:.1f}°C mean ({s.temp_peak_c:.1f}°C peak)")
    print(f"  Outliers:    {s.n_outliers_removed} / {len(result.latencies_ms)} removed")
    print(f"  Valid:       {s.n_valid} samples")

    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        f"tflite_{args.threads}t.json",
    )
    save_result(result, output_path)

    # Also save machine info
    info_path = output_path.replace(".json", "_machine.json")
    with open(info_path, "w") as f:
        json.dump(get_machine_info(), f, indent=2)

    return result


if __name__ == "__main__":
    main()
