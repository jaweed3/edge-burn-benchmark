#!/usr/bin/env python3
"""
ONNX Runtime benchmark for edge-burn-benchmark.

Loads MobileNetV2 ONNX model, runs inference, measures:
  - End-to-end latency
  - CPU utilization
  - Peak memory consumption
  - Thermal behavior
"""

import os
import sys
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_common import (
    BenchmarkConfig, run_benchmark, save_result, get_machine_info,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "mobilenetv2-7.onnx")
INPUT_SIZE = 224


# ---------------------------------------------------------------------------
# Inference harness
# ---------------------------------------------------------------------------

class ONNXInference:
    """Wraps an ONNX Runtime session for benchmarking."""

    def __init__(self, model_path: str, n_threads: int = 1):
        import onnxruntime as ort

        # Session options
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = n_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type

        # Verify input shape
        expected = (1, 3, INPUT_SIZE, INPUT_SIZE)  # ONNX uses NCHW
        actual = self.input_shape
        # Handle dynamic dims (None in shape)
        assert actual[0] in (1, None), f"Expected batch=1, got {actual[0]}"
        assert actual[1] in (3, None), f"Expected 3 channels, got {actual[1]}"

        print(f"  ONNX model loaded: {model_path}")
        print(f"  Input name: {self.input_name}, shape: {self.input_shape}")
        print(f"  Threads: {n_threads}")

    def infer(self, input_data: np.ndarray | None) -> float:
        """Run single inference."""
        if input_data is not None:
            _ = self.session.run(None, {self.input_name: input_data})
        else:
            _ = self.session.run(None, {self.input_name: self._dummy})
        return 0.0


def preprocess_input() -> np.ndarray:
    """Create synthetic input tensor in NCHW format for ONNX.

    Shape: (1, 3, 224, 224) float32.
    Normalized to match standard ImageNet preprocessing.
    """
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, size=(1, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    # Standardize: (x - mean) / std  with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    return (raw - mean) / std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .onnx model")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup iterations")
    parser.add_argument("--measured", type=int, default=1000, help="Measured iterations")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--output", default="", help="Output JSON path")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run 'bash models/download_models.sh' first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ONNX Runtime Benchmark — {args.threads} thread(s)")
    print(f"{'='*60}")
    engine = ONNXInference(model_path, n_threads=args.threads)

    config = BenchmarkConfig(
        framework="onnx",
        model=os.path.basename(model_path),
        input_shape=(1, 3, 224, 224),
        n_warmup=args.warmup,
        n_measured=args.measured,
        n_threads=args.threads,
    )

    result = run_benchmark(
        config=config,
        inference_fn=engine.infer,
        preprocess_fn=preprocess_input,
        n_warmup=args.warmup,
        n_measured=args.measured,
    )

    # Print
    s = result
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

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        f"onnx_{args.threads}t.json",
    )
    save_result(result, output_path)

    info_path = output_path.replace(".json", "_machine.json")
    with open(info_path, "w") as f:
        json.dump(get_machine_info(), f, indent=2)

    return result


if __name__ == "__main__":
    main()
