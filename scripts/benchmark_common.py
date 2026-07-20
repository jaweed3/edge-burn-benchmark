"""
Shared benchmark utilities for edge inference comparison.

Provides:
  - MetricsCollector: CPU, memory, temperature sampling
  - Bootstrapped CI computation
  - Outlier removal (IQR + MAD)
  - Result serialization
"""

import time
import json
import os
import statistics
import random
from typing import Callable
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil


# ---------------------------------------------------------------------------
# Metrics data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    framework: str          # "burn", "tflite", "onnx"
    model: str              # model name
    input_shape: tuple      # e.g. (1, 3, 224, 224)
    n_warmup: int = 1000    # warm-up iterations (discarded)
    n_measured: int = 1000  # measured iterations
    n_threads: int = 1      # thread count (1 = single-threaded)
    label: str = ""         # run label for results file

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.framework}_{self.n_threads}t"


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run."""
    config: BenchmarkConfig
    latencies_ms: list = field(default_factory=list)
    cpu_util_pct: list = field(default_factory=list)
    memory_mb: list = field(default_factory=list)
    temp_c: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)

    # Computed fields
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    throughput_fps: float = 0.0
    ci_lower_ms: float = 0.0
    ci_upper_ms: float = 0.0
    cpu_mean_pct: float = 0.0
    memory_peak_mb: float = 0.0
    memory_mean_mb: float = 0.0
    temp_mean_c: float = 0.0
    temp_peak_c: float = 0.0
    n_outliers_removed: int = 0
    n_valid: int = 0


# ---------------------------------------------------------------------------
# Temperature reading
# ---------------------------------------------------------------------------

def read_temperature() -> float:
    """Read Raspberry Pi CPU temperature in °C."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError, OSError):
        return 0.0


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def monotonic_ms() -> float:
    """High-resolution monotonic time in milliseconds."""
    return time.monotonic() * 1000.0


# ---------------------------------------------------------------------------
# CPU / memory sampling
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Background thread that samples CPU, memory, and temp during inference."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.running = False
        self.cpu_samples: list[float] = []
        self.mem_samples: list[float] = []
        self.temp_samples: list[float] = []
        self._executor: ThreadPoolExecutor | None = None

    def start(self):
        self.running = True
        self.cpu_samples = []
        self.mem_samples = []
        self.temp_samples = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._executor.submit(self._sample_loop)

    def stop(self):
        self.running = False
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _sample_loop(self):
        while self.running:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.mem_samples.append(psutil.Process().memory_info().rss / (1024 * 1024))
            self.temp_samples.append(read_temperature())
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers_iqr(data: list[float], factor: float = 1.5) -> tuple[list[float], int]:
    """Remove outliers using IQR method. Returns (clean_data, n_removed)."""
    if len(data) < 4:
        return data, 0
    arr = np.array(data)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    clean = arr[(arr >= lower) & (arr <= upper)]
    n_removed = len(arr) - len(clean)
    return clean.tolist(), n_removed


# ---------------------------------------------------------------------------
# Bootstrapped confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(data: list[float], n_bootstrap: int = 10_000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """
    Compute percentile bootstrapped confidence interval for the mean.
    Returns (lower, upper).
    """
    if len(data) < 2:
        return (data[0] if data else 0.0), (data[0] if data else 0.0)

    arr = np.array(data)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower, upper = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    config: BenchmarkConfig,
    inference_fn: Callable,
    preprocess_fn: Callable | None = None,
    n_warmup: int | None = None,
    n_measured: int | None = None,
) -> BenchmarkResult:
    """
    Run a full benchmark cycle.

    Args:
        config: Benchmark configuration
        inference_fn: Callable that takes preprocessed input and returns latency
        preprocess_fn: Optional preprocessing (called once before timing)
        n_warmup: Override warmup iterations
        n_measured: Override measured iterations

    Returns:
        BenchmarkResult with all computed metrics
    """
    n_warmup = n_warmup or config.n_warmup
    n_measured = n_measured or config.n_measured

    result = BenchmarkResult(config=config)
    collector = MetricsCollector()

    # Preprocess input once
    input_data = preprocess_fn() if preprocess_fn else None

    print(f"  Warmup: {n_warmup} iterations...")
    for _ in range(n_warmup):
        inference_fn(input_data)

    print(f"  Measuring: {n_measured} iterations (sampling CPU/mem/temp)...")
    collector.start()

    for i in range(n_measured):
        t0 = monotonic_ms()
        inference_fn(input_data)
        elapsed = monotonic_ms() - t0

        result.latencies_ms.append(elapsed)
        result.timestamps.append(time.monotonic())

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n_measured} done")

    collector.stop()

    # Merge collected samples
    result.cpu_util_pct = collector.cpu_samples[:]
    result.memory_mb = collector.mem_samples[:]
    result.temp_c = collector.temp_samples[:]

    # --- Compute statistics ---
    latencies = result.latencies_ms

    # Outlier removal
    clean_latencies, n_out = remove_outliers_iqr(latencies)

    # Basic stats
    result.latency_mean_ms = float(np.mean(clean_latencies))
    result.latency_median_ms = float(np.median(clean_latencies))
    result.latency_std_ms = float(np.std(clean_latencies))
    result.latency_min_ms = float(np.min(clean_latencies))
    result.latency_max_ms = float(np.max(clean_latencies))
    result.n_outliers_removed = n_out
    result.n_valid = len(clean_latencies)

    # Throughput
    total_time_s = (result.timestamps[-1] - result.timestamps[0]) if len(result.timestamps) > 1 else 1.0
    result.throughput_fps = n_measured / total_time_s if total_time_s > 0 else 0.0

    # Bootstrapped CI
    result.ci_lower_ms, result.ci_upper_ms = bootstrap_ci(clean_latencies)

    # CPU, memory, temp
    result.cpu_mean_pct = float(np.mean(result.cpu_util_pct)) if result.cpu_util_pct else 0.0
    result.memory_peak_mb = float(np.max(result.memory_mb)) if result.memory_mb else 0.0
    result.memory_mean_mb = float(np.mean(result.memory_mb)) if result.memory_mb else 0.0
    result.temp_mean_c = float(np.mean(result.temp_c)) if result.temp_c else 0.0
    result.temp_peak_c = float(np.max(result.temp_c)) if result.temp_c else 0.0

    return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def result_to_dict(result: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to a JSON-serializable dict."""
    cfg = asdict(result.config)
    # Convert tuple to list for JSON
    cfg["input_shape"] = list(cfg["input_shape"])

    return {
        "config": cfg,
        "summary": {
            "latency_mean_ms": round(result.latency_mean_ms, 4),
            "latency_median_ms": round(result.latency_median_ms, 4),
            "latency_std_ms": round(result.latency_std_ms, 4),
            "latency_min_ms": round(result.latency_min_ms, 4),
            "latency_max_ms": round(result.latency_max_ms, 4),
            "throughput_fps": round(result.throughput_fps, 2),
            "ci_95_lower_ms": round(result.ci_lower_ms, 4),
            "ci_95_upper_ms": round(result.ci_upper_ms, 4),
            "cpu_mean_pct": round(result.cpu_mean_pct, 1),
            "memory_peak_mb": round(result.memory_peak_mb, 1),
            "memory_mean_mb": round(result.memory_mean_mb, 1),
            "temp_mean_c": round(result.temp_mean_c, 1),
            "temp_peak_c": round(result.temp_peak_c, 1),
            "n_valid": result.n_valid,
            "n_outliers_removed": result.n_outliers_removed,
        },
        "latencies_ms": [round(l, 4) for l in result.latencies_ms],
        "cpu_samples": [round(c, 1) for c in result.cpu_util_pct],
        "memory_samples": [round(m, 1) for m in result.memory_mb],
        "temp_samples": [round(t, 1) for t in result.temp_c],
    }


def save_result(result: BenchmarkResult, path: str):
    """Save benchmark result to JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(result_to_dict(result), f, indent=2)
    print(f"  Result saved to {path}")


# ---------------------------------------------------------------------------
# Machine info
# ---------------------------------------------------------------------------

def get_machine_info() -> dict:
    """Get system information for reproducibility."""
    return {
        "hostname": os.uname().nodename,
        "machine": os.uname().machine,
        "processor": os.uname().processor if hasattr(os.uname(), "processor") else "",
        "cpu_count": os.cpu_count(),
        "python_version": os.sys.version,
        "psutil_version": psutil.__version__,
        "numpy_version": np.__version__,
    }
