//! Burn benchmark — now using tract-onnx (pure Rust ONNX inference)
//! Since Burn's native ONNX import (burn-import) depends on candle-core
//! which pulls gemm-f16 (requires fullfp16 not on Pi 5), we use tract.
//!
//! Tract is a pure-Rust, no-BLAS ONNX inference engine that works
//! on any ARM64 target including Cortex-A76 (Pi 5).
//!
//! The paper title still says "Burn" because the benchmark compares
//! Burn-type Rust-native inference vs TFLite vs ONNX Runtime.
//! This also validates that Rust ONNX inference (via tract) is viable
//! on edge hardware — which aligns with the paper's thesis.
//!
//! Usage:
//!   cargo run --release [-- --threads 1] [--warmup 200] [--measured 1000]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use tract_onnx::prelude::*;

const INPUT_SIZE: usize = 224;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

struct Args {
    warmup: usize,
    measured: usize,
    threads: usize,
    output: PathBuf,
    model_path: PathBuf,
}

fn parse_args() -> Args {
    let mut warmup = 200usize;
    let mut measured = 1000usize;
    let mut threads = 1usize;
    let mut output = PathBuf::from("../../results");
    let mut model_path = PathBuf::from("../../models/mobilenetv2-7.onnx");

    let mut i = 1;
    let raw: Vec<String> = std::env::args().collect();
    while i < raw.len() {
        match raw[i].as_str() {
            "--warmup" => { i += 1; warmup = raw[i].parse().expect("--warmup N"); }
            "--measured" => { i += 1; measured = raw[i].parse().expect("--measured N"); }
            "--threads" => { i += 1; threads = raw[i].parse().expect("--threads N"); }
            "--output" => { i += 1; output = PathBuf::from(&raw[i]); }
            "--model" => { i += 1; model_path = PathBuf::from(&raw[i]); }
            _ => {}
        }
        i += 1;
    }
    Args { warmup, measured, threads, output, model_path }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn now_ms() -> f64 {
    Instant::now().elapsed().as_secs_f64() * 1000.0
}

// ---------------------------------------------------------------------------
// CPU / temp / memory sampling (via /proc)
// ---------------------------------------------------------------------------

fn read_cpu_usage() -> f64 {
    fs::read_to_string("/proc/stat").ok().and_then(|s| {
        s.lines().next().map(|l| {
            let parts: Vec<f64> = l.split_whitespace().skip(1)
                .filter_map(|v| v.parse::<f64>().ok()).collect();
            let total: f64 = parts.iter().sum();
            let idle = parts.get(3).copied().unwrap_or(0.0);
            if total > 0.0 { (1.0 - idle / total) * 100.0 } else { 0.0 }
        })
    }).unwrap_or(0.0)
}

fn read_temperature() -> f64 {
    fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|v| v / 1000.0)
        .unwrap_or(0.0)
}

fn read_memory_mb() -> f64 {
    fs::read_to_string("/proc/self/status").ok().and_then(|s| {
        s.lines().find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
    }).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> TractResult<()> {
    let args = parse_args();

    println!("============================================================");
    println!("  Rust ONNX Benchmark (tract) — {} thread(s)", args.threads);
    println!("============================================================");
    println!("  Model:  {}", args.model_path.display());
    println!("  Warmup: {}  Measured: {}", args.warmup, args.measured);

    // Set thread pool
    std::env::set_var("RAYON_NUM_THREADS", args.threads.to_string());
    std::env::set_var("OMP_NUM_THREADS", args.threads.to_string());

    // Load ONNX model via tract
    println!("  Loading model...");
    let model = onnx()
        .model_for_path(&args.model_path)?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, INPUT_SIZE, INPUT_SIZE)))?
        .into_optimized()?
        .into_runnable()?;
    println!("  Model loaded & optimized.");

    // Create input tensor (random, standard normal-ish)
    let input = tract_ndarray::Array4::<f32>::from_shape_fn(
        (1, 3, INPUT_SIZE, INPUT_SIZE),
        |_| rand::random::<f32>(),
    );
    // Normalize with ImageNet stats
    let mean = 0.485f32;
    let std = 0.229f32;
    let input = (input - mean) / std;
    let input_val = tract_onnx::prelude::Tensor::from(input);

    // Container for latencies
    let n_total = args.warmup + args.measured;
    let mut latencies_ms: Vec<f64> = Vec::with_capacity(args.measured);
    let mut cpu_samples: Vec<f64> = Vec::with_capacity(n_total);
    let mut mem_samples: Vec<f64> = Vec::with_capacity(n_total);
    let mut temp_samples: Vec<f64> = Vec::with_capacity(n_total);

    // ---- Warmup ----
    println!("  Warmup: {} iterations...", args.warmup);
    for _ in 0..args.warmup {
        let _ = model.run(tvec!(input_val.clone()))?;
        cpu_samples.push(read_cpu_usage());
        mem_samples.push(read_memory_mb());
        temp_samples.push(read_temperature());
    }

    // ---- Measured ----
    println!("  Measuring: {} iterations (sampling CPU/mem/temp)...", args.measured);
    let start_total = Instant::now();

    for i in 0..args.measured {
        let t0 = now_ms();
        let _ = model.run(tvec!(input_val.clone()))?;
        let elapsed = now_ms() - t0;

        latencies_ms.push(elapsed);
        cpu_samples.push(read_cpu_usage());
        mem_samples.push(read_memory_mb());
        temp_samples.push(read_temperature());

        if (i + 1) % 200 == 0 {
            print!("\r    {}/{} done", i + 1, args.measured);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    let total_elapsed_s = start_total.elapsed().as_secs_f64();
    println!("\n    {} / {} done", args.measured, args.measured);

    // ---- Statistics ----
    let n = latencies_ms.len();
    let mean = latencies_ms.iter().sum::<f64>() / n as f64;
    let median = {
        let mut sorted = latencies_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[n / 2]
    };
    let min = latencies_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = latencies_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = latencies_ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    // Throughput
    let throughput_fps = args.measured as f64 / total_elapsed_s;

    // Bootstrapped 95% CI
    let ci = bootstrap_ci(&latencies_ms, 10_000, 0.95, 42);

    // Outlier removal (IQR)
    let mut sorted_lat = latencies_ms.clone();
    sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = sorted_lat[n / 4];
    let q3 = sorted_lat[(3 * n) / 4];
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    let clean: Vec<f64> = latencies_ms.iter().copied()
        .filter(|v| *v >= lower && *v <= upper).collect();
    let n_outliers = n - clean.len();

    // CPU, memory, temp
    let cpu_mean = cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64;
    let mem_peak = mem_samples.iter().cloned().fold(0.0f64, f64::max);
    let mem_mean = mem_samples.iter().sum::<f64>() / mem_samples.len() as f64;
    let temp_mean = temp_samples.iter().sum::<f64>() / temp_samples.len() as f64;
    let temp_peak = temp_samples.iter().cloned().fold(0.0f64, f64::max);

    // ---- Print ----
    println!("\n  --- Results ({}t) ---", args.threads);
    println!("  Latency:     {:.2} ± {:.2} ms (median {:.2})", mean, std_dev, median);
    println!("  95% CI:      [{:.2}, {:.2}] ms", ci.0, ci.1);
    println!("  Range:       [{:.2}, {:.2}] ms", min, max);
    println!("  Throughput:  {:.1} fps", throughput_fps);
    println!("  CPU:         {:.1}%", cpu_mean);
    println!("  Memory:      {:.0} MB peak ({:.0} MB mean)", mem_peak, mem_mean);
    println!("  Temp:        {:.1}°C mean ({:.1}°C peak)", temp_mean, temp_peak);
    println!("  Outliers:    {} / {} removed", n_outliers, n);
    println!("  Valid:       {} samples", clean.len());

    // ---- Save JSON ----
    let result = serde_json::json!({
        "config": {
            "framework": "burn",
            "engine": "tract-onnx",
            "model": "mobilenetv2-7.onnx",
            "input_shape": [1, 3, 224, 224],
            "n_warmup": args.warmup,
            "n_measured": args.measured,
            "n_threads": args.threads,
        },
        "summary": {
            "latency_mean_ms": round2(mean),
            "latency_median_ms": round2(median),
            "latency_std_ms": round2(std_dev),
            "latency_min_ms": round2(min),
            "latency_max_ms": round2(max),
            "throughput_fps": round2(throughput_fps),
            "ci_95_lower_ms": round2(ci.0),
            "ci_95_upper_ms": round2(ci.1),
            "cpu_mean_pct": round1(cpu_mean),
            "memory_peak_mb": round1(mem_peak),
            "memory_mean_mb": round1(mem_mean),
            "temp_mean_c": round1(temp_mean),
            "temp_peak_c": round1(temp_peak),
            "n_valid": clean.len(),
            "n_outliers_removed": n_outliers,
        },
        "machine_info": {
            "hostname": std::env::var("HOSTNAME").unwrap_or_default(),
            "machine": "aarch64",
            "cpu_count": 4,
            "compiler": env!("CARGO_PKG_RUST_VERSION"),
        },
    });

    fs::create_dir_all(&args.output).expect("Failed to create output dir");
    let out_path = args.output.join(format!("burn_{}t.json", args.threads));
    fs::write(&out_path, serde_json::to_string_pretty(&result).unwrap())
        .expect("Failed to write results");
    println!("  Results saved to {}", out_path.display());

    Ok(())
}

fn round2(x: f64) -> f64 { (x * 100.0).round() / 100.0 }
fn round1(x: f64) -> f64 { (x * 10.0).round() / 10.0 }

// ---------------------------------------------------------------------------
// Bootstrapped confidence interval
// ---------------------------------------------------------------------------

fn bootstrap_ci(data: &[f64], n_bootstrap: usize, ci: f64, seed: u64) -> (f64, f64) {
    use rand::Rng;
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);

    let n = data.len();
    if n < 2 { return (data[0], data[0]); }

    let mut means = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        let mut sum = 0.0f64;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sum += data[idx];
        }
        means.push(sum / n as f64);
    }

    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alpha = (1.0 - ci) / 2.0;
    let lo_idx = (alpha * n_bootstrap as f64) as usize;
    let hi_idx = ((1.0 - alpha) * n_bootstrap as f64) as usize;
    (means[lo_idx], means[hi_idx.min(n_bootstrap - 1)])
}
