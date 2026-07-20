#!/usr/bin/env python3
"""Generate paper-quality figures from edge-burn-benchmark results."""

import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load results
def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)

results = {}
for fname in ['burn_1t', 'burn_4t', 'tflite_1t', 'tflite_4t', 'onnx_1t', 'onnx_4t']:
    results[fname] = load_json(f'{fname}.json')

# Color palette — colorblind-friendly, high contrast
C_1T = '#1f77b4'   # blue
C_4T = '#ff7f0e'   # orange
C_BURN = '#1f77b4'
C_TFLITE = '#2ca02c'
C_ONNX = '#d62728'

LABELS = ['Burn\n(tract-onnx)', 'TensorFlow\nLite', 'ONNX\nRuntime']
X = np.arange(len(LABELS))
W = 0.30  # bar width

def get(fw, field, threads):
    return results[f'{fw}_{threads}t']['summary'][field]

# ---------------------------------------------------------------------------
# Figure 1: Latency (grouped bars with CI error bars)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

lat_1t = [get('burn', 'latency_mean_ms', 1), get('tflite', 'latency_mean_ms', 1), get('onnx', 'latency_mean_ms', 1)]
lat_4t = [get('burn', 'latency_mean_ms', 4), get('tflite', 'latency_mean_ms', 4), get('onnx', 'latency_mean_ms', 4)]
ci_low_1t = [lat_1t[i] - get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_lower_ms', 1 if i==0 else 1) for i in range(3)]
ci_err_1t = [get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_upper_ms', 1) - get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_lower_ms', 1) for i in range(3)]
ci_err_4t = [get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_upper_ms', 4) - get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_lower_ms', 4) for i in range(3)]

# Use asymmetric error bars (mean - lower, upper - mean)
yerr_1t = np.array([[lat_1t[i] - get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_lower_ms', 1 if i==0 else (1 if i==1 else 1)) for i in range(3)],
                    [get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_upper_ms', 1 if i==0 else (1 if i==1 else 1)) - lat_1t[i] for i in range(3)]])
yerr_4t = np.array([[lat_4t[i] - get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_lower_ms', 4 if i==0 else (4 if i==1 else 4)) for i in range(3)],
                    [get('burn' if i==0 else ('tflite' if i==1 else 'onnx'), 'ci_95_upper_ms', 4 if i==0 else (4 if i==1 else 4)) - lat_4t[i] for i in range(3)]])

bars1 = ax.bar(X - W/2, lat_1t, W, label='1 thread', color=C_1T, edgecolor='black', linewidth=0.5,
               yerr=yerr_1t, capsize=3, error_kw={'elinewidth': 1})
bars4 = ax.bar(X + W/2, lat_4t, W, label='4 threads', color=C_4T, edgecolor='black', linewidth=0.5,
               yerr=yerr_4t, capsize=3, error_kw={'elinewidth': 1})

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars4:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(X)
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylabel('Mean Latency (ms)', fontsize=11)
ax.set_title('Inference Latency — MobileNetV2 @ 224×224\nRaspberry Pi 5 (Cortex-A76)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0, 125)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_latency.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_latency.png'), dpi=200, bbox_inches='tight')
print('  fig1_latency')

# ---------------------------------------------------------------------------
# Figure 2: Throughput (horizontal bars are clearer for this comparison)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

tput_1t = [get('burn', 'throughput_fps', 1), get('tflite', 'throughput_fps', 1), get('onnx', 'throughput_fps', 1)]
tput_4t = [get('burn', 'throughput_fps', 4), get('tflite', 'throughput_fps', 4), get('onnx', 'throughput_fps', 4)]

y = np.arange(len(LABELS))
h = 0.30

ax.barh(y - h/2, tput_1t, h, label='1 thread', color=C_1T, edgecolor='black', linewidth=0.5)
ax.barh(y + h/2, tput_4t, h, label='4 threads', color=C_4T, edgecolor='black', linewidth=0.5)

for i, (v1, v4) in enumerate(zip(tput_1t, tput_4t)):
    ax.text(v1 + 1, i - h/2, f'{v1:.1f}', va='center', fontsize=8, fontweight='bold')
    ax.text(v4 + 1, i + h/2, f'{v4:.1f}', va='center', fontsize=8, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(LABELS, fontsize=10)
ax.set_xlabel('Throughput (fps)', fontsize=11)
ax.set_title('Inference Throughput — MobileNetV2 @ 224×224\nRaspberry Pi 5 (Cortex-A76)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 110)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_throughput.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_throughput.png'), dpi=200, bbox_inches='tight')
print('  fig2_throughput')

# ---------------------------------------------------------------------------
# Figure 3: Memory (bars) + Temperature (lines) — dual axis
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7, 4.5))

mem_1t = [get('burn', 'memory_peak_mb', 1), get('tflite', 'memory_peak_mb', 1), get('onnx', 'memory_peak_mb', 1)]
mem_4t = [get('burn', 'memory_peak_mb', 4), get('tflite', 'memory_peak_mb', 4), get('onnx', 'memory_peak_mb', 4)]

bars_m1 = ax1.bar(X - W/2, mem_1t, W, label='Mem (1t)', color=C_1T, edgecolor='black', linewidth=0.5, alpha=0.85)
bars_m4 = ax1.bar(X + W/2, mem_4t, W, label='Mem (4t)', color=C_4T, edgecolor='black', linewidth=0.5, alpha=0.85)

for bar in bars_m1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{bar.get_height():.0f}', ha='center', fontsize=8, fontweight='bold', color=C_1T)
for bar in bars_m4:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{bar.get_height():.0f}', ha='center', fontsize=8, fontweight='bold', color=C_4T)

ax1.set_xticks(X)
ax1.set_xticklabels(LABELS, fontsize=10)
ax1.set_ylabel('Peak Memory (MB)', fontsize=11, color=C_1T)
ax1.tick_params(axis='y', labelcolor=C_1T)

# Temperature on secondary axis
ax2 = ax1.twinx()
temp_1t = [get('burn', 'temp_peak_c', 1), get('tflite', 'temp_peak_c', 1), get('onnx', 'temp_peak_c', 1)]
temp_4t = [get('burn', 'temp_peak_c', 4), get('tflite', 'temp_peak_c', 4), get('onnx', 'temp_peak_c', 4)]

line_t1 = ax2.plot(X, temp_1t, 'o-', color='#d62728', linewidth=2, markersize=7, label='Peak temp (1t)')
line_t4 = ax2.plot(X, temp_4t, 's--', color='#8c564b', linewidth=2, markersize=7, label='Peak temp (4t)')

for i, (t1, t4) in enumerate(zip(temp_1t, temp_4t)):
    offset = 0.08 * (max(temp_1t + temp_4t) - min(temp_1t + temp_4t))
    ax2.annotate(f'{t1:.0f}°C', (X[i], t1), textcoords='offset points', xytext=(5, 5), fontsize=7, color='#d62728')
    ax2.annotate(f'{t4:.0f}°C', (X[i], t4), textcoords='offset points', xytext=(5, -10), fontsize=7, color='#8c564b')

ax2.set_ylabel('Peak Temperature (°C)', fontsize=11, color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')

# Combined legend
lines = [bars_m1, bars_m4, line_t1[0], line_t4[0]]
labels = ['Peak mem (1t)', 'Peak mem (4t)', 'Peak temp (1t)', 'Peak temp (4t)']
ax1.legend(lines, labels, fontsize=8, loc='upper left')

ax1.set_title('Peak Memory and Temperature — MobileNetV2 @ 224×224\nRaspberry Pi 5 (Cortex-A76)', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_memory_temp.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_memory_temp.png'), dpi=200, bbox_inches='tight')
print('  fig3_memory_temp')

# ---------------------------------------------------------------------------
# Figure 4: Multi-thread speedup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))

speedups = [
    get('burn', 'latency_mean_ms', 1) / get('burn', 'latency_mean_ms', 4),
    get('tflite', 'latency_mean_ms', 1) / get('tflite', 'latency_mean_ms', 4),
    get('onnx', 'latency_mean_ms', 1) / get('onnx', 'latency_mean_ms', 4),
]

colors = [C_BURN, C_TFLITE, C_ONNX]
bars = ax.bar(LABELS, speedups, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.4, linewidth=1)
ax.axhline(y=4.0, color='gray', linestyle=':', alpha=0.3, linewidth=1, label='Ideal 4×')

for bar, s in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'{s:.2f}×', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Speedup (1t → 4t)', fontsize=11)
ax.set_title('Multi-thread Scaling', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(speedups) * 1.35)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_speedup.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_speedup.png'), dpi=200, bbox_inches='tight')
print('  fig4_speedup')

print(f'\nDone → {OUTPUT_DIR}/')
