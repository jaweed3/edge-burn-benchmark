#!/usr/bin/env python3
"""Generate paper-quality figures from edge-burn-benchmark results."""

import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load results
def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as f:
        return json.load(f)

results = {}
for fname in ['burn_1t', 'burn_4t', 'tflite_1t', 'tflite_4t', 'onnx_1t', 'onnx_4t']:
    results[fname] = load_json(f'{fname}.json')

# Config
LABELS = ['Burn\n(tract)', 'TensorFlow\nLite', 'ONNX\nRuntime']
COLORS = {'burn': '#E84855', 'tflite': '#3B8EA5', 'onnx': '#F4A261'}
BAR_1T = np.arange(len(LABELS))
BAR_4T = BAR_1T + 0.35
WIDTH = 0.30

def get(fw, field, threads):
    key = f'{fw}_{threads}t'
    return results[key]['summary'][field]

# ---------------------------------------------------------------------------
# Figure 1: Latency comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

lat_1t = [get('burn', 'latency_mean_ms', 1), get('tflite', 'latency_mean_ms', 1), get('onnx', 'latency_mean_ms', 1)]
lat_4t = [get('burn', 'latency_mean_ms', 4), get('tflite', 'latency_mean_ms', 4), get('onnx', 'latency_mean_ms', 4)]
err_1t = [get('burn', 'latency_std_ms', 1), get('tflite', 'latency_std_ms', 1), get('onnx', 'latency_std_ms', 1)]
err_4t = [get('burn', 'latency_std_ms', 4), get('tflite', 'latency_std_ms', 4), get('onnx', 'latency_std_ms', 4)]

bars1 = ax.bar(BAR_1T, lat_1t, WIDTH, label='1 thread', color='#4A4E69', yerr=err_1t, capsize=4, edgecolor='black', linewidth=0.5)
bars4 = ax.bar(BAR_4T, lat_4t, WIDTH, label='4 threads', color='#9A8C98', yerr=err_4t, capsize=4, edgecolor='black', linewidth=0.5)

# Add value labels
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars4:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(BAR_1T + WIDTH/2)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('Inference Latency — MobileNetV2 @ 224×224\nRaspberry Pi 5 (Cortex-A76)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, max(lat_1t) * 1.25)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_latency.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_latency.png'), dpi=150, bbox_inches='tight')
print(f'  fig1_latency.pdf / .png')

# ---------------------------------------------------------------------------
# Figure 2: Throughput
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

tput_1t = [get('burn', 'throughput_fps', 1), get('tflite', 'throughput_fps', 1), get('onnx', 'throughput_fps', 1)]
tput_4t = [get('burn', 'throughput_fps', 4), get('tflite', 'throughput_fps', 4), get('onnx', 'throughput_fps', 4)]

bars1 = ax.bar(BAR_1T, tput_1t, WIDTH, label='1 thread', color='#4A4E69', edgecolor='black', linewidth=0.5)
bars4 = ax.bar(BAR_4T, tput_4t, WIDTH, label='4 threads', color='#9A8C98', edgecolor='black', linewidth=0.5)

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars4:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(BAR_1T + WIDTH/2)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel('Throughput (fps)', fontsize=12)
ax.set_title('Inference Throughput — MobileNetV2 @ 224×224\nRaspberry Pi 5 (Cortex-A76)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_throughput.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_throughput.png'), dpi=150, bbox_inches='tight')
print(f'  fig2_throughput.pdf / .png')

# ---------------------------------------------------------------------------
# Figure 3: Memory + Temperature dual-axis
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))

x = np.arange(len(LABELS))
w = 0.25

mem_1t = [get('burn', 'memory_peak_mb', 1), get('tflite', 'memory_peak_mb', 1), get('onnx', 'memory_peak_mb', 1)]
mem_4t = [get('burn', 'memory_peak_mb', 4), get('tflite', 'memory_peak_mb', 4), get('onnx', 'memory_peak_mb', 4)]

ax1.bar(x - w/2, mem_1t, w, label='1 thread', color='#4A4E69', edgecolor='black', linewidth=0.5)
ax1.bar(x + w/2, mem_4t, w, label='4 threads', color='#9A8C98', edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Peak Memory (MB)', fontsize=12, color='#4A4E69')
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, fontsize=11)

# Add memory labels
for i, (m1, m4) in enumerate(zip(mem_1t, mem_4t)):
    ax1.text(x[i] - w/2, m1 + 10, f'{m1:.0f}', ha='center', fontsize=8, fontweight='bold', color='#4A4E69')
    ax1.text(x[i] + w/2, m4 + 10, f'{m4:.0f}', ha='center', fontsize=8, fontweight='bold', color='#9A8C98')

ax2 = ax1.twinx()
temp_1t = [get('burn', 'temp_peak_c', 1), get('tflite', 'temp_peak_c', 1), get('onnx', 'temp_peak_c', 1)]
temp_4t = [get('burn', 'temp_peak_c', 4), get('tflite', 'temp_peak_c', 4), get('onnx', 'temp_peak_c', 4)]

ax2.plot(x, temp_1t, 'o-', color='#E84855', linewidth=2, markersize=8, label='Peak temp (1t)')
ax2.plot(x, temp_4t, 's--', color='#E84855', linewidth=2, markersize=8, label='Peak temp (4t)')
ax2.set_ylabel('Peak Temperature (°C)', fontsize=12, color='#E84855')
ax2.tick_params(axis='y', labelcolor='#E84855')

# Add temp labels
for i, (t1, t4) in enumerate(zip(temp_1t, temp_4t)):
    ax2.text(x[i] + 0.1, t1 + 0.5, f'{t1:.0f}°C', fontsize=8, color='#E84855')
    ax2.text(x[i] + 0.1, t4 - 0.5, f'{t4:.0f}°C', fontsize=8, color='#E84855')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

ax1.set_title('Peak Memory & Temperature — MobileNetV2 @ 224×224\nRaspberry Pi 5 (Cortex-A76)', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_memory_temp.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_memory_temp.png'), dpi=150, bbox_inches='tight')
print(f'  fig3_memory_temp.pdf / .png')

# ---------------------------------------------------------------------------
# Figure 4: Speedup factor (4t / 1t)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

speedups = [
    get('burn', 'latency_mean_ms', 1) / get('burn', 'latency_mean_ms', 4),
    get('tflite', 'latency_mean_ms', 1) / get('tflite', 'latency_mean_ms', 4),
    get('onnx', 'latency_mean_ms', 1) / get('onnx', 'latency_mean_ms', 4),
]

bar_colors = [COLORS['burn'], COLORS['tflite'], COLORS['onnx']]
ax.bar(LABELS, speedups, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.5)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.3, label='Ideal 2× (4 cores)')
ax.axhline(y=4.0, color='green', linestyle=':', alpha=0.2, label='Perfect 4×')

for i, s in enumerate(speedups):
    ax.text(i, s + 0.05, f'{s:.2f}×', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Speedup Factor (1t → 4t)', fontsize=12)
ax.set_title('Multi-thread Scaling', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(speedups) * 1.4)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_speedup.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_speedup.png'), dpi=150, bbox_inches='tight')
print(f'  fig4_speedup.pdf / .png')

print(f'\nAll figures saved to {OUTPUT_DIR}/')
