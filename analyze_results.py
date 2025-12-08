#!/usr/bin/env python3
"""
Visualize FedALA results
Structure: {method: {dataset: [accuracies]}}
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Load results
try:
    results = np.load('fedala_results.npy', allow_pickle=True).item()
except FileNotFoundError:
    print("❌ Error: fedala_results.npy not found!")
    sys.exit(1)

methods = list(results.keys())
datasets = list(results[methods[0]].keys())

# Transpose to dataset-first
results_by_dataset = {}
for dataset in datasets:
    results_by_dataset[dataset] = {}
    for method in methods:
        results_by_dataset[dataset][method] = results[method][dataset]

method_labels = ['FedAvg', 'FedALA', 'FedALA-MP']
colors = ['#2ecc71', '#3498db', '#e74c3c']

# Plot 1: Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    
    # Extract mean accuracies
    means = []
    stds = []
    for method in methods:
        data = results_by_dataset[dataset][method]
        means.append(np.mean(data))
        stds.append(np.std(data))
    
    # Bar plot
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add improvement percentage for FedALA methods
        if i > 0:  # Not FedAvg
            improvement = ((mean - means[0]) / means[0]) * 100
            if improvement > 0.1:  # Only show if improvement > 0.1%
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'+{improvement:.1f}%',
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Formatting
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=11)
    ax.set_ylim([min(means) - 0.03, max(means) + max(stds) + 0.03])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('FedALA: Comparison of Methods Across Datasets',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fedala_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: fedala_comparison.png")

# Plot 2: Improvement plot
fig, ax = plt.subplots(figsize=(10, 6))

improvements = {method: [] for method in ['fedala', 'fedala_mp']}
for dataset in datasets:
    fedavg_mean = np.mean(results_by_dataset[dataset]['fedavg'])
    for method in ['fedala', 'fedala_mp']:
        method_mean = np.mean(results_by_dataset[dataset][method])
        improvement = ((method_mean - fedavg_mean) / fedavg_mean) * 100
        improvements[method].append(improvement)

x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, improvements['fedala'], width,
               label='FedALA', color='#3498db', alpha=0.8,
               edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, improvements['fedala_mp'], width,
               label='FedALA-MP', color='#e74c3c', alpha=0.8,
               edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Improvement over FedAvg (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_title('FedALA: Improvements over FedAvg Baseline',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(fontsize=11, frameon=True, shadow=True, loc='upper left')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fedala_improvements.png', dpi=300, bbox_inches='tight')
print("✅ Saved: fedala_improvements.png")

# Plot 3: Detailed per-seed results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

seeds = [42, 123, 456]
seed_colors = ['#3498db', '#e74c3c', '#f39c12']

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    
    x = np.arange(len(methods))
    width = 0.25
    
    for seed_idx, seed in enumerate(seeds):
        values = []
        for method in methods:
            data = results_by_dataset[dataset][method]
            values.append(data[seed_idx] if seed_idx < len(data) else 0)
        
        offset = (seed_idx - 1) * width
        ax.bar(x + offset, values, width, label=f'Seed {seed}',
               color=seed_colors[seed_idx], alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('FedALA: Per-Seed Results', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fedala_per_seed.png', dpi=300, bbox_inches='tight')
print("✅ Saved: fedala_per_seed.png")

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
for dataset in datasets:
    print(f"\n{dataset}:")
    fedavg = np.mean(results_by_dataset[dataset]['fedavg'])
    fedala = np.mean(results_by_dataset[dataset]['fedala'])
    fedala_mp = np.mean(results_by_dataset[dataset]['fedala_mp'])
    
    print(f"  FedAvg:    {fedavg:.4f} ({fedavg*100:.2f}%)")
    print(f"  FedALA:    {fedala:.4f} ({fedala*100:.2f}%) [{((fedala-fedavg)/fedavg*100):+.2f}%]")
    print(f"  FedALA-MP: {fedala_mp:.4f} ({fedala_mp*100:.2f}%) [{((fedala_mp-fedavg)/fedavg*100):+.2f}%]")

print("\n✅ All visualizations saved!")
print("   - fedala_comparison.png (main comparison)")
print("   - fedala_improvements.png (improvement bars)")
print("   - fedala_per_seed.png (per-seed details)")
