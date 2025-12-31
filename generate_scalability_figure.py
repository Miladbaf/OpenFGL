"""
Ultra-compact 2-column figure for IEEE paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Ultra-compact style
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6

# Load results
results = np.load('scalability_results.npy', allow_pickle=True).item()

METHODS = ['fedavg', 'fedala', 'fedala_r']
METHOD_LABELS = {'fedavg': 'FedAvg', 'fedala': 'FedALA', 'fedala_r': 'FedALA-R'}
COLORS = {'fedavg': '#1f77b4', 'fedala': '#ff7f0e', 'fedala_r': '#2ca02c'}
MARKERS = {'fedavg': 'o', 'fedala': 's', 'fedala_r': '^'}
DATASETS = ['Cora', 'CiteSeer', 'PubMed']
CLIENT_COUNTS = [5, 10, 15, 20]

# Create compact 1x3 figure - fits in two columns
fig, axes = plt.subplots(1, 3, figsize=(7, 2))
fig.subplots_adjust(wspace=0.35)

for idx, (ax, dataset) in enumerate(zip(axes, DATASETS)):
    for method in METHODS:
        means = []
        stds = []
        
        for num_clients in CLIENT_COUNTS:
            accs = [a for a in results[method][dataset][num_clients] if not np.isnan(a)]
            if accs:
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(np.nan)
                stds.append(0)
        
        ax.errorbar(CLIENT_COUNTS, means, yerr=stds,
                   label=METHOD_LABELS[method], marker=MARKERS[method],
                   markersize=4, linewidth=1.2, capsize=2.5, capthick=0.8,
                   color=COLORS[method], alpha=0.9)
    
    ax.set_xlabel('Clients', fontsize=7)
    if idx == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=7)
    ax.set_title(f'{dataset}', fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.set_xticks(CLIENT_COUNTS)
    
    if idx == 2:
        ax.legend(loc='lower left', framealpha=0.9, fontsize=6, ncol=1)

plt.tight_layout()
plt.savefig('scalability_twoCol.pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.savefig('scalability_twoCol.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
print("✅ Two-column figure saved!")

# LaTeX for two-column figure
print("\n\\begin{figure*}[t]")
print("\\centering")
print("\\includegraphics[width=\\textwidth]{scalability_twoCol.pdf}")
print("\\caption{Scalability analysis: Accuracy vs. number of clients. FedALA-R maintains improvements as federation size increases.}")
print("\\label{fig:scalability}")
print("\\end{figure*}")
