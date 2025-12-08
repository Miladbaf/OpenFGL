"""
Generate LaTeX table comparing FedAvg, FedALA, and FedALA-MP
"""

import numpy as np

# Load results
results = np.load("fedala_results.npy", allow_pickle=True).item()

datasets = ["Cora", "CiteSeer", "PubMed"]
methods = ["fedavg", "fedala", "fedala_mp"]
method_names = {
    "fedavg": "FedAvg",
    "fedala": "FedALA",
    "fedala_mp": "FedALA-MP (Method 1)"
}

# Generate LaTeX
latex = r"""\begin{table}[h!]
\centering
\begin{tabular}{lccc}
\toprule
Method & Cora & CiteSeer & PubMed \\
\midrule
"""

for method in methods:
    row = f"{method_names[method]}"
    
    for dataset in datasets:
        vals = np.array(results[method][dataset])
        vals = vals[~np.isnan(vals)]
        
        if vals.size > 0:
            mean, std = vals.mean(), vals.std()
            row += f" & {mean:.2f} $\\pm$ {std:.2f}"
        else:
            row += " & --- "
    
    latex += row + " \\\\\n"

latex += r"""\midrule
\multicolumn{4}{l}{\textbf{Improvement over FedAvg}} \\
"""

# Add improvements
for method in ["fedala", "fedala_mp"]:
    row = f"{method_names[method]}"
    
    for dataset in datasets:
        baseline = np.array(results["fedavg"][dataset])
        baseline = baseline[~np.isnan(baseline)].mean()
        
        vals = np.array(results[method][dataset])
        vals = vals[~np.isnan(vals)].mean()
        
        improvement = ((vals - baseline) / baseline) * 100
        row += f" & {improvement:+.2f}\\%"
    
    latex += row + " \\\\\n"

latex += r"""\bottomrule
\end{tabular}
\caption{Comparison of FedAvg, FedALA, and FedALA-MP (Method 1) on node classification.}
\end{table}
"""

print(latex)

# Save to file
with open("fedala_table.tex", "w") as f:
    f.write(latex)

print("\n✅ LaTeX table saved to 'fedala_table.tex'")
