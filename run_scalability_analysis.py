"""
Scalability Analysis: FedAvg vs FedALA vs FedALA-R with varying client counts
Tests: 5, 10, 15, 20 clients
For paper section: Scalability Insights
"""

import os
from contextlib import redirect_stdout, redirect_stderr
import io
import numpy as np
import torch
import traceback

# PyTorch 2.6+ Compatibility
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
print("✓ Patched torch.load for PyTorch 2.6+ compatibility\n")

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# Setup
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO_ROOT, 'data')

# Configuration
METHODS = ['fedavg', 'fedala', 'fedala_r']
METHOD_LABELS = {
    'fedavg': 'FedAvg',
    'fedala': 'FedALA',
    'fedala_r': 'FedALA-R'
}

DATASETS = ['Cora', 'CiteSeer', 'PubMed']
CLIENT_COUNTS = [5, 10, 15, 20]  # Scalability test
SEEDS = [42, 123, 456]

# Results storage: results[method][dataset][num_clients] = [accuracies]
results = {
    method: {
        dataset: {
            num_clients: [] for num_clients in CLIENT_COUNTS
        } for dataset in DATASETS
    } for method in METHODS
}

print("=" * 80)
print("🚀 SCALABILITY ANALYSIS: Varying Number of Clients")
print("=" * 80)
print(f"📁 Data directory: {DATA_ROOT}")
print(f"📊 Methods: {list(METHOD_LABELS.values())}")
print(f"📦 Datasets: {DATASETS}")
print(f"👥 Client counts: {CLIENT_COUNTS}")
print(f"🎲 Seeds: {SEEDS}")
print(f"🔢 Total runs: {len(METHODS) * len(DATASETS) * len(CLIENT_COUNTS) * len(SEEDS)}")
print("=" * 80 + "\n")

# Track errors
errors = []

# Run experiments
run_count = 0
total_runs = len(METHODS) * len(DATASETS) * len(CLIENT_COUNTS) * len(SEEDS)

for num_clients in CLIENT_COUNTS:
    print(f"\n{'='*80}")
    print(f"🔬 TESTING WITH {num_clients} CLIENTS")
    print(f"{'='*80}\n")
    
    for method in METHODS:
        print(f"\n  Method: {METHOD_LABELS[method]}")
        print(f"  {'-'*70}")
        
        for dataset in DATASETS:
            for seed in SEEDS:
                run_count += 1
                tag = f"K={num_clients:2d} | {METHOD_LABELS[method]:12s} | {dataset:8s} | seed={seed}"
                print(f"  [{run_count:03d}/{total_runs:03d}] {tag} ... ", end='', flush=True)
                
                f_stdout = io.StringIO()
                f_stderr = io.StringIO()
                
                try:
                    with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                        args = config.args
                        
                        # Dataset configuration
                        args.root = DATA_ROOT
                        args.dataset = [dataset]
                        args.simulation_mode = "subgraph_fl_louvain"
                        args.num_clients = num_clients
                        args.fl_algorithm = method
                        args.model = ["gcn"]
                        
                        # Training configuration
                        args.num_rounds = 100
                        args.local_epoch = 5
                        args.lr = 0.01
                        args.weight_decay = 5e-4
                        args.metrics = ["accuracy"]
                        args.seed = seed
                        
                        # Train
                        trainer = FGLTrainer(args)
                        trainer.train()
                        
                        # Extract accuracy
                        acc = np.nan
                        if hasattr(trainer, "evaluation_result"):
                            metric_name = args.metrics[0]
                            task = getattr(args, "task", None)
                            
                            if task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
                                key = f"best_test_{metric_name}"
                            else:
                                key = f"best_{metric_name}"
                            
                            acc = trainer.evaluation_result.get(key, np.nan)
                        
                        # Convert to percentage
                        if not np.isnan(acc) and acc < 1.0:
                            acc = acc * 100
                        
                        results[method][dataset][num_clients].append(float(acc))
                        print(f"✓ {acc:.2f}%")
                        
                except Exception as e:
                    error_short = str(e)[:60]
                    print(f"✗ FAILED: {error_short}")
                    results[method][dataset][num_clients].append(np.nan)
                    
                    errors.append({
                        'num_clients': num_clients,
                        'method': method,
                        'dataset': dataset,
                        'seed': seed,
                        'error': str(e)
                    })

# ============================================================================
# ANALYSIS & RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("📊 SCALABILITY RESULTS SUMMARY")
print("=" * 80 + "\n")

# Summary table for each dataset
for dataset in DATASETS:
    print(f"\n{dataset}:")
    print("-" * 80)
    print(f"{'Method':<12} {'5 clients':>12} {'10 clients':>12} {'15 clients':>12} {'20 clients':>12}")
    print("-" * 80)
    
    for method in METHODS:
        row = [METHOD_LABELS[method]]
        for num_clients in CLIENT_COUNTS:
            accs = [a for a in results[method][dataset][num_clients] if not np.isnan(a)]
            if len(accs) > 0:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                row.append(f"{mean_acc:.2f}±{std_acc:.2f}")
            else:
                row.append("FAILED")
        
        print(f"{row[0]:<12} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")
    print()

# Improvement analysis
print("\n" + "=" * 80)
print("📈 FEDALA-R IMPROVEMENT OVER FedAvg")
print("=" * 80 + "\n")

improvement_data = {}

for dataset in DATASETS:
    print(f"\n{dataset}:")
    print("-" * 80)
    print(f"{'Clients':<12} {'FedAvg':>12} {'FedALA-R':>12} {'Improvement':>15}")
    print("-" * 80)
    
    improvement_data[dataset] = {}
    
    for num_clients in CLIENT_COUNTS:
        fedavg_accs = [a for a in results['fedavg'][dataset][num_clients] if not np.isnan(a)]
        fedala_r_accs = [a for a in results['fedala_r'][dataset][num_clients] if not np.isnan(a)]
        
        if len(fedavg_accs) > 0 and len(fedala_r_accs) > 0:
            fedavg_mean = np.mean(fedavg_accs)
            fedala_r_mean = np.mean(fedala_r_accs)
            improvement = fedala_r_mean - fedavg_mean
            
            improvement_data[dataset][num_clients] = {
                'fedavg': fedavg_mean,
                'fedala_r': fedala_r_mean,
                'improvement': improvement
            }
            
            symbol = "⭐⭐" if improvement > 2 else "⭐" if improvement > 1 else "✓" if improvement > 0 else "✗"
            print(f"{num_clients:<12} {fedavg_mean:>11.2f}% {fedala_r_mean:>11.2f}% {improvement:>+14.2f}% {symbol}")
    print()

# LaTeX table generation
print("\n" + "=" * 80)
print("📄 LaTeX TABLE CODE: Scalability Results")
print("=" * 80 + "\n")

print("\\begin{table}[h!]")
print("\\centering")
print("\\scriptsize")
print("\\caption{Scalability analysis: Performance across varying numbers of clients.}")
print("\\begin{tabular}{llcccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Method} & \\textbf{5 clients} & \\textbf{10 clients} & \\textbf{15 clients} & \\textbf{20 clients} \\\\")
print("\\midrule")

for dataset in DATASETS:
    print(f"\\multirow{{3}}{{*}}{{{dataset}}}")
    for method in METHODS:
        method_str = " & " + METHOD_LABELS[method]
        
        for num_clients in CLIENT_COUNTS:
            accs = [a for a in results[method][dataset][num_clients] if not np.isnan(a)]
            if len(accs) > 0:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                method_str += f" & {mean_acc:.2f}$\\pm${std_acc:.2f}"
            else:
                method_str += " & --"
        
        method_str += " \\\\"
        print(method_str)
    
    if dataset != DATASETS[-1]:
        print("\\midrule")

print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:scalability}")
print("\\end{table}")

# Trend analysis
print("\n" + "=" * 80)
print("📊 TREND ANALYSIS")
print("=" * 80 + "\n")

print("How accuracy evolves with more clients:")
print("-" * 80)

for dataset in DATASETS:
    print(f"\n{dataset}:")
    
    for method in METHODS:
        accuracies = []
        for num_clients in CLIENT_COUNTS:
            accs = [a for a in results[method][dataset][num_clients] if not np.isnan(a)]
            if accs:
                accuracies.append(np.mean(accs))
        
        if len(accuracies) == len(CLIENT_COUNTS):
            trend = "increasing" if accuracies[-1] > accuracies[0] else "decreasing"
            change = accuracies[-1] - accuracies[0]
            print(f"  {METHOD_LABELS[method]:<12}: {accuracies[0]:.2f}% → {accuracies[-1]:.2f}% ({change:+.2f}%, {trend})")

print("\n" + "-" * 80)
print("FedALA-R improvement trend:")
print("-" * 80)

for dataset in DATASETS:
    if dataset in improvement_data:
        improvements = [improvement_data[dataset][k]['improvement']
                       for k in CLIENT_COUNTS if k in improvement_data[dataset]]
        
        if len(improvements) == len(CLIENT_COUNTS):
            trend = "increasing" if improvements[-1] > improvements[0] else "decreasing"
            change = improvements[-1] - improvements[0]
            print(f"{dataset:12}: {improvements[0]:+.2f}% → {improvements[-1]:+.2f}% ({trend}, Δ={change:+.2f}%)")

# Key insights
print("\n" + "=" * 80)
print("🔍 KEY INSIGHTS FOR PAPER")
print("=" * 80 + "\n")

print("1. Accuracy Evolution:")
print("   - Does accuracy increase or decrease with more clients?")
print("   - Trade-off: More data (↑) vs. More heterogeneity (↓)")
print()

print("2. FedALA-R Benefit:")
print("   - Does improvement increase with more clients?")
print("   - More clients = more collective consensus to leverage")
print()

print("3. Optimal Client Count:")
for dataset in DATASETS:
    best_k = None
    best_acc = -1
    for num_clients in CLIENT_COUNTS:
        accs = [a for a in results['fedala_r'][dataset][num_clients] if not np.isnan(a)]
        if accs:
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_k = num_clients
    if best_k:
        print(f"   - {dataset}: Best with {best_k} clients ({best_acc:.2f}%)")

print()
print("4. Computational Overhead:")
print("   - Server: O(Kd) scales linearly with clients")
print("   - Communication: 2d per round (constant per client)")

# Error summary
if errors:
    print("\n" + "=" * 80)
    print(f"⚠️  ERRORS: {len(errors)} runs failed")
    print("=" * 80)
    error_summary = {}
    for err in errors:
        key = (err['num_clients'], err['method'], err['dataset'])
        error_summary[key] = error_summary.get(key, 0) + 1
    
    for (k, m, d), count in sorted(error_summary.items()):
        print(f"  K={k}, {m}, {d}: {count} failures")

# Save results
np.save('scalability_results.npy', results)
print("\n✅ Results saved to 'scalability_results.npy'")

print("\n" + "=" * 80)
print("🎯 NEXT STEPS")
print("=" * 80)
print("1. Copy LaTeX table into paper Section 6.3")
print("2. Analyze trends and write discussion")
print("3. Create plots using saved results if needed")
print("4. Focus discussion on key findings above")
print("=" * 80)
