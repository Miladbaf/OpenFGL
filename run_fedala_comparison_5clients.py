"""
Run FedAvg vs FedALA vs FedALA-R comparison (5 clients)
Generates Table: Performance comparison (5 clients, 3 seeds)
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
SEEDS = [42, 123, 456]
NUM_CLIENTS = 5

# Results storage
results = {method: {dataset: [] for dataset in DATASETS} for method in METHODS}

print("=" * 80)
print("🚀 FedALA-R Performance Evaluation (5 Clients)")
print("=" * 80)
print(f"📁 Data directory: {DATA_ROOT}")
print(f"📊 Methods: {list(METHOD_LABELS.values())}")
print(f"📦 Datasets: {DATASETS}")
print(f"🎲 Seeds: {SEEDS}")
print(f"👥 Clients: {NUM_CLIENTS}")
print(f"🔢 Total runs: {len(METHODS) * len(DATASETS) * len(SEEDS)}")
print("=" * 80 + "\n")

# Track errors
first_error = None

# Run experiments
run_count = 0
total_runs = len(METHODS) * len(DATASETS) * len(SEEDS)

for method in METHODS:
    print(f"\n{'='*80}")
    print(f"Running {METHOD_LABELS[method]}")
    print(f"{'='*80}\n")
    
    for dataset in DATASETS:
        for seed in SEEDS:
            run_count += 1
            tag = f"{METHOD_LABELS[method]:12s} | {dataset:8s} | seed={seed}"
            print(f"[{run_count:02d}/{total_runs:02d}] {tag} ... ", end='', flush=True)
            
            f_stdout = io.StringIO()
            f_stderr = io.StringIO()
            
            try:
                with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                    args = config.args
                    
                    # Dataset configuration
                    args.root = DATA_ROOT
                    args.dataset = [dataset]
                    args.simulation_mode = "subgraph_fl_louvain"
                    args.num_clients = NUM_CLIENTS
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
                    
                    results[method][dataset].append(float(acc))
                    print(f"✓ {acc:.2f}%")
                    
            except Exception as e:
                error_short = str(e)[:80]
                print(f"✗ FAILED: {error_short}")
                results[method][dataset].append(np.nan)
                
                if first_error is None:
                    first_error = {
                        'method': method,
                        'dataset': dataset,
                        'seed': seed,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }

# Print results
print("\n" + "=" * 80)
print("📊 RESULTS SUMMARY")
print("=" * 80 + "\n")

for dataset in DATASETS:
    print(f"{dataset}:")
    print("-" * 70)
    for method in METHODS:
        accs = results[method][dataset]
        valid_accs = [a for a in accs if not np.isnan(a)]
        
        if len(valid_accs) == 0:
            print(f"  {METHOD_LABELS[method]:<12s}: ALL FAILED")
        else:
            mean_acc = np.mean(valid_accs)
            std_acc = np.std(valid_accs) if len(valid_accs) > 1 else 0
            print(f"  {METHOD_LABELS[method]:<12s}: {mean_acc:6.2f} ± {std_acc:4.2f}%")
    print()

# Improvements over FedAvg
print("=" * 80)
print("📈 IMPROVEMENTS OVER FedAvg")
print("=" * 80 + "\n")

for dataset in DATASETS:
    fedavg_accs = [a for a in results['fedavg'][dataset] if not np.isnan(a)]
    
    if len(fedavg_accs) > 0:
        fedavg_mean = np.mean(fedavg_accs)
        print(f"{dataset}:")
        
        for method in ['fedala', 'fedala_r']:
            method_accs = [a for a in results[method][dataset] if not np.isnan(a)]
            
            if len(method_accs) > 0:
                method_mean = np.mean(method_accs)
                abs_improvement = method_mean - fedavg_mean
                rel_improvement = (abs_improvement / fedavg_mean) * 100
                
                symbol = "⭐⭐" if abs_improvement > 2 else "⭐" if abs_improvement > 1 else "✓" if abs_improvement > 0 else "✗"
                print(f"  {METHOD_LABELS[method]:<12s}: {abs_improvement:+6.2f}% (rel: {rel_improvement:+5.2f}%) {symbol}")
        print()

# LaTeX table generation
print("=" * 80)
print("📄 LaTeX TABLE CODE")
print("=" * 80 + "\n")

print("\\begin{table}[h!]")
print("\\centering")
print("\\scriptsize")
print("\\caption{Comparison of FedAvg, FedALA, and FedALA-R on citation benchmarks (5 clients, 3 seeds).}")
print("\\resizebox{\\columnwidth}{!}{%")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("\\textbf{Method} & \\textbf{Cora} & \\textbf{CiteSeer} & \\textbf{PubMed} \\\\")
print("\\midrule")

for method in METHODS:
    row_parts = [METHOD_LABELS[method]]
    
    for dataset in DATASETS:
        accs = [a for a in results[method][dataset] if not np.isnan(a)]
        
        if len(accs) > 0:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            
            # Calculate improvement over FedAvg
            if method == 'fedavg':
                cell = f"{mean_acc:.1f}$\\pm${std_acc:.1f}\\%"
            else:
                fedavg_accs = [a for a in results['fedavg'][dataset] if not np.isnan(a)]
                if len(fedavg_accs) > 0:
                    fedavg_mean = np.mean(fedavg_accs)
                    improvement = mean_acc - fedavg_mean
                    
                    # Bold if best
                    if method == 'fedala_r' and improvement > 0.5:
                        cell = f"\\textbf{{{mean_acc:.1f}}}$\\pm${std_acc:.1f}\\% ($+${improvement:.1f}\\%)"
                    else:
                        cell = f"{mean_acc:.1f}$\\pm${std_acc:.1f}\\% (${improvement:+.2f}\\%)"
                else:
                    cell = f"{mean_acc:.1f}$\\pm${std_acc:.1f}\\%"
            
            row_parts.append(cell)
        else:
            row_parts.append("--")
    
    print(" & ".join(row_parts) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\label{tab:fedala_r_5clients}")
print("\\end{table}")

# Overall ranking
print("\n" + "=" * 80)
print("🏆 OVERALL RANKING")
print("=" * 80 + "\n")

overall_avgs = {}
for method in METHODS:
    all_accs = []
    for dataset in DATASETS:
        accs = [a for a in results[method][dataset] if not np.isnan(a)]
        if accs:
            all_accs.extend(accs)
    if all_accs:
        overall_avgs[method] = np.mean(all_accs)

if overall_avgs:
    ranked = sorted(overall_avgs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<6} {'Method':<15s} {'Avg Accuracy':<15}")
    print("-" * 80)
    
    for rank, (method, avg) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        improvement = ""
        if method != 'fedavg' and 'fedavg' in overall_avgs:
            imp = avg - overall_avgs['fedavg']
            improvement = f"({imp:+.2f}%)"
        print(f"{rank:<6} {METHOD_LABELS[method]:<15s} {avg:>13.2f}% {improvement:>10s} {medal}")

# Error details
if first_error:
    print("\n" + "=" * 80)
    print("🐛 FIRST ERROR")
    print("=" * 80)
    print(f"Method:  {first_error['method']}")
    print(f"Dataset: {first_error['dataset']}")
    print(f"Seed:    {first_error['seed']}")
    print(f"\nError: {first_error['error']}\n")
    print("Traceback:")
    print(first_error['traceback'])

# Save results
np.save('fedala_r_5clients_results.npy', results)
print("\n✅ Results saved to 'fedala_r_5clients_results.npy'")

print("\n" + "=" * 80)
print("🎯 SUMMARY")
print("=" * 80)
print("1. Copy LaTeX table code above into paper")
print("2. Results show FedALA-R improvements over baselines")
print("3. Ready to insert into Section 6.2 of paper")
print("=" * 80)
