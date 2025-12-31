"""
Compare FedALA vs FedALA-Prox
Shows the benefit of adding proximal regularization to adaptive aggregation
"""

import os
from contextlib import redirect_stdout, redirect_stderr
import io
import numpy as np
import torch
import traceback

# ========== PyTorch 2.6+ Compatibility ==========
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
METHODS = ['fedavg', 'fedala', 'fedala_prox']
METHOD_LABELS = {
    'fedavg': 'FedAvg (Baseline)',
    'fedala': 'FedALA (Adaptive Init)',
    'fedala_prox': 'FedALA-Prox (Adaptive + Proximal)'
}

DATASETS = ['Cora', 'CiteSeer', 'PubMed']
SEEDS = [42, 123, 456]

# Results storage
results = {method: {dataset: [] for dataset in DATASETS} for method in METHODS}

print("🚀 FedALA vs FedALA-Prox Comparison (Total: {} runs)".format(len(METHODS) * len(DATASETS) * len(SEEDS)))
print(f"📁 Data directory: {DATA_ROOT}")
print(f"📊 Methods:")
for method in METHODS:
    print(f"   - {method}: {METHOD_LABELS[method]}")
print(f"📦 Datasets: {DATASETS}")
print(f"🎲 Seeds: {SEEDS}\n")
print("=" * 70 + "\n")

# Track errors
first_error = None

# Run experiments
run_count = 0
total_runs = len(METHODS) * len(DATASETS) * len(SEEDS)

for method in METHODS:
    for dataset in DATASETS:
        for seed in SEEDS:
            run_count += 1
            tag = f"{method:12s} | {dataset:8s} | seed={seed}"
            print(f"[{run_count:02d}/{total_runs:02d}] {tag} ... ", end='', flush=True)
            
            f_stdout = io.StringIO()
            f_stderr = io.StringIO()
            
            try:
                with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                    args = config.args
                    
                    # Setup
                    args.root = DATA_ROOT
                    args.dataset = [dataset]
                    args.simulation_mode = "subgraph_fl_louvain"
                    args.num_clients = 5
                    args.fl_algorithm = method
                    args.model = ["gcn"]
                    
                    # Training
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
print("\n" + "=" * 70)
print("📊 RESULTS SUMMARY")
print("=" * 70 + "\n")

for dataset in DATASETS:
    print(f"{dataset}:")
    print("-" * 50)
    for method in METHODS:
        accs = results[method][dataset]
        valid_accs = [a for a in accs if not np.isnan(a)]
        
        if len(valid_accs) == 0:
            print(f"  {METHOD_LABELS[method]:<35s}: ALL FAILED")
        else:
            mean_acc = np.mean(valid_accs)
            std_acc = np.std(valid_accs) if len(valid_accs) > 1 else 0
            print(f"  {METHOD_LABELS[method]:<35s}: {mean_acc:6.2f} ± {std_acc:4.2f}")
    print()

# Improvements analysis
print("=" * 70)
print("📈 IMPROVEMENTS OVER FedAvg")
print("=" * 70 + "\n")

for dataset in DATASETS:
    print(f"{dataset}:")
    fedavg_accs = [a for a in results['fedavg'][dataset] if not np.isnan(a)]
    
    if len(fedavg_accs) > 0:
        fedavg_mean = np.mean(fedavg_accs)
        
        for method in ['fedala', 'fedala_prox']:
            method_accs = [a for a in results[method][dataset] if not np.isnan(a)]
            
            if len(method_accs) > 0:
                method_mean = np.mean(method_accs)
                improvement = ((method_mean - fedavg_mean) / fedavg_mean) * 100
                
                symbol = "⭐⭐" if improvement > 5 else "⭐" if improvement > 3 else "✓" if improvement > 1 else ""
                print(f"  {METHOD_LABELS[method]:<35s}: {improvement:+6.2f}% {symbol}")
    print()

# FedALA-Prox vs FedALA comparison
print("=" * 70)
print("🎯 PROXIMAL TERM BENEFIT (FedALA-Prox vs FedALA)")
print("=" * 70 + "\n")

print(f"{'Dataset':<12} {'FedALA':>10} {'FedALA-Prox':>13} {'Benefit':>10}")
print("-" * 70)

total_benefit = []
for dataset in DATASETS:
    fedala_accs = [a for a in results['fedala'][dataset] if not np.isnan(a)]
    prox_accs = [a for a in results['fedala_prox'][dataset] if not np.isnan(a)]
    
    if len(fedala_accs) > 0 and len(prox_accs) > 0:
        fedala_mean = np.mean(fedala_accs)
        prox_mean = np.mean(prox_accs)
        benefit = prox_mean - fedala_mean
        total_benefit.append(benefit)
        
        symbol = "⭐⭐⭐" if benefit > 5 else "⭐⭐" if benefit > 3 else "⭐" if benefit > 1 else "✓" if benefit > 0 else "✗"
        print(f"{dataset:<12} {fedala_mean:>9.2f}% {prox_mean:>12.2f}% {benefit:>+9.2f}% {symbol}")

if total_benefit:
    avg_benefit = np.mean(total_benefit)
    print("-" * 70)
    print(f"{'AVERAGE':<12} {'':<10} {'':<13} {avg_benefit:>+9.2f}%")

print("\n" + "=" * 70)
print("📊 OVERALL RANKING")
print("=" * 70 + "\n")

# Calculate overall averages
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
    
    print(f"{'Rank':<6} {'Method':<35s} {'Avg Accuracy':<15}")
    print("-" * 70)
    
    for rank, (method, avg) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        print(f"{rank:<6} {METHOD_LABELS[method]:<35s} {avg:>13.2f}% {medal}")

# Error details
if first_error:
    print("\n" + "=" * 70)
    print("🐛 FIRST ERROR")
    print("=" * 70)
    print(f"Method:  {first_error['method']}")
    print(f"Dataset: {first_error['dataset']}")
    print(f"Seed:    {first_error['seed']}")
    print(f"\nError: {first_error['error']}\n")
    print("Traceback:")
    print(first_error['traceback'])

# Save results
np.save('fedala_prox_results.npy', results)
print("\n✅ Complete! Results saved to 'fedala_prox_results.npy'")

# Final summary
print("\n" + "=" * 70)
print("🎯 KEY FINDINGS")
print("=" * 70)
print("\nFedALA-Prox combines two complementary strategies:")
print("  1. FedALA: Adaptive initialization (smart starting point)")
print("  2. FedProx: Proximal regularization (stable training)")
print("\nThe proximal term prevents client drift during local training,")
print("maintaining the benefits of FedALA's adaptive initialization.")
if total_benefit and np.mean(total_benefit) > 1:
    print(f"\n✅ Average benefit: +{np.mean(total_benefit):.2f}% over FedALA alone!")
else:
    print("\n⚠️  If benefit is low, try increasing μ (proximal coefficient)")
    print("   Edit fedala_prox_client.py: self.mu = 0.05 or 0.1")
print("=" * 70)
