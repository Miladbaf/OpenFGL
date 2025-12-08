"""
Run all 4 FedALA configurations:
1. Baseline FedALA (parameter-level)
2. FedALA-MP (Method 1: message-passing aware)
3. FedALA-R (Method 2: residual-based)
4. FedALA-MP-R (Combined: Methods 1+2)
"""

import os
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import io
import numpy as np
import torch
import traceback

# ========== CRITICAL: PyTorch 2.6+ Compatibility Fix ==========
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
print("✓ Patched torch.load to use weights_only=False for PyTorch 2.6+ compatibility\n")
# ========== End Compatibility Fix ==========

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# Setup
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO_ROOT, 'data')

# Configuration - ALL 4 METHODS
METHODS = ['fedavg', 'fedala', 'fedala_mp', 'fedala_r', 'fedala_mpr']
METHOD_LABELS = {
    'fedavg': 'Baseline (FedAvg)',
    'fedala': 'FedALA (parameter-level)',
    'fedala_mp': 'FedALA-MP (Method 1: message-passing)',
    'fedala_r': 'FedALA-R (Method 2: residual)',
    'fedala_mpr': 'FedALA-MP-R (Combined: 1+2)'
}

DATASETS = ['Cora', 'CiteSeer', 'PubMed']
SEEDS = [42, 123, 456]

# Results storage
results = {method: {dataset: [] for dataset in DATASETS} for method in METHODS}

print("🚀 Starting Complete FedALA Evaluation (Total: {} runs)".format(len(METHODS) * len(DATASETS) * len(SEEDS)))
print(f"📁 Data directory: {DATA_ROOT}")
print(f"📊 Methods: {len(METHODS)}")
for method in METHODS:
    print(f"   - {method}: {METHOD_LABELS[method]}")
print(f"📦 Datasets: {DATASETS}")
print(f"🎲 Seeds: {SEEDS}\n")
print("=" * 70 + "\n")

# Track first error
first_error = None
first_full_error = None

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
                    
                    # Create trainer and train
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
                
                # Capture first error for debugging
                if first_full_error is None:
                    first_full_error = {
                        'method': method,
                        'dataset': dataset,
                        'seed': seed,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }

# Print results summary
print("\n" + "=" * 70)
print("📊 RESULTS SUMMARY (mean ± std)")
print("=" * 70 + "\n")

for dataset in DATASETS:
    print(f"{dataset}:")
    print("-" * 50)
    for method in METHODS:
        accs = results[method][dataset]
        valid_accs = [a for a in accs if not np.isnan(a)]
        
        if len(valid_accs) == 0:
            print(f"  {method:12s}: ALL FAILED")
        elif len(valid_accs) < len(SEEDS):
            mean_acc = np.mean(valid_accs)
            print(f"  {method:12s}: {mean_acc:6.2f} (PARTIAL: {len(valid_accs)}/{len(SEEDS)} runs)")
        else:
            mean_acc = np.mean(valid_accs)
            std_acc = np.std(valid_accs)
            print(f"  {method:12s}: {mean_acc:6.2f} ± {std_acc:4.2f}")
    print()

# Improvements over FedAvg
print("=" * 70)
print("📈 IMPROVEMENTS OVER FedAvg")
print("=" * 70 + "\n")

for dataset in DATASETS:
    print(f"{dataset}:")
    fedavg_accs = [a for a in results['fedavg'][dataset] if not np.isnan(a)]
    
    if len(fedavg_accs) > 0:
        fedavg_mean = np.mean(fedavg_accs)
        
        for method in ['fedala', 'fedala_mp', 'fedala_r', 'fedala_mpr']:
            method_accs = [a for a in results[method][dataset] if not np.isnan(a)]
            
            if len(method_accs) > 0:
                method_mean = np.mean(method_accs)
                improvement = ((method_mean - fedavg_mean) / fedavg_mean) * 100
                label = METHOD_LABELS[method].split(':')[0]
                print(f"  {label:25s}: {improvement:+.2f}%")
            else:
                print(f"  {method:25s}: N/A (all failed)")
    else:
        print("  No FedAvg baseline available")
    print()

# Method contribution analysis
print("=" * 70)
print("📊 METHOD CONTRIBUTION ANALYSIS")
print("=" * 70 + "\n")

for dataset in DATASETS:
    print(f"{dataset}:")
    print("-" * 50)
    
    fedavg_accs = [a for a in results['fedavg'][dataset] if not np.isnan(a)]
    fedala_accs = [a for a in results['fedala'][dataset] if not np.isnan(a)]
    fedala_mp_accs = [a for a in results['fedala_mp'][dataset] if not np.isnan(a)]
    fedala_r_accs = [a for a in results['fedala_r'][dataset] if not np.isnan(a)]
    fedala_mpr_accs = [a for a in results['fedala_mpr'][dataset] if not np.isnan(a)]
    
    if all([len(x) > 0 for x in [fedavg_accs, fedala_accs, fedala_mp_accs, fedala_r_accs, fedala_mpr_accs]]):
        baseline = np.mean(fedavg_accs)
        fedala_gain = np.mean(fedala_accs) - baseline
        mp_contribution = np.mean(fedala_mp_accs) - np.mean(fedala_accs)
        r_contribution = np.mean(fedala_r_accs) - np.mean(fedala_accs)
        combined_gain = np.mean(fedala_mpr_accs) - baseline
        synergy = combined_gain - (fedala_gain + mp_contribution + r_contribution)
        
        print(f"  Baseline FedALA gain:        {fedala_gain:+.4f} ({fedala_gain/baseline*100:+.2f}%)")
        print(f"  Method 1 (MP) contribution:  {mp_contribution:+.4f}")
        print(f"  Method 2 (R) contribution:   {r_contribution:+.4f}")
        print(f"  Combined (MP-R) total gain:  {combined_gain:+.4f} ({combined_gain/baseline*100:+.2f}%)")
        print(f"  Synergy effect:              {synergy:+.4f}")
        if synergy > 0:
            print(f"  → Methods are complementary! ✓")
        else:
            print(f"  → No additional synergy")
    print()

# Print error details if any
if first_full_error:
    print("=" * 70)
    print("🐛 FIRST ERROR DETAILS (for debugging)")
    print("=" * 70)
    print(f"Method:  {first_full_error['method']}")
    print(f"Dataset: {first_full_error['dataset']}")
    print(f"Seed:    {first_full_error['seed']}\n")
    print(f"Error: {first_full_error['error']}\n")
    print("Full Traceback:")
    print(first_full_error['traceback'])
    print()

# Save results
np.save('fedala_complete_results.npy', results)
print("✅ Experiments complete! Results saved to 'fedala_complete_results.npy'")
