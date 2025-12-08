"""
Run FedALA-R experiments only
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

# Configuration - FedALA-R ONLY
METHODS = ['fedala_r']
DATASETS = ['Cora', 'CiteSeer', 'PubMed']
SEEDS = [42, 123, 456]

# Results storage
results = {method: {dataset: [] for dataset in DATASETS} for method in METHODS}

print("🚀 Starting FedALA-R Experiments (Total: {} runs)".format(len(METHODS) * len(DATASETS) * len(SEEDS)))
print(f"📁 Data directory: {DATA_ROOT}")
print(f"📊 Methods: {METHODS}")
print(f"📦 Datasets: {DATASETS}")
print(f"🎲 Seeds: {SEEDS}\n")
print("=" * 70 + "\n")

# Track first error
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
                    
                    # Create trainer and train
                    trainer = FGLTrainer(args)
                    trainer.train()
                    
                    # Extract accuracy (same as working script)
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
                error_msg = str(e)
                print(f"✗ FAILED: {error_msg}")
                results[method][dataset].append(np.nan)
                
                # Capture first error for debugging
                if first_error is None:
                    first_error = {
                        'method': method,
                        'dataset': dataset,
                        'seed': seed,
                        'error': error_msg,
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

# Print error details if any
if first_error:
    print("=" * 70)
    print("🐛 FIRST ERROR DETAILS (for debugging)")
    print("=" * 70)
    print(f"Method:  {first_error['method']}")
    print(f"Dataset: {first_error['dataset']}")
    print(f"Seed:    {first_error['seed']}\n")
    print(f"Error: {first_error['error']}\n")
    print("Full Traceback:")
    print(first_error['traceback'])
    print()

# Save results
np.save('fedala_r_results.npy', results)
print("✅ Experiments complete! Results saved to 'fedala_r_results.npy'")
