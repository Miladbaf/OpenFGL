"""
Run FedALA experiments and compare with baselines
Includes PyTorch 2.6+ compatibility fix for weights_only loading
"""

import os
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import io
import numpy as np
import torch
import traceback

# ========== CRITICAL: PyTorch 2.6+ Compatibility Fix ==========
# Monkey-patch torch.load to disable weights_only requirement
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False"""
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
DATA_ROOT = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_ROOT, exist_ok=True)

DATASETS = ["Cora", "CiteSeer", "PubMed"]
METHODS = [
    "fedavg",      # Baseline
    "fedala",      # FedALA (parameter-level)
    "fedala_mp",   # Method 1 (message-passing aware)
    "fedala_r"
]
SEEDS = [42, 123, 456]

results = defaultdict(lambda: defaultdict(list))
total_runs = len(METHODS) * len(DATASETS) * len(SEEDS)
run_idx = 0
first_full_error = None

print(f"🚀 Starting FedALA Experiments (Total: {total_runs} runs)")
print(f"📁 Data directory: {DATA_ROOT}")
print(f"📊 Methods: {METHODS}")
print(f"📦 Datasets: {DATASETS}")
print(f"🎲 Seeds: {SEEDS}\n")
print("="*70 + "\n")

for method in METHODS:
    for dataset in DATASETS:
        for seed in SEEDS:
            run_idx += 1
            tag = f"{method:12s} | {dataset:8s} | seed={seed}"
            print(f"[{run_idx:02d}/{total_runs}] {tag} ... ", end="", flush=True)
            
            f_stdout = io.StringIO()
            f_stderr = io.StringIO()
            
            try:
                with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                    args = config.args
                    
                    # Setup
                    args.root = DATA_ROOT
                    args.dataset = [dataset]
                    args.simulation_mode = "subgraph_fl_louvain"
                    args.num_clients = 10
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
                
                # Capture first full error for debugging
                if first_full_error is None:
                    first_full_error = {
                        'method': method,
                        'dataset': dataset,
                        'seed': seed,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                
                results[method][dataset].append(np.nan)

# Save results
results_dict = {
    m: {d: np.array(v).tolist() for d, v in ds.items()}
    for m, ds in results.items()
}
np.save("fedala_results.npy", results_dict, allow_pickle=True)

# Print summary
print("\n" + "="*70)
print("📊 RESULTS SUMMARY (mean ± std)")
print("="*70)

for dataset in DATASETS:
    print(f"\n{dataset}:")
    print("-" * 50)
    
    for method in METHODS:
        vals = np.array(results[method][dataset])
        vals = vals[~np.isnan(vals)]
        
        if vals.size > 0:
            mean, std = vals.mean(), vals.std()
            print(f"  {method:12s}: {mean:6.2f} ± {std:4.2f}")
        else:
            print(f"  {method:12s}: ALL FAILED")

# Compute improvements
print("\n" + "="*70)
print("📈 IMPROVEMENTS OVER FedAvg")
print("="*70)

for dataset in DATASETS:
    print(f"\n{dataset}:")
    baseline_vals = np.array(results["fedavg"][dataset])
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
    
    if baseline_vals.size > 0:
        baseline = baseline_vals.mean()
        
        for method in ["fedala", "fedala_mp"]:
            vals = np.array(results[method][dataset])
            vals = vals[~np.isnan(vals)]
            
            if vals.size > 0:
                method_mean = vals.mean()
                improvement = ((method_mean - baseline) / baseline) * 100
                print(f"  {method:12s}: {improvement:+6.2f}%")
            else:
                print(f"  {method:12s}: N/A (all failed)")
    else:
        print(f"  Baseline failed - cannot compute improvements")

# Print first error details if any failures occurred
if first_full_error is not None:
    print("\n" + "="*70)
    print("🐛 FIRST ERROR DETAILS (for debugging)")
    print("="*70)
    print(f"Method:  {first_full_error['method']}")
    print(f"Dataset: {first_full_error['dataset']}")
    print(f"Seed:    {first_full_error['seed']}")
    print(f"\nError: {first_full_error['error']}")
    print(f"\nFull Traceback:")
    print(first_full_error['traceback'])

print("\n✅ Experiments complete! Results saved to 'fedala_results.npy'")
