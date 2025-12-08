"""
run_openfgl_grid.py

Grid runner for OpenFGL:
- Datasets: Cora, CiteSeer, PubMed
- Methods:  fedavg, fedprox, gcfl_plus, fedsage_plus, adafgl, fgssl
- Seeds:    42, 123, 456

Collects best test accuracy from FGLTrainer.evaluation_result and prints
a compact progress log plus a summary at the end.
"""

import os
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import io

import numpy as np
import torch
import torch.serialization

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# Try to import PyG safe globals (only needed on newer PyTorch)
try:
    import torch_geometric.data.data as pyg_data
    import torch_geometric.data.storage as pyg_storage

    torch.serialization.add_safe_globals([
        getattr(pyg_data, "DataEdgeAttr", object),
        getattr(pyg_data, "DataTensorAttr", object),
        getattr(pyg_storage, "GlobalStorage", object),
        getattr(pyg_storage, "NodeStorage", object),
        getattr(pyg_storage, "EdgeStorage", object),
    ])
except Exception:
    # If this fails (e.g., older PyTorch), just ignore; it’s only for PyTorch>=2.6
    pass


# ==============================================================================
# Experiment configuration
# ==============================================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_ROOT, exist_ok=True)

DATASETS = ["Cora", "CiteSeer", "PubMed"]
METHODS = ["fedavg", "fedsage_plus", "adafgl", "fedtad", "fedgta"]
SEEDS = [42, 123, 456]

# results[method][dataset] -> list of accuracies (one per seed)
results = defaultdict(lambda: defaultdict(list))

total_runs = len(METHODS) * len(DATASETS) * len(SEEDS)
run_idx = 0
first_error_msg = None

print(f"Total runs: {total_runs}")

for method in METHODS:
    for dataset in DATASETS:
        for seed in SEEDS:
            run_idx += 1
            tag = f"{method} | {dataset} | seed={seed}"
            print(f"[{run_idx:02d}/{total_runs}] Running {tag} ... ", end="", flush=True)

            f_stdout = io.StringIO()
            f_stderr = io.StringIO()

            try:
                # Redirect all internal prints to keep notebook/terminal output small
                with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                    args = config.args

                    # Core paths & data setup
                    args.root = DATA_ROOT
                    args.dataset = [dataset]
                    args.simulation_mode = "subgraph_fl_louvain"
                    args.num_clients = 10

                    # Algorithm & model
                    args.fl_algorithm = method
                    args.model = ["gcn"]

                    # Training hyperparameters
                    args.num_rounds = 100
                    args.local_epoch = 3
                    args.lr = 0.01
                    args.weight_decay = 5e-4
                    args.metrics = ["accuracy"]
                    args.seed = seed

                    # Make sure evaluation mode is set (default in repo is usually OK,
                    # but we can be explicit if needed):
                    # args.evaluation_mode = "local_model_on_local_data"
                    # args.task is defined in config based on dataset; we don’t touch it.

                    if method == "fedprox":
                        args.mu = 0.01

                    trainer = FGLTrainer(args)
                    trainer.train()  # returns None; metrics are in trainer.evaluation_result

                # ---- Extract accuracy from trainer.evaluation_result ----
                acc = np.nan
                if hasattr(trainer, "evaluation_result") and isinstance(
                    trainer.evaluation_result, dict
                ):
                    # Primary metric name (we only use the first)
                    metric_name = args.metrics[0]

                    # For tasks graph_cls, graph_reg, node_cls, link_pred the trainer uses
                    # best_val_<metric> and best_test_<metric>.
                    # For node_clust it uses best_<metric>.
                    task = getattr(args, "task", None)
                    if task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
                        key = f"best_test_{metric_name}"
                    else:
                        key = f"best_{metric_name}"

                    acc = trainer.evaluation_result.get(key, np.nan)

                results[method][dataset].append(float(acc))
                print("OK")

            except Exception as e:
                err_short = str(e)
                if first_error_msg is None:
                    first_error_msg = err_short
                print(f"FAIL ({err_short})")
                results[method][dataset].append(np.nan)

# Convert defaultdict -> normal dict for saving
results_serializable = {
    m: {d: np.array(vals, dtype=float).tolist()
        for d, vals in ds.items()}
    for m, ds in results.items()
}

np.save("results_cache.npy", results_serializable, allow_pickle=True)

# ==============================================================================
# Summary
# ==============================================================================

print("\nSummary over seeds (mean ± std, NaNs ignored):")
for method in METHODS:
    for dataset in DATASETS:
        vals = np.array(results[method][dataset], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            print(f"{method:10s} | {dataset:8s} -> {vals.mean():.4f} ± {vals.std():.4f}")
        else:
            print(f"{method:10s} | {dataset:8s} -> all runs failed")

if first_error_msg is not None:
    print("\nFirst error (for debugging):")
    print(first_error_msg)

