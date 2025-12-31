"""
run_openfgl_grid_graphfl_simple.py

Simple grid runner for OpenFGL Graph-FL / graph_cls:
- Dataset: DD (TUDataset)
- Simulation: graph_fl_label_skew (alpha=1.0), K=10
- Methods: fedavg, fedprox, scaffold, moon
- Seeds: 42, 123, 456

Prints:
- per-run best test accuracy immediately
- running mean over completed seeds for each (method,dataset)
- final summary
Saves: results_cache.npy
"""

import os
import io
from copy import deepcopy
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch
import torch.serialization

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# -------------------------------------------------------------------
# Optional: PyTorch 2.6+ safe globals for PyG
# -------------------------------------------------------------------
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
    pass

# ==============================================================================
# Experiment configuration
# ==============================================================================

FEDPROX_CFG = {"fedprox_mu": 1e-3}
MOON_CFG = {"moon_mu": 1, "temperature": 0.5}

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_ROOT, exist_ok=True)

DATASETS = ["PROTEINS"]
METHODS = ["fedavg", "fedprox", "scaffold", "moon"]
SEEDS = [42, 123, 456]

# results[method][dataset] -> list of accuracies (one per seed)
results = defaultdict(lambda: defaultdict(list))

total_runs = len(METHODS) * len(DATASETS) * len(SEEDS)
run_idx = 0
first_error_msg = None

print(f"Total runs: {total_runs}")

for method in METHODS:
    for dataset in DATASETS:
        setup_vals = []  # keep for per-method/dataset running mean

        print(f"\n================ SETUP: {method} | {dataset} ================\n")

        for seed_i, seed in enumerate(SEEDS, start=1):
            run_idx += 1
            tag = f"{method} | {dataset} | seed={seed}"
            print(f"[{run_idx:02d}/{total_runs}] Running {tag} ...", flush=True)

            f_stdout = io.StringIO()
            f_stderr = io.StringIO()

            try:
                # capture trainer logs to reduce spam
                with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                    args = deepcopy(config.args)  # IMPORTANT: isolate runs

                    # ---- Force correct setting for DD (Graph-FL, graph classification) ----
                    args.root = DATA_ROOT
                    args.scenario = "graph_fl"
                    args.task = "graph_cls"
                    args.dataset = [dataset]

                    # ---- Dataset simulation (match Graph-FL label skew default) ----
                    args.simulation_mode = "graph_fl_label_skew"
                    args.num_clients = 10
                    args.client_frac = 1.0

                    args.dirichlet_alpha = 1.0
                    args.skew_alpha = 1.0              # required for processed_dir naming
                    args.dirichlet_try_cnt = getattr(args, "dirichlet_try_cnt", 100)
                    args.least_samples = getattr(args, "least_samples", 5)

                    # ---- Avoid accidental perturbations if your config has them ----
                    args.processing = "raw"

                    # ---- Algorithm & model ----
                    args.fl_algorithm = method
                    args.model = ["gin"]

                    # ---- Training hyperparameters ----
                    args.num_rounds = 100
                    args.num_epochs = 1
                    args.lr = 1e-3
                    args.weight_decay = 5e-4
                    args.metrics = ["accuracy"]
                    args.seed = seed

                    # Method-specific knobs
                    if method == "fedprox":
                        args.mu = FEDPROX_CFG["fedprox_mu"]
                        args.fedprox_mu = FEDPROX_CFG["fedprox_mu"]
                    elif method == "moon":
                        args.mu = MOON_CFG["moon_mu"]
                        args.moon_mu = MOON_CFG["moon_mu"]
                        args.temperature = MOON_CFG["temperature"]

                    # Seed everything deterministically
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    trainer = FGLTrainer(args)
                    trainer.train()

                # ---- Extract accuracy ----
                acc = np.nan
                if hasattr(trainer, "evaluation_result") and isinstance(trainer.evaluation_result, dict):
                    # In this repo, you previously saw "best_test_accuracy"
                    acc = trainer.evaluation_result.get("best_test_accuracy", np.nan)
                    # Fallback if naming differs:
                    if np.isnan(acc):
                        acc = trainer.evaluation_result.get("best_test_accuracy".lower(), np.nan)
                    if np.isnan(acc):
                        acc = trainer.evaluation_result.get("best_test_accuracy".upper(), np.nan)

                results[method][dataset].append(float(acc))
                setup_vals.append(float(acc))

                # ---- Print per-run + running mean so far ----
                clean = np.array(setup_vals, dtype=float)
                clean = clean[~np.isnan(clean)]
                running_mean = float(clean.mean()) if clean.size else np.nan

                print(f"  [result] seed={seed} best_test_accuracy={acc:.4f}", flush=True)
                print(f"  [running mean] after {seed_i}/{len(SEEDS)} seeds -> {running_mean:.4f}", flush=True)

            except Exception as e:
                err_short = str(e)
                if first_error_msg is None:
                    first_error_msg = err_short

                # Optional: show tail of captured error for debugging
                tail_err = f_stderr.getvalue()[-1200:]
                if tail_err.strip():
                    print("  [captured stderr tail]")
                    print(tail_err)

                print(f"  [result] FAILED ({err_short})", flush=True)
                results[method][dataset].append(np.nan)
                setup_vals.append(np.nan)

        # ---- Per-setup summary over 3 seeds ----
        vals = np.array(setup_vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            print(f"\n[setup summary] {method:10s} | {dataset:8s} -> {vals.mean():.4f} ± {vals.std():.4f} (n={vals.size})")
        else:
            print(f"\n[setup summary] {method:10s} | {dataset:8s} -> all runs failed")

# ---- Save results ----
results_serializable = {
    m: {d: np.array(vals, dtype=float).tolist() for d, vals in ds.items()}
    for m, ds in results.items()
}
np.save("results_cache.npy", results_serializable, allow_pickle=True)
print("\nSaved results_cache.npy")

# ==============================================================================
# Final summary
# ==============================================================================
print("\n================ FINAL SUMMARY (mean ± std, NaNs ignored) ================")
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
