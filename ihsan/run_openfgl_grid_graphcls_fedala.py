# run_openfgl_grid_graphcls_fedala.py

import os
import io
import random
from copy import deepcopy
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.serialization

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# Optional PyTorch 2.6+ safe globals (PyG)
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# IMPORTANT: point this to your clean dataset root (your data_table6_graphfl_a1)
DATA_ROOT = os.path.join(REPO_ROOT, "data_table6_graphfl_a1")

DATASETS = ["MUTAG", "BZR", "COX2", "AIDS"]
METHODS = ["fedala"]
SEEDS = [55, 160, 234]

results = defaultdict(lambda: defaultdict(list))

total_runs = len(METHODS) * len(DATASETS) * len(SEEDS)
run_idx = 0

for method in METHODS:
    for dataset in DATASETS:
        for seed in SEEDS:
            run_idx += 1
            tag = f"{method} | {dataset} | seed={seed}"
            print(f"[{run_idx:02d}/{total_runs}] {tag}", flush=True)

            f_stdout = io.StringIO()
            f_stderr = io.StringIO()

            try:
                with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                    args = deepcopy(config.args)

                    args.root = DATA_ROOT
                    args.scenario = "graph_fl"
                    args.task = "graph_cls_2"
                    args.dataset = [dataset]

                    args.simulation_mode = "graph_fl_label_skew"
                    args.num_clients = 10
                    args.client_frac = 1.0
                    args.dirichlet_alpha = 1.0
                    args.skew_alpha = 1.0
                    args.processing = "raw"

                    args.fl_algorithm = method
                    args.model = ["gin"]

                    # Training
                    args.num_rounds = 100
                    args.num_epochs = 1
                    args.lr = 1e-3
                    args.weight_decay = 5e-4
                    args.metrics = ["accuracy"]
                    args.seed = seed

                    # FedALA hyperparameters (tunable)
                    # Keep these explicit to avoid “silent defaults”.
                    if method == "fedala":
                        args.ala_batch_size = getattr(args, "batch_size", 32)
                        args.ala_rand_percent = 30.0
                        args.ala_layer_idx = 0          # 0 => adapt all parameter groups as "top"
                        args.ala_eta = 1.0
                        args.ala_std_threshold = 0.1
                        args.ala_num_pre_loss = 10

                    seed_everything(seed)

                    trainer = FGLTrainer(args)
                    trainer.train()

                # Extract best test accuracy from trainer.evaluation_result
                acc = np.nan
                if isinstance(getattr(trainer, "evaluation_result", None), dict):
                    # trainer.py stores best_test_{metric}, so for accuracy => best_test_accuracy
                    acc = trainer.evaluation_result.get("best_test_accuracy", np.nan)

                results[method][dataset].append(float(acc))
                print(f"  best_test_accuracy={acc:.4f}", flush=True)

            except Exception as e:
                tail_err = f_stderr.getvalue()[-1200:]
                if tail_err.strip():
                    print("  [stderr tail]")
                    print(tail_err)
                print(f"  FAILED: {e}", flush=True)
                results[method][dataset].append(np.nan)

# summary
print("\n================ FINAL SUMMARY (mean ± std, NaNs ignored) ================")
for method in METHODS:
    for dataset in DATASETS:
        vals = np.array(results[method][dataset], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            print(f"{method:8s} | {dataset:6s} -> {vals.mean():.4f} ± {vals.std():.4f} (n={vals.size})")
        else:
            print(f"{method:8s} | {dataset:6s} -> all runs failed")

np.save("fedala_vs_fedavg_results2.npy", {m: dict(d) for m, d in results.items()}, allow_pickle=True)
print("\nSaved fedala_vs_fedavg_results.npy")
