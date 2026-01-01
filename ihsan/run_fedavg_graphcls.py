"""
run_openfgl_grid_graphfl_table6.py

Deterministic-ish grid runner for OpenFGL Graph-FL / graph classification.

Key behaviors:
- Uses graph_fl_label_skew with (alpha=1.0, K=10).
- Deletes split cache ONCE per (dataset, seed) so repeats are independent.
- Reuses the same split across methods for a given (dataset, seed) for fair comparison.
- Uses task=graph_cls_2 to avoid empty val/test failures in label-skew clients.
- Validates that the expected processed_dir exists and contains data_0.pt...data_9.pt
- Extracts best_test_accuracy (i.e., best_test_{metric}) from trainer.evaluation_result.
- Saves results_cache.npy and results.csv
"""

import os
import io
import json
import shutil
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
from openfgl.utils.basic_utils import seed_everything

# -------------------------------------------------------------------
# Optional: PyTorch 2.6+ safe globals for PyG torch.load
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

# -------------------------------------------------------------------
# Method configs (NOTE: FedProx/MOON use module-level config dicts in this repo)
# -------------------------------------------------------------------
FEDPROX_MU = 1e-3
MOON_MU = 1.0
MOON_TEMPERATURE = 0.5

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Point this to your dedicated Table-6 baseline root (the one containing /distrib and /global).
# Example: OpenFGL/data_table6_graphfl_a1
DATA_ROOT = os.path.join(REPO_ROOT, "data_table6_graphfl_a1")  # <-- adjust if needed

# -------------------------------------------------------------------
# Experiment grid
# -------------------------------------------------------------------
DATASETS = ["NCI1"]
METHODS = ["fedavg"]
SEEDS = random.sample(range(30, 20000), 3)
METRIC = "accuracy"  # must match args.metrics[0]

NPY_PATH = "results_cache_graphFL.npy"
CSV_PATH = "results_cache_graphFL.csv"

# results[method][dataset] -> list of metrics (one per seed)
results = defaultdict(lambda: defaultdict(list))

# -------------------------------------------------------------------
# Helper: reproduce OpenFGL processed_dir naming (without instantiating FGLDataset)
# -------------------------------------------------------------------
def expected_processed_dir(args):
    if args.simulation_mode in ["subgraph_fl_label_skew", "graph_fl_label_skew"]:
        simulation_name = f"{args.simulation_mode}_{args.skew_alpha:.2f}"
    elif args.simulation_mode in ["subgraph_fl_louvain_plus", "subgraph_fl_louvain"]:
        simulation_name = f"{args.simulation_mode}_{args.louvain_resolution}"
    elif args.simulation_mode in ["subgraph_fl_metis_plus"]:
        simulation_name = f"{args.simulation_mode}_{args.metis_num_coms}"
    else:
        simulation_name = args.simulation_mode

    fmt_dataset_list = sorted(list(args.dataset))
    return os.path.join(args.root, "distrib", "_".join([simulation_name, "_".join(fmt_dataset_list), f"client_{args.num_clients}"]))

def assert_partition_exists(processed_dir, num_clients):
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    missing = []
    for i in range(num_clients):
        fp = os.path.join(processed_dir, f"data_{i}.pt")
        if not os.path.isfile(fp):
            missing.append(fp)
    if missing:
        raise FileNotFoundError("Missing client partition files:\n" + "\n".join(missing))

def assert_description_matches(processed_dir, expected):
    desc_path = os.path.join(processed_dir, "description.txt")
    if not os.path.isfile(desc_path):
        raise FileNotFoundError(f"description.txt not found: {desc_path}")
    with open(desc_path, "r", encoding="utf-8") as f:
        desc = json.load(f)

    # Minimal invariants to prevent silent mismatch
    for k, v in expected.items():
        if k not in desc:
            raise KeyError(f"description.txt missing key '{k}' in {desc_path}")
        if desc[k] != v:
            raise ValueError(f"description mismatch in {desc_path}: expected {k}={v}, found {desc[k]}")

def delete_default_split(processed_dir):
    split_dir = os.path.join(processed_dir, "graph_cls", "default_split")
    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
total_runs = len(DATASETS) * len(SEEDS) * len(METHODS)
run_idx = 0
first_error_msg = None

print(f"DATA_ROOT: {DATA_ROOT}")
print(f"Total runs: {total_runs}")

# iterate in (dataset -> seed -> method) order so split deletion is naturally “once per seed”
for dataset in DATASETS:
    for seed in SEEDS:
        print(f"\n================ SETUP: {dataset} | seed={seed} ================\n", flush=True)

        # Configure args once per (dataset,seed) so processed_dir computation is consistent
        base_args = deepcopy(config.args)
        base_args.root = DATA_ROOT
        base_args.scenario = "graph_fl"
        base_args.task = "graph_cls_2"        # important: avoids empty val/test failures
        base_args.dataset = [dataset]

        base_args.simulation_mode = "graph_fl_label_skew"
        base_args.num_clients = 10
        base_args.client_frac = 1.0
        base_args.dirichlet_alpha = 1.0
        base_args.skew_alpha = 1.0            # required for naming in this repo
        base_args.processing = "raw"

        # Paper-default-ish training hyperparams for Graph-FL in OpenFGL paper
        base_args.num_rounds = 100
        base_args.num_epochs = 1
        base_args.lr = 1e-3
        base_args.weight_decay = 5e-4
        base_args.batch_size = 128
        base_args.dropout = 0.5
        base_args.optim = "adam"

        base_args.metrics = [METRIC]
        base_args.evaluation_mode = "local_model_on_local_data"
        base_args.model = ["gin"]
        base_args.seed = seed

        # Validate partition existence AND description invariants (prevents accidental regen/mismatch)
        proc_dir = expected_processed_dir(base_args)
        assert_partition_exists(proc_dir, base_args.num_clients)
        assert_description_matches(
            proc_dir,
            expected={
                "scenario": "graph_fl",
                "simulation_mode": "graph_fl_label_skew",
                "num_clients": 10,
                "dirichlet_alpha": 1.0,
                "skew_alpha": 1.0,
            },
        )

        # Delete split cache ONCE per (dataset,seed) so this repeat is independent
        delete_default_split(proc_dir)

        # Seed everything ONCE per (dataset,seed). Split generation happens during trainer init.
        seed_everything(seed)

        for method in METHODS:
            run_idx += 1
            tag = f"{dataset} | seed={seed} | {method}"
            print(f"[{run_idx:02d}/{total_runs}] Running {tag} ...", flush=True)

            f_stdout = io.StringIO()
            f_stderr = io.StringIO()

            try:
                with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                    args = deepcopy(base_args)
                    args.fl_algorithm = method

                    # Method-specific config in THIS repo is module-level, not args-level.
                    if method == "fedprox":
                        from openfgl.flcore.fedprox.fedprox_config import config as fedprox_config
                        fedprox_config["fedprox_mu"] = FEDPROX_MU
                    elif method == "moon":
                        from openfgl.flcore.moon.moon_config import config as moon_config
                        moon_config["moon_mu"] = MOON_MU
                        moon_config["temperature"] = MOON_TEMPERATURE

                    trainer = FGLTrainer(args)
                    trainer.train()

                # Extract best test metric
                acc = np.nan
                if hasattr(trainer, "evaluation_result") and isinstance(trainer.evaluation_result, dict):
                    acc = trainer.evaluation_result.get(f"best_test_{METRIC}", np.nan)

                results[method][dataset].append(float(acc))
                print(f"  [result] best_test_{METRIC}={acc:.4f}", flush=True)

            except Exception as e:
                err_short = str(e)
                if first_error_msg is None:
                    first_error_msg = err_short

                tail_err = f_stderr.getvalue()[-2000:]
                if tail_err.strip():
                    print("  [captured stderr tail]")
                    print(tail_err)

                print(f"  [result] FAILED ({err_short})", flush=True)
                results[method][dataset].append(np.nan)

# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------
results_serializable = {
    m: {d: np.array(vals, dtype=float).tolist() for d, vals in ds.items()}
    for m, ds in results.items()
}
np.save(NPY_PATH, results_serializable, allow_pickle=True)

# Also write a simple CSV for paper/table generation
csv_lines = ["method,dataset,seed,best_test_accuracy"]
for method in METHODS:
    for dataset in DATASETS:
        for seed, val in zip(SEEDS, results[method][dataset]):
            csv_lines.append(f"{method},{dataset},{seed},{val}")
with open(CSV_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(csv_lines))

print(f"\nSaved {NPY_PATH} and {CSV_PATH}")

# -------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------
print("\n================ FINAL SUMMARY (mean ± std, NaNs ignored) ================")
for method in METHODS:
    for dataset in DATASETS:
        vals = np.array(results[method][dataset], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            print(f"{method:10s} | {dataset:10s} -> {vals.mean():.4f} ± {vals.std():.4f} (n={vals.size})")
        else:
            print(f"{method:10s} | {dataset:10s} -> all runs failed")

if first_error_msg is not None:
    print("\nFirst error (for debugging):")
    print(first_error_msg)
