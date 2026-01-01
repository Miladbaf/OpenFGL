"""
run_openfgl_grid_graphfl_table6_v3.py

Grid runner for OpenFGL Graph-FL / graph classification over MULTIPLE dataset instances.

Assumptions about your downloaded data layout (from your multi-instance downloader):
    <BASE_DATA_ROOT>/
        inst_01/
            distrib/graph_fl_label_skew_1.00_<DATASET>_client_10/...
        inst_02/
            distrib/...
        inst_03/
            distrib/...

Behavior:
- For each (dataset, instance, seed):
    - deletes split cache ONCE (graph_cls/default_split)
    - seeds everything ONCE
    - runs all METHODS reusing the same split across methods (fair comparison)
- For each (dataset, method), aggregates results across (instances × seeds):
    - e.g., MUTAG with 2 instances and 3 seeds => 6 runs per method
- Saves a single .npy containing:
    - config
    - per-run records
    - grouped values
    - summary stats
"""

import os
import io
import json
import shutil
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

# -------------------------------------------------------------------
# Make repo imports work even if this script is inside "ihsan/"
# -------------------------------------------------------------------
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
# Method configs (repo-specific: FedProx/MOON use module-level config dicts)
# -------------------------------------------------------------------
FEDPROX_MU = 1e-3
MOON_MU = 1.0
MOON_TEMPERATURE = 0.5

# -------------------------------------------------------------------
# Data root (multi-instance)
# -------------------------------------------------------------------
BASE_DATA_ROOT = os.path.join(REPO_ROOT, "data_table6_graphfl_a1_multi")  # adjust if needed

# -------------------------------------------------------------------
# Experiment grid
# -------------------------------------------------------------------
K = 10
ALPHA = 1.0
TASK = "graph_cls_2"           # safer split handling; still uses split dir "graph_cls/default_split"
METRIC = "accuracy"

DATASET_INSTANCES = {
    "MUTAG": 2,
    "COX2": 2,
    "BZR": 2,
    "AIDS": 1,
    "PROTEINS": 3,
}

METHODS = ["moon"]  # adjust as needed
SEEDS = [42, 123, 456]                               # adjust as needed

OUT_NPY = "results_baseline_moon.npy"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def instance_root(instance_idx: int) -> str:
    return os.path.join(BASE_DATA_ROOT, f"inst_{instance_idx:02d}")

def expected_processed_dir(args) -> str:
    # matches openfgl/data/distributed_dataset_loader.py processed_dir logic
    if args.simulation_mode in ["subgraph_fl_label_skew", "graph_fl_label_skew"]:
        simulation_name = f"{args.simulation_mode}_{args.skew_alpha:.2f}"
    elif args.simulation_mode in ["subgraph_fl_louvain_plus", "subgraph_fl_louvain"]:
        simulation_name = f"{args.simulation_mode}_{args.louvain_resolution}"
    elif args.simulation_mode in ["subgraph_fl_metis_plus"]:
        simulation_name = f"{args.simulation_mode}_{args.metis_num_coms}"
    else:
        simulation_name = args.simulation_mode

    fmt_dataset_list = sorted(list(args.dataset))
    return os.path.join(args.root, "distrib", "_".join([
        simulation_name, "_".join(fmt_dataset_list), f"client_{args.num_clients}"
    ]))

def assert_partition_exists(processed_dir: str, num_clients: int):
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    missing = []
    for i in range(num_clients):
        fp = os.path.join(processed_dir, f"data_{i}.pt")
        if not os.path.isfile(fp):
            missing.append(fp)
    if missing:
        raise FileNotFoundError("Missing client partition files:\n" + "\n".join(missing))

def assert_description_matches(processed_dir: str, expected: dict):
    desc_path = os.path.join(processed_dir, "description.txt")
    if not os.path.isfile(desc_path):
        raise FileNotFoundError(f"description.txt not found: {desc_path}")
    with open(desc_path, "r", encoding="utf-8") as f:
        desc = json.load(f)
    for k, v in expected.items():
        if k not in desc:
            raise KeyError(f"description.txt missing key '{k}' in {desc_path}")
        if desc[k] != v:
            raise ValueError(f"description mismatch in {desc_path}: expected {k}={v}, found {desc[k]}")

def delete_default_split(processed_dir: str):
    # graph_cls_2 still stores splits under ".../graph_cls/default_split" in this repo
    split_dir = os.path.join(processed_dir, "graph_cls", "default_split")
    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)

# -------------------------------------------------------------------
# Storage
# -------------------------------------------------------------------
# grouped_values[method][dataset] -> list of best_test_accuracy values across (instances × seeds)
grouped_values = defaultdict(lambda: defaultdict(list))

# per-run records (for later analysis)
runs = []

# -------------------------------------------------------------------
# Run loop
# -------------------------------------------------------------------
datasets = list(DATASET_INSTANCES.keys())
total_runs = sum(DATASET_INSTANCES[d] for d in datasets) * len(SEEDS) * len(METHODS)
run_idx = 0
first_error_msg = None

print(f"BASE_DATA_ROOT: {BASE_DATA_ROOT}")
print(f"Total runs (instances × seeds × methods): {total_runs}")

for dataset in datasets:
    n_inst = DATASET_INSTANCES[dataset]

    for inst_idx in range(1, n_inst + 1):
        inst_root = instance_root(inst_idx)

        for seed in SEEDS:
            print(f"\n================ SETUP: {dataset} | inst={inst_idx:02d}/{n_inst} | seed={seed} ================\n", flush=True)

            # Build base args ONCE per (dataset, instance, seed)
            base_args = deepcopy(config.args)
            base_args.root = inst_root
            base_args.scenario = "graph_fl"
            base_args.task = TASK
            base_args.dataset = [dataset]

            base_args.simulation_mode = "graph_fl_label_skew"
            base_args.num_clients = K
            base_args.client_frac = 1.0
            base_args.dirichlet_alpha = ALPHA
            base_args.skew_alpha = ALPHA
            base_args.processing = "raw"

            # Training hyperparams (keep consistent with your baseline scripts)
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

            # Validate partitions exist for this instance/dataset
            proc_dir = expected_processed_dir(base_args)
            assert_partition_exists(proc_dir, base_args.num_clients)
            assert_description_matches(
                proc_dir,
                expected={
                    "scenario": "graph_fl",
                    "simulation_mode": "graph_fl_label_skew",
                    "num_clients": K,
                    "dirichlet_alpha": ALPHA,
                    "skew_alpha": ALPHA,
                },
            )

            # Delete split cache ONCE per (dataset, instance, seed)
            delete_default_split(proc_dir)

            # Seed everything ONCE per (dataset, instance, seed)
            seed_everything(seed)

            for method in METHODS:
                run_idx += 1
                tag = f"{method} | {dataset} | inst={inst_idx:02d} | seed={seed}"
                print(f"[{run_idx:04d}/{total_runs}] Running {tag} ...", flush=True)

                f_stdout = io.StringIO()
                f_stderr = io.StringIO()

                try:
                    with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                        args = deepcopy(base_args)
                        args.fl_algorithm = method

                        # Method-specific module configs (repo-specific)
                        if method == "fedprox":
                            from openfgl.flcore.fedprox.fedprox_config import config as fedprox_config
                            fedprox_config["fedprox_mu"] = FEDPROX_MU
                        elif method == "moon":
                            from openfgl.flcore.moon.moon_config import config as moon_config
                            moon_config["moon_mu"] = MOON_MU
                            moon_config["temperature"] = MOON_TEMPERATURE

                        trainer = FGLTrainer(args)
                        trainer.train()

                    best = np.nan
                    if hasattr(trainer, "evaluation_result") and isinstance(trainer.evaluation_result, dict):
                        best = trainer.evaluation_result.get(f"best_test_{METRIC}", np.nan)

                    grouped_values[method][dataset].append(float(best))
                    runs.append({
                        "dataset": dataset,
                        "instance": inst_idx,
                        "seed": seed,
                        "method": method,
                        f"best_test_{METRIC}": float(best),
                    })

                    print(f"  [result] best_test_{METRIC}={best:.4f}", flush=True)

                except Exception as e:
                    err_short = str(e)
                    if first_error_msg is None:
                        first_error_msg = err_short

                    tail_err = f_stderr.getvalue()[-2000:]
                    if tail_err.strip():
                        print("  [captured stderr tail]")
                        print(tail_err)

                    grouped_values[method][dataset].append(np.nan)
                    runs.append({
                        "dataset": dataset,
                        "instance": inst_idx,
                        "seed": seed,
                        "method": method,
                        f"best_test_{METRIC}": float("nan"),
                        "error": err_short,
                    })

                    print(f"  [result] FAILED ({err_short})", flush=True)

# -------------------------------------------------------------------
# Summaries
# -------------------------------------------------------------------
summary = {}
for method in METHODS:
    summary[method] = {}
    for dataset in datasets:
        vals = np.array(grouped_values[method][dataset], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            summary[method][dataset] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "n": int(vals.size),
            }
        else:
            summary[method][dataset] = {
                "mean": float("nan"),
                "std": float("nan"),
                "n": 0,
            }

print("\n================ FINAL SUMMARY (mean ± std, NaNs ignored) ================")
for method in METHODS:
    for dataset in datasets:
        s = summary[method][dataset]
        if s["n"] > 0:
            print(f"{method:10s} | {dataset:10s} -> {s['mean']:.4f} ± {s['std']:.4f} (n={s['n']})")
        else:
            print(f"{method:10s} | {dataset:10s} -> all runs failed")

if first_error_msg is not None:
    print("\nFirst error (for debugging):")
    print(first_error_msg)

# -------------------------------------------------------------------
# Save .npy (single artifact for later table generation / analysis)
# -------------------------------------------------------------------
payload = {
    "config": {
        "K": K,
        "alpha": ALPHA,
        "task": TASK,
        "metric": METRIC,
        "methods": METHODS,
        "seeds": SEEDS,
        "dataset_instances": DATASET_INSTANCES,
        "base_data_root": BASE_DATA_ROOT,
    },
    "runs": runs,  # list[dict]
    "grouped_values": {
        m: {d: np.array(v, dtype=float).tolist() for d, v in ds.items()}
        for m, ds in grouped_values.items()
    },
    "summary": summary,
}

np.save(OUT_NPY, payload, allow_pickle=True)
print(f"\nSaved {OUT_NPY}")
