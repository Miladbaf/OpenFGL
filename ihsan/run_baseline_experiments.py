"""
ihsan/run_graphfl_table6_baselines.py

Baseline grid runner for OpenFGL Graph-FL (graph_cls_2) under graph_fl_label_skew.

Runs (per dataset):
  num_instances(dataset) * len(SEEDS) * len(METHODS)

Baselines (Table-6 style):
  FedAvg, FedProx, Scaffold, GCFL+, FedStar

Key behaviors:
- Uses your downloaded multi-instance roots:
    data_table6_graphfl_a1_multi/inst_01, inst_02, inst_03, ...
- Uses task=graph_cls_2 for stability (avoids empty val/test in skewed clients).
- Deletes split cache ONCE per (dataset, instance, seed) so each seed is independent,
  and all methods share the same split for fair comparison.
- Sets module-level config dicts for FedProx / GCFL+ / FedStar explicitly.

Output:
- Saves a single npy payload with runs + summary + raw results.

Run:
  python -m ihsan.run_graphfl_table6_baselines
or
  python ihsan/run_graphfl_table6_baselines.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from copy import deepcopy
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.serialization

# -----------------------------------------------------------------------------
# Repo import path (so scripts can live under ihsan/)
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from openfgl.utils.basic_utils import seed_everything

# -----------------------------------------------------------------------------
# Optional: PyTorch safe globals for PyG torch.load (your torch is 2.5.1 but safe)
# -----------------------------------------------------------------------------
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

# =============================================================================
# USER CONFIG
# =============================================================================
DATASETS = ["MUTAG", "BZR", "COX2", "AIDS"]

# IMPORTANT: method names must match basic_utils.load_client/load_server
METHODS = ["fedavg", "fedprox", "fedstar"]

SEEDS = [540, 204, 350]
METRIC = "accuracy"

K_CLIENTS = 10
DIRICHLET_ALPHA = 1.0

# Multi-instance root created by data_download_multi.py
BASE_ROOT = (REPO_ROOT / "data_table6_graphfl_a1_multi")

# How many independent instances exist per dataset (your UPDATED mapping)
DATASET_INSTANCES: Dict[str, int] = {
    "MUTAG": 3,
    "COX2": 2,
    "BZR": 2,
    "AIDS": 1,
}

# Output
OUT_NPY = REPO_ROOT / "results_graphfl_table6_baselines2.npy"

# Training defaults (paper-default-ish; keep consistent with your baseline runner)
TRAINING_DEFAULTS = dict(
    num_rounds=100,
    num_epochs=1,
    lr=1e-3,
    weight_decay=5e-4,
    batch_size=128,
    dropout=0.5,
    optim="adam",
    model=["gin"],
    evaluation_mode="local_model_on_local_data",
)

# Module-level config overrides (set explicitly per run)
FEDPROX_MU = 1e-3

GCFL_PLUS_DEFAULTS = dict(
    eps1=0.05,
    eps2=0.1,
    seq_length=5,
    standardize=True,
)

FEDSTAR_DEFAULTS = dict(
    fedstar_beta=0.2,
    fedstar_tau=0.05,
    fedstar_window=1,
    fedstar_max_iter=200,
    fedstar_num_samples=100,
    fedstar_base_lr=0.3,
    fedstar_base_optim="sgd",
    fedstar_personalized_lr=0.01,
    fedstar_personalized_epochs=5,
    fedstar_personalized_optim="adam",
)

# Logging controls
SHOW_ROUND_LOGS = False
STORE_STDIO_TAILS = True
STDIO_TAIL_CHARS = 2000

# =============================================================================
# Helpers
# =============================================================================
def inst_root(inst_index_1based: int) -> Path:
    return BASE_ROOT / f"inst_{inst_index_1based:02d}"

def processed_dir_from_args(args) -> Path:
    # matches openfgl/data/distributed_dataset_loader.py naming
    if args.simulation_mode in ["subgraph_fl_label_skew", "graph_fl_label_skew"]:
        simulation_name = f"{args.simulation_mode}_{args.skew_alpha:.2f}"
    else:
        simulation_name = args.simulation_mode

    ds = sorted(list(args.dataset))
    folder = "_".join([simulation_name, "_".join(ds), f"client_{args.num_clients}"])
    return Path(args.root) / "distrib" / folder

def assert_partitions_exist(proc_dir: Path, num_clients: int) -> None:
    if not proc_dir.is_dir():
        raise FileNotFoundError(f"processed_dir not found: {proc_dir}")
    missing = []
    for cid in range(num_clients):
        fp = proc_dir / f"data_{cid}.pt"
        if not fp.is_file():
            missing.append(str(fp))
    if missing:
        raise FileNotFoundError("Missing client partition files:\n" + "\n".join(missing))

def delete_default_split(proc_dir: Path) -> None:
    # graph_cls_2 uses graph_cls/default_split path internally
    split_dir = proc_dir / "graph_cls" / "default_split"
    if split_dir.is_dir():
        shutil.rmtree(split_dir)

def set_method_module_configs(method: str) -> None:
    # FedProx reads mu from module-level dict
    if method == "fedprox":
        from openfgl.flcore.fedprox.fedprox_config import config as fedprox_config
        fedprox_config["fedprox_mu"] = float(FEDPROX_MU)

    # GCFL+ reads eps/seq_length/standardize from module-level dict
    if method == "gcfl_plus":
        from openfgl.flcore.gcfl_plus.gcfl_plus_config import config as gcflp_config
        for k, v in GCFL_PLUS_DEFAULTS.items():
            gcflp_config[k] = v

    # FedStar reads its knobs from module-level dict
    if method == "fedstar":
        from openfgl.flcore.fedstar.fedstar_config import config as fedstar_config
        for k, v in FEDSTAR_DEFAULTS.items():
            fedstar_config[k] = v

def make_args(dataset: str, inst_root_path: Path, seed: int, method: str):
    args = deepcopy(config.args)

    # core
    args.root = str(inst_root_path)
    args.scenario = "graph_fl"
    args.task = "graph_cls_2"
    args.dataset = [dataset]
    args.processing = "raw"

    # simulation
    args.simulation_mode = "graph_fl_label_skew"
    args.num_clients = int(K_CLIENTS)
    args.client_frac = 1.0
    args.dirichlet_alpha = float(DIRICHLET_ALPHA)
    args.skew_alpha = float(DIRICHLET_ALPHA)

    # windows-safe dataloading
    args.num_workers = 0
    args.persistent_workers = False

    # training
    args.num_rounds = int(TRAINING_DEFAULTS["num_rounds"])
    args.num_epochs = int(TRAINING_DEFAULTS["num_epochs"])
    args.lr = float(TRAINING_DEFAULTS["lr"])
    args.weight_decay = float(TRAINING_DEFAULTS["weight_decay"])
    args.batch_size = int(TRAINING_DEFAULTS["batch_size"])
    args.dropout = float(TRAINING_DEFAULTS["dropout"])
    args.optim = str(TRAINING_DEFAULTS["optim"])
    args.model = list(TRAINING_DEFAULTS["model"])
    args.evaluation_mode = str(TRAINING_DEFAULTS["evaluation_mode"])

    # evaluation metric
    args.metrics = [METRIC]

    # algorithm + seed
    args.fl_algorithm = method
    args.seed = int(seed)

    return args

def extract_best_test(trainer: FGLTrainer, metric: str) -> float:
    key = f"best_test_{metric}"
    d = getattr(trainer, "evaluation_result", None)
    if isinstance(d, dict) and key in d:
        try:
            return float(d[key])
        except Exception:
            return float(np.nan)
    return float(np.nan)

def print_experiment_config() -> Dict[str, Any]:
    cfg = {
        "repo_root": str(REPO_ROOT),
        "base_root": str(BASE_ROOT),
        "datasets": DATASETS,
        "dataset_instances": DATASET_INSTANCES,
        "methods": METHODS,
        "seeds": SEEDS,
        "scenario": "graph_fl",
        "task": "graph_cls_2",
        "simulation_mode": "graph_fl_label_skew",
        "num_clients": K_CLIENTS,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "training_defaults": TRAINING_DEFAULTS,
        "fedprox_mu": FEDPROX_MU,
        "gcfl_plus_defaults": GCFL_PLUS_DEFAULTS,
        "fedstar_defaults": FEDSTAR_DEFAULTS,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "show_round_logs": SHOW_ROUND_LOGS,
    }
    print("\n================ EXPERIMENT CONFIG ================")
    print(json.dumps(cfg, indent=2, sort_keys=True))
    print("===================================================\n")
    return cfg

# =============================================================================
# Main
# =============================================================================
def main():
    runner_cfg = print_experiment_config()

    total_runs = sum(DATASET_INSTANCES[d] for d in DATASETS) * len(SEEDS) * len(METHODS)
    run_idx = 0

    # raw_results[method][dataset] -> list of floats (len = instances*seeds)
    raw_results = defaultdict(lambda: defaultdict(list))
    runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        n_inst = int(DATASET_INSTANCES[dataset])

        for inst_i in range(1, n_inst + 1):
            inst_path = inst_root(inst_i)

            for seed in SEEDS:
                print(f"\n================ SETUP: {dataset} | inst={inst_i:02d}/{n_inst} | seed={seed} ================\n", flush=True)

                # ensure partitions exist
                args_probe = make_args(dataset, inst_path, seed, method="fedavg")
                proc_dir = processed_dir_from_args(args_probe)
                assert_partitions_exist(proc_dir, args_probe.num_clients)

                # delete split cache ONCE per (dataset,inst,seed)
                delete_default_split(proc_dir)

                # seed ONCE per (dataset,inst,seed) so the regenerated split is seed-dependent
                seed_everything(seed)

                for method in METHODS:
                    run_idx += 1
                    tag = f"[{run_idx:04d}/{total_runs}] {method} | {dataset} | inst={inst_i:02d} | seed={seed}"
                    print(f"{tag} ...", flush=True)

                    # method config overrides (module-level dicts)
                    set_method_module_configs(method)

                    # build args
                    args = make_args(dataset, inst_path, seed, method)
                    proc_dir = processed_dir_from_args(args)

                    # capture stdout/stderr unless requested otherwise
                    f_out = io.StringIO()
                    f_err = io.StringIO()
                    ctx_out = nullcontext() if SHOW_ROUND_LOGS else redirect_stdout(f_out)
                    ctx_err = nullcontext() if SHOW_ROUND_LOGS else redirect_stderr(f_err)

                    try:
                        with ctx_out, ctx_err:
                            trainer = FGLTrainer(args)
                            trainer.train()

                        acc = extract_best_test(trainer, METRIC)
                        raw_results[method][dataset].append(acc)

                        rec = {
                            "method": method,
                            "dataset": dataset,
                            "instance_index": inst_i,
                            "instance_root": str(inst_path),
                            "seed": seed,
                            "processed_dir": str(proc_dir),
                            "best_test_accuracy": acc,
                        }
                        if STORE_STDIO_TAILS and (not SHOW_ROUND_LOGS):
                            rec["stdout_tail"] = f_out.getvalue()[-STDIO_TAIL_CHARS:]
                            rec["stderr_tail"] = f_err.getvalue()[-STDIO_TAIL_CHARS:]
                        runs.append(rec)

                        print(f"  [result] best_test_accuracy={acc:.4f}", flush=True)

                    except Exception as e:
                        err = str(e)
                        raw_results[method][dataset].append(float(np.nan))
                        failures.append({
                            "method": method,
                            "dataset": dataset,
                            "instance_index": inst_i,
                            "instance_root": str(inst_path),
                            "seed": seed,
                            "processed_dir": str(proc_dir),
                            "error": err,
                            "stdout_tail": f_out.getvalue()[-STDIO_TAIL_CHARS:],
                            "stderr_tail": f_err.getvalue()[-STDIO_TAIL_CHARS:],
                        })
                        print(f"  [result] FAILED ({err})", flush=True)
                        if (not SHOW_ROUND_LOGS) and f_err.getvalue().strip():
                            print("  [captured stderr tail]")
                            print(f_err.getvalue()[-STDIO_TAIL_CHARS:], flush=True)

    # Summary (Table-6 style numbers are in %; we store both)
    print("\n================ FINAL SUMMARY (mean ± std, NaNs ignored) ================")
    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for method in METHODS:
        summary[method] = {}
        for dataset in DATASETS:
            vals = np.array(raw_results[method][dataset], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size:
                mean = float(vals.mean())
                std = float(vals.std())
                summary[method][dataset] = {
                    "mean": mean,
                    "std": std,
                    "n": int(vals.size),
                    "mean_pct": mean * 100.0,
                    "std_pct": std * 100.0,
                }
                print(f"{method:10s} | {dataset:10s} -> {mean*100.0:.2f} ± {std*100.0:.2f} (n={int(vals.size)})")
            else:
                summary[method][dataset] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "n": 0,
                    "mean_pct": float("nan"),
                    "std_pct": float("nan"),
                }
                print(f"{method:10s} | {dataset:10s} -> all runs failed")

    payload = {
        "meta": {
            "runner_config": runner_cfg,
            "note": "Accuracies stored in [0,1]; *_pct fields are [%]."
        },
        "runs": runs,
        "failures": failures,
        "raw_results": {m: {d: raw_results[m][d] for d in DATASETS} for m in METHODS},
        "summary": summary,
    }

    np.save(str(OUT_NPY), payload, allow_pickle=True)
    print(f"\nSaved results to: {OUT_NPY}", flush=True)


if __name__ == "__main__":
    main()
