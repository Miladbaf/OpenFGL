"""
ihsan/run_graphfl_fedala_grid.py

Runs FedALA and FedALA-R on Graph-FL graph classification (graph_cls_2)
with graph_fl_label_skew (alpha=1.0, K=10) across multiple dataset instances.

Outputs:
- .npy payload containing:
  - meta config
  - per-run results
  - per-method/dataset summary (mean/std)
"""

from __future__ import annotations

import io
import json
import sys
import shutil
from copy import deepcopy
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.serialization

# Repo import path
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from openfgl.utils.basic_utils import seed_everything

# Optional: PyTorch 2.6+ safe globals for PyG torch.load
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

# USER CONFIG
DATASETS = [ "BZR", "AIDS", "COX2"]
METHODS  = ["fedala", "fedala_r"]
#SEEDS    = [42, 123, 456]
SEEDS    = [540, 204, 350]
METRIC   = "accuracy"

K_CLIENTS = 10
DIRICHLET_ALPHA = 1.0

DATASET_INSTANCES = {
    "MUTAG": 3,
    "COX2": 2,
    "BZR": 2,
    "AIDS": 1,
}

# Instance roots: BASE/inst_01, inst_02, ...
BASE_MULTI_ROOT = REPO_ROOT / "data_table6_graphfl_a1_multi"

OUT_NPY = str(REPO_ROOT / "ihsan" / "graphfl_fedala_fedalar_results.npy")

TRAINING_DEFAULTS = dict(
    num_rounds=100,
    num_epochs=2,
    lr=1e-3,
    weight_decay=5e-4,
    batch_size=128,
    dropout=0.5,
    optim="adam",
    model=["gin"],
    evaluation_mode="local_model_on_local_data",
)

ALA_DEFAULTS = dict(
    ala_batch_size=32,
    ala_rand_percent=40.0,
    ala_layer_idx=1,
    ala_eta=0.05,
    ala_std_threshold=0.02,
    ala_num_pre_loss=5,
    ala_max_warmup_passes=5,
)

RESIDUAL_DEFAULTS = dict(
    residual_gamma=0.01,
    residual_beta=0.95,
    residual_clip_norm=1.0,
    residual_start_round=20,
)

SHOW_ROUND_LOGS = False
STDIO_TAIL_CHARS = 2000

# Helpers
def processed_dir_from_args(args) -> Path:
    if args.simulation_mode in ["graph_fl_label_skew", "subgraph_fl_label_skew"]:
        sim_name = f"{args.simulation_mode}_{args.skew_alpha:.2f}"
    else:
        sim_name = args.simulation_mode

    ds = sorted(list(args.dataset))
    folder = "_".join([sim_name, "_".join(ds), f"client_{args.num_clients}"])
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
    # OpenFGL caches splits here
    split_dir = proc_dir / "graph_cls" / "default_split"
    if split_dir.is_dir():
        shutil.rmtree(split_dir)

def make_args(dataset: str, inst_root: Path, seed: int, method: str):
    args = deepcopy(config.args)

    # scenario/task
    args.root = str(inst_root)
    args.scenario = "graph_fl"
    args.task = "graph_cls_2"
    args.dataset = [dataset]
    args.processing = "raw"

    # simulation
    args.simulation_mode = "graph_fl_label_skew"
    args.num_clients = K_CLIENTS
    args.client_frac = 1.0
    args.dirichlet_alpha = DIRICHLET_ALPHA
    args.skew_alpha = DIRICHLET_ALPHA

    # Windows stability knobs
    args.num_workers = 0
    args.persistent_workers = False

    # training
    for k, v in TRAINING_DEFAULTS.items():
        setattr(args, k, v)

    # eval
    args.metrics = [METRIC]

    # algorithm + seed
    args.fl_algorithm = method
    args.seed = seed

    # ALA knobs
    for k, v in ALA_DEFAULTS.items():
        setattr(args, k, v)

    # residual knobs
    if method == "fedala_r":
        for k, v in RESIDUAL_DEFAULTS.items():
            setattr(args, k, v)

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

# Main
def main():
    cfg = {
        "repo_root": str(REPO_ROOT),
        "datasets": DATASETS,
        "methods": METHODS,
        "seeds": SEEDS,
        "dataset_instances": DATASET_INSTANCES,
        "base_multi_root": str(BASE_MULTI_ROOT),
        "training_defaults": TRAINING_DEFAULTS,
        "ala_defaults": ALA_DEFAULTS,
        "residual_defaults": RESIDUAL_DEFAULTS,
        "scenario": "graph_fl",
        "task": "graph_cls_2",
        "simulation_mode": "graph_fl_label_skew",
        "num_clients": K_CLIENTS,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "torch_version": getattr(torch, "__version__", "unknown"),
    }

    print("\n================ EXPERIMENT CONFIG ================")
    print(json.dumps(cfg, indent=2, sort_keys=True))
    print("===================================================\n")

    total_runs = sum(DATASET_INSTANCES[d] for d in DATASETS) * len(SEEDS) * len(METHODS)
    run_idx = 0

    results = defaultdict(lambda: defaultdict(list))
    runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        n_inst = DATASET_INSTANCES[dataset]
        for inst_i in range(n_inst):
            inst_root = BASE_MULTI_ROOT / f"inst_{inst_i+1:02d}"

            for seed in SEEDS:
                print(f"\n================ SETUP: {dataset} | inst={inst_i+1:02d}/{n_inst} | seed={seed} ================\n", flush=True)

                # Ensure methods share the same split for this dataset/inst/seed
                base_args = make_args(dataset, inst_root, seed, METHODS[0])
                proc_dir = processed_dir_from_args(base_args)
                assert_partitions_exist(proc_dir, K_CLIENTS)
                delete_default_split(proc_dir)

                seed_everything(seed)

                for method in METHODS:
                    run_idx += 1
                    print(f"[{run_idx:04d}/{total_runs}] {method} | {dataset} | inst={inst_i+1:02d} | seed={seed} ...", flush=True)

                    args = make_args(dataset, inst_root, seed, method)
                    f_out, f_err = io.StringIO(), io.StringIO()

                    try:
                        ctx_out = nullcontext() if SHOW_ROUND_LOGS else redirect_stdout(f_out)
                        ctx_err = nullcontext() if SHOW_ROUND_LOGS else redirect_stderr(f_err)

                        with ctx_out, ctx_err:
                            trainer = FGLTrainer(args)
                            trainer.train()

                        acc = extract_best_test(trainer, METRIC)
                        results[method][dataset].append(acc)

                        runs.append({
                            "method": method,
                            "dataset": dataset,
                            "instance_index": inst_i + 1,
                            "instance_root": str(inst_root),
                            "seed": seed,
                            "processed_dir": str(proc_dir),
                            "best_test_accuracy": acc,
                            "stdout_tail": f_out.getvalue()[-STDIO_TAIL_CHARS:],
                            "stderr_tail": f_err.getvalue()[-STDIO_TAIL_CHARS:],
                        })

                        print(f"  [result] best_test_accuracy={acc:.4f}", flush=True)

                    except Exception as e:
                        err = str(e)
                        failures.append({
                            "method": method,
                            "dataset": dataset,
                            "instance_index": inst_i + 1,
                            "instance_root": str(inst_root),
                            "seed": seed,
                            "processed_dir": str(proc_dir),
                            "error": err,
                            "stdout_tail": f_out.getvalue()[-STDIO_TAIL_CHARS:],
                            "stderr_tail": f_err.getvalue()[-STDIO_TAIL_CHARS:],
                        })
                        results[method][dataset].append(float(np.nan))
                        print(f"  [result] FAILED ({err})", flush=True)
                        tail = f_err.getvalue()[-STDIO_TAIL_CHARS:].strip()
                        if tail:
                            print("  [captured stderr tail]")
                            print(tail, flush=True)

    # Summary
    print("\n================ FINAL SUMMARY (mean ± std, NaNs ignored) ================")
    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method in METHODS:
        summary[method] = {}
        for dataset in DATASETS:
            vals = np.array(results[method][dataset], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size:
                m, s = float(vals.mean()), float(vals.std())
                summary[method][dataset] = {"mean": m, "std": s, "n": int(vals.size)}
                print(f"{method:10s} | {dataset:10s} -> {m*100:.2f} ± {s*100:.2f} (n={int(vals.size)})")
            else:
                summary[method][dataset] = {"mean": float('nan'), "std": float('nan'), "n": 0}
                print(f"{method:10s} | {dataset:10s} -> all runs failed")

    payload = {
        "meta": cfg,
        "runs": runs,
        "summary": summary,
        "failures": failures,
        "raw_results": {m: {d: results[m][d] for d in DATASETS} for m in METHODS},
    }
    np.save(OUT_NPY, payload, allow_pickle=True)
    print(f"\nSaved detailed results to: {OUT_NPY}", flush=True)

if __name__ == "__main__":
    main()
