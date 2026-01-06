"""
ihsan/run_scalability_graphfl_clients.py

Scalability runner for OpenFGL Graph-FL:
- Runs accuracy vs. number of clients (K) for graph classification.
- Methods: fedavg, fedala, fedala_r
- Datasets: MUTAG, BZR, COX2, AIDS
- Seeds: fixed list (user-provided)
- Uses already-downloaded multi-instance partitions under:
    <REPO_ROOT>/data_table6_graphfl_a1_multi/inst_XX/distrib/...

Key behaviors:
- For each (dataset, inst, K, seed), deletes split cache ONCE, then runs methods.
  => methods share same split for fair comparison, but different seeds regenerate splits.
- Validates that the expected processed_dir exists and includes data_0.pt..data_{K-1}.pt
- Extracts best_test_accuracy from trainer.evaluation_result.
- Saves detailed payload to a .npy (runs + summary).

Run:
  python -m ihsan.run_scalability_graphfl_clients
or
  python ihsan/run_scalability_graphfl_clients.py
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.serialization

# -------------------------------------------------------------------
# Repo import path: keep scripts inside /ihsan without moving them
# -------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from openfgl.utils.basic_utils import seed_everything

# -------------------------------------------------------------------
# Optional: PyTorch safe globals for PyG torch.load (if needed)
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

# =============================================================================
# USER CONFIG
# =============================================================================
DATASETS = ["BZR", "COX2", "AIDS"]
METHODS  = ["fedavg", "fedala", "fedala_r"]

# You requested these seeds explicitly
SEEDS = [540, 204, 350]

# Client counts for scalability curve (edit if desired)
CLIENT_COUNTS = [5, 10, 15, 20]

METRIC = "accuracy"
DIRICHLET_ALPHA = 1.0  # fixed for your downloaded partitions

# Your new instance counts
DATASET_INSTANCES: Dict[str, int] = {
    "MUTAG": 3,
    "COX2": 2,
    "BZR": 2,
    "AIDS": 1,
}

# Base root containing inst_01, inst_02, ...
MULTI_ROOT = REPO_ROOT / "data_table6_graphfl_a1_multi"

OUT_NPY = str(REPO_ROOT / "results_scalability_graphfl_clients.npy")

# Training defaults (keep aligned with what you used for baselines/FedALA comparisons)
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

# FedALA knobs (fedala + fedala_r)
ALA_DEFAULTS = dict(
    ala_batch_size=32,
    ala_rand_percent=40.0,
    ala_layer_idx=1,
    ala_eta=0.05,
    ala_std_threshold=0.02,
    ala_num_pre_loss=5,
    ala_max_warmup_passes=5,
)

# FedALA-R server/client knobs (kept consistent with your repo conventions)
# Server-side likely uses residual_*; client currently uses r_res_scale.
RESIDUAL_DEFAULTS = dict(
    residual_gamma=0.05,
    residual_beta=0.95,
    residual_clip_norm=1.0,
    residual_start_round=10,
    r_res_scale=1.0,
)

SHOW_ROUND_LOGS = False
STORE_STDIO_TAILS = True
STDIO_TAIL_CHARS = 2000

# =============================================================================
# Helpers
# =============================================================================
def processed_dir_from_args(args) -> Path:
    # matches OpenFGL distributed_dataset_loader naming
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


def delete_default_split_cache(proc_dir: Path) -> None:
    # In this repo, graph tasks cache under graph_cls/default_split even for graph_cls_2
    split_dir = proc_dir / "graph_cls" / "default_split"
    if split_dir.is_dir():
        shutil.rmtree(split_dir)


def extract_best_test_accuracy_percent(trainer: FGLTrainer) -> float:
    d = getattr(trainer, "evaluation_result", None)
    if not isinstance(d, dict):
        return float(np.nan)
    key = f"best_test_{METRIC}"
    if key not in d:
        return float(np.nan)
    try:
        v = float(d[key])
    except Exception:
        return float(np.nan)
    # OpenFGL often reports in [0,1]; convert to percent for Table-style reporting
    return v * 100.0 if v <= 1.0 else v


def make_args(dataset: str, inst_root: Path, num_clients: int, seed: int, method: str):
    args = deepcopy(config.args)

    # scenario/task
    args.root = str(inst_root)
    args.scenario = "graph_fl"
    args.task = "graph_cls_2"
    args.dataset = [dataset]
    args.processing = "raw"

    # simulation
    args.simulation_mode = "graph_fl_label_skew"
    args.num_clients = int(num_clients)
    args.client_frac = 1.0
    args.dirichlet_alpha = float(DIRICHLET_ALPHA)
    args.skew_alpha = float(DIRICHLET_ALPHA)

    # speed knobs (Windows stability)
    args.num_workers = 0
    args.persistent_workers = False

    # training
    for k, v in TRAINING_DEFAULTS.items():
        setattr(args, k, v)

    # eval
    args.metrics = [METRIC]

    # algorithm + seed
    args.fl_algorithm = method
    args.seed = int(seed)

    # ALA knobs
    for k, v in ALA_DEFAULTS.items():
        setattr(args, k, v)

    # Residual knobs (fedala_r only; harmless if unused elsewhere)
    if method == "fedala_r":
        for k, v in RESIDUAL_DEFAULTS.items():
            setattr(args, k, v)

    return args


def print_experiment_config() -> Dict[str, Any]:
    cfg = {
        "repo_root": str(REPO_ROOT),
        "multi_root": str(MULTI_ROOT),
        "datasets": DATASETS,
        "methods": METHODS,
        "seeds": SEEDS,
        "client_counts": CLIENT_COUNTS,
        "scenario": "graph_fl",
        "task": "graph_cls_2",
        "simulation_mode": "graph_fl_label_skew",
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "dataset_instances": DATASET_INSTANCES,
        "training_defaults": TRAINING_DEFAULTS,
        "ala_defaults": ALA_DEFAULTS,
        "residual_defaults": RESIDUAL_DEFAULTS,
        "torch_version": getattr(torch, "__version__", "unknown"),
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

    # inst roots: inst_01, inst_02, ...
    max_inst = max(DATASET_INSTANCES.values())
    base_instance_roots: List[Path] = [
        MULTI_ROOT / f"inst_{i:02d}" for i in range(1, max_inst + 1)
    ]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    total_runs = sum(
        DATASET_INSTANCES[ds] * len(SEEDS) * len(CLIENT_COUNTS) * len(METHODS)
        for ds in DATASETS
    )
    run_idx = 0

    for dataset in DATASETS:
        n_inst = DATASET_INSTANCES[dataset]

        for inst_i in range(n_inst):
            inst_root = base_instance_roots[inst_i]
            if not inst_root.is_dir():
                raise FileNotFoundError(f"Instance root not found: {inst_root}")

            for k in CLIENT_COUNTS:
                for seed in SEEDS:
                    print(f"\n================ SETUP: {dataset} | inst={inst_i+1:02d}/{n_inst} | K={k} | seed={seed} ================\n", flush=True)

                    # Build args for partition existence check (method-independent)
                    args_check = make_args(dataset, inst_root, k, seed, METHODS[0])
                    proc_dir = processed_dir_from_args(args_check)

                    # If this (dataset,K) partition set is missing, skip cleanly
                    try:
                        assert_partitions_exist(proc_dir, k)
                    except Exception as e:
                        msg = str(e)
                        print(f"[skip] Missing partitions for {dataset} inst={inst_i+1:02d} K={k}: {msg}", flush=True)
                        skipped.append({
                            "dataset": dataset,
                            "instance_index": inst_i + 1,
                            "instance_root": str(inst_root),
                            "K": k,
                            "seed": seed,
                            "processed_dir": str(proc_dir),
                            "reason": msg,
                        })
                        continue

                    # Delete split cache ONCE per (dataset, inst, K, seed)
                    delete_default_split_cache(proc_dir)

                    # Seed everything ONCE per (dataset, inst, K, seed)
                    seed_everything(seed)

                    for method in METHODS:
                        run_idx += 1
                        print(f"[{run_idx:04d}/{total_runs}] {method} | {dataset} | inst={inst_i+1:02d} | K={k} | seed={seed} ...", flush=True)

                        args = make_args(dataset, inst_root, k, seed, method)

                        f_out = io.StringIO()
                        f_err = io.StringIO()

                        try:
                            ctx_out = nullcontext() if SHOW_ROUND_LOGS else redirect_stdout(f_out)
                            ctx_err = nullcontext() if SHOW_ROUND_LOGS else redirect_stderr(f_err)

                            with ctx_out, ctx_err:
                                trainer = FGLTrainer(args)
                                trainer.train()

                            acc_pct = extract_best_test_accuracy_percent(trainer)
                            results[method][dataset][k].append(acc_pct)

                            rec = {
                                "method": method,
                                "dataset": dataset,
                                "instance_index": inst_i + 1,
                                "instance_root": str(inst_root),
                                "K": int(k),
                                "seed": int(seed),
                                "processed_dir": str(proc_dir),
                                "best_test_accuracy_percent": float(acc_pct),
                            }
                            if STORE_STDIO_TAILS and (not SHOW_ROUND_LOGS):
                                rec["stdout_tail"] = f_out.getvalue()[-STDIO_TAIL_CHARS:]
                                rec["stderr_tail"] = f_err.getvalue()[-STDIO_TAIL_CHARS:]

                            runs.append(rec)
                            print(f"  [result] best_test_accuracy={acc_pct:.2f}%", flush=True)

                        except Exception as e:
                            err = str(e)
                            failures.append({
                                "method": method,
                                "dataset": dataset,
                                "instance_index": inst_i + 1,
                                "instance_root": str(inst_root),
                                "K": int(k),
                                "seed": int(seed),
                                "processed_dir": str(proc_dir),
                                "error": err,
                                "stdout_tail": f_out.getvalue()[-STDIO_TAIL_CHARS:],
                                "stderr_tail": f_err.getvalue()[-STDIO_TAIL_CHARS:],
                            })
                            results[method][dataset][k].append(float(np.nan))
                            print(f"  [result] FAILED ({err})", flush=True)
                            if (not SHOW_ROUND_LOGS) and f_err.getvalue().strip():
                                print("  [captured stderr tail]")
                                print(f_err.getvalue()[-STDIO_TAIL_CHARS:], flush=True)

    # Build summary: mean ± std per (method, dataset, K)
    summary: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    print("\n================ FINAL SUMMARY (mean ± std, NaNs ignored) ================")
    for method in METHODS:
        summary[method] = {}
        for dataset in DATASETS:
            summary[method][dataset] = {}
            for k in CLIENT_COUNTS:
                vals = np.array(results[method][dataset].get(k, []), dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size:
                    m, s = float(vals.mean()), float(vals.std())
                    summary[method][dataset][k] = {"mean": m, "std": s, "n": int(vals.size)}
                    print(f"{method:9s} | {dataset:8s} | K={k:2d} -> {m:.2f} ± {s:.2f} (n={int(vals.size)})")
                else:
                    summary[method][dataset][k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                    print(f"{method:9s} | {dataset:8s} | K={k:2d} -> no runs")

    payload = {
        "meta": {"runner_config": runner_cfg},
        "runs": runs,
        "summary": summary,
        "raw_results": {
            m: {d: {int(k): results[m][d].get(k, []) for k in CLIENT_COUNTS} for d in DATASETS}
            for m in METHODS
        },
        "failures": failures,
        "skipped": skipped,
    }
    np.save(OUT_NPY, payload, allow_pickle=True)
    print(f"\nSaved detailed results to: {OUT_NPY}", flush=True)


if __name__ == "__main__":
    main()
