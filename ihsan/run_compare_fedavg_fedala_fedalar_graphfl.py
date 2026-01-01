"""
ihsan/run_compare_graphfl_methods.py

Minimal grid runner for OpenFGL Graph-FL (graph_cls_2) under graph_fl_label_skew.

Logging:
- EXPERIMENT CONFIG block
- SETUP header per (dataset, inst, seed)
- Running line per method
- [config] per run
- [result] per run
- Optional per-round logs (SHOW_ROUND_LOGS)

Run:
  python -m ihsan.run_compare_graphfl_methods
or
  python ihsan/run_compare_graphfl_methods.py
"""

from __future__ import annotations

import io
import json
import sys
from copy import deepcopy
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.serialization

# -----------------------------------------------------------------------------
# Repo import path
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from openfgl.utils.basic_utils import seed_everything

# -----------------------------------------------------------------------------
# Optional: PyTorch 2.6+ safe globals for PyG torch.load
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
DATASETS = ["MUTAG", "COX2", "BZR", "AIDS", "PROTEINS"]
METHODS = ["fedala", "fedala_r"]
SEEDS = [42, 123, 456]

METRIC = "accuracy"
K_CLIENTS = 10
DIRICHLET_ALPHA = 1.0

BASE_INSTANCE_ROOTS: List[Path] = [
    REPO_ROOT / "data_table6_graphfl_a1_multi" / "inst_01",
    REPO_ROOT / "data_table6_graphfl_a1_multi" / "inst_02",
    REPO_ROOT / "data_table6_graphfl_a1_multi" / "inst_03",
]

DATASET_INSTANCES: Dict[str, int] = {
    "MUTAG": 2,
    "COX2": 2,
    "BZR": 2,
    "AIDS": 1,
    "PROTEINS": 3,
}

OUT_NPY = "./results_compare_graphfl_methods_4.npy"

TRAINING_DEFAULTS = dict(
    num_rounds=100,
    num_epochs=2,
    lr=0.001,
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
    ala_max_warmup_epochs = 2
)

RESIDUAL_DEFAULTS = dict(
    residual_gamma=0.05,
    residual_beta=0.95,
    residual_clip_norm=1.0,
    residual_start_round=10,
)

SHOW_ROUND_LOGS = False

# If capturing logs, store only tails to keep output small.
STORE_STDIO_TAILS = True
STDIO_TAIL_CHARS = 2000


# =============================================================================
# Minimal helpers
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

    # speed knobs (avoid dataloader worker persistence weirdness on Windows)
    args.num_workers = 0
    args.persistent_workers = False

    # training
    args.num_rounds = TRAINING_DEFAULTS["num_rounds"]
    args.num_epochs = TRAINING_DEFAULTS["num_epochs"]
    args.lr = TRAINING_DEFAULTS["lr"]
    args.weight_decay = TRAINING_DEFAULTS["weight_decay"]
    args.batch_size = TRAINING_DEFAULTS["batch_size"]
    args.dropout = TRAINING_DEFAULTS["dropout"]
    args.optim = TRAINING_DEFAULTS["optim"]
    args.model = TRAINING_DEFAULTS["model"]
    args.evaluation_mode = TRAINING_DEFAULTS["evaluation_mode"]

    # eval
    args.metrics = [METRIC]

    # algorithm + seed
    args.fl_algorithm = method
    args.seed = seed

    # ALA knobs (both fedala and fedala_r)
    args.ala_batch_size = ALA_DEFAULTS["ala_batch_size"] or args.batch_size
    args.ala_rand_percent = ALA_DEFAULTS["ala_rand_percent"]
    args.ala_layer_idx = ALA_DEFAULTS["ala_layer_idx"]
    args.ala_eta = ALA_DEFAULTS["ala_eta"]
    args.ala_std_threshold = ALA_DEFAULTS["ala_std_threshold"]
    args.ala_num_pre_loss = ALA_DEFAULTS["ala_num_pre_loss"]

    # residual knobs (fedala_r only)
    if method == "fedala_r":
        args.residual_gamma = RESIDUAL_DEFAULTS["residual_gamma"]
        args.residual_beta = RESIDUAL_DEFAULTS["residual_beta"]
        args.residual_clip_norm = RESIDUAL_DEFAULTS["residual_clip_norm"]
        args.residual_start_round = RESIDUAL_DEFAULTS["residual_start_round"]

    return args


def effective_hparams_dict(args, method: str) -> Dict[str, Any]:
    eff = {
        "training": {
            "num_rounds": int(args.num_rounds),
            "num_epochs": int(args.num_epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "dropout": float(args.dropout),
            "optim": str(args.optim),
            "model": list(args.model) if isinstance(args.model, (list, tuple)) else args.model,
            "evaluation_mode": str(args.evaluation_mode),
        },
        "ala": {
            "ala_batch_size": int(args.ala_batch_size),
            "ala_rand_percent": float(args.ala_rand_percent),
            "ala_layer_idx": int(args.ala_layer_idx),
            "ala_eta": float(args.ala_eta),
            "ala_std_threshold": float(args.ala_std_threshold),
            "ala_num_pre_loss": int(args.ala_num_pre_loss),
        },
        "residual": None,
    }
    if method == "fedala_r":
        eff["residual"] = {
            "residual_gamma": float(getattr(args, "residual_gamma", 0.0)),
            "residual_beta": float(getattr(args, "residual_beta", 0.0)),
            "residual_clip_norm": float(getattr(args, "residual_clip_norm", 0.0)),
            "residual_start_round": int(getattr(args, "residual_start_round", 0)),
        }
    return eff


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
        "datasets": DATASETS,
        "methods": METHODS,
        "seeds": SEEDS,
        "scenario": "graph_fl",
        "task": "graph_cls_2",
        "simulation_mode": "graph_fl_label_skew",
        "num_clients": K_CLIENTS,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "dataset_instances": DATASET_INSTANCES,
        "base_instance_roots": [str(p) for p in BASE_INSTANCE_ROOTS],
        "training_defaults": TRAINING_DEFAULTS,
        "ala_defaults": ALA_DEFAULTS,
        "residual_defaults": RESIDUAL_DEFAULTS,
        "show_round_logs": SHOW_ROUND_LOGS,
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

    total_runs = sum(DATASET_INSTANCES[ds] for ds in DATASETS) * len(SEEDS) * len(METHODS)
    run_idx = 0

    results = defaultdict(lambda: defaultdict(list))
    runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        n_inst = DATASET_INSTANCES[dataset]

        for inst_i in range(n_inst):
            inst_root = BASE_INSTANCE_ROOTS[inst_i]

            for seed in SEEDS:
                print(f"\n================ SETUP: {dataset} | inst={inst_i+1:02d}/{n_inst} | seed={seed} ================\n", flush=True)

                # One deterministic seed per (dataset, inst, seed) is enough.
                seed_everything(seed)

                for method in METHODS:
                    run_idx += 1

                    args = make_args(dataset, inst_root, seed, method)
                    proc_dir = processed_dir_from_args(args)
                    assert_partitions_exist(proc_dir, args.num_clients)

                    eff_cfg = effective_hparams_dict(args, method)

                    print(f"[{run_idx:04d}/{total_runs}] Running {method} | {dataset} | inst={inst_i+1:02d} | seed={seed} ...", flush=True)
                    print("[config] " + json.dumps({
                        "dataset": dataset,
                        "instance": inst_i + 1,
                        "seed": seed,
                        "method": method,
                        "processed_dir": str(proc_dir),
                        "effective_hparams": eff_cfg,
                    }, sort_keys=True), flush=True)

                    f_out = io.StringIO()
                    f_err = io.StringIO()

                    try:
                        ctx = nullcontext() if SHOW_ROUND_LOGS else redirect_stdout(f_out)
                        ctx2 = nullcontext() if SHOW_ROUND_LOGS else redirect_stderr(f_err)

                        with ctx, ctx2:
                            trainer = FGLTrainer(args)
                            trainer.train()

                        acc = extract_best_test(trainer, METRIC)
                        results[method][dataset].append(acc)

                        rec = {
                            "method": method,
                            "dataset": dataset,
                            "instance_index": inst_i + 1,
                            "instance_root": str(inst_root),
                            "seed": seed,
                            "processed_dir": str(proc_dir),
                            "best_test_accuracy": acc,
                            "effective_hparams": eff_cfg,
                        }
                        if STORE_STDIO_TAILS and (not SHOW_ROUND_LOGS):
                            rec["stdout_tail"] = f_out.getvalue()[-STDIO_TAIL_CHARS:]
                            rec["stderr_tail"] = f_err.getvalue()[-STDIO_TAIL_CHARS:]

                        runs.append(rec)
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
                        if (not SHOW_ROUND_LOGS) and f_err.getvalue().strip():
                            print("  [captured stderr tail]")
                            print(f_err.getvalue()[-STDIO_TAIL_CHARS:], flush=True)

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
                print(f"{method:10s} | {dataset:10s} -> {m:.4f} ± {s:.4f} (n={int(vals.size)})")
            else:
                summary[method][dataset] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                print(f"{method:10s} | {dataset:10s} -> all runs failed")

    payload = {
        "meta": {"runner_config": runner_cfg},
        "runs": runs,
        "summary": summary,
        "failures": failures,
        "raw_results": {m: {d: results[m][d] for d in DATASETS} for m in METHODS},
    }
    np.save(str(OUT_NPY), payload, allow_pickle=True)
    print(f"\nSaved detailed results to: {OUT_NPY}", flush=True)


if __name__ == "__main__":
    main()
