"""
ihsan/run_runtime_graphfl_clients.py

Runtime + overhead scalability runner for OpenFGL Graph-FL (graph classification).

Goal:
- Measure (1) accuracy and (2) runtime per round as K varies.
- Produce artifacts needed for plots like:
    (a) Runtime vs #clients (mean ± std)
    (b) Overhead vs FedAvg (FedALA−FedAvg, FedALA-R−FedAvg)

Scope:
- Datasets: BZR, COX2, AIDS   (MUTAG omitted)
- Methods:  fedavg, fedala, fedala_r
- Client counts (K): [5, 10, 15, 20]
- Seeds: [540, 204, 350]
- Usesalready-downloaded multi-instance partitions under:
    <REPO_ROOT>/data_table6_graphfl_a1_multi/inst_XX/distrib/...

Key behavior for fairness:
- For each (dataset, inst, K, seed), delete split cache ONCE, then run all methods
  so they share the same split for that seed.

Outputs:
- Saves a .npy payload (runs + summary + failures + skipped) at:
    <REPO_ROOT>/results_runtime_graphfl_clients.npy

Run:
  python -m ihsan.run_runtime_graphfl_clients
or
  python ihsan/run_runtime_graphfl_clients.py
"""

from __future__ import annotations

import io
import json
import sys
import shutil
import time
from copy import deepcopy
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.serialization


# PyTorch 2.6+ compatibility: avoid weights_only=True default surprises
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load


# Repo import path: keep scripts inside /ihsan without moving them
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from openfgl.utils.basic_utils import seed_everything


# Optional: safe globals for torch.load with PyG objects
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
DATASETS = ["BZR", "COX2", "AIDS"]
METHODS  = ["fedavg", "fedala", "fedala_r"]

SEEDS = [540, 204, 350]
CLIENT_COUNTS = [5, 10, 15, 20]

METRIC = "accuracy"
DIRICHLET_ALPHA = 1.0

DATASET_INSTANCES: Dict[str, int] = {
    "COX2": 1,
    "BZR": 1,
    "AIDS": 1,
}

# Base root containing inst_01, inst_02, ...
MULTI_ROOT = REPO_ROOT / "data_table6_graphfl_a1_multi"

OUT_NPY = str(REPO_ROOT / "results_runtime_graphfl_clients.npy")

TRAINING_DEFAULTS = dict(
    num_rounds=100,
    num_epochs=2,          # (OpenFGL uses num_epochs for local epochs in this scenario)
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

# FedALA-R knobs
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

WARMUP_RUNS = 0
REPEATS = 1


# Helpers
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

    return v * 100.0 if v <= 1.0 else v


def _cuda_sync_if_needed() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def timed_train(trainer: FGLTrainer) -> float:
    """
    Returns wall-clock seconds for trainer.train().
    Uses CUDA sync when available for more reliable timing.
    """
    _cuda_sync_if_needed()
    t0 = time.perf_counter()
    trainer.train()
    _cuda_sync_if_needed()
    t1 = time.perf_counter()
    return float(t1 - t0)


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

    # speed knobs (esp. Windows)
    args.num_workers = 0
    args.persistent_workers = False

    # training
    for k, v in TRAINING_DEFAULTS.items():
        setattr(args, k, v)

    # metrics
    args.metrics = [METRIC]

    # algorithm + seed
    args.fl_algorithm = method
    args.seed = int(seed)

    # ALA knobs
    for k, v in ALA_DEFAULTS.items():
        setattr(args, k, v)

    # Residual knobs (fedala_r only)
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
        "warmup_runs": WARMUP_RUNS,
        "repeats": REPEATS,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    print("\n================ EXPERIMENT CONFIG ================")
    print(json.dumps(cfg, indent=2, sort_keys=True))
    print("===================================================\n")
    return cfg


def summarize(values: List[float]) -> Tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    return mean, std, int(arr.size)


# Main
def main():
    runner_cfg = print_experiment_config()

    max_inst = max(DATASET_INSTANCES.values())
    base_instance_roots: List[Path] = [
        MULTI_ROOT / f"inst_{i:02d}" for i in range(1, max_inst + 1)
    ]

    # Aggregate collectors
    acc_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))      # [method][dataset][K] -> [acc%]
    tpr_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))      # time per round seconds
    ttot_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))     # total seconds

    runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    total_runs = sum(
        DATASET_INSTANCES[ds] * len(SEEDS) * len(CLIENT_COUNTS) * len(METHODS) * REPEATS
        for ds in DATASETS
    )
    run_idx = 0

    # For overhead computation: match on (dataset, inst, K, seed, repeat)
    fedavg_time_per_round: Dict[Tuple[str, int, int, int, int], float] = {}

    for dataset in DATASETS:
        n_inst = DATASET_INSTANCES[dataset]

        for inst_i in range(n_inst):
            inst_root = base_instance_roots[inst_i]
            if not inst_root.is_dir():
                raise FileNotFoundError(f"Instance root not found: {inst_root}")

            for k in CLIENT_COUNTS:
                for seed in SEEDS:
                    print(
                        f"\n================ SETUP: {dataset} | inst={inst_i+1:02d}/{n_inst} | K={k} | seed={seed} ================\n",
                        flush=True,
                    )

                    # Partition existence check
                    args_check = make_args(dataset, inst_root, k, seed, METHODS[0])
                    proc_dir = processed_dir_from_args(args_check)

                    try:
                        assert_partitions_exist(proc_dir, k)
                    except Exception as e:
                        msg = str(e)
                        print(f"[skip] Missing partitions for {dataset} inst={inst_i+1:02d} K={k}: {msg}", flush=True)
                        skipped.append({
                            "dataset": dataset,
                            "instance_index": inst_i + 1,
                            "instance_root": str(inst_root),
                            "K": int(k),
                            "seed": int(seed),
                            "processed_dir": str(proc_dir),
                            "reason": msg,
                        })
                        continue

                    # Delete split cache ONCE per (dataset, inst, K, seed)
                    delete_default_split_cache(proc_dir)

                    # Seed everything ONCE per (dataset, inst, K, seed)
                    seed_everything(seed)

                    for repeat in range(REPEATS + WARMUP_RUNS):
                        is_warmup = repeat < WARMUP_RUNS
                        rep_id = repeat - WARMUP_RUNS  # 0..REPEATS-1 for real runs

                        for method in METHODS:
                            if is_warmup:
                                tag = f"(warmup) {method} | {dataset} | inst={inst_i+1:02d} | K={k} | seed={seed}"
                            else:
                                tag = f"{method} | {dataset} | inst={inst_i+1:02d} | K={k} | seed={seed} | rep={rep_id}"

                            run_idx += 1
                            print(f"[{run_idx:04d}/{total_runs:04d}] {tag} ...", flush=True)

                            args = make_args(dataset, inst_root, k, seed, method)

                            f_out = io.StringIO()
                            f_err = io.StringIO()

                            try:
                                ctx_out = nullcontext() if SHOW_ROUND_LOGS else redirect_stdout(f_out)
                                ctx_err = nullcontext() if SHOW_ROUND_LOGS else redirect_stderr(f_err)

                                with ctx_out, ctx_err:
                                    trainer = FGLTrainer(args)
                                    wall_sec = timed_train(trainer)

                                acc_pct = extract_best_test_accuracy_percent(trainer)

                                # discard warmup from stats storage but keep in failures visibility if needed
                                if not is_warmup:
                                    per_round_sec = wall_sec / float(getattr(args, "num_rounds", 1) or 1)

                                    acc_store[method][dataset][k].append(float(acc_pct))
                                    ttot_store[method][dataset][k].append(float(wall_sec))
                                    tpr_store[method][dataset][k].append(float(per_round_sec))

                                rec = {
                                    "method": method,
                                    "dataset": dataset,
                                    "instance_index": inst_i + 1,
                                    "instance_root": str(inst_root),
                                    "K": int(k),
                                    "seed": int(seed),
                                    "repeat": None if is_warmup else int(rep_id),
                                    "is_warmup": bool(is_warmup),
                                    "processed_dir": str(proc_dir),
                                    "num_rounds": int(getattr(args, "num_rounds", 0)),
                                    "num_epochs": int(getattr(args, "num_epochs", 0)),
                                    "lr": float(getattr(args, "lr", float("nan"))),
                                    "best_test_accuracy_percent": float(acc_pct),
                                    "wall_time_sec_total": float(wall_sec),
                                    "wall_time_sec_per_round": float(wall_sec / float(getattr(args, "num_rounds", 1) or 1)),
                                }

                                # overhead vs FedAvg for the *same* (dataset, inst, K, seed, rep)
                                if not is_warmup:
                                    key = (dataset, inst_i + 1, int(k), int(seed), int(rep_id))
                                    if method == "fedavg":
                                        fedavg_time_per_round[key] = float(rec["wall_time_sec_per_round"])
                                        rec["overhead_sec_per_round_vs_fedavg"] = 0.0
                                    else:
                                        base = fedavg_time_per_round.get(key, float("nan"))
                                        rec["overhead_sec_per_round_vs_fedavg"] = (
                                            float(rec["wall_time_sec_per_round"]) - float(base)
                                            if not np.isnan(base) else float("nan")
                                        )

                                if STORE_STDIO_TAILS and (not SHOW_ROUND_LOGS):
                                    rec["stdout_tail"] = f_out.getvalue()[-STDIO_TAIL_CHARS:]
                                    rec["stderr_tail"] = f_err.getvalue()[-STDIO_TAIL_CHARS:]

                                runs.append(rec)

                                if is_warmup:
                                    print(f"  [warmup] done (time={wall_sec:.3f}s)", flush=True)
                                else:
                                    print(
                                        f"  [result] acc={acc_pct:.2f}% | time/round={rec['wall_time_sec_per_round']:.4f}s "
                                        f"| overhead_vs_fedavg={rec.get('overhead_sec_per_round_vs_fedavg', float('nan')):.4f}s",
                                        flush=True,
                                    )

                            except Exception as e:
                                err = str(e)
                                failures.append({
                                    "method": method,
                                    "dataset": dataset,
                                    "instance_index": inst_i + 1,
                                    "instance_root": str(inst_root),
                                    "K": int(k),
                                    "seed": int(seed),
                                    "repeat": None if is_warmup else int(rep_id),
                                    "is_warmup": bool(is_warmup),
                                    "processed_dir": str(proc_dir),
                                    "error": err,
                                    "stdout_tail": f_out.getvalue()[-STDIO_TAIL_CHARS:],
                                    "stderr_tail": f_err.getvalue()[-STDIO_TAIL_CHARS:],
                                })
                                # only count failures into stores for real runs (keeps array lengths consistent)
                                if not is_warmup:
                                    acc_store[method][dataset][k].append(float(np.nan))
                                    ttot_store[method][dataset][k].append(float(np.nan))
                                    tpr_store[method][dataset][k].append(float(np.nan))

                                print(f"  [result] FAILED ({err})", flush=True)
                                if (not SHOW_ROUND_LOGS) and f_err.getvalue().strip():
                                    print("  [captured stderr tail]")
                                    print(f_err.getvalue()[-STDIO_TAIL_CHARS:], flush=True)

    # Summaries (accuracy + runtime)
    summary: Dict[str, Any] = {
        "accuracy_percent": {},
        "time_per_round_sec": {},
        "total_time_sec": {},
    }

    print("\n================ FINAL SUMMARY: Accuracy (mean ± std, NaNs ignored) ================")
    for method in METHODS:
        summary["accuracy_percent"][method] = {}
        for dataset in DATASETS:
            summary["accuracy_percent"][method][dataset] = {}
            for k in CLIENT_COUNTS:
                m, s, n = summarize(acc_store[method][dataset].get(k, []))
                summary["accuracy_percent"][method][dataset][k] = {"mean": m, "std": s, "n": n}
                if n > 0:
                    print(f"{method:9s} | {dataset:6s} | K={k:2d} -> {m:.2f} ± {s:.2f} (n={n})")
                else:
                    print(f"{method:9s} | {dataset:6s} | K={k:2d} -> no runs")

    print("\n================ FINAL SUMMARY: Time/round (sec, mean ± std, NaNs ignored) ================")
    for method in METHODS:
        summary["time_per_round_sec"][method] = {}
        for dataset in DATASETS:
            summary["time_per_round_sec"][method][dataset] = {}
            for k in CLIENT_COUNTS:
                m, s, n = summarize(tpr_store[method][dataset].get(k, []))
                summary["time_per_round_sec"][method][dataset][k] = {"mean": m, "std": s, "n": n}
                if n > 0:
                    print(f"{method:9s} | {dataset:6s} | K={k:2d} -> {m:.4f} ± {s:.4f} (n={n})")
                else:
                    print(f"{method:9s} | {dataset:6s} | K={k:2d} -> no runs")

    # Optional: overhead summaries computed directly from run records (safe + simple)
    # (plotting scripts can also recompute these)
    overhead_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # [method][dataset][K] -> [overhead/round]
    for r in runs:
        if r.get("is_warmup"):
            continue
        m = r["method"]
        if m not in ("fedala", "fedala_r"):
            continue
        ov = r.get("overhead_sec_per_round_vs_fedavg", float("nan"))
        overhead_store[m][r["dataset"]][int(r["K"])].append(float(ov))

    summary["overhead_sec_per_round_vs_fedavg"] = {}
    print("\n================ FINAL SUMMARY: Overhead vs FedAvg (sec/round, mean ± std) ================")
    for method in ("fedala", "fedala_r"):
        summary["overhead_sec_per_round_vs_fedavg"][method] = {}
        for dataset in DATASETS:
            summary["overhead_sec_per_round_vs_fedavg"][method][dataset] = {}
            for k in CLIENT_COUNTS:
                m, s, n = summarize(overhead_store[method][dataset].get(k, []))
                summary["overhead_sec_per_round_vs_fedavg"][method][dataset][k] = {"mean": m, "std": s, "n": n}
                if n > 0:
                    print(f"{method:9s} | {dataset:6s} | K={k:2d} -> {m:.4f} ± {s:.4f} (n={n})")
                else:
                    print(f"{method:9s} | {dataset:6s} | K={k:2d} -> no runs")

    payload = {
        "meta": {"runner_config": runner_cfg},
        "runs": runs,
        "summary": summary,
        "failures": failures,
        "skipped": skipped,
    }

    np.save(OUT_NPY, payload, allow_pickle=True)
    print(f"\nSaved detailed results to: {OUT_NPY}", flush=True)


if __name__ == "__main__":
    main()
