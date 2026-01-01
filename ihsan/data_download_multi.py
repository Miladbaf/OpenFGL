# download_graphfl_labelskew_a1_k10_multi.py
#
# Creates multiple *independent* Graph-FL label-skew instances per dataset by
# varying the simulation seed and writing each instance to a distinct root.
#
# - K = 10 clients
# - Dirichlet alpha = 1.0
# - Instances:
#     MUTAG:     2
#     COX2:      2
#     BZR:       2
#     AIDS:      1
#     PROTEINS:  3

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import copy
import random
import numpy as np
import torch

from openfgl import config
from openfgl.data.distributed_dataset_loader import FGLDataset

# ----------------------------
# CONFIG
# ----------------------------
K = 10
ALPHA = 1.0

# Base directory (one subfolder per dataset instance)
BASE_ROOT = "../data_table6_graphfl_a1_multi"

# How many independent instances to generate per dataset
DATASET_INSTANCES = {
    "MUTAG": 2,
    "COX2": 2,
    "BZR": 2,
    "AIDS": 1,
    "PROTEINS": 3,
}

# Fixed list of seeds used to create different dataset instances.
# (You can change these, but keep them fixed once you start reporting results.)
INSTANCE_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_args(dataset_name: str, instance_idx: int, instance_seed: int):
    # IMPORTANT: config.args is a shared singleton -> deepcopy to isolate per dataset
    args = copy.deepcopy(config.args)

    # Isolated root per instance so nothing is overwritten / mixed
    instance_root = os.path.join(BASE_ROOT, f"inst_{instance_idx:02d}")
    os.makedirs(instance_root, exist_ok=True)

    # Core
    args.root = instance_root
    args.scenario = "graph_fl"
    args.task = "graph_cls"   # keep graph_cls unless you *explicitly* need graph_cls_2 here
    args.dataset = [dataset_name]
    args.num_clients = K

    # Label-skew simulation (Dirichlet)
    args.simulation_mode = "graph_fl_label_skew"
    args.dirichlet_alpha = ALPHA
    args.skew_alpha = ALPHA  # used in processed_dir naming in OpenFGL

    # Reproducibility / avoid hidden defaults
    if not hasattr(args, "dirichlet_try_cnt"):
        args.dirichlet_try_cnt = 100
    if not hasattr(args, "least_samples"):
        args.least_samples = 5

    # If your dataset generation code uses args.seed or args.random_seed,
    # set both; otherwise seeding is still enforced via seed_all().
    args.seed = instance_seed
    if hasattr(args, "random_seed"):
        args.random_seed = instance_seed

    return args


if __name__ == "__main__":
    # Ensure we have enough seeds
    max_needed = max(DATASET_INSTANCES.values())
    if len(INSTANCE_SEEDS) < max_needed:
        raise ValueError(f"Need at least {max_needed} INSTANCE_SEEDS, got {len(INSTANCE_SEEDS)}")

    for ds_name, n_instances in DATASET_INSTANCES.items():
        for inst in range(1, n_instances + 1):
            seed = INSTANCE_SEEDS[inst - 1]

            print(f"\n=== Processing {ds_name} | instance {inst}/{n_instances} | alpha={ALPHA} | K={K} | seed={seed} ===")
            seed_all(seed)

            args = make_args(ds_name, inst, seed)
            FGLDataset(args)

    print("\nDone.")
    print(f"Created under: {BASE_ROOT}/inst_<NN>/distrib/graph_fl_label_skew_<alpha>_<DATASET>_client_{K}/")
    print("Example path:")
    print(f"  {BASE_ROOT}/inst_01/distrib/graph_fl_label_skew_1.00_MUTAG_client_10/")
