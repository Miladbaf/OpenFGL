# download_graphfl_labelskew_a1_k10.py

import copy
from openfgl import config
from openfgl.data.distributed_dataset_loader import FGLDataset

import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_ROOT = "../data_table6_graphfl_a1"
DATASETS = ["NCI1"]

K = 10
ALPHA = 1.0  # Dirichlet alpha for label skew


def make_args(dataset_name: str):
    # IMPORTANT: config.args is a shared singleton -> deepcopy to isolate per dataset
    args = copy.deepcopy(config.args)

    # Core
    args.root = DATA_ROOT
    args.scenario = "graph_fl"
    args.task = "graph_cls"
    args.dataset = [dataset_name]
    args.num_clients = K

    # Label-skew simulation (Dirichlet)
    args.simulation_mode = "graph_fl_label_skew"
    args.dirichlet_alpha = ALPHA

    args.skew_alpha = ALPHA

    # Good to set explicitly for reproducibility / to avoid hidden defaults
    if not hasattr(args, "dirichlet_try_cnt"):
        args.dirichlet_try_cnt = 100
    if not hasattr(args, "least_samples"):
        args.least_samples = 5

    return args


if __name__ == "__main__":
    for ds in DATASETS:
        print(f"\n=== Processing {ds} | Graph-FL label skew | alpha={ALPHA} | K={K} ===")
        FGLDataset(make_args(ds))

    print("\nDone.")
    print("Created under: ./data/distrib/graph_fl_label_skew_<alpha>_<DATASET>_client_10")
    print("Example path will look like: ./data/distrib/graph_fl_label_skew_1.00_MUTAG_client_10/")
