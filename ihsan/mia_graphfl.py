# ==============================================================================
# SIMPLE PRIVACY AUDIT (BLACK-BOX MIA) — GRAPH-FL (GRAPH CLASSIFICATION)
# OpenFGL (config.args + FGLTrainer) — FedAvg / FedALA / FedALA-R
#
# Membership unit: GRAPH instances (not nodes).
# Members = all client train graphs; Non-members = all client test (or val) graphs.
#
# Reports:
#  - Loss-threshold MIA AUC (score=-loss)
#  - Logistic regression attacker AUC on [loss, confidence(true label), entropy]
#
# Outputs:
#  - DataFrame printed + saved: mia_graphfl_results.csv and mia_graphfl_results_mean.csv
# ==============================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# (1) SAFE torch.load patch (idempotent; avoids recursion in notebooks)
torch.load = torch.serialization.load
if not hasattr(torch, "_openfgl_original_torch_load"):
    torch._openfgl_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return torch._openfgl_original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
print("✓ torch.load patched safely (idempotent)")

# (2) Patch FedALA + FedALA-R loss_fn bug
def patch_get_ala_loss_fn(module, fn_name="_get_ala_loss_fn"):
    orig_name = f"_orig{fn_name}"
    if not hasattr(module, orig_name):
        setattr(module, orig_name, getattr(module, fn_name))
    orig = getattr(module, orig_name)

    def fixed(task):
        fn = getattr(task, "default_loss_fn", None)
        if isinstance(fn, nn.Module):
            return lambda logits, labels: fn(logits, labels)
        if isinstance(fn, type) and issubclass(fn, nn.Module):
            crit = fn()
            return lambda logits, labels: crit(logits, labels)
        if callable(fn):
            crit = fn()
            if isinstance(crit, nn.Module):
                return lambda logits, labels: crit(logits, labels)
            if callable(crit):
                return lambda logits, labels: crit(logits, labels)
        return orig(task)

    setattr(module, fn_name, fixed)

try:
    import openfgl.flcore.fedala.client_ihsan as ala_mod
    import openfgl.flcore.fedala_r.client_ihsan as alar_mod
    patch_get_ala_loss_fn(ala_mod)
    patch_get_ala_loss_fn(alar_mod)
    print("Patched _get_ala_loss_fn for FedALA and FedALA-R")
except Exception as e:
    print("⚠Could not patch FedALA modules. Error:", e)


# OpenFGL imports
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer


# USER SETTINGS (Graph-FL)
METHODS  = ["fedavg", "fedala", "fedala_r"]
DATASETS = ["BZR", "COX2", "AIDS"]
SEEDS    = [540, 204, 350]

NUM_CLIENTS  = 10
NUM_ROUNDS   = 50
LOCAL_EPOCHS = 2
LR           = 1e-3
WEIGHT_DECAY = 5e-4

PREFER_NONMEMBER = "test"     # "test" or "val"
MAX_POINTS_PER_CLASS = 5000   # cap members and non-members each (balanced)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("✓ Device:", DEVICE)

REPO_ROOT = os.getcwd()
DATA_ROOT = os.path.join(REPO_ROOT, "data_table6_graphfl_a1_multi", "inst_01")  # adjust if needed


# (4) Robust model/logits helpers
def find_first_module(obj, max_depth=10):
    visited = set()
    def _walk(x, depth):
        if x is None: return None
        xid = id(x)
        if xid in visited: return None
        visited.add(xid)
        if isinstance(x, nn.Module):
            return x
        if depth <= 0: return None
        if isinstance(x, dict):
            for v in x.values():
                m = _walk(v, depth-1)
                if m is not None: return m
        if isinstance(x, (list, tuple, set)):
            for v in x:
                m = _walk(v, depth-1)
                if m is not None: return m
        if hasattr(x, "__dict__"):
            for v in x.__dict__.values():
                m = _walk(v, depth-1)
                if m is not None: return m
        return None
    return _walk(obj, max_depth)

def pick_global_model(trainer):
    # prefer server/global model if present
    for obj in [getattr(trainer, "server", None), trainer]:
        if obj is None: continue
        for name in ["global_model", "model", "net", "gnn", "backbone"]:
            if hasattr(obj, name) and isinstance(getattr(obj, name), nn.Module):
                return getattr(obj, name)
    m = find_first_module(getattr(trainer, "server", trainer), max_depth=10)
    if m is None:
        m = find_first_module(trainer, max_depth=10)
    if m is None:
        raise AttributeError("Could not find any torch.nn.Module inside trainer/server.")
    return m

def coerce_logits(out):
    if out is None:
        raise RuntimeError("Model forward returned None.")
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item):
                return item
        for item in out:
            if isinstance(item, dict):
                for k in ["logits", "out", "pred", "y_hat"]:
                    if k in item and torch.is_tensor(item[k]):
                        return item[k]
        raise TypeError("Forward returned tuple/list without tensor logits.")
    if isinstance(out, dict):
        for k in ["logits", "out", "pred", "y_hat"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise TypeError("Forward returned dict but no tensor logits found.")
    if hasattr(out, "logits") and torch.is_tensor(out.logits):
        return out.logits
    raise TypeError(f"Unsupported forward output type: {type(out)}")

def forward_logits_graph_batch(model, batch):
    model.eval()
    with torch.no_grad():
        # Most graph-cls models in PyG accept batch directly
        try:
            return coerce_logits(model(batch))
        except Exception:
            pass
        # Common alt signatures
        try:
            return coerce_logits(model(batch.x, batch.edge_index, batch.batch))
        except Exception:
            pass
        try:
            return coerce_logits(model(batch.x, batch.edge_index))
        except Exception as e:
            raise RuntimeError(f"Could not forward model for graph batch. Last error: {e}")


# (5) Extract client loaders (Graph-FL)
def pick_clients(trainer):
    for obj in [getattr(trainer, "server", None), trainer]:
        if obj is None: continue
        for name in ["clients", "client_list", "client_pool"]:
            if hasattr(obj, name):
                c = getattr(obj, name)
                if isinstance(c, (list, tuple)) and len(c) > 0:
                    return list(c)
    raise AttributeError("Could not locate clients list in trainer/server.")

def pick_client_loader(client, which="train"):
    # try common names
    names = {
        "train": ["train_loader", "train_dataloader", "loader_train"],
        "test":  ["test_loader", "test_dataloader", "loader_test"],
        "val":   ["val_loader", "valid_loader", "val_dataloader", "valid_dataloader"],
    }[which]
    for n in names:
        if hasattr(client, n):
            return getattr(client, n)
    return None

def collect_member_nonmember_loaders(trainer, prefer_nonmember="test"):
    """
    Graph-FL (graph classification) membership source:
      - members: per-client train_dataloader
      - non-members: per-client test_dataloader (or val_dataloader)

    Returns:
      train_loaders: list of DataLoaders
      non_loaders:   list of DataLoaders
    """
    non_name = "test_dataloader" if prefer_nonmember == "test" else "val_dataloader"

    def _looks_like_client(x):
        return hasattr(x, "task") and x.task is not None

    def _extract_loaders_from_client(c):
        task = getattr(c, "task", None)
        if task is None:
            return (None, None)

        tr = getattr(task, "train_dataloader", None)
        non = getattr(task, non_name, None)

        # some OpenFGL tasks stash loaders in processed_data
        processed = getattr(task, "processed_data", None)
        if isinstance(processed, dict):
            tr = tr or processed.get("train_dataloader", None)
            non = non or processed.get(non_name, None)

        return (tr, non)

    # 1) try standard locations
    candidates = []
    for name in ["clients", "client_list", "client_pool", "client_objs", "client_objects"]:
        obj = getattr(trainer, name, None)
        if isinstance(obj, list) and obj and all(_looks_like_client(x) for x in obj):
            candidates = obj
            break

    if not candidates:
        srv = getattr(trainer, "server", None)
        if srv is not None:
            for name in ["clients", "client_list", "client_pool"]:
                obj = getattr(srv, name, None)
                if isinstance(obj, list) and obj and all(_looks_like_client(x) for x in obj):
                    candidates = obj
                    break

    # 2) fallback: recursive search for any list of client-like objects
    if not candidates:
        visited = set()

        def _walk(o, depth=6):
            if o is None or depth < 0:
                return None
            oid = id(o)
            if oid in visited:
                return None
            visited.add(oid)

            if isinstance(o, list) and o and all(_looks_like_client(x) for x in o):
                return o

            if isinstance(o, dict):
                for v in o.values():
                    r = _walk(v, depth - 1)
                    if r is not None:
                        return r

            if isinstance(o, (tuple, set)):
                for v in o:
                    r = _walk(v, depth - 1)
                    if r is not None:
                        return r

            if hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    r = _walk(v, depth - 1)
                    if r is not None:
                        return r

            return None

        candidates = _walk(trainer, depth=8) or []

    # 3) collect loaders
    train_loaders, non_loaders = [], []
    for c in candidates:
        tr, non = _extract_loaders_from_client(c)
        if tr is not None:
            train_loaders.append(tr)
        if non is not None:
            non_loaders.append(non)

    if not train_loaders:
        raise AttributeError(
            "No client train loaders found. "
            "This means the trainer does not keep client objects after training, "
            "or the task does not expose train_dataloader for this scenario/task."
        )
    if not non_loaders:
        raise AttributeError(
            f"No client non-member loaders found ({non_name}). "
            "Check whether task provides test_dataloader/val_dataloader."
        )

    return train_loaders, non_loaders


# (6) Compute per-graph features from loaders
def features_from_loaders(model, loaders, device, max_points, seed=0):
    rng = np.random.default_rng(seed)
    losses, confs, ents = [], [], []

    # Stream all samples; downsample at end for balance
    for loader in loaders:
        for batch in loader:
            batch = batch.to(device)
            logits = forward_logits_graph_batch(model, batch)  # [B,C]
            y = batch.y.view(-1).long()

            l = F.cross_entropy(logits, y, reduction="none")  # [B]
            p = F.softmax(logits, dim=-1)
            c = p.gather(1, y.view(-1, 1)).squeeze(1)
            e = -(p * p.clamp_min(1e-12).log()).sum(dim=1)

            losses.append(l.detach().cpu().numpy())
            confs.append(c.detach().cpu().numpy())
            ents.append(e.detach().cpu().numpy())

    if len(losses) == 0:
        return None

    losses = np.concatenate(losses, axis=0)
    confs  = np.concatenate(confs,  axis=0)
    ents   = np.concatenate(ents,   axis=0)

    n = min(len(losses), int(max_points))
    if n <= 0:
        return None

    idx = rng.choice(np.arange(len(losses)), size=n, replace=False) if len(losses) > n else np.arange(len(losses))
    return losses[idx], confs[idx], ents[idx]



# (7) MIA computations (same attackers as Subgraph-FL)
def mia_from_features(mem_feat, non_feat, seed=0):
    mem_loss, mem_conf, mem_ent = mem_feat
    non_loss, non_conf, non_ent = non_feat

    n = min(len(mem_loss), len(non_loss))
    mem_loss, mem_conf, mem_ent = mem_loss[:n], mem_conf[:n], mem_ent[:n]
    non_loss, non_conf, non_ent = non_loss[:n], non_conf[:n], non_ent[:n]

    X = np.concatenate([
        np.stack([mem_loss, mem_conf, mem_ent], axis=1),
        np.stack([non_loss, non_conf, non_ent], axis=1)
    ], axis=0)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)], axis=0)

    # loss-threshold AUC (score = -loss)
    scores = np.concatenate([-mem_loss, -non_loss], axis=0)
    auc_loss = roc_auc_score(y, scores)

    thr = np.median(np.concatenate([mem_loss, non_loss]))
    pred_thr = (np.concatenate([mem_loss, non_loss]) < thr).astype(int)
    acc_thr = accuracy_score(y, pred_thr)

    # logreg attacker on [loss, conf, ent]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    auc_lr = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

    return {
        "auc_loss_threshold": float(auc_loss),
        "acc_threshold_median_loss": float(acc_thr),
        "auc_logreg_features": float(auc_lr),
        "n_members": int(n),
        "n_nonmembers": int(n),
    }


# (8) Run training + audit (Graph-FL)
rows = []

for method in METHODS:
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"\n=== {method.upper()} | {dataset} | seed={seed} ===")

            args = deepcopy(config.args)

            # Graph-FL scenario/task
            args.root = DATA_ROOT
            args.scenario = "graph_fl"
            args.task = "graph_cls_2"
            args.dataset = [dataset]
            args.processing = "raw"

            # Simulation
            args.simulation_mode = "graph_fl_label_skew"
            args.num_clients = int(NUM_CLIENTS)
            args.client_frac = 1.0
            args.dirichlet_alpha = 1.0
            args.skew_alpha = 1.0

            # Model/training
            args.model = ["gin"]
            args.num_rounds = int(NUM_ROUNDS)
            args.num_epochs = int(LOCAL_EPOCHS)
            args.lr = float(LR)
            args.weight_decay = float(WEIGHT_DECAY)
            args.metrics = ["accuracy"]
            args.seed = int(seed)

            # optional stability
            args.num_workers = 0
            args.persistent_workers = False

            t0 = time.time()
            trainer = FGLTrainer(args)
            trainer.train()
            print(f"✓ Training finished in {time.time()-t0:.2f}s")

            model = pick_global_model(trainer).to(DEVICE)

            # Loaders define membership in Graph-FL
            train_loaders, non_loaders = collect_member_nonmember_loaders(trainer, prefer_nonmember=PREFER_NONMEMBER)

            mem_feat = features_from_loaders(model, train_loaders, DEVICE, MAX_POINTS_PER_CLASS, seed=seed)
            non_feat = features_from_loaders(model, non_loaders, DEVICE, MAX_POINTS_PER_CLASS, seed=seed)

            if mem_feat is None or non_feat is None:
                print("Skipped: could not extract member/non-member features.")
                continue

            audit = mia_from_features(mem_feat, non_feat, seed=seed)

            row = {"method": method, "dataset": dataset, "seed": seed, **audit}
            rows.append(row)

            print(f"  AUC(loss-threshold): {row['auc_loss_threshold']:.3f} | "
                  f"AUC(logreg): {row['auc_logreg_features']:.3f} | "
                  f"n={row['n_members']}+{row['n_nonmembers']}")

df = pd.DataFrame(rows)

print("\n" + "="*110)
print("GRAPH-FL MIA RESULTS (higher AUC = worse privacy)")
print("="*110)

if len(df) > 0:
    mean_df = df.groupby(["dataset", "method"])[["auc_loss_threshold", "auc_logreg_features"]].mean().reset_index()
    print("\nMean AUC per dataset/method:")

    df.to_csv("mia_graphfl_results.csv", index=False)
    mean_df.to_csv("mia_graphfl_results_mean.csv", index=False)
    print("\nSaved: mia_graphfl_results.csv and mia_graphfl_results_mean.csv")
