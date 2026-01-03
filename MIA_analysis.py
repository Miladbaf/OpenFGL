# ==============================================================================
# SIMPLE PRIVACY AUDIT (BLACK-BOX MEMBERSHIP INFERENCE) — FULL NOTEBOOK CELL
# OpenFGL (config.args + FGLTrainer) — FedAvg / FedALA / FedALA-R
#
# Fixes included:
#  - SAFE torch.load patch (no recursion in Jupyter)
#  - Patch FedALA and FedALA-R loss_fn bug (CrossEntropyLoss instance)
#  - Robustly find model + global PyG Data even if nested
#  - Robustly get membership split even if no train_mask/test_mask (idx or split dict)
#  - Robust logits extraction even if model returns tuple/dict/object
#
# What it reports:
#  - Loss-threshold MIA AUC (score=-loss)
#  - Logistic regression attacker AUC on [loss, confidence(true label), entropy]
#
# Outputs:
#  - DataFrame printed + saved: mia_results.csv and mia_results_mean.csv
# ==============================================================================

import os, time, warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# (0) Clean warnings (optional)
# -----------------------------
warnings.filterwarnings(
    "ignore",
    message="It is not recommended to directly access the internal storage format `data` of an 'InMemoryDataset'.*"
)

# -----------------------------
# (1) SAFE torch.load patch (idempotent; avoids recursion in notebooks)
# -----------------------------
torch.load = torch.serialization.load
if not hasattr(torch, "_openfgl_original_torch_load"):
    torch._openfgl_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return torch._openfgl_original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
print("✓ torch.load patched safely (idempotent)")

# -----------------------------
# (2) Patch FedALA + FedALA-R bug: task.default_loss_fn can be CrossEntropyLoss() instance
# -----------------------------
def patch_get_ala_loss_fn(module, fn_name="_get_ala_loss_fn"):
    orig_name = f"_orig{fn_name}"
    if not hasattr(module, orig_name):
        setattr(module, orig_name, getattr(module, fn_name))
    orig = getattr(module, orig_name)

    def fixed(task):
        fn = getattr(task, "default_loss_fn", None)

        if isinstance(fn, nn.Module):  # criterion instance
            return lambda logits, labels: fn(logits, labels)

        if isinstance(fn, type) and issubclass(fn, nn.Module):  # criterion class
            crit = fn()
            return lambda logits, labels: crit(logits, labels)

        if callable(fn):  # factory function returning criterion/callable
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
    print("✓ Patched _get_ala_loss_fn for FedALA and FedALA-R")
except Exception as e:
    print("⚠️ Could not patch FedALA modules (repo may differ). Error:", e)

# -----------------------------
# (3) OpenFGL imports
# -----------------------------
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# ==============================================================================
# USER SETTINGS (edit here)
# ==============================================================================
METHODS  = ["fedavg", "fedala", "fedala_r"]
DATASETS = ["Cora", "CiteSeer", "PubMed"]
SEEDS    = [42, 123, 456]          # simple run; add more later

NUM_CLIENTS  = 5
NUM_ROUNDS   = 30        # quick audit (set 100 if you want)
LOCAL_EPOCH  = 5
LR           = 0.01
WEIGHT_DECAY = 5e-4

MAX_POINTS_PER_CLASS = 5000   # members and non-members each (balanced)
PREFER_NONMEMBER = "test"     # "test" or "val"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("✓ Device:", DEVICE)

REPO_ROOT = os.getcwd()
DATA_ROOT = os.path.join(REPO_ROOT, "data")

# ==============================================================================
# (4) Robust utilities: find model and global_data even if nested
# ==============================================================================
def find_first_module(obj, max_depth=10):
    visited = set()

    def _walk(x, depth):
        if x is None:
            return None
        xid = id(x)
        if xid in visited:
            return None
        visited.add(xid)

        if isinstance(x, nn.Module):
            return x
        if depth <= 0:
            return None

        if isinstance(x, dict):
            for v in x.values():
                m = _walk(v, depth-1)
                if m is not None:
                    return m

        if isinstance(x, (list, tuple, set)):
            for v in x:
                m = _walk(v, depth-1)
                if m is not None:
                    return m

        if hasattr(x, "__dict__"):
            for v in x.__dict__.values():
                m = _walk(v, depth-1)
                if m is not None:
                    return m

        return None

    return _walk(obj, max_depth)

def pick_model_from_trainer_anywhere(trainer):
    for obj in [trainer, getattr(trainer, "server", None)]:
        if obj is None:
            continue
        for name in ["model", "global_model", "net", "gnn", "backbone"]:
            if hasattr(obj, name) and isinstance(getattr(obj, name), nn.Module):
                return getattr(obj, name)

    m = find_first_module(getattr(trainer, "server", trainer), max_depth=10)
    if m is None:
        m = find_first_module(trainer, max_depth=10)
    if m is None:
        raise AttributeError("Could not find any torch.nn.Module inside trainer/server.")
    return m

def pick_global_data(trainer):
    candidates = []
    if hasattr(trainer, "server"):
        candidates.append(trainer.server)
    candidates.append(trainer)

    for obj in candidates:
        for name in ["global_data", "data"]:
            if hasattr(obj, name):
                cand = getattr(obj, name)
                if hasattr(cand, "x") and hasattr(cand, "edge_index") and hasattr(cand, "y"):
                    return cand

    visited = set()
    def _walk(x, depth):
        if x is None: return None
        xid = id(x)
        if xid in visited: return None
        visited.add(xid)

        if hasattr(x, "x") and hasattr(x, "edge_index") and hasattr(x, "y"):
            return x
        if depth <= 0: return None

        if isinstance(x, dict):
            for v in x.values():
                r = _walk(v, depth-1)
                if r is not None: return r
        if isinstance(x, (list, tuple, set)):
            for v in x:
                r = _walk(v, depth-1)
                if r is not None: return r
        if hasattr(x, "__dict__"):
            for v in x.__dict__.values():
                r = _walk(v, depth-1)
                if r is not None: return r
        return None

    r = _walk(trainer, 10)
    if r is None:
        r = _walk(getattr(trainer, "server", None), 10)

    if r is None:
        raise AttributeError("Could not locate global PyG Data (x, edge_index, y) in trainer/server.")
    return r

# ==============================================================================
# (5) Robust logits extraction (fixes your latest error)
# ==============================================================================
def coerce_logits(out):
    """
    Convert forward output to a [N, C] torch.Tensor.
    Handles Tensor / tuple/list / dict / object.logits
    """
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
                for k in ["logits", "out", "pred", "prediction", "y_hat"]:
                    if k in item and torch.is_tensor(item[k]):
                        return item[k]
        raise TypeError(f"Forward returned tuple/list without tensor logits. Types: {[type(x) for x in out]}")

    if isinstance(out, dict):
        for k in ["logits", "out", "pred", "prediction", "y_hat"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise TypeError(f"Forward returned dict but no tensor logits found. Keys: {list(out.keys())}")

    if hasattr(out, "logits") and torch.is_tensor(out.logits):
        return out.logits

    raise TypeError(f"Unsupported forward output type: {type(out)}")

def forward_logits(model, data):
    model.eval()
    with torch.no_grad():
        # model(data)
        try:
            out = model(data)
            return coerce_logits(out)
        except Exception:
            pass
        # model(x, edge_index)
        try:
            out = model(data.x, data.edge_index)
            return coerce_logits(out)
        except Exception:
            pass
        # model(x, edge_index, edge_attr)
        try:
            out = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
            return coerce_logits(out)
        except Exception as e:
            raise RuntimeError(f"Could not forward model with common signatures. Last error: {e}")

# ==============================================================================
# (6) Split extraction: masks OR indices OR split dict (OGB style)
# ==============================================================================
def _to_1d_index(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        x = torch.tensor(x)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.long().view(-1)

def _make_mask(n, idx):
    m = torch.zeros(n, dtype=torch.bool)
    m[idx] = True
    return m

def get_membership_masks(data, prefer_nonmember="test"):
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        n = int(data.num_nodes)
    elif hasattr(data, "y"):
        n = int(data.y.shape[0])
    elif hasattr(data, "x"):
        n = int(data.x.shape[0])
    else:
        raise AttributeError("Cannot determine num nodes from data (no num_nodes/y/x).")

    train_mask = None
    non_mask = None

    # 1) direct masks
    if hasattr(data, "train_mask"):
        tm = data.train_mask
        if torch.is_tensor(tm) and tm.dim() == 2:
            tm = tm[:, 0]
        train_mask = tm.bool().view(-1)

    non_name = "test_mask" if prefer_nonmember == "test" else "val_mask"
    if hasattr(data, non_name):
        nm = getattr(data, non_name)
        if torch.is_tensor(nm) and nm.dim() == 2:
            nm = nm[:, 0]
        non_mask = nm.bool().view(-1)

    # 2) index style
    if train_mask is None:
        for name in ["train_idx", "train_index", "train_indices", "idx_train", "train_nodes"]:
            if hasattr(data, name):
                train_mask = _make_mask(n, _to_1d_index(getattr(data, name)))
                break

    if non_mask is None:
        names = (["test_idx","test_index","test_indices","idx_test","test_nodes"]
                 if prefer_nonmember == "test"
                 else ["val_idx","val_index","val_indices","idx_val","val_nodes"])
        for name in names:
            if hasattr(data, name):
                non_mask = _make_mask(n, _to_1d_index(getattr(data, name)))
                break

    # 3) split dict style
    if (train_mask is None) or (non_mask is None):
        for name in ["split_idx", "splits", "node_split", "split"]:
            if hasattr(data, name):
                sd = getattr(data, name)
                if isinstance(sd, dict):
                    if train_mask is None and "train" in sd:
                        train_mask = _make_mask(n, _to_1d_index(sd["train"]))
                    key = "test" if prefer_nonmember == "test" else "valid"
                    if non_mask is None and key in sd:
                        non_mask = _make_mask(n, _to_1d_index(sd[key]))
                break

    info = []
    if train_mask is None:
        idx = torch.randperm(n)
        half = n // 2
        train_mask = _make_mask(n, idx[:half])
        info.append("⚠️ No split found: using RANDOM train (audit not meaningful).")
    else:
        info.append("✓ Found train split.")

    if non_mask is None:
        non_mask = ~train_mask
        info.append("⚠️ No test/val found: using (NOT train) as non-members.")
    else:
        info.append("✓ Found non-member split.")

    overlap = (train_mask & non_mask).sum().item()
    if overlap > 0:
        non_mask = non_mask & (~train_mask)
        info.append(f"⚠️ Overlap {overlap} removed from non-members.")

    info.append(f"train={int(train_mask.sum())}, nonmember={int(non_mask.sum())}, N={n}")
    return train_mask, non_mask, " | ".join(info)

# ==============================================================================
# (7) MIA computations
# ==============================================================================
def membership_audit_from_logits(logits, y, train_mask, non_mask, max_points=5000, seed=0):
    """
    logits: torch.Tensor [N, C]
    y: torch.Tensor [N]
    """
    if not torch.is_tensor(logits):
        raise TypeError(f"logits must be torch.Tensor, got {type(logits)}")

    rng = np.random.default_rng(seed)

    mem_idx = torch.where(train_mask)[0].cpu().numpy()
    non_idx = torch.where(non_mask)[0].cpu().numpy()
    if len(mem_idx) == 0 or len(non_idx) == 0:
        return None

    n = min(len(mem_idx), len(non_idx), max_points)
    mem_sel = rng.choice(mem_idx, size=n, replace=False)
    non_sel = rng.choice(non_idx, size=n, replace=False)

    idx = np.concatenate([mem_sel, non_sel])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    perm = rng.permutation(len(idx))
    idx = idx[perm]
    labels = labels[perm]

    # IMPORTANT: use torch indexing with torch.LongTensor
    idx_t = torch.as_tensor(idx, device=logits.device, dtype=torch.long)

    L = logits.index_select(0, idx_t)   # [2n, C]
    Y = y.index_select(0, idx_t)        # [2n]

    loss = F.cross_entropy(L, Y, reduction="none").detach().cpu().numpy()
    probs = F.softmax(L, dim=-1)

    conf = probs.gather(1, Y.view(-1, 1)).squeeze(1).detach().cpu().numpy()
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).detach().cpu().numpy()

    auc_loss = roc_auc_score(labels, -loss)

    thr = np.median(loss)
    pred_thr = (loss < thr).astype(int)
    acc_thr = accuracy_score(labels, pred_thr)

    X = np.stack([loss, conf, ent], axis=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=seed, stratify=labels)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    score = clf.predict_proba(X_te)[:, 1]
    auc_lr = roc_auc_score(y_te, score)

    return {
        "auc_loss_threshold": float(auc_loss),
        "acc_threshold_median_loss": float(acc_thr),
        "auc_logreg_features": float(auc_lr),
        "n_members": int(n),
        "n_nonmembers": int(n),
    }

# ==============================================================================
# (8) Run training + audit
# ==============================================================================
rows = []

for method in METHODS:
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"\n=== {method.upper()} | {dataset} | seed={seed} ===")

            args = config.args
            args.root = DATA_ROOT
            args.dataset = [dataset]
            args.simulation_mode = "subgraph_fl_louvain"
            args.num_clients = NUM_CLIENTS
            args.fl_algorithm = method
            args.model = ["gcn"]

            args.num_rounds = NUM_ROUNDS
            args.local_epoch = LOCAL_EPOCH
            args.lr = LR
            args.weight_decay = WEIGHT_DECAY
            args.metrics = ["accuracy"]
            args.seed = int(seed)

            t0 = time.time()
            trainer = FGLTrainer(args)
            trainer.train()
            t1 = time.time()
            print(f"✓ Training finished in {t1-t0:.2f}s")

            model = pick_model_from_trainer_anywhere(trainer).to(DEVICE)
            data = pick_global_data(trainer).to(DEVICE)

            train_mask, non_mask, split_info = get_membership_masks(data, prefer_nonmember=PREFER_NONMEMBER)
            print("Split info:", split_info)

            logits = forward_logits(model, data)
            print("logits:", type(logits), tuple(logits.shape))

            audit = membership_audit_from_logits(
                logits=logits,
                y=data.y,
                train_mask=train_mask.to(logits.device),
                non_mask=non_mask.to(logits.device),
                max_points=MAX_POINTS_PER_CLASS,
                seed=seed
            )

            if audit is None:
                print("⚠️ Skipped: no member/non-member nodes found.")
                continue

            row = {"method": method, "dataset": dataset, "seed": seed, **audit}
            rows.append(row)

            print(f"  AUC(loss-threshold): {row['auc_loss_threshold']:.3f} | "
                  f"AUC(logreg): {row['auc_logreg_features']:.3f} | "
                  f"n={row['n_members']}+{row['n_nonmembers']}")

df = pd.DataFrame(rows)

print("\n" + "="*110)
print("MIA RESULTS (higher AUC = worse privacy)")
print("="*110)
display(df)

if len(df) > 0:
    mean_df = df.groupby(["dataset", "method"])[["auc_loss_threshold", "auc_logreg_features"]].mean().reset_index()
    print("\nMean AUC per dataset/method:")
    display(mean_df)

    df.to_csv("mia_results.csv", index=False)
    mean_df.to_csv("mia_results_mean.csv", index=False)
    print("\n✅ Saved: mia_results.csv and mia_results_mean.csv")
