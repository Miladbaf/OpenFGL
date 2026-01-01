import copy
import torch
import torch.nn as nn

from openfgl.flcore.base import BaseClient
from openfgl.utils.ala_utils import (
    AdaptiveLocalAggregation,
    AdaptiveLocalAggregationGraphFL,
)


def _is_pyg_graph_sample(sample) -> bool:
    """
    Heuristic: PyG graph classification sample usually has edge_index and y.
    Works for Graph-FL and Subgraph-FL graph_cls / graph_cls_2.
    """
    if sample is None:
        return False
    return hasattr(sample, "edge_index") and hasattr(sample, "y")


def _get_train_dataset_for_ala(task):
    """
    Return an indexable training dataset ONLY when it looks like a PyG graph dataset.
    Otherwise return None so ALA is disabled (safe default for node_cls etc.).
    """
    dl = getattr(task, "train_dataloader", None)
    ds = getattr(dl, "dataset", None) if dl is not None else None

    if ds is None:
        processed = getattr(task, "processed_data", None)
        if isinstance(processed, dict):
            dl = processed.get("train_dataloader", None)
            ds = getattr(dl, "dataset", None) if dl is not None else None

    if ds is None:
        return None

    try:
        sample0 = ds[0]
    except Exception:
        return None

    return ds if _is_pyg_graph_sample(sample0) else None


def _get_ala_loss_fn(task):
    """
    ALA expects: loss_fn(logits, labels) -> scalar
    Returns a callable loss function.
    """
    # Try getting the loss function from task
    loss_fn_attr = getattr(task, "default_loss_fn", None)
    
    # Case 1: It's already a loss function instance (nn.Module)
    if isinstance(loss_fn_attr, nn.Module):
        return lambda logits, labels: loss_fn_attr(logits, labels)
    
    # Case 2: It's a callable that returns a loss function
    if callable(loss_fn_attr):
        try:
            crit = loss_fn_attr()  # Call it to get the instance
            if isinstance(crit, nn.Module):
                return lambda logits, labels: crit(logits, labels)
        except:
            pass
    
    # Case 3: Try task.criterion
    crit = getattr(task, "criterion", None)
    if isinstance(crit, nn.Module):
        return lambda logits, labels: crit(logits, labels)
    
    # Case 4: Fallback to default CrossEntropyLoss
    crit = nn.CrossEntropyLoss()
    return lambda logits, labels: crit(logits, labels)


class FedALAClient(BaseClient):
    """
    FedALA (Adaptive Local Aggregation):
      - Build a global_model snapshot from server weights
      - Learn per-parameter interpolation weights on a local subset (ALA)
      - Initialize local model using those learned weights
      - Train locally
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)

        train_data = _get_train_dataset_for_ala(self.task)
        loss_fn = _get_ala_loss_fn(self.task)

        ala_batch_size = int(getattr(args, "ala_batch_size", getattr(args, "batch_size", 32)))
        ala_rand_percent = float(getattr(args, "ala_rand_percent", 30.0))
        ala_layer_idx = int(getattr(args, "ala_layer_idx", 0))
        ala_eta = float(getattr(args, "ala_eta", 1.0))
        ala_std_threshold = float(getattr(args, "ala_std_threshold", 0.1))
        ala_num_pre_loss = int(getattr(args, "ala_num_pre_loss", 10))
        ala_max_warmup_passes = int(getattr(args, "ala_max_warmup_passes", 5))

        if train_data is None:
            self.ala = None
        else:
            # IMPORTANT: graph datasets must use the PyG-aware ALA
            self.ala = AdaptiveLocalAggregationGraphFL(
                cid=client_id,
                loss_fn=loss_fn,
                train_data=train_data,
                batch_size=ala_batch_size,
                rand_percent=ala_rand_percent,
                layer_idx=ala_layer_idx,
                eta=ala_eta,
                device=device,
                std_threshold=ala_std_threshold,
                num_pre_loss=ala_num_pre_loss,
                max_warmup_passes=ala_max_warmup_passes,
            )

    def execute(self):
        # Build a global model snapshot from message_pool
        global_model = copy.deepcopy(self.task.model).to(self.device)
        with torch.no_grad():
            for p, g in zip(global_model.parameters(), self.message_pool["server"]["weight"]):
                g_t = g.data if isinstance(g, torch.nn.Parameter) else g
                p.data.copy_(g_t.to(p.device))

        # FedALA init (or FedAvg fallback)
        if self.ala is not None:
            self.ala.adaptive_local_aggregation(global_model, self.task.model)
        else:
            with torch.no_grad():
                for p_local, p_global in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                    g_t = p_global.data if isinstance(p_global, torch.nn.Parameter) else p_global
                    p_local.data.copy_(g_t.to(p_local.device))

        # Standard local training
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
