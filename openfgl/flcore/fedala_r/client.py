# openfgl/flcore/fedala_r/client_ihsan.py

import copy
import torch
import torch.nn as nn

from openfgl.flcore.base import BaseClient
from openfgl.utils.ala_utils import (AdaptiveLocalAggregationGraphFL)


def _is_pyg_graph_sample(sample) -> bool:
    return sample is not None and hasattr(sample, "edge_index") and hasattr(sample, "y")


def _get_train_dataset_for_ala(task):
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
    if hasattr(task, "default_loss_fn") and callable(task.default_loss_fn):
        crit = task.default_loss_fn()
        if isinstance(crit, nn.Module):
            return lambda logits, labels: crit(logits, labels)

    crit = getattr(task, "criterion", None)
    if isinstance(crit, nn.Module):
        return lambda logits, labels: crit(logits, labels)

    crit = nn.CrossEntropyLoss()
    return lambda logits, labels: crit(logits, labels)


class FedALARClient(BaseClient):
    """
    FedALA-R client (Graph-FL + Subgraph-FL compatible):

      1) FedALA init via AdaptiveLocalAggregationGraphFL (learned per-parameter weights)
      2) Residual injection from server:
            Θ_local <- Θ_local + beta * R
      3) Standard local training
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)

        train_data = _get_train_dataset_for_ala(self.task)
        loss_fn = _get_ala_loss_fn(self.task)

        # FedALA hyperparameters
        ala_batch_size = int(getattr(args, "ala_batch_size", getattr(args, "batch_size", 32)))
        ala_rand_percent = float(getattr(args, "ala_rand_percent", 30.0))
        ala_layer_idx = int(getattr(args, "ala_layer_idx", 0))
        ala_eta = float(getattr(args, "ala_eta", 1.0))
        ala_std_threshold = float(getattr(args, "ala_std_threshold", 0.1))
        ala_num_pre_loss = int(getattr(args, "ala_num_pre_loss", 10))
        ala_max_warmup_passes = int(getattr(args, "ala_max_warmup_passes", 5))

        self.ala = None
        if train_data is not None:
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

        # residual scale (beta)
        self.res_scale = float(getattr(args, "r_res_scale", 1.0))

        # OPTIONAL EMA / schedule hooks (kept commented by request)
        # self.res_scale_init = float(getattr(args, "r_res_scale_init", 1.0))
        # self.res_scale_final = float(getattr(args, "r_res_scale_final", 1.0))
        # self.res_scale_decay_rounds = int(getattr(args, "r_res_scale_decay_rounds", 0))

    def execute(self):
        # rebuild a global model instance from server weights
        global_model = copy.deepcopy(self.task.model).to(self.device)
        with torch.no_grad():
            for p, g in zip(global_model.parameters(), self.message_pool["server"]["weight"]):
                g_t = g.data if hasattr(g, "data") else g
                p.data.copy_(g_t.to(p.device))

        # FedALA init (learned weights) or fallback
        if self.ala is not None:
            self.ala.adaptive_local_aggregation(global_model, self.task.model)
        else:
            with torch.no_grad():
                for p_local, p_global in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                    g_t = p_global.data if hasattr(p_global, "data") else p_global
                    p_local.data.copy_(g_t.to(p_local.device))

        # residual add (delta residual from server_ihsan)
        residual = self.message_pool["server"].get("residual", None)
        if residual is not None:
            beta = self.res_scale
            with torch.no_grad():
                for p, r in zip(self.task.model.parameters(), residual):
                    r_t = r.data if hasattr(r, "data") else r
                    p.data.add_(r_t.to(p.data.device), alpha=beta)

        # local training
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
