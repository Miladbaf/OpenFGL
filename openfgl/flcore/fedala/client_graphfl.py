# openfgl/flcore/fedala/client_graphfl.py

import copy
import torch
import torch.nn as nn

from openfgl.flcore.base import BaseClient
from openfgl.utils.ala_utils import AdaptiveLocalAggregationGraphFL


class FedALAGraphFLClient(BaseClient):
    """
    FedALA client specialized for Graph-FL graph classification (PyG).

    Uses AdaptiveLocalAggregationGraphFL to compute parameter-wise weights w in [0,1]
    on a random subset of the client's local train graphs each round.
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)

        # GraphClsTask default is CrossEntropyLoss; use CE on logits directly.
        loss_fn = nn.CrossEntropyLoss()

        # Build indexable local train graph list (GraphClsTask / graph_cls_2 provides train_mask + data)
        train_graphs = None
        try:
            if hasattr(self.task, "train_mask") and hasattr(self.task, "data"):
                train_graphs = [g for g in self.task.data[self.task.train_mask]]
        except Exception:
            train_graphs = None

        ala_batch_size = getattr(args, "ala_batch_size", getattr(args, "batch_size", 32))
        ala_rand_percent = getattr(args, "ala_rand_percent", 30.0)
        ala_layer_idx = getattr(args, "ala_layer_idx", 0)
        ala_eta = getattr(args, "ala_eta", 1.0)
        ala_std_threshold = getattr(args, "ala_std_threshold", 0.1)
        ala_num_pre_loss = getattr(args, "ala_num_pre_loss", 10)

        self.ala = None
        if train_graphs is not None and len(train_graphs) > 0:
            self.ala = AdaptiveLocalAggregationGraphFL(
                cid=client_id,
                loss_fn=loss_fn,
                train_data=train_graphs,
                batch_size=ala_batch_size,
                rand_percent=ala_rand_percent,
                layer_idx=ala_layer_idx,
                eta=ala_eta,
                device=device,
                std_threshold=ala_std_threshold,
                num_pre_loss=ala_num_pre_loss,
            )

    def execute(self):
        # Build a global model clone and load server weights
        global_model = copy.deepcopy(self.task.model).to(self.device)
        with torch.no_grad():
            for p, g in zip(global_model.parameters(), self.message_pool["server"]["weight"]):
                p.data.copy_(g)

        # FedALA init (or FedAvg fallback)
        if self.ala is not None:
            self.ala.adaptive_local_aggregation(global_model, self.task.model)
        else:
            with torch.no_grad():
                for p_local, p_global in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                    p_local.data.copy_(p_global)

        # Standard local training
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
