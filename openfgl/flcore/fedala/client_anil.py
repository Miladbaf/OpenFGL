# openfgl/flcore/fedala/client.py

import copy
import torch
import torch.nn as nn

from openfgl.flcore.base import BaseClient
from openfgl.utils.ala_utils import AdaptiveLocalAggregation


class FedALAClient(BaseClient):
    """
    FedALA client implementation for OpenFGL.

    It differs from FedAvgClient in how it initializes the local model each
    round: instead of hard-copying the global model, it uses Adaptive Local
    Aggregation (ALA) to interpolate between the previous local model and the
    new global model in a parameter-wise way.
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)

        # ---- pick loss function from the task if possible ----
        if hasattr(self.task, "loss_fn"):
            loss_fn = self.task.loss_fn
        elif hasattr(self.task, "criterion"):
            loss_fn = self.task.criterion
        else:
            # Fall back to a generic classification loss
            loss_fn = nn.CrossEntropyLoss()

        # ---- find the local training data container for this client ----
        train_data = None
        # Many implementations store a "train_dataset" or similar
        for attr in ["train_dataset", "train_data", "train_set"]:
            if hasattr(self.task, attr):
                train_data = getattr(self.task, attr)
                break

        # Some OpenFGL tasks keep a "splitted_data" dict; you may need to adapt this.
        if train_data is None and hasattr(self.task, "splitted_data"):
            sd = getattr(self.task, "splitted_data")
            if isinstance(sd, dict) and "train" in sd:
                train_data = sd["train"]

        # FedALA hyperparameters; define them in your config / args
        ala_batch_size = getattr(args, "ala_batch_size", getattr(args, "batch_size", 32))
        ala_rand_percent = getattr(args, "ala_rand_percent", 30.0)
        ala_layer_idx = getattr(args, "ala_layer_idx", 0)
        ala_eta = getattr(args, "ala_eta", 1.0)
        ala_std_threshold = getattr(args, "ala_std_threshold", 0.1)
        ala_num_pre_loss = getattr(args, "ala_num_pre_loss", 10)

        if train_data is None:
            # ALA needs access to local training samples;
            # if we can't find them, we silently fall back to FedAvg-style init.
            self.ala = None
        else:
            self.ala = AdaptiveLocalAggregation(
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
            )

    def execute(self):
        """
        One local round:

        1. Build a temporary global model using weights from the server.
        2. Use ALA to adaptively initialize self.task.model.
        3. Run the task's standard local training routine.
        """
        # Rebuild a "global model" object from the parameter list in the message pool
        global_model = copy.deepcopy(self.task.model).to(self.device)
        with torch.no_grad():
            for p, g in zip(global_model.parameters(),
                           self.message_pool["server"]["weight"]):
                p.data.copy_(g.data if isinstance(g, torch.Tensor) else g)

        # Either do FedALA-style initialization or fall back to FedAvg
        if self.ala is not None:
            self.ala.adaptive_local_aggregation(global_model, self.task.model)
        else:
            # FedAvg fallback: just copy global weights
            with torch.no_grad():
                for p_local, p_global in zip(
                    self.task.model.parameters(),
                    self.message_pool["server"]["weight"],
                ):
                    p_local.data.copy_(p_global.data if isinstance(p_global, torch.Tensor) else p_global)

        # Standard OpenFGL local training (same as FedAvgClient)
        self.task.train()

    def send_message(self):
        """
        Send updated model parameters and num_samples back to the server.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
