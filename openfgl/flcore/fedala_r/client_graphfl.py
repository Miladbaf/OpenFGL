# openfgl/flcore/fedala_r/client_graphfl.py

from __future__ import annotations

import copy
from typing import Optional, List, Sequence

import torch
import torch.nn as nn

from openfgl.flcore.base import BaseClient
from openfgl.utils.ala_utils import AdaptiveLocalAggregationGraphFL


class FedALARGraphFLClient(BaseClient):
    """
    Graph-FL FedALA-R client:

      1) Receive global weights (theta) and optional residual buffer (R) from server
      2) Construct effective global weights: theta_eff = theta + gamma * R (after residual_start_round)
      3) Run FedALA (ALA) to initialize local model toward theta_eff
      4) Standard local training
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)

        # ----- FedALA hyperparameters -----
        self.ala_batch_size = int(getattr(args, "ala_batch_size", getattr(args, "batch_size", 32)))
        self.ala_rand_percent = float(getattr(args, "ala_rand_percent", 30.0))
        self.ala_layer_idx = int(getattr(args, "ala_layer_idx", 0))
        self.ala_eta = float(getattr(args, "ala_eta", 1.0))
        self.ala_std_threshold = float(getattr(args, "ala_std_threshold", 0.1))
        self.ala_num_pre_loss = int(getattr(args, "ala_num_pre_loss", 10))

        # ----- Residual hyperparameters -----
        self.residual_gamma = float(getattr(args, "residual_gamma", 1.0))
        self.residual_start_round = int(getattr(args, "residual_start_round", 1))

        # Optional debugging control (safe default: off)
        self.debug_rounds = set(getattr(args, "debug_rounds", (0, 1, 5)))

        # Loss for graph classification
        self.loss_fn = nn.CrossEntropyLoss()

        # Build train graphs for ALA (robust across mask formats)
        train_graphs = self._get_train_graphs_safely()

        self.ala: Optional[AdaptiveLocalAggregationGraphFL] = None
        if train_graphs is not None and len(train_graphs) > 0:
            self.ala = AdaptiveLocalAggregationGraphFL(
                cid=client_id,
                loss_fn=self.loss_fn,
                train_data=train_graphs,
                batch_size=self.ala_batch_size,
                rand_percent=self.ala_rand_percent,
                layer_idx=self.ala_layer_idx,
                eta=self.ala_eta,
                device=self.device,
                std_threshold=self.ala_std_threshold,
                num_pre_loss=self.ala_num_pre_loss,
            )

    # -------------------------------------------------------------------------
    # Data helpers
    # -------------------------------------------------------------------------

    def _get_train_graphs_safely(self) -> Optional[List]:
        """
        Returns a list of train graphs (PyG Data objects) if available.
        Supports multiple task layouts:
          - task.splitted_data["train"]
          - task.data + task.train_mask (bool tensor / bool list / index list)
        """
        try:
            # Preferred: some tasks expose splitted_data
            if hasattr(self.task, "splitted_data") and isinstance(self.task.splitted_data, dict):
                tr = self.task.splitted_data.get("train", None)
                if tr is not None:
                    # Ensure it's list-like
                    return list(tr)

            if not (hasattr(self.task, "data") and hasattr(self.task, "train_mask")):
                return None

            data = self.task.data
            mask = self.task.train_mask

            # data should be indexable
            if data is None:
                return None

            idxs: List[int] = []

            # torch bool tensor mask
            if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                idxs = mask.nonzero(as_tuple=False).view(-1).tolist()

            # torch int tensor indices
            elif isinstance(mask, torch.Tensor) and mask.dtype in (torch.int32, torch.int64):
                idxs = mask.view(-1).tolist()

            # python list/tuple mask
            elif isinstance(mask, (list, tuple)):
                if len(mask) == 0:
                    return None
                # list of bools
                if all(isinstance(x, (bool, int)) for x in mask) and len(mask) == len(data):
                    idxs = [i for i, v in enumerate(mask) if bool(v)]
                # list of indices
                elif all(isinstance(x, int) for x in mask):
                    idxs = list(mask)
                else:
                    return None
            else:
                return None

            if not idxs:
                return None

            return [data[i] for i in idxs]

        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Model/weight helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _assert_param_count(model: nn.Module, weights: Sequence[torch.Tensor]) -> None:
        n_params = len(list(model.parameters()))
        if len(weights) != n_params:
            raise ValueError(f"Weight length mismatch: got {len(weights)} tensors, model has {n_params} params.")

    def _load_weights_into_model(self, model: nn.Module, weights: List[torch.Tensor]) -> None:
        self._assert_param_count(model, weights)
        with torch.no_grad():
            for p, w in zip(model.parameters(), weights):
                if not isinstance(w, torch.Tensor):
                    raise TypeError(f"Server weight is not a Tensor: {type(w)}")
                if p.data.shape != w.shape:
                    raise ValueError(f"Shape mismatch: param {tuple(p.data.shape)} vs weight {tuple(w.shape)}")
                p.data.copy_(w.to(p.device))

    def _apply_residual(
            self,
            weights: List[torch.Tensor],
            residual: Optional[List[torch.Tensor]],
            gamma: float,
    ) -> List[torch.Tensor]:
        if residual is None or (not isinstance(residual, list)) or len(residual) != len(weights):
            return weights

        # Apply residual ONLY to the same "top" groups as ALA
        num_groups = len(weights)
        top_k = self.ala_layer_idx if self.ala_layer_idx > 0 else num_groups
        top_k = min(max(1, top_k), num_groups)
        start = num_groups - top_k

        out = []
        with torch.no_grad():
            for i, (w, r) in enumerate(zip(weights, residual)):
                if (not isinstance(w, torch.Tensor)) or (not isinstance(r, torch.Tensor)) or (w.shape != r.shape):
                    out.append(w)
                    continue
                if i < start:
                    out.append(w)  # no residual on lower layers
                else:
                    out.append(w + gamma * r.to(w.device))
        return out


    # -------------------------------------------------------------------------
    # FL hooks
    # -------------------------------------------------------------------------

    def execute(self):
        round_id = int(self.message_pool.get("round", 0))

        server_pack = self.message_pool.get("server", {})
        global_w = server_pack.get("weight", None)
        residual = server_pack.get("residual", None)

        # Defensive fallback: if server weights missing, just train locally
        if global_w is None:
            self.task.train()
            return

        if not isinstance(global_w, list) or len(global_w) == 0:
            raise ValueError("server['weight'] is not a non-empty list of tensors.")

        # Debug (ONLY once)
        if self.client_id == 0 and round_id in self.debug_rounds:
            has_res = isinstance(residual, list) and len(residual) == len(global_w)
            msg = f"[debug] round_id={round_id} has_residual={has_res} gamma={self.residual_gamma} start={self.residual_start_round}"
            if has_res and len(residual) > 0 and isinstance(residual[0], torch.Tensor):
                msg += f" residual_norm0={float(residual[0].norm().item()):.6f}"
            print(msg, flush=True)

        # ---------------------------------------------------------------------
        # 1) Build effective global weights = global + gamma * residual (if enabled)
        # ---------------------------------------------------------------------
        eff_w = global_w

        if (
                isinstance(residual, list)
                and round_id >= self.residual_start_round
                and self.residual_gamma != 0.0
        ):
            ramp = min(1.0, (round_id - self.residual_start_round + 1) / 10.0)  # 10-round ramp
            gamma_eff = float(self.residual_gamma) * ramp
            eff_w = self._apply_residual(global_w, residual, gamma_eff)

        # ---------------------------------------------------------------------
        # 2) Build effective global model
        # ---------------------------------------------------------------------
        effective_global_model = copy.deepcopy(self.task.model).to(self.device)
        self._load_weights_into_model(effective_global_model, eff_w)

        # ---------------------------------------------------------------------
        # 3) FedALA init toward EFFECTIVE global (or FedAvg fallback)
        # ---------------------------------------------------------------------
        if self.ala is not None:
            self.ala.adaptive_local_aggregation(effective_global_model, self.task.model)
        else:
            with torch.no_grad():
                for p_local, p_eff in zip(self.task.model.parameters(), effective_global_model.parameters()):
                    p_local.data.copy_(p_eff.data)

        # ---------------------------------------------------------------------
        # 4) Standard local training
        # ---------------------------------------------------------------------
        self.task.train()

    def send_message(self):
        # IMPORTANT: always send detached clones, never live Parameter references
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": int(self.task.num_samples),
            "weight": [p.detach().clone() for p in self.task.model.parameters()],
        }
