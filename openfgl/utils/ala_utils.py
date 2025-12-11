# openfgl/utils/ala_utils.py

from __future__ import annotations
from typing import Any, Sequence, Optional
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class AdaptiveLocalAggregation:
    """
    Implements the weight-learning step of FedALA on top of a local model.

    It learns, for each selected layer, an element-wise weight in [0, 1] that
    interpolates between the previous local model and the new global model.

    For a parameter tensor θ:
        θ_init = θ_local + w ⊙ (θ_global - θ_local)
    where w is learned to minimize a local loss on a small subset of data.
    """

    def __init__(
        self,
        cid: int,
        loss_fn: nn.Module,
        train_data: Any,
        batch_size: int,
        rand_percent: float = 30.0,
        layer_idx: int = 0,
        eta: float = 1.0,
        device: torch.device | str = "cpu",
        std_threshold: float = 0.1,
        num_pre_loss: int = 10,
    ) -> None:
        """
        Args:
            cid: client id (for logging/debug).
            loss_fn: local objective (e.g. CrossEntropyLoss).
            train_data: indexable local training dataset (e.g. torch Dataset
                        or a list of (x, y)).
            batch_size: mini-batch size for the weight-learning loop.
            rand_percent: percentage of local data to sample for ALA.
            layer_idx: number of *top* parameter groups to adapt. If 0,
                       all layers are treated as "top" layers.
            eta: learning rate for the aggregation weights.
            device: cuda / cpu.
            std_threshold: stop weight-learning when std of recent losses
                           drops below this.
            num_pre_loss: how many recent losses to use for the std check.
        """
        self.cid = cid
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.device = torch.device(device)
        self.std_threshold = std_threshold
        self.num_pre_loss = num_pre_loss

        # Learned element-wise weights for the selected layers
        self._weights: Optional[list[torch.Tensor]] = None
        # During the first global rounds we allow more iterations;
        # afterwards we only do one pass over the sampled data.
        self._warmup_phase: bool = True

    def _build_random_loader(self) -> Optional[DataLoader]:
        if self.train_data is None:
            return None

        try:
            n = len(self.train_data)
        except TypeError:
            # train_data not indexable → user must adapt this part
            return None

        if n == 0 or self.rand_percent <= 0:
            return None

        subset_size = max(1, int(n * self.rand_percent / 100.0))
        indices = random.sample(range(n), subset_size)
        subset = [self.train_data[i] for i in indices]

        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def adaptive_local_aggregation(
        self,
        global_model: nn.Module,
        local_model: nn.Module,
    ) -> None:
        """
        Modify `local_model` in-place to obtain FedALA's **initialized** local model.

        Args:
            global_model: model with freshly received global parameters.
            local_model: model containing the previous local parameters.
                This object will be updated in-place.
        """
        rand_loader = self._build_random_loader()
        if rand_loader is None:
            # no data / unsupported container → fall back silently
            return

        # Collect parameter lists for convenience
        params_g = list(global_model.parameters())
        params_l = list(local_model.parameters())

        if len(params_g) == 0:
            return

        # If models are identical (e.g. very first round), do nothing
        with torch.no_grad():
            if torch.sum(params_g[0] - params_l[0]).abs().item() == 0.0:
                return

        # Decide how many "top" parameter groups to adapt
        num_groups = len(params_l)
        top_k = self.layer_idx if self.layer_idx > 0 else num_groups
        top_k = min(max(1, top_k), num_groups)

        # Lower layers: we simply take the global update
        low_l = params_l[:-top_k]
        low_g = params_g[:-top_k]
        with torch.no_grad():
            for p_l, p_g in zip(low_l, low_g):
                p_l.data.copy_(p_g.data)

        # Build a temporary model for weight-learning on the top layers
        model_tmp = copy.deepcopy(local_model).to(self.device)
        params_tmp = list(model_tmp.parameters())
        low_tmp = params_tmp[:-top_k]
        top_tmp = params_tmp[-top_k:]
        top_l = params_l[-top_k:]
        top_g = params_g[-top_k:]

        # Freeze lower layers in tmp model
        for p in low_tmp:
            p.requires_grad_(False)

        # "Optimizer" with lr=0, we only use it to accumulate gradients.
        optimizer = torch.optim.SGD(top_tmp, lr=0.0)

        # Initialize per-parameter weights (all ones) on first use
        if self._weights is None:
            self._weights = [
                torch.ones_like(p.data, device=self.device)
                for p in top_l
            ]

        # Initialize tmp model's top layers with current weights
        with torch.no_grad():
            for p_t, p_l, p_g, w in zip(top_tmp, top_l, top_g, self._weights):
                p_t.data.copy_(p_l.data + (p_g.data - p_l.data) * w.data)

        losses: list[float] = []
        steps = 0

        # Weight-learning loop
        while True:
            for batch in rand_loader:
                # ---- unpack (x, y); adapt this if your task uses another format ----
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                else:
                    # For graph tasks you may need something like:
                    #   data = batch
                    #   x, y = data.x, data.y
                    # Adjust as needed.
                    x, y = batch

                if isinstance(x, (list, tuple)):
                    x = [t.to(self.device) for t in x]
                else:
                    x = x.to(self.device)

                y = y.to(self.device)

                optimizer.zero_grad()
                out = model_tmp(x)
                loss = self.loss_fn(out, y)
                loss.backward()

                with torch.no_grad():
                    # Gradient w.r.t. weights follows from:
                    # param_t = param_l + (param_g - param_l) * w
                    for w, p_t, p_l, p_g in zip(
                        self._weights, top_tmp, top_l, top_g
                    ):
                        grad_w = p_t.grad * (p_g.data - p_l.data)
                        w.data.add_(-self.eta * grad_w).clamp_(0.0, 1.0)
                        # Update tmp param according to new weights
                        p_t.data.copy_(p_l.data + (p_g.data - p_l.data) * w.data)

                losses.append(float(loss.item()))
                steps += 1

            # If we're not in the warmup phase, stop after one pass
            if not self._warmup_phase:
                break

            # Check simple convergence criterion on last `num_pre_loss` losses
            if len(losses) >= self.num_pre_loss:
                tail = losses[-self.num_pre_loss:]
                if np.std(tail) < self.std_threshold:
                    # print(f"[ALA] client {self.cid}: std={np.std(tail):.4f}, steps={steps}")
                    break

        # After learning weights, copy the adapted top layers back to the *real* local model
        with torch.no_grad():
            for p_l, p_t in zip(top_l, top_tmp):
                p_l.data.copy_(p_t.data)

        # From now on, only one pass per round
        self._warmup_phase = False
