# openfgl/flcore/fedala_r/server_graphfl.py

from __future__ import annotations

from typing import List, Optional

import torch

from openfgl.flcore.base import BaseServer


class FedALARGraphFLServer(BaseServer):
    """
    Graph-FL FedALA-R server:
      - FedAvg aggregation (weighted average by client num_samples)
      - Residual buffer R is an EMA of global deltas:
            delta_t = theta_t - theta_{t-1}
            R_t = beta * R_{t-1} + (1-beta) * delta_t
      - Sends both theta_t and R_t to clients each round.
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device)

        self.beta = float(getattr(args, "residual_beta", 0.9))            # EMA factor
        self.gamma = float(getattr(args, "residual_gamma", 1.0))          # client-side scaling (for logging)
        self.clip_norm = float(getattr(args, "residual_clip_norm", 0.0))  # 0 => no clipping

        self.prev_global: Optional[List[torch.Tensor]] = None
        self.residual: Optional[List[torch.Tensor]] = None

        # IMPORTANT: initialize with CLONED tensors (no aliasing)
        init_w = [p.detach().clone() for p in self.task.model.parameters()]

        self.message_pool["server"] = {
            "num_samples": int(getattr(self.task, "num_samples", 0)),
            "weight": init_w,
            "residual": None,
            "residual_beta": float(self.beta),
            "residual_gamma": float(self.gamma),
        }

    @staticmethod
    def _clone_params(params: List[torch.Tensor]) -> List[torch.Tensor]:
        return [p.detach().clone() for p in params]

    def _compute_weighted_average(self, sampled_clients: List[int]) -> List[torch.Tensor]:
        total = 0
        client_states = []

        for cid in sampled_clients:
            pack = self.message_pool.get(f"client_{cid}", None)
            if pack is None:
                continue

            n = int(pack.get("num_samples", 0))
            w = pack.get("weight", None)
            if n <= 0 or w is None:
                continue
            if not isinstance(w, list) or len(w) == 0:
                continue

            total += n
            client_states.append((n, w))

        # fallback: return cloned current server weights (NO live params)
        if total == 0 or not client_states:
            return [p.detach().clone() for p in self.task.model.parameters()]

        # initialize accumulator on server model device
        with torch.no_grad():
            avg = [torch.zeros_like(p, device=p.device) for p in self.task.model.parameters()]

            for n, w in client_states:
                frac = float(n) / float(total)
                for i, w_i in enumerate(w):
                    if not isinstance(w_i, torch.Tensor):
                        raise TypeError(f"Client weight is not a Tensor (cid={cid}, idx={i}): {type(w_i)}")
                    avg[i].add_(frac * w_i.to(avg[i].device))

        return avg

    def _maybe_clip_residual(self, residual: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.clip_norm is None or self.clip_norm <= 0:
            return residual

        with torch.no_grad():
            sq = 0.0
            for r in residual:
                sq += float(torch.sum(r * r).item())
            norm = sq ** 0.5
            if norm <= 1e-12 or norm <= self.clip_norm:
                return residual

            scale = self.clip_norm / norm
            return [r * scale for r in residual]

    def execute(self):
        sampled_clients = self.message_pool.get("sampled_clients", list(range(self.args.num_clients)))

        # 1) FedAvg aggregation -> new global weights
        new_global = self._compute_weighted_average(sampled_clients)

        # 2) Compute delta and update residual buffer (EMA)
        if self.prev_global is None:
            self.prev_global = self._clone_params(new_global)
            self.residual = [torch.zeros_like(t) for t in new_global]
        else:
            delta = [ng.detach() - pg.detach() for ng, pg in zip(new_global, self.prev_global)]

            if self.residual is None:
                self.residual = [torch.zeros_like(t) for t in new_global]

            self.residual = [
                self.beta * r + (1.0 - self.beta) * d
                for r, d in zip(self.residual, delta)
            ]
            self.residual = self._maybe_clip_residual(self.residual)

            self.prev_global = self._clone_params(new_global)

        # 3) Load new global into the server model
        with torch.no_grad():
            for p, w in zip(self.task.model.parameters(), new_global):
                if not isinstance(w, torch.Tensor):
                    raise TypeError(f"new_global contains non-Tensor weight: {type(w)}")
                p.data.copy_(w.to(p.device))

    def send_message(self):
        # round is managed by FGLTrainer.train()
        round_id = int(self.message_pool.get("round", 0))

        weights = [p.detach().clone() for p in self.task.model.parameters()]
        residual = None if self.residual is None else [r.detach().clone() for r in self.residual]

        self.message_pool["server"] = {
            "num_samples": int(getattr(self.task, "num_samples", 0)),
            "weight": weights,
            "residual": residual,
            "residual_beta": float(self.beta),
            "residual_gamma": float(self.gamma),
            "round": round_id,  # optional: for debugging/logging only
        }

        if round_id in (0, 1, 5):
            has_res = residual is not None
            print(f"[server debug] round={round_id} has_residual={has_res}", flush=True)

