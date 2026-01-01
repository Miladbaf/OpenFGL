# openfgl/flcore/fedala_r/server_ihsan.py
"""
FedALA-R Server (Graph-FL + Subgraph-FL compatible)

- Aggregation: FedAvg (sample-weighted)
- Residual (DEFAULT): delta residual
    R^t = Θ^t - Θ^{t-1}
- (OPTIONAL / COMMENTED): EMA residual
    R_ema^t = ema * R_ema^{t-1} + (1-ema) * (Θ^t - Θ^{t-1})
"""

import torch
from openfgl.flcore.base import BaseServer


class FedALARServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device, personalized=False)
        self.prev_global_params = None

        # ---- EMA option (kept, but not active by default) ----
        # self.ema_decay = float(getattr(args, "residual_ema_decay", 0.9))
        # self.ema_residual = None

    @torch.no_grad()
    def execute(self):
        # ----------------------------
        # 1) FedAvg aggregation
        # ----------------------------
        sampled_clients = self.message_pool.get(
            "sampled_clients", list(range(self.args.num_clients))
        )

        # total samples for weighting
        num_tot = 0
        for cid in sampled_clients:
            num_tot += int(self.message_pool[f"client_{cid}"]["num_samples"])

        # guard: avoid division by zero
        if num_tot == 0:
            raise ValueError("FedALARServer: total num_samples across sampled clients is 0.")

        # initialize accumulator tensors
        aggregated = [torch.zeros_like(p.data) for p in self.task.model.parameters()]

        # accumulate weighted client parameters
        for cid in sampled_clients:
            msg = self.message_pool[f"client_{cid}"]
            n = int(msg["num_samples"])
            w = n / float(num_tot)

            client_params = msg["weight"]
            for i, cp in enumerate(client_params):
                cp_t = cp.data if hasattr(cp, "data") else cp
                aggregated[i].add_(cp_t.to(aggregated[i].device), alpha=w)

        # write back to global model
        for p, a in zip(self.task.model.parameters(), aggregated):
            p.data.copy_(a)

        # ----------------------------
        # 2) Residual computation (DELTA)
        # ----------------------------
        if self.prev_global_params is None:
            delta_residual = [torch.zeros_like(p.data) for p in self.task.model.parameters()]
        else:
            delta_residual = [
                (p.data - prev.to(p.data.device))
                for p, prev in zip(self.task.model.parameters(), self.prev_global_params)
            ]

        # update prev snapshot for next round
        self.prev_global_params = [p.data.clone() for p in self.task.model.parameters()]

        # ----------------------------
        # 3) OPTIONAL: EMA residual (COMMENTED OUT)
        # ----------------------------
        # if self.ema_residual is None:
        #     self.ema_residual = [r.clone() for r in delta_residual]
        # else:
        #     for i in range(len(delta_residual)):
        #         self.ema_residual[i].mul_(self.ema_decay).add_(delta_residual[i], alpha=(1.0 - self.ema_decay))
        #
        # residual_to_send = self.ema_residual

        # DEFAULT (delta)
        residual_to_send = delta_residual

        # ----------------------------
        # 4) publish to message pool
        # ----------------------------
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}

        self.message_pool["server"]["residual"] = residual_to_send

    def send_message(self):
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}
        self.message_pool["server"]["weight"] = list(self.task.model.parameters())
        # residual is produced in execute()
