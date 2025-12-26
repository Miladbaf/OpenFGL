"""
FedALA-R: Residual Adaptive Aggregation (Server)

Computes a global residual as the FedAvg-weighted update:
  R^t = Θ_global^t - Θ_global^{t-1}

Stores:
  - message_pool["server"]["weight"]   = current global model parameters
  - message_pool["server"]["residual"] = list of residual tensors R^t
"""

import torch
from openfgl.flcore.fedavg.server import FedAvgServer


class FedALARServer(FedAvgServer):
    """
    FedALA-R server extends FedAvg with residual computation.

    Steps per round:
      1. Aggregate client models via FedAvg.
      2. Compute global residual R^t = Θ_global^t - Θ_global^{t-1}.
      3. Update global model and store residual in message_pool["server"].
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALARServer, self).__init__(
            args, global_data, data_dir, message_pool, device
        )

        # Store previous global model parameters for residual computation
        self.previous_global_params = None

    def execute(self):
        """
        Server execution:
          - Perform FedAvg aggregation.
          - Compute residual as difference between new and previous global model.
        """
        # Ensure server message dict exists
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}

        # Initialize previous_global_params on first round
        if self.previous_global_params is None:
            self.previous_global_params = [
                param.data.clone() for param in self.task.model.parameters()
            ]

        # 1) Collect client models and sample counts
        sampled_clients = self.message_pool["sampled_clients"]

        clients_weight_list = [
            self.message_pool[f"client_{client_id}"]["weight"]
            for client_id in sampled_clients
        ]
        clients_sample_nums = [
            self.message_pool[f"client_{client_id}"]["num_samples"]
            for client_id in sampled_clients
        ]

        total_samples = sum(clients_sample_nums)

        # 2) FedAvg aggregation: Θ_global^t
        aggregated_params = []
        num_clients = len(clients_weight_list)

        for param_idx in range(len(clients_weight_list[0])):
            # Weighted average by num_samples
            agg_param = 0.0
            for client_idx in range(num_clients):
                client_param = clients_weight_list[client_idx][param_idx].data
                weight = clients_sample_nums[client_idx] / total_samples
                agg_param = agg_param + client_param * weight
            aggregated_params.append(agg_param)

        # 3) Compute residual: R^t = Θ_global^t - Θ_global^{t-1}
        global_residual = []
        for prev_param, agg_param in zip(self.previous_global_params, aggregated_params):
            global_residual.append(agg_param - prev_param)

        # 4) Update global model parameters
        with torch.no_grad():
            for param, aggregated in zip(self.task.model.parameters(), aggregated_params):
                param.data.copy_(aggregated)

        # 5) Update previous_global_params for next round
        self.previous_global_params = [
            param.data.clone() for param in self.task.model.parameters()
        ]

        # 6) Store residual in message pool for clients
        self.message_pool["server"]["residual"] = global_residual

    def send_message(self):
        """
        Broadcast global model and residual to clients.
        """
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}

        # Current global model
        self.message_pool["server"]["weight"] = list(self.task.model.parameters())
        # Residual already set in execute()
