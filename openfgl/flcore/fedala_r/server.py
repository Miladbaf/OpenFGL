"""
FedALA-R Server: FedAvg aggregation with residual computation
"""

import torch
from openfgl.flcore.fedavg.server import FedAvgServer


class FedALARServer(FedAvgServer):
    """FedALA-R server computes global residual R^t = Θ^t - Θ^{t-1}"""

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALARServer, self).__init__(
            args, global_data, data_dir, message_pool, device
        )
        self.previous_global_params = None
        self.personalized = False

    def execute(self):
        """Aggregate clients and compute residual"""
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}

        if self.previous_global_params is None:
            self.previous_global_params = [
                param.data.clone() for param in self.task.model.parameters()
            ]

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

        aggregated_params = []
        for param_idx in range(len(clients_weight_list[0])):
            agg_param = 0.0
            for client_idx, client_weight in enumerate(clients_weight_list):
                client_param = client_weight[param_idx].data
                weight = clients_sample_nums[client_idx] / total_samples
                agg_param = agg_param + client_param * weight
            aggregated_params.append(agg_param)

        global_residual = []
        for prev_param, agg_param in zip(self.previous_global_params, aggregated_params):
            global_residual.append(agg_param - prev_param)

        with torch.no_grad():
            for param, aggregated in zip(self.task.model.parameters(), aggregated_params):
                param.data.copy_(aggregated)

        self.previous_global_params = [
            param.data.clone() for param in self.task.model.parameters()
        ]

        self.message_pool["server"]["residual"] = global_residual

    def send_message(self):
        """Broadcast global model and residual"""
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}
        self.message_pool["server"]["weight"] = list(self.task.model.parameters())
