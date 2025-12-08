"""
FedALA-R: Residual Low-Rank Adaptive Aggregation (Server)
Computes and maintains global residual for cross-client consensus
"""

import torch
from openfgl.flcore.fedavg.server import FedAvgServer


class FedALARServer(FedAvgServer):
    """
    FedALA-R server extends FedAvg with residual computation.
    
    Additional operations:
    1. Compute global residual: R^t = (1/K) Σ(Θ_i^t - Θ^(t-1))
    2. Broadcast residual along with global model
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALARServer, self).__init__(
            args, global_data, data_dir, message_pool, device
        )
        
        # Store previous global model for residual computation
        self.previous_global_params = None
    
    def execute(self):
        """
        Server execution:
        1. Standard FedAvg aggregation
        2. Compute and store global residual
        """
        # Initialize server message pool
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}
        
        # Store previous global model before aggregation
        if self.previous_global_params is None:
            # First round: initialize
            self.previous_global_params = [
                param.data.clone() for param in self.task.model.parameters()
            ]
        
        # Get client models
        clients_weight_list = [
            self.message_pool[f"client_{client_id}"]["weight"]
            for client_id in self.message_pool["sampled_clients"]
        ]
        
        clients_sample_nums = [
            self.message_pool[f"client_{client_id}"]["num_samples"]
            for client_id in self.message_pool["sampled_clients"]
        ]
        
        total_samples = sum(clients_sample_nums)
        aggregated_params = []
        
        for param_idx in range(len(clients_weight_list[0])):
            # Weighted average
            aggregated = sum(
                clients_weight_list[client_idx][param_idx].data * (clients_sample_nums[client_idx] / total_samples)
                for client_idx in range(len(clients_weight_list))
            )
            aggregated_params.append(aggregated)
        
        # Compute global residual: R^t = (1/K) Σ(Θ_i^t - Θ^(t-1))
        K = len(clients_weight_list)
        global_residual = []
        
        for param_idx in range(len(self.previous_global_params)):
            # Average of client updates
            avg_update = sum(
                clients_weight_list[client_idx][param_idx].data - self.previous_global_params[param_idx]
                for client_idx in range(K)
            ) / K
            global_residual.append(avg_update)
        
        # Update global model
        with torch.no_grad():
            for param, aggregated in zip(self.task.model.parameters(), aggregated_params):
                param.data.copy_(aggregated)
        
        # Store current as previous for next round
        self.previous_global_params = [
            param.data.clone() for param in self.task.model.parameters()
        ]
        
        # Store residual in message pool for clients
        self.message_pool["server"]["residual"] = global_residual
    
    def send_message(self):
        """Broadcast global model and residual to clients"""
        # Ensure server key exists
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}
        
        self.message_pool["server"]["weight"] = list(self.task.model.parameters())
        # Residual already stored in execute()
