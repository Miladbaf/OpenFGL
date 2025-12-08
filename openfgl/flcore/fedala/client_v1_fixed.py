"""
FedALA: Federated Learning with Adaptive Local Aggregation
Simplified version with fixed adaptive weight
"""

import torch
from openfgl.flcore.base import BaseClient


class FedALAClient(BaseClient):
    """
    FedALA client with adaptive parameter blending.
    Uses fixed adaptive weight w=0.5 for simplicity.
    
    FedALA formula: θ_new = θ_local + w * (θ_global - θ_local)
    - w=0: Keep all local knowledge (ignore global)
    - w=1: Accept all global knowledge (FedAvg)
    - w=0.5: Balance between local and global
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device
        )
        
        # Adaptive weight (can be tuned as hyperparameter)
        self.adaptive_weight = 0.5
    
    def execute(self):
        """
        Execute FedALA training:
        1. Adaptively blend global model with current local model
        2. Perform standard local training
        """
        # Get global model parameters from server
        global_params = self.message_pool["server"]["weight"]
        
        # Apply adaptive aggregation: θ_new = θ_local + w * (θ_global - θ_local)
        with torch.no_grad():
            for local_param, global_param in zip(
                self.task.model.parameters(),
                global_params
            ):
                # Calculate difference between global and local
                diff = global_param - local_param.data
                # Blend: move towards global by factor w
                local_param.data.add_(self.adaptive_weight * diff)
        
        # Standard local training
        self.task.train()
    
    def send_message(self):
        """Send trained model parameters to server"""
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
