"""
FedALA: Federated Learning with Adaptive Local Aggregation
Improved version with round-dependent adaptive weight
"""

import torch
from openfgl.flcore.base import BaseClient


class FedALAClient(BaseClient):
    """
    FedALA client with adaptive parameter blending.
    
    Key improvements:
    - Higher adaptive weight in early rounds (accept global)
    - Lower adaptive weight in later rounds (preserve local)
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device
        )
        
        # Adaptive weight schedule
        self.w_initial = 0.8  # Early rounds: accept more global
        self.w_final = 0.3    # Later rounds: preserve more local
        self.decay_rounds = 50  # Rounds to decay from initial to final
    
    def execute(self):
        """
        Execute FedALA training with round-dependent adaptive weight
        """
        # Get current round
        current_round = self.message_pool.get("round", 0)
        
        # Compute adaptive weight (decay from w_initial to w_final)
        if current_round < self.decay_rounds:
            alpha = current_round / self.decay_rounds
            w = self.w_initial + alpha * (self.w_final - self.w_initial)
        else:
            w = self.w_final
        
        # Get global model parameters from server
        global_params = self.message_pool["server"]["weight"]
        
        # Apply adaptive aggregation: θ_new = θ_local + w * (θ_global - θ_local)
        with torch.no_grad():
            for local_param, global_param in zip(
                self.task.model.parameters(),
                global_params
            ):
                diff = global_param - local_param.data
                local_param.data.add_(w * diff)
        
        # Standard local training
        self.task.train()
    
    def send_message(self):
        """Send trained model parameters to server"""
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
