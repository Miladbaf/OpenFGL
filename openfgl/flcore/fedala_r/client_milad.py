"""
FedALA-R: Residual Low-Rank Adaptive Aggregation (Client)
Combines adaptive local aggregation with global residual consensus
"""

import torch
from openfgl.flcore.base import BaseClient


class FedALARClient(BaseClient):
    """
    FedALA-R client with residual-enhanced adaptive aggregation.
    
    Strategy:
    - Round-dependent adaptive weights (w: 0.8 → 0.3)
    - Global residual term for cross-client consensus
    - Formula: Θ̃ = Θ_local + R + (Θ_global - Θ_local) ⊙ w
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALARClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device
        )
        
        # Adaptive weight schedule
        self.w_initial = 0.8  # Early rounds: accept more global
        self.w_final = 0.3    # Later rounds: preserve more local
        self.decay_rounds = 50
    
    def execute(self):
        """
        Execute FedALA-R training with residual-enhanced aggregation
        """
        # Get current round
        current_round = self.message_pool.get("round", 0)
        
        # Compute adaptive weight (decay from w_initial to w_final)
        if current_round < self.decay_rounds:
            alpha = current_round / self.decay_rounds
            w = self.w_initial + alpha * (self.w_final - self.w_initial)
        else:
            w = self.w_final
        
        # Get global model and residual from server
        global_params = self.message_pool["server"]["weight"]
        global_residual = self.message_pool["server"].get("residual", None)
        
        # Apply residual-enhanced adaptive aggregation
        with torch.no_grad():
            for idx, (local_param, global_param) in enumerate(zip(
                self.task.model.parameters(),
                global_params
            )):
                # Step 1: Apply adaptive aggregation
                diff = global_param - local_param.data
                local_param.data.add_(w * diff)
                
                # Step 2: Add global residual if available
                if global_residual is not None and idx < len(global_residual):
                    local_param.data.add_(global_residual[idx])
        
        # Standard local training
        self.task.train()
    
    def send_message(self):
        """Send trained model parameters to server"""
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
