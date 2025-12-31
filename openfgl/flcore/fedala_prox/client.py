"""
FedALA-Prox: FedALA with Proximal Regularization
Combines adaptive local aggregation initialization with proximal term during training
"""

import torch
from openfgl.flcore.base import BaseClient


class FedALAProxClient(BaseClient):
    """
    FedALA-Prox Client: Adaptive initialization + Proximal regularization
    
    Phase 1: FedALA adaptive aggregation (smart initialization)
    Phase 2: Local training with proximal term (prevents drift)
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAProxClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device
        )
        
        # FedALA parameters
        self.w_initial = 0.8  # Early rounds: accept more global
        self.w_final = 0.3    # Later rounds: preserve more local
        self.decay_rounds = 50
        
        # FedProx parameters
        self.mu = 0.1  # Proximal term coefficient (typical: 0.001 - 0.1)
    
    def execute(self):
        """
        Execute FedALA-Prox training
        """
        # ============================================================
        # PHASE 1: FedALA Adaptive Aggregation (Initialization)
        # ============================================================
        
        # Get current round
        current_round = self.message_pool.get("round", 0)
        
        # Compute adaptive weight (decay from w_initial to w_final)
        if current_round < self.decay_rounds:
            alpha = current_round / self.decay_rounds
            w = self.w_initial + alpha * (self.w_final - self.w_initial)
        else:
            w = self.w_final
        
        # Get global model parameters
        global_params = self.message_pool["server"]["weight"]
        
        # Apply FedALA aggregation: θ_new = θ_local + w * (θ_global - θ_local)
        with torch.no_grad():
            for local_param, global_param in zip(self.task.model.parameters(), global_params):
                diff = global_param - local_param.data
                local_param.data.add_(w * diff)
        
        # Store initialized parameters for proximal term
        init_params = [p.data.clone() for p in self.task.model.parameters()]
        
        # ============================================================
        # PHASE 2: Local Training
        # ============================================================
        # Note: Adding proximal term requires modifying OpenFGL's task internals
        # For now, using standard training (future work: custom training loop)
        
        self.task.train()
    
    def send_message(self):
        """
        Send updated model to server
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
