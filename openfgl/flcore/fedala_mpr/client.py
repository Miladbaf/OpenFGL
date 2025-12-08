"""
FedALA-MP-R: Combined Message-Passing Aware + Residual Aggregation
Combines layer-specific adaptive weights with global residual consensus
"""

import torch
from openfgl.flcore.base import BaseClient


class FedALAMPRClient(BaseClient):
    """
    FedALA-MP-R: Combines both innovations
    - Layer-specific adaptive weights (from FedALA-MP)
    - Global residual term (from FedALA-R)
    
    Formula: Θ̃ = Θ_local + R + (Θ_global - Θ_local) ⊙ W_layer
    where W_layer varies by parameter type
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAMPRClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device
        )
        
        # Layer-specific adaptive weights (from FedALA-MP)
        self.weight_conv_initial = 0.9    # Conv: strongly global initially
        self.weight_conv_final = 0.5      # Conv: balanced later
        self.weight_linear = 0.1          # Linear: strongly local (fixed)
        self.weight_default = 0.3         # Others: mostly local
        self.decay_rounds = 50
    
    def execute(self):
        """
        Execute FedALA-MP-R: layer-aware adaptive aggregation with global residual
        """
        # Get current round
        current_round = self.message_pool.get("round", 0)
        
        # Compute conv weight (decay over time)
        if current_round < self.decay_rounds:
            alpha = current_round / self.decay_rounds
            w_conv = self.weight_conv_initial + alpha * (self.weight_conv_final - self.weight_conv_initial)
        else:
            w_conv = self.weight_conv_final
        
        # Get global model and residual from server
        global_params = self.message_pool["server"]["weight"]
        global_residual = self.message_pool["server"].get("residual", None)
        
        # Apply combined aggregation: layer-aware weights + residual
        with torch.no_grad():
            for idx, ((name, local_param), global_param) in enumerate(zip(
                self.task.model.named_parameters(),
                global_params
            )):
                # Step 1: Select layer-specific adaptive weight
                name_lower = name.lower()
                
                if 'conv' in name_lower or 'gnn' in name_lower or 'sage' in name_lower:
                    # GNN layers: favor global (decaying)
                    w = w_conv
                elif 'linear' in name_lower or 'fc' in name_lower or 'classifier' in name_lower:
                    # Classifier/FC layers: preserve local
                    w = self.weight_linear
                elif 'bias' in name_lower:
                    # Biases: mostly local
                    w = self.weight_default
                else:
                    # Other parameters
                    w = self.weight_default
                
                # Step 2: Apply layer-aware adaptive aggregation
                diff = global_param - local_param.data
                local_param.data.add_(w * diff)
                
                # Step 3: Add global residual (if available)
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
