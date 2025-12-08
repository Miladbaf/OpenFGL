"""
FedALA-MP: Message-Passing Aware Adaptive Aggregation
Improved version with more aggressive layer-specific weights
"""

import torch
from openfgl.flcore.base import BaseClient


class FedALAMPClient(BaseClient):
    """
    FedALA-MP with aggressive layer-aware blending.
    
    Strategy:
    - Conv layers: STRONGLY favor global (w=0.9) - graph structure is shared
    - Linear layers: STRONGLY preserve local (w=0.1) - client patterns unique
    - Round-dependent decay for conv layers
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAMPClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device
        )
        
        # Layer-specific adaptive weights (more aggressive)
        self.weight_conv_initial = 0.9    # Conv: strongly global initially
        self.weight_conv_final = 0.5      # Conv: balanced later
        self.weight_linear = 0.1          # Linear: strongly local (fixed)
        self.weight_default = 0.3         # Others: mostly local
        self.decay_rounds = 50
    
    def execute(self):
        """
        Execute FedALA-MP with adaptive layer-wise blending
        """
        # Get current round
        current_round = self.message_pool.get("round", 0)
        
        # Compute conv weight (decay over time)
        if current_round < self.decay_rounds:
            alpha = current_round / self.decay_rounds
            w_conv = self.weight_conv_initial + alpha * (self.weight_conv_final - self.weight_conv_initial)
        else:
            w_conv = self.weight_conv_final
        
        # Get global model parameters from server
        global_params = self.message_pool["server"]["weight"]
        
        # Apply layer-aware adaptive aggregation
        with torch.no_grad():
            for (name, local_param), global_param in zip(
                self.task.model.named_parameters(),
                global_params
            ):
                # Select adaptive weight based on parameter type
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
                
                # Apply adaptive aggregation
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
