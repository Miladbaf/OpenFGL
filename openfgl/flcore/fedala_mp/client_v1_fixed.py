"""
FedALA-MP: Message-Passing Aware Adaptive Aggregation
Layer-specific adaptive weights based on parameter type
"""

import torch
from openfgl.flcore.base import BaseClient


class FedALAMPClient(BaseClient):
    """
    FedALA-MP client with layer-aware adaptive parameter blending.
    
    Hypothesis:
    - GNN conv layers: Learn more from global (w=0.7) - shared graph structure
    - Linear/FC layers: Preserve more local (w=0.3) - client-specific patterns
    - Other parameters: Balanced (w=0.5)
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAMPClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device
        )
        
        # Layer-specific adaptive weights
        self.weight_conv = 0.7      # GNN/Conv layers: more global
        self.weight_linear = 0.3    # Linear/FC layers: more local
        self.weight_default = 0.5   # Other parameters: balanced
    
    def execute(self):
        """
        Execute FedALA-MP training:
        1. Apply layer-aware adaptive aggregation
        2. Perform standard local training
        """
        # Get global model parameters from server
        global_params = self.message_pool["server"]["weight"]
        
        # Apply layer-aware adaptive aggregation
        with torch.no_grad():
            for (name, local_param), global_param in zip(
                self.task.model.named_parameters(),
                global_params
            ):
                # Select adaptive weight based on parameter name/type
                if 'conv' in name.lower() or 'gnn' in name.lower():
                    # GNN convolutional layers: favor global knowledge
                    w = self.weight_conv
                elif 'linear' in name.lower() or 'fc' in name.lower():
                    # Fully connected layers: preserve local patterns
                    w = self.weight_linear
                else:
                    # Other parameters (bias, normalization, etc.): balanced
                    w = self.weight_default
                
                # Apply: θ_new = θ_local + w * (θ_global - θ_local)
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
