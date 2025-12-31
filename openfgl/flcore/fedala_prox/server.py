"""
FedALA-Prox Server: Standard FedAvg aggregation
The innovation is entirely client-side (adaptive init + proximal training)
"""

from openfgl.flcore.fedavg.server import FedAvgServer


class FedALAProxServer(FedAvgServer):
    """
    FedALA-Prox server uses standard FedAvg aggregation.
    
    The method's innovation is client-side:
    - Clients use FedALA for adaptive initialization
    - Clients use FedProx for stable local training
    - Server simply averages the results
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALAProxServer, self).__init__(
            args, global_data, data_dir, message_pool, device
        )
        # Not a personalized method (all clients get same global model)
        self.personalized = False
    
    def execute(self):
        """
        Standard FedAvg aggregation
        """
        # Initialize server message pool
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}
        
        # Call parent's execute (standard FedAvg)
        super().execute()
    
    def send_message(self):
        """
        Broadcast global model to clients
        """
        # Initialize server key if needed
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}
        
        # Call parent's send_message
        super().send_message()
