"""
FedALA-Gradient: Gradient-based adaptive aggregation
Server uses standard FedAvg
"""

from openfgl.flcore.fedavg.server import FedAvgServer


class FedALAGradientServer(FedAvgServer):
    """Standard FedAvg server - innovation is client-side"""
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device)
        self.personalized = False
    
    def execute(self):
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}
        super().execute()
    
    def send_message(self):
        if "server" not in self.message_pool:
            self.message_pool["server"] = {}
        super().send_message()
