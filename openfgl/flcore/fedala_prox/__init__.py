"""
FedALA-Prox: Adaptive Local Aggregation with Proximal Regularization
Combines FedALA's smart initialization with FedProx's stable training
"""

from .client import FedALAProxClient
from .server import FedALAProxServer

__all__ = ['FedALAProxClient', 'FedALAProxServer']
