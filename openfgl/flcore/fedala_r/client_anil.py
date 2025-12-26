"""
FedALA-R: Residual Adaptive Local Aggregation (Client)

Idea:
  Θ_local' = (1 - w_t) * Θ_local + w_t * Θ_global + β_t * R

Where:
  - w_t decays from w_initial → w_final over 'decay_rounds' rounds
  - β_t controls the strength of the residual term from the server
  - R is a list of per-parameter residual tensors provided by the server
"""

import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient


class FedALARClient(BaseClient):
    """
    FedALA-R client with residual-enhanced adaptive aggregation.

    Strategy:
      - Round-dependent adaptive weight w_t (global vs local mix)
      - Optional round-dependent residual scaling β_t
      - Uses a residual tensor per parameter from the server (if provided)

    Configurable via args (with defaults):
      - args.r_w_initial      (float, default=0.8)
      - args.r_w_final        (float, default=0.3)
      - args.r_decay_rounds   (int,   default=50)
      - args.r_res_scale_init (float, default=1.0)   # residual scale at round 0
      - args.r_res_scale_final(float, default=0.5)   # residual scale at round >= decay_rounds
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALARClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device
        )

        # ----- Adaptive weight schedule for FedALA part -----
        self.w_initial = getattr(args, "r_w_initial", 0.8)
        self.w_final = getattr(args, "r_w_final", 0.3)
        self.decay_rounds = getattr(args, "r_decay_rounds", 50)

        # ----- Residual scaling schedule (β_t) -----
        self.res_scale_initial = getattr(args, "r_res_scale_init", 1.0)
        self.res_scale_final = getattr(args, "r_res_scale_final", 0.5)

        self.device = device

    # ----------------------------------------------------------------------
    # Main FedALA-R logic
    # ----------------------------------------------------------------------
    def execute(self):
        """
        Execute FedALA-R training with residual-enhanced aggregation.

        Steps:
          1. Compute round-dependent w_t and β_t.
          2. For each parameter:
               θ_local ← (1-w_t) θ_local + w_t θ_global
               θ_local ← θ_local + β_t * R_i      (if residual R_i provided)
          3. Run standard local training.
        """
        # 1) Round & schedules
        current_round = self.message_pool.get("round", 0)

        if self.decay_rounds > 0 and current_round < self.decay_rounds:
            alpha = current_round / float(self.decay_rounds)
        else:
            alpha = 1.0

        # FedALA global-vs-local mix
        w_t = self.w_initial + alpha * (self.w_final - self.w_initial)

        # Residual scaling factor β_t
        beta_t = self.res_scale_initial + alpha * (
            self.res_scale_final - self.res_scale_initial
        )

        # 2) Get global parameters and residuals from server
        server_msg = self.message_pool["server"]
        global_params = server_msg["weight"]
        global_residual = server_msg.get("residual", None)  # expected: list of tensors or None

        # 3) Apply FedALA-R update per parameter
        with torch.no_grad():
            for idx, (local_param, global_param) in enumerate(
                zip(self.task.model.parameters(), global_params)
            ):
                # Get global tensor data on correct device
                if isinstance(global_param, nn.Parameter):
                    g_data = global_param.data
                else:
                    g_data = global_param
                g_data = g_data.to(local_param.device)

                # --- FedALA-style interpolation: θ ← (1-w_t) θ + w_t θ_global ---
                local_param.data.lerp_(g_data, w_t)

                # --- Residual term: θ ← θ + β_t * R_i (if provided) ---
                if global_residual is not None and idx < len(global_residual):
                    r_i = global_residual[idx]
                    if isinstance(r_i, nn.Parameter):
                        r_i = r_i.data
                    r_i = r_i.to(local_param.device)

                    # Optional safety check: same shape
                    if r_i.shape != local_param.data.shape:
                        raise ValueError(
                            f"FedALARClient: residual shape mismatch at idx={idx}: "
                            f"param shape={local_param.data.shape}, residual shape={r_i.shape}"
                        )

                    local_param.data.add_(beta_t * r_i)

        # 4) Standard local training
        self.task.train()

    # ----------------------------------------------------------------------
    # Message sending
    # ----------------------------------------------------------------------
    def send_message(self):
        """
        Send trained model parameters to the server.
        (Residuals, if any, are assumed to be computed/handled on the server side.)
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
