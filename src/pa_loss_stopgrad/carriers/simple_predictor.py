#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Phase 1.8 SimplePredictor.

Minimal multimodal trajectory predictor (Master Plan Phase 0 carrier, see
``idea/Planning-GAZE_Code_Design.md``). It exists to test whether the
Planning-Aware Loss can train a predictor at all, and to support the
``Full > stop_gradient(predicted_futures)`` hard gate -- it does NOT depend on
GameFormer.

Output ``predicted_futures`` are per-step deltas ``[B, N, M, T, 2]``; absolute
trajectories are recovered downstream by cumsum.
"""

import torch
import torch.nn as nn


class SimplePredictor(nn.Module):
    def __init__(self, state_dim=9, hidden_dim=128, num_modes=6, horizon=6):
        super().__init__()
        self.encoder = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.traj_head = nn.Linear(hidden_dim, num_modes * horizon * 2)
        self.score_head = nn.Linear(hidden_dim, num_modes)
        self.num_modes = num_modes
        self.horizon = horizon

    def forward(self, agent_history):
        # agent_history: [B, N, H, state_dim]
        b, n, h, d = agent_history.shape
        x = agent_history.reshape(b * n, h, d)
        _, hidden = self.encoder(x)
        hidden = hidden[-1]
        futures = self.traj_head(hidden).view(b, n, self.num_modes, self.horizon, 2)
        scores = self.score_head(hidden).view(b, n, self.num_modes)
        return {"predicted_futures": futures, "prediction_scores": scores}
