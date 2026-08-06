#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains the ego-vehicle planner.

It takes the output of the perception and prediction modules and generates a safe and
comfortable trajectory for the ego-vehicle.
"""

import torch.nn as nn

class EgoPlanner(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, horizon=12, num_modes=6):
        """Initializes the EgoPlanner."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_modes = num_modes

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.trajectory_head = nn.Linear(hidden_dim, self.num_modes * self.horizon * 3)
        self.score_head = nn.Linear(hidden_dim, self.num_modes)

    def forward(self, context_features):
        """
        Generates candidate ego-trajectories.

        Args:
            context_features (torch.Tensor): Features from the scene context.

        Returns:
            dict: Candidate trajectories ``[B, K, T, 3]`` and scores ``[B, K]``.
        """
        if context_features.ndim == 1:
            context_features = context_features.unsqueeze(0)
        if context_features.ndim != 2:
            raise ValueError("context_features must have shape [B, input_dim]")

        batch_size = context_features.shape[0]
        features = self.backbone(context_features)
        trajectories = self.trajectory_head(features).view(
            batch_size, self.num_modes, self.horizon, 3
        )
        scores = self.score_head(features)

        return {
            "trajectories": trajectories,
            "scores": scores,
        }
