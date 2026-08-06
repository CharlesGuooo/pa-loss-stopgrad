#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Phase 0 Planning-Aware Loss (synthetic / real-data risk probe).

Risk-weighted, mode-level collision objective used by the Phase 0 probes
(``scripts/risk_probe_train.py``, ``planning_gaze/local_risk_trainer.py``,
``scripts/check_pa_loss_gradients.py``, ``planning_gaze/phase0_20_allocation_sanity.py``).
It consumes absolute predicted agent futures and an ego plan, allocates a weight
to each prediction mode, and supports the ``stop_gradient(predicted_futures)``
ablation via ``detach_predicted_futures``.

Allocation modes
----------------
- ``softmax`` (default): ``alpha_m = risk_weights = softmax(per_mode_cost / tau)``
  -- dangerous (closer-to-ego) modes get more safety weight even at low
  probability.
- ``wta``: ``alpha_m`` is one-hot at the most likely predicted mode
  (``argmax(prediction_scores)``) -- the "true WTA" safety allocation that only
  protects the top-scoring mode.

Inputs
------
- ``predicted_futures``: ``[B, N, M, T, 2]`` absolute agent positions.
- ``ego_plans``: dict with ``trajectories`` ``[B, K, T, 3]`` (x, y, heading) and
  ``scores`` ``[B, K]``.
- ``drivable_area_map``: optional, used by the offroad term (None -> 0).
- ``prediction_scores``: optional ``[B, N, M]`` (required for WTA allocation).
- ``detach_predicted_futures``: detach predictions inside the loss so planning
  risk cannot train the predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanningAwareLoss(nn.Module):
    def __init__(self, w_collision=1.0, w_comfort=1.0, w_offroad=1.0, tau=1.0, sigma=1.0):
        super().__init__()
        self.w_collision = w_collision
        self.w_comfort = w_comfort
        self.w_offroad = w_offroad
        self.tau = tau
        self.sigma = sigma

    def forward(
        self,
        predicted_futures,
        ego_plans,
        drivable_area_map=None,
        prediction_scores=None,
        detach_predicted_futures=False,
        allocation_mode="softmax",
    ):
        num_modes = predicted_futures.shape[2]
        agent_xy = predicted_futures[..., :2]
        if detach_predicted_futures:
            agent_xy = agent_xy.detach()

        trajectories = ego_plans["trajectories"] if isinstance(ego_plans, dict) else ego_plans
        ego_xy = trajectories[..., :2]
        horizon = min(agent_xy.shape[-2], ego_xy.shape[-2])
        agent_xy = agent_xy[..., :horizon, :]              # [B, N, M, T', 2]
        ego_xy = ego_xy[..., :horizon, :]                  # [B, K, T', 2]

        # Pointwise distance between every agent mode and every ego plan step.
        delta = agent_xy[:, :, :, None, :, :] - ego_xy[:, None, None, :, :, :]
        squared_distance = delta.square().sum(dim=-1)      # [B, N, M, K, T']
        cost = torch.exp(-squared_distance / (2.0 * self.sigma ** 2))
        per_mode_cost = cost.mean(dim=(3, 4))              # [B, N, M]

        # Risk weighting: higher collision cost (closer to ego) -> higher weight.
        risk_weights = F.softmax(per_mode_cost / self.tau, dim=-1)  # [B, N, M]

        if allocation_mode == "wta":
            if prediction_scores is not None:
                selected_mode = prediction_scores.argmax(dim=-1)       # [B, N]
            else:
                selected_mode = per_mode_cost.argmax(dim=-1)
            alpha_m = F.one_hot(selected_mode, num_classes=num_modes).to(per_mode_cost.dtype)
        else:
            selected_mode = risk_weights.argmax(dim=-1)
            alpha_m = risk_weights

        collision_loss = (alpha_m * per_mode_cost).sum(dim=-1).mean()

        comfort_loss = self._comfort_loss(trajectories) if self.w_comfort else collision_loss.new_zeros(())
        offroad_loss = (
            self._offroad_loss(agent_xy, drivable_area_map)
            if (self.w_offroad and drivable_area_map is not None)
            else collision_loss.new_zeros(())
        )

        total_loss = (
            self.w_collision * collision_loss
            + self.w_comfort * comfort_loss
            + self.w_offroad * offroad_loss
        )
        # Zero-coupled connector: keeps total_loss in the autograd graph (so
        # callers can always differentiate it) while contributing exactly zero
        # gradient to the predictions. Under detach_predicted_futures this is the
        # only path back to predicted_futures, yielding a zero (not missing)
        # prediction gradient.
        total_loss = total_loss + predicted_futures.sum() * 0.0

        return {
            "total_loss": total_loss,
            "collision_loss": collision_loss,
            "comfort_loss": comfort_loss,
            "offroad_loss": offroad_loss,
            "risk_weights": risk_weights,
            "alpha_m": alpha_m,
            "selected_mode": selected_mode,
            "per_mode_cost": per_mode_cost,
        }

    @staticmethod
    def _comfort_loss(trajectories):
        # Huber-wrapped jerk of the ego plan (finite differences over time).
        xy = trajectories[..., :2]
        if xy.shape[-2] < 4:
            return xy.new_zeros(())
        vel = xy[..., 1:, :] - xy[..., :-1, :]
        acc = vel[..., 1:, :] - vel[..., :-1, :]
        jerk = acc[..., 1:, :] - acc[..., :-1, :]
        return F.huber_loss(jerk, torch.zeros_like(jerk), delta=1.0)

    @staticmethod
    def _offroad_loss(agent_xy, drivable_area_map):
        # Placeholder differentiable offroad penalty: mean squared excursion of
        # predicted positions outside a [-r, r] box derived from the map extent.
        radius = float(getattr(drivable_area_map, "radius", 50.0))
        excursion = (agent_xy.abs() - radius).clamp(min=0.0)
        return excursion.square().mean()
