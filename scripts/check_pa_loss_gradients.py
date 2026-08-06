#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Synthetic Full vs stop-gradient check for PlanningAwareLoss."""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pa_loss_stopgrad.pa_loss.planning_loss import PlanningAwareLoss


def run_gradient_check(batch_size=2, num_agents=3, num_modes=4, horizon=6):
    torch.manual_seed(7)
    loss_fn = PlanningAwareLoss(w_collision=1.0, w_comfort=0.0, w_offroad=0.0)

    full_predictions = torch.randn(
        batch_size, num_agents, num_modes, horizon, 2, requires_grad=True
    )
    ego_plans = {
        "trajectories": torch.zeros(batch_size, 1, horizon, 3, requires_grad=True),
        "scores": torch.zeros(batch_size, 1),
    }
    full_loss = loss_fn(full_predictions, ego_plans, None)["total_loss"]
    full_loss.backward()
    full_grad_norm = float(full_predictions.grad.norm().item())

    stop_predictions = full_predictions.detach().clone().requires_grad_(True)
    stop_ego_plans = {
        "trajectories": torch.zeros(batch_size, 1, horizon, 3, requires_grad=True),
        "scores": torch.zeros(batch_size, 1),
    }
    stop_loss = loss_fn(
        stop_predictions,
        stop_ego_plans,
        None,
        detach_predicted_futures=True,
    )["total_loss"]
    stop_loss.backward()
    stop_grad_norm = 0.0
    if stop_predictions.grad is not None:
        stop_grad_norm = float(stop_predictions.grad.norm().item())

    return {
        "full_loss": float(full_loss.item()),
        "stop_gradient_loss": float(stop_loss.item()),
        "full_prediction_grad_norm": full_grad_norm,
        "stop_gradient_prediction_grad_norm": stop_grad_norm,
        "passed": full_grad_norm > 0.0 and stop_grad_norm == 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    result = run_gradient_check(
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        num_modes=args.num_modes,
        horizon=args.horizon,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
