#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Synthetic Phase 0.1 risk probe training loop."""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pa_loss_stopgrad.pa_loss.planning_loss import PlanningAwareLoss
from pa_loss_stopgrad.eval.risk_metrics import (
    collision_proxy_risk,
    high_conflict_metrics,
    mode_diversity,
    summarize_epoch,
    wta_l1_error,
)
from pa_loss_stopgrad.carriers.simple_predictor import SimplePredictor
from pa_loss_stopgrad._internal.synthetic_scenarios import SCENARIO_TYPES, make_synthetic_interaction_batch


def make_synthetic_batch(batch_size=4, num_agents=3, history_len=5, horizon=6, state_dim=9):
    return make_synthetic_interaction_batch(
        batch_size=batch_size,
        num_agents=num_agents,
        history_len=history_len,
        horizon=horizon,
        state_dim=state_dim,
        seed=13,
    )


def winner_take_all_l1(predicted_futures, gt_futures):
    return wta_l1_error(predicted_futures, gt_futures)


def run_risk_probe_epoch(
    mode="full",
    batch_size=4,
    num_steps=2,
    scenario_mix=None,
    seed=17,
):
    if mode not in {"no_pa", "full", "stop_grad", "wta_only"}:
        raise ValueError("mode must be one of: no_pa, full, stop_grad, wta_only")
    if scenario_mix is None:
        scenario_mix = list(SCENARIO_TYPES)

    torch.manual_seed(seed)
    predictor = SimplePredictor(state_dim=9, hidden_dim=32, num_modes=4, horizon=6)
    loss_fn = PlanningAwareLoss(w_collision=1.0, w_comfort=0.0, w_offroad=0.0)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    records = []
    for step_idx in range(num_steps):
        batch = make_synthetic_interaction_batch(
            batch_size=batch_size,
            scenario_mix=scenario_mix,
            seed=seed + step_idx,
        )
        output = predictor(batch["agent_history"])
        predicted_futures = output["predicted_futures"]
        predicted_futures.retain_grad()
        wta_loss = wta_l1_error(predicted_futures, batch["gt_futures"])

        pa_loss = predicted_futures.sum() * 0.0
        pa_prediction_grad_norm = 0.0
        if mode in {"no_pa", "wta_only"}:
            loss = wta_loss
        else:
            pa_loss = loss_fn(
                predicted_futures,
                batch["ego_plans"],
                None,
                prediction_scores=output["prediction_scores"],
                detach_predicted_futures=(mode == "stop_grad"),
            )["total_loss"]
            pa_grad = torch.autograd.grad(
                pa_loss,
                predicted_futures,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if pa_grad is not None:
                pa_prediction_grad_norm = float(pa_grad.norm().item())
            loss = wta_loss + 0.5 * pa_loss

        optimizer.zero_grad()
        loss.backward()
        prediction_grad_norm = 0.0
        if predicted_futures.grad is not None:
            prediction_grad_norm = float(predicted_futures.grad.norm().item())
        optimizer.step()

        per_sample_metrics = {
            "collision_risk": collision_proxy_risk(
                predicted_futures.detach(),
                batch["ego_plans"],
                per_sample=True,
            ),
            "wta_l1": wta_l1_error(
                predicted_futures.detach(),
                batch["gt_futures"],
                per_sample=True,
            ),
        }
        conflict_summary = high_conflict_metrics(per_sample_metrics, batch["is_high_conflict"])
        records.append(
            {
                "loss": float(loss.detach().item()),
                "wta_l1": float(wta_loss.detach().item()),
                "pa_loss": float(pa_loss.detach().item()),
                "collision_risk": float(per_sample_metrics["collision_risk"].mean().item()),
                "mode_diversity": float(mode_diversity(predicted_futures.detach()).item()),
                "prediction_grad_norm": prediction_grad_norm,
                "pa_prediction_grad_norm": pa_prediction_grad_norm,
                **conflict_summary,
            }
        )

    summary = summarize_epoch(records)
    return {
        "mode": mode,
        **summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no_pa", "full", "stop_grad", "wta_only"], default="full")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--scenario-mix", nargs="+", choices=list(SCENARIO_TYPES), default=list(SCENARIO_TYPES))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    result = run_risk_probe_epoch(
        mode=args.mode,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        scenario_mix=args.scenario_mix,
        seed=args.seed,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
