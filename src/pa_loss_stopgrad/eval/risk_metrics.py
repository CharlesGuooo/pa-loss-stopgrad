#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Risk and prediction metrics for Phase 0.1 synthetic probes."""

import torch


def wta_l1_error(predicted_futures, gt_futures, per_sample=False):
    errors = (predicted_futures - gt_futures.unsqueeze(2)).abs().mean(dim=(-1, -2))
    sample_errors = errors.min(dim=-1).values.mean(dim=-1)
    return sample_errors if per_sample else sample_errors.mean()


def collision_proxy_risk(predicted_futures, ego_plans, sigma=1.0, per_sample=False):
    trajectories = ego_plans["trajectories"] if isinstance(ego_plans, dict) else ego_plans
    ego_xy = trajectories[..., :2]
    horizon = min(predicted_futures.shape[-2], ego_xy.shape[-2])
    agent_xy = predicted_futures[..., :horizon, :]
    ego_xy = ego_xy[..., :horizon, :]

    delta = agent_xy[:, :, :, None, :, :] - ego_xy[:, None, None, :, :, :]
    squared_distance = delta.square().sum(dim=-1)
    risk = torch.exp(-squared_distance / (2.0 * sigma**2)).mean(dim=(1, 2, 3, 4))
    return risk if per_sample else risk.mean()


def mode_diversity(predicted_futures, per_sample=False):
    if predicted_futures.shape[2] < 2:
        zeros = predicted_futures.new_zeros(predicted_futures.shape[0])
        return zeros if per_sample else zeros.mean()

    mode_centers = predicted_futures.mean(dim=-2)
    num_modes = mode_centers.shape[2]
    pairwise_delta = mode_centers[:, :, :, None, :] - mode_centers[:, :, None, :, :]
    same_agent = pairwise_delta.norm(dim=-1)
    upper = torch.triu(torch.ones(num_modes, num_modes, device=predicted_futures.device), diagonal=1)
    counts = upper.sum().clamp_min(1.0)
    sample_diversity = (same_agent * upper).sum(dim=(-1, -2)).mean(dim=-1) / counts
    return sample_diversity if per_sample else sample_diversity.mean()


def high_conflict_metrics(metrics, is_high_conflict):
    result = {}
    high_mask = is_high_conflict.bool()
    low_mask = ~high_mask

    for name, values in metrics.items():
        tensor_values = torch.as_tensor(values, dtype=torch.float32)
        if high_mask.any():
            result[f"high_conflict_{name}"] = float(tensor_values[high_mask].mean().item())
        else:
            result[f"high_conflict_{name}"] = 0.0
        if low_mask.any():
            result[f"low_conflict_{name}"] = float(tensor_values[low_mask].mean().item())
        else:
            result[f"low_conflict_{name}"] = 0.0

    return result


def summarize_epoch(records):
    if not records:
        return {}

    summary = {}
    keys = records[0].keys()
    for key in keys:
        values = [record[key] for record in records if isinstance(record.get(key), (int, float))]
        if values:
            summary[key] = float(torch.tensor(values, dtype=torch.float32).mean().item())
    return summary
