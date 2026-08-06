#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Minimal high-conflict scene diagnostics for Phase 0."""

import argparse

import torch


def pairwise_time_aligned_distance(ego_future, agent_futures):
    ego_xy = ego_future[..., :2]
    agent_xy = agent_futures[..., :2]
    return (agent_xy - ego_xy.unsqueeze(0)).norm(dim=-1)


def min_time_to_collision(ego_future, agent_futures, distance_threshold=1.5):
    distances = pairwise_time_aligned_distance(ego_future, agent_futures)
    close = distances <= distance_threshold
    if not bool(close.any()):
        return float("inf")
    return float(close.float().argmax(dim=-1).min().item())


def velocity_angle_score(ego_future, agent_futures):
    ego_velocity = ego_future[-1, :2] - ego_future[0, :2]
    agent_velocity = agent_futures[:, -1, :2] - agent_futures[:, 0, :2]
    ego_norm = ego_velocity.norm().clamp_min(1e-6)
    agent_norm = agent_velocity.norm(dim=-1).clamp_min(1e-6)
    cos_angle = (agent_velocity @ ego_velocity) / (agent_norm * ego_norm)
    return 1.0 - cos_angle.abs()


def classify_scene(
    ego_future,
    agent_futures,
    min_distance_threshold=1.5,
    ttc_threshold=4.0,
    angle_threshold=0.3,
):
    distances = pairwise_time_aligned_distance(ego_future, agent_futures)
    min_distance = float(distances.min().item())
    ttc = min_time_to_collision(ego_future, agent_futures, min_distance_threshold)
    angle_score = float(velocity_angle_score(ego_future, agent_futures).max().item())

    if min_distance <= min_distance_threshold * 0.5 and ttc <= ttc_threshold:
        return "high_conflict"
    if min_distance <= min_distance_threshold and ttc <= ttc_threshold and angle_score >= angle_threshold:
        return "high_conflict"
    return "low_conflict"


def classify_batch(ego_futures, agent_futures, **kwargs):
    labels = []
    for batch_idx in range(ego_futures.shape[0]):
        labels.append(classify_scene(ego_futures[batch_idx], agent_futures[batch_idx], **kwargs))
    return labels


def summarize_conflict_split(records):
    total = len(records)
    high_count = sum(1 for record in records if record.get("label") == "high_conflict")
    low_count = total - high_count
    return {
        "total": total,
        "high_conflict": high_count,
        "low_conflict": low_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run a synthetic crossing example.")
    args = parser.parse_args()

    if args.demo:
        ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        agent_futures = torch.tensor([[[2.0, -2.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0]]])
        print(classify_scene(ego_future, agent_futures))
    else:
        print("eval_high_conflict provides TTC, min-distance, and velocity-angle helpers.")


if __name__ == "__main__":
    main()
