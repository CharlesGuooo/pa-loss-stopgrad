#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Continuous conflict diagnostics for real-data mining calibration."""

import math

import torch


def compute_conflict_diagnostics(ego_future, agent_futures, ego_history=None):
    ego_future = torch.as_tensor(ego_future, dtype=torch.float32)
    agent_futures = torch.as_tensor(agent_futures, dtype=torch.float32)
    distances = (agent_futures[..., :2] - ego_future.unsqueeze(0)[..., :2]).norm(dim=-1)
    sync_min_distance = float(distances.min().item()) if distances.numel() else float("inf")
    temporal = _temporal_nearest_distance(ego_future, agent_futures)
    angle_score = float(_velocity_angle_score(ego_future, agent_futures).max().item()) if agent_futures.numel() else 0.0
    overlap = _path_overlap_diagnostics(ego_future, agent_futures, temporal.get("closest_agent_index", -1))

    ego_xy = torch.as_tensor(ego_history, dtype=torch.float32) if ego_history is not None else ego_future
    speed = _ego_speed(ego_xy)
    return {
        "min_distance": sync_min_distance,
        "sync_min_distance": sync_min_distance,
        **temporal,
        **overlap,
        "ttc": _time_to_collision(distances),
        "angle_score": angle_score,
        "num_agents": int(agent_futures.shape[0]) if agent_futures.ndim >= 3 else 0,
        "ego_speed_mean": float(speed.mean().item()) if speed.numel() else 0.0,
        "ego_speed_max": float(speed.max().item()) if speed.numel() else 0.0,
    }


def classify_conflict_diagnostics(
    diagnostics,
    min_distance_threshold=1.5,
    ttc_threshold=4.0,
    angle_threshold=0.3,
):
    min_distance = diagnostics["min_distance"]
    ttc = diagnostics["ttc"]
    angle_score = diagnostics["angle_score"]
    if min_distance <= min_distance_threshold * 0.5 and ttc <= ttc_threshold:
        return "high_conflict"
    if min_distance <= min_distance_threshold and ttc <= ttc_threshold and angle_score >= angle_threshold:
        return "high_conflict"
    return "low_conflict"


def classify_near_miss_diagnostics(
    diagnostics,
    temporal_distance_threshold=2.5,
    max_time_gap=3,
    min_angle_score=0.1,
    min_distance_threshold=1.5,
    ttc_threshold=4.0,
    angle_threshold=0.3,
):
    sync_label = classify_conflict_diagnostics(
        diagnostics,
        min_distance_threshold=min_distance_threshold,
        ttc_threshold=ttc_threshold,
        angle_threshold=angle_threshold,
    )
    if sync_label == "high_conflict":
        return sync_label
    if (
        diagnostics.get("temporal_min_distance", float("inf")) <= temporal_distance_threshold
        and diagnostics.get("temporal_time_gap", float("inf")) <= max_time_gap
        and diagnostics.get("angle_score", 0.0) >= min_angle_score
    ):
        return "near_miss"
    return "low_conflict"


def classify_filtered_near_miss_diagnostics(
    diagnostics,
    temporal_distance_threshold=3.5,
    max_time_gap=4,
    min_crossing_angle_degrees=25.0,
    max_path_overlap_ratio=0.5,
    max_sync_min_distance=8.0,
    min_distance_threshold=1.5,
    ttc_threshold=4.0,
    angle_threshold=0.3,
):
    sync_label = classify_conflict_diagnostics(
        diagnostics,
        min_distance_threshold=min_distance_threshold,
        ttc_threshold=ttc_threshold,
        angle_threshold=angle_threshold,
    )
    if sync_label == "high_conflict":
        return sync_label
    if diagnostics.get("temporal_min_distance", float("inf")) > temporal_distance_threshold:
        return "low_conflict"
    if diagnostics.get("temporal_time_gap", float("inf")) > max_time_gap:
        return "low_conflict"
    if diagnostics.get("path_overlap_ratio", 0.0) > max_path_overlap_ratio:
        return "low_conflict"
    angle_deg = diagnostics.get("trajectory_angle_degrees", 0.0)
    if angle_deg < min_crossing_angle_degrees or angle_deg > (180.0 - min_crossing_angle_degrees):
        return "low_conflict"
    if diagnostics.get("sync_min_distance", diagnostics.get("min_distance", float("inf"))) > max_sync_min_distance:
        return "low_conflict"
    return "near_miss"


def summarize_numeric(values):
    finite_values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite_values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
        }
    return {
        "count": len(finite_values),
        "min": finite_values[0],
        "max": finite_values[-1],
        "mean": sum(finite_values) / len(finite_values),
        "q25": _quantile(finite_values, 0.25),
        "q50": _quantile(finite_values, 0.50),
        "q75": _quantile(finite_values, 0.75),
    }


def summarize_diagnostic_distribution(records):
    keys = [
        "min_distance",
        "sync_min_distance",
        "temporal_min_distance",
        "temporal_time_gap",
        "path_overlap_ratio",
        "trajectory_angle_degrees",
        "ttc",
        "angle_score",
        "num_agents",
        "ego_speed_mean",
        "ego_speed_max",
    ]
    return {
        key: summarize_numeric(record[key] for record in records if key in record)
        for key in keys
    }


def _velocity_angle_score(ego_future, agent_futures):
    ego_velocity = ego_future[-1, :2] - ego_future[0, :2]
    agent_velocity = agent_futures[:, -1, :2] - agent_futures[:, 0, :2]
    ego_norm = ego_velocity.norm().clamp_min(1e-6)
    agent_norm = agent_velocity.norm(dim=-1).clamp_min(1e-6)
    cos_angle = (agent_velocity @ ego_velocity) / (agent_norm * ego_norm)
    return 1.0 - cos_angle.abs()


def _time_to_collision(distances, distance_threshold=1.5):
    if not distances.numel():
        return float("inf")
    close = distances <= distance_threshold
    if not bool(close.any()):
        return float("inf")
    return float(close.float().argmax(dim=-1).min().item())


def _temporal_nearest_distance(ego_future, agent_futures):
    if not agent_futures.numel():
        return {
            "temporal_min_distance": float("inf"),
            "temporal_ego_timestep": -1,
            "temporal_agent_timestep": -1,
            "temporal_time_gap": float("inf"),
            "closest_agent_index": -1,
        }
    delta = agent_futures[:, None, :, :2] - ego_future[None, :, None, :2]
    distances = delta.norm(dim=-1)
    flat_index = int(distances.argmin().item())
    _, ego_steps, agent_steps = distances.shape
    closest_agent_index = flat_index // (ego_steps * agent_steps)
    remainder = flat_index % (ego_steps * agent_steps)
    ego_timestep = remainder // agent_steps
    agent_timestep = remainder % agent_steps
    return {
        "temporal_min_distance": float(distances.flatten()[flat_index].item()),
        "temporal_ego_timestep": int(ego_timestep),
        "temporal_agent_timestep": int(agent_timestep),
        "temporal_time_gap": int(abs(ego_timestep - agent_timestep)),
        "closest_agent_index": int(closest_agent_index),
    }


def _path_overlap_diagnostics(ego_future, agent_futures, closest_agent_index, overlap_distance_threshold=1.5):
    if (
        not agent_futures.numel()
        or closest_agent_index < 0
        or closest_agent_index >= agent_futures.shape[0]
    ):
        return {
            "trajectory_cosine": 0.0,
            "trajectory_angle_degrees": 0.0,
            "path_overlap_ratio": 0.0,
            "same_path_overlap": False,
        }

    ego_velocity = ego_future[-1, :2] - ego_future[0, :2]
    agent_future = agent_futures[closest_agent_index]
    agent_velocity = agent_future[-1, :2] - agent_future[0, :2]
    ego_norm = ego_velocity.norm().clamp_min(1e-6)
    agent_norm = agent_velocity.norm().clamp_min(1e-6)
    trajectory_cosine = float((agent_velocity @ ego_velocity / (agent_norm * ego_norm)).item())
    trajectory_angle_degrees = float(math.degrees(math.acos(max(-1.0, min(1.0, abs(trajectory_cosine))))))

    ego_points = ego_future[:, :2]
    agent_points = agent_future[:, :2]
    pairwise = (ego_points[:, None, :] - agent_points[None, :, :]).norm(dim=-1)
    overlap_mask = pairwise.min(dim=1).values <= overlap_distance_threshold
    path_overlap_ratio = float(overlap_mask.float().mean().item()) if overlap_mask.numel() else 0.0

    same_direction = trajectory_angle_degrees <= 25.0
    opposite_direction = trajectory_angle_degrees >= 155.0
    same_path_overlap = path_overlap_ratio >= 0.5 and (same_direction or opposite_direction)

    return {
        "trajectory_cosine": trajectory_cosine,
        "trajectory_angle_degrees": trajectory_angle_degrees,
        "path_overlap_ratio": path_overlap_ratio,
        "same_path_overlap": same_path_overlap,
    }


def _ego_speed(ego_xy):
    if ego_xy.shape[0] < 2:
        return torch.zeros(0, dtype=torch.float32)
    return (ego_xy[1:, :2] - ego_xy[:-1, :2]).norm(dim=-1)


def _quantile(sorted_values, q):
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
