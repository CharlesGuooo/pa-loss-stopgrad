#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Closest-agent trajectory quality metrics for Phase 0.12."""

from collections import Counter

import torch

from pa_loss_stopgrad.eval.conflict_metrics import compute_conflict_diagnostics
from pa_loss_stopgrad.data.nuscenes_trajectories import build_scene_tracks
from pa_loss_stopgrad._internal.benchmark import hash_events
from pa_loss_stopgrad._internal.real_data_risk_probe import find_event_window


DEFAULT_QUALITY_THRESHOLDS = {
    "static_displacement_threshold": 0.5,
    "min_path_length": 1.0,
    "max_sync_distance": 3.0,
    "max_time_gap": 2.0,
    "min_crossing_angle_degrees": 25.0,
    "max_path_overlap_ratio": 0.75,
}


def compute_event_trajectory_quality(
    nusc,
    event,
    history_len=5,
    future_len=6,
    max_agents=16,
    thresholds=None,
):
    thresholds = {**DEFAULT_QUALITY_THRESHOLDS, **(thresholds or {})}
    scene_tracks = build_scene_tracks(nusc, event["scene_token"])
    match = find_event_window(
        nusc,
        event,
        history_len=history_len,
        future_len=future_len,
        max_agents=max_agents,
    )
    if match is None:
        return {
            "event": dict(event),
            "decision": "skipped",
            "reason": "window_not_found",
            "metrics": {},
        }

    window = match["window"]
    target_index = match["target_agent_index"]
    agent_future = window["agent_futures"][target_index]
    agent_history = window["agent_history"][target_index]
    metrics = {
        **compute_agent_kinematics(agent_future, agent_history),
        **compute_scene_track_stats(scene_tracks, event.get("closest_agent_token"), history_len, future_len),
        **compute_interaction_quality(window, target_index),
        "target_agent_index": target_index,
        "closest_agent_token": event.get("closest_agent_token"),
        "min_required_future_steps": future_len,
    }
    decision, reason = classify_trajectory_quality(metrics, thresholds)
    return {
        "event": dict(event),
        "decision": decision,
        "reason": reason,
        "metrics": metrics,
    }


def compute_agent_kinematics(agent_future, agent_history=None):
    future = torch.as_tensor(agent_future, dtype=torch.float32)
    finite_steps = torch.isfinite(future).all(dim=-1)
    if future.shape[0] == 0:
        displacement = 0.0
        path_length = 0.0
        speeds = torch.empty(0)
    else:
        displacement = float((future[-1, :2] - future[0, :2]).norm().item())
        deltas = future[1:, :2] - future[:-1, :2]
        speeds = deltas.norm(dim=-1)
        path_length = float(speeds.sum().item())
    return {
        "agent_future_displacement": displacement,
        "agent_future_path_length": path_length,
        "agent_speed_mean": float(speeds.mean().item()) if speeds.numel() else 0.0,
        "agent_speed_max": float(speeds.max().item()) if speeds.numel() else 0.0,
        "valid_future_steps": int(finite_steps.sum().item()) if finite_steps.numel() else 0,
        "is_static": displacement < DEFAULT_QUALITY_THRESHOLDS["static_displacement_threshold"],
    }


def compute_scene_track_stats(scene_tracks, agent_token, history_len=5, future_len=6):
    track = scene_tracks["agent_tracks"].get(agent_token, [])
    track_length = sum(1 for point in track if point is not None)
    min_required = history_len + future_len
    return {
        "agent_track_length_samples": track_length,
        "min_required_track_samples": min_required,
        "is_short_track": track_length < min_required,
    }


def compute_interaction_quality(window, target_agent_index):
    diagnostics = compute_conflict_diagnostics(
        window["ego_future"],
        window["agent_futures"][target_agent_index : target_agent_index + 1],
        ego_history=window["ego_history"],
    )
    angle = float(diagnostics.get("trajectory_angle_degrees", 0.0))
    sync = float(diagnostics.get("sync_min_distance", diagnostics.get("min_distance", 10.0)))
    gap = float(diagnostics.get("temporal_time_gap", 10.0))
    overlap = float(diagnostics.get("path_overlap_ratio", 1.0))
    crossing_score = max(0.0, min(angle / 90.0, 1.0)) * (1.0 / (1.0 + gap)) * (1.0 - min(overlap, 1.0))
    return {
        "resolved_sync_min_distance": sync,
        "resolved_temporal_min_distance": float(diagnostics.get("temporal_min_distance", sync)),
        "resolved_temporal_time_gap": gap,
        "resolved_trajectory_angle_degrees": angle,
        "resolved_path_overlap_ratio": overlap,
        "crossing_quality_score": crossing_score,
    }


def classify_trajectory_quality(metrics, thresholds=None):
    thresholds = {**DEFAULT_QUALITY_THRESHOLDS, **(thresholds or {})}
    if metrics.get("valid_future_steps", 0) < metrics.get("min_required_future_steps", 0):
        return "reject", "incomplete_future"
    if metrics.get("is_static") or metrics.get("agent_future_displacement", 0.0) < thresholds["static_displacement_threshold"]:
        return "reject", "static_agent"
    if metrics.get("agent_future_path_length", 0.0) < thresholds["min_path_length"]:
        return "reject", "short_future_path"
    if metrics.get("is_short_track"):
        return "reject", "short_scene_track"
    if metrics.get("resolved_sync_min_distance", 10.0) > thresholds["max_sync_distance"]:
        return "borderline", "weak_sync_distance"
    if metrics.get("resolved_temporal_time_gap", 10.0) > thresholds["max_time_gap"]:
        return "borderline", "weak_temporal_alignment"
    if metrics.get("resolved_trajectory_angle_degrees", 0.0) < thresholds["min_crossing_angle_degrees"]:
        return "borderline", "weak_crossing_angle"
    if metrics.get("resolved_path_overlap_ratio", 1.0) > thresholds["max_path_overlap_ratio"]:
        return "borderline", "same_path_overlap"
    return "keep", "moving_interaction_quality_ok"


def filter_events_by_trajectory_quality(
    nusc,
    events,
    history_len=5,
    future_len=6,
    max_agents=16,
    thresholds=None,
    parent_event_hash=None,
):
    events = list(events)
    kept = []
    rejected = []
    borderline = []
    skipped = []
    decisions = []

    for event in events:
        result = compute_event_trajectory_quality(
            nusc,
            event,
            history_len=history_len,
            future_len=future_len,
            max_agents=max_agents,
            thresholds=thresholds,
        )
        updated = {
            **result["event"],
            **{f"phase0_12_{key}": value for key, value in result["metrics"].items()},
            "phase0_12_quality_decision": result["decision"],
            "phase0_12_quality_reason": result["reason"],
        }
        decisions.append(
            {
                "event_id": event.get("event_id"),
                "decision": result["decision"],
                "reason": result["reason"],
                "label": event.get("label"),
            }
        )
        if result["decision"] == "keep":
            kept.append(updated)
        elif result["decision"] == "reject":
            rejected.append(updated)
        elif result["decision"] == "borderline":
            borderline.append(updated)
        else:
            skipped.append(updated)

    high_conflict_input = sum(1 for event in events if event.get("label") == "high_conflict")
    high_conflict_kept = sum(1 for event in kept if event.get("label") == "high_conflict")
    report = {
        "parent_event_hash": parent_event_hash,
        "current_event_hash": hash_events(events),
        "num_input_events": len(events),
        "num_hq_events": len(kept),
        "num_rejected_events": len(rejected),
        "num_borderline_events": len(borderline),
        "num_skipped_events": len(skipped),
        "high_conflict_input_count": high_conflict_input,
        "high_conflict_kept_count": high_conflict_kept,
        "high_conflict_downgraded_count": high_conflict_input - high_conflict_kept,
        "decision_counts": dict(Counter(item["decision"] for item in decisions)),
        "reason_counts": dict(Counter(item["reason"] for item in decisions)),
        "thresholds": {**DEFAULT_QUALITY_THRESHOLDS, **(thresholds or {})},
        "decisions": decisions,
    }
    return {
        "hq_events": kept,
        "rejected_events": rejected,
        "borderline_events": borderline,
        "skipped_events": skipped,
        "report": report,
    }
