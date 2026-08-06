#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Real-data risk-probe batch construction from Phase 0.5d event indices."""

import json
from pathlib import Path

import torch

from pa_loss_stopgrad.data.nuscenes_trajectories import build_scene_tracks, iter_trajectory_windows


def load_event_index(path, limit=None):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    return events[:limit] if limit is not None else events


def find_event_window(nusc, event, history_len=5, future_len=6, max_agents=16):
    tracks = build_scene_tracks(nusc, event["scene_token"])
    for window in iter_trajectory_windows(
        tracks,
        history_len=history_len,
        future_len=future_len,
        max_agents=max_agents,
    ):
        if window["sample_token"] != event.get("sample_token"):
            continue
        closest_agent_token = event.get("closest_agent_token")
        if closest_agent_token not in window["agent_tokens"]:
            return None
        return {
            "event": event,
            "window": window,
            "target_agent_index": window["agent_tokens"].index(closest_agent_token),
        }
    return None


def build_real_data_risk_batch(nusc, events, history_len=5, future_len=6, max_agents=16):
    matches = []
    skipped = []
    for event in events:
        match = find_event_window(
            nusc,
            event,
            history_len=history_len,
            future_len=future_len,
            max_agents=max_agents,
        )
        if match is None:
            skipped.append(
                {
                    "event_id": event.get("event_id"),
                    "scene_token": event.get("scene_token"),
                    "sample_token": event.get("sample_token"),
                    "closest_agent_token": event.get("closest_agent_token"),
                    "reason": "window_not_found",
                }
            )
            continue
        matches.append(match)

    if not matches:
        return _empty_batch(skipped)

    windows = [match["window"] for match in matches]
    padded_history, padded_futures, agent_mask = _pad_agent_tensors(windows, max_agents=max_agents)
    ego_future_xy = torch.stack([window["ego_future"] for window in windows])
    ego_plan_xyz = torch.zeros(
        ego_future_xy.shape[0],
        1,
        ego_future_xy.shape[1],
        3,
        dtype=ego_future_xy.dtype,
    )
    ego_plan_xyz[:, 0, :, :2] = ego_future_xy

    return {
        "agent_history": padded_history,
        "gt_futures": padded_futures,
        "agent_mask": agent_mask,
        "ego_plans": {
            "trajectories": ego_plan_xyz,
            "scores": torch.zeros(ego_future_xy.shape[0], 1, dtype=ego_future_xy.dtype),
        },
        "is_high_conflict": torch.tensor(
            [match["event"].get("label") == "high_conflict" for match in matches],
            dtype=torch.bool,
        ),
        "event_ids": [match["event"].get("event_id") for match in matches],
        "scene_tokens": [match["event"].get("scene_token") for match in matches],
        "sample_tokens": [match["event"].get("sample_token") for match in matches],
        "closest_agent_tokens": [match["event"].get("closest_agent_token") for match in matches],
        "risk_strata": [match["event"].get("risk_stratum", "unstratified") for match in matches],
        "target_agent_indices": [match["target_agent_index"] for match in matches],
        "num_loaded_events": len(matches),
        "num_skipped_events": len(skipped),
        "skipped_events": skipped,
    }


def _empty_batch(skipped):
    return {
        "agent_history": torch.empty(0, 0, 0, 9),
        "gt_futures": torch.empty(0, 0, 0, 2),
        "agent_mask": torch.empty(0, 0, dtype=torch.bool),
        "ego_plans": {
            "trajectories": torch.empty(0, 1, 0, 3),
            "scores": torch.empty(0, 1),
        },
        "is_high_conflict": torch.empty(0, dtype=torch.bool),
        "event_ids": [],
        "scene_tokens": [],
        "sample_tokens": [],
        "closest_agent_tokens": [],
        "risk_strata": [],
        "target_agent_indices": [],
        "num_loaded_events": 0,
        "num_skipped_events": len(skipped),
        "skipped_events": skipped,
    }


def _pad_agent_tensors(windows, max_agents):
    batch_size = len(windows)
    num_agents = min(max(len(window["agent_tokens"]) for window in windows), max_agents)
    history_len = windows[0]["agent_history"].shape[1]
    future_len = windows[0]["agent_futures"].shape[1]
    state_dim = windows[0]["agent_history"].shape[2]
    dtype = windows[0]["agent_history"].dtype

    padded_history = torch.zeros(batch_size, num_agents, history_len, state_dim, dtype=dtype)
    padded_futures = torch.zeros(batch_size, num_agents, future_len, 2, dtype=dtype)
    agent_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool)

    for batch_index, window in enumerate(windows):
        count = min(window["agent_history"].shape[0], num_agents)
        padded_history[batch_index, :count] = window["agent_history"][:count]
        padded_futures[batch_index, :count] = window["agent_futures"][:count]
        agent_mask[batch_index, :count] = True

    return padded_history, padded_futures, agent_mask
