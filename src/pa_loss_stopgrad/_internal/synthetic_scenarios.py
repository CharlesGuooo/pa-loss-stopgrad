#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Deterministic synthetic interaction scenarios for Phase 0.1 risk probes."""

import torch


SCENARIO_TYPES = ("crossing", "cut_in", "safe_following", "near_collision")


def make_ego_plan(batch_size=1, horizon=6, speed=1.0):
    steps = torch.arange(horizon, dtype=torch.float32)
    trajectory = torch.zeros(batch_size, 1, horizon, 3, dtype=torch.float32)
    trajectory[:, 0, :, 0] = steps * speed
    return {
        "trajectories": trajectory,
        "scores": torch.zeros(batch_size, 1, dtype=torch.float32),
    }


def make_synthetic_interaction_batch(
    batch_size=4,
    scenario_mix=None,
    history_len=5,
    horizon=6,
    state_dim=9,
    num_agents=3,
    seed=0,
    noise_std=0.0,
    time_jitter=0.0,
    conflict_distance_scale=1.0,
):
    if scenario_mix is None:
        scenario_mix = list(SCENARIO_TYPES)
    if state_dim < 4:
        raise ValueError("state_dim must be at least 4 for x, y, vx, vy")

    torch.manual_seed(seed)
    scenario_types = [scenario_mix[idx % len(scenario_mix)] for idx in range(batch_size)]
    ego_plans = make_ego_plan(batch_size=batch_size, horizon=horizon)
    gt_futures = torch.zeros(batch_size, num_agents, horizon, 2, dtype=torch.float32)
    agent_history = torch.zeros(batch_size, num_agents, history_len, state_dim, dtype=torch.float32)
    is_high_conflict = torch.zeros(batch_size, dtype=torch.bool)

    for batch_idx, scenario_type in enumerate(scenario_types):
        primary_future, high_conflict = _primary_agent_future(
            scenario_type,
            horizon,
            conflict_distance_scale=conflict_distance_scale,
            time_jitter=time_jitter,
        )
        primary_future = _apply_seeded_noise(primary_future, noise_std)
        gt_futures[batch_idx, 0] = primary_future
        is_high_conflict[batch_idx] = high_conflict

        for agent_idx in range(1, num_agents):
            offset = torch.tensor([8.0 + 3.0 * agent_idx, 5.0 + float(agent_idx)])
            background_future = offset + _linear_steps(horizon, torch.tensor([0.2, 0.0]))
            gt_futures[batch_idx, agent_idx] = _apply_seeded_noise(background_future, noise_std)

        for agent_idx in range(num_agents):
            agent_history[batch_idx, agent_idx] = _history_from_future(
                gt_futures[batch_idx, agent_idx],
                history_len=history_len,
                state_dim=state_dim,
            )

    return {
        "agent_history": agent_history,
        "gt_futures": gt_futures,
        "ego_plans": ego_plans,
        "scenario_type": scenario_types,
        "is_high_conflict": is_high_conflict,
    }


def _primary_agent_future(scenario_type, horizon, conflict_distance_scale=1.0, time_jitter=0.0):
    conflict_distance_scale = float(conflict_distance_scale)
    time_shift = float(time_jitter)
    if scenario_type == "crossing":
        x = torch.full((horizon,), 2.0 * conflict_distance_scale)
        y = torch.linspace(-2.0 + time_shift, 3.0 + time_shift, horizon)
        return torch.stack([x, y], dim=-1), True
    if scenario_type == "cut_in":
        x = torch.linspace(1.0, float(horizon), horizon)
        y = torch.linspace(2.5 * conflict_distance_scale, 0.0, horizon)
        return torch.stack([x, y], dim=-1), True
    if scenario_type == "safe_following":
        x = torch.arange(horizon, dtype=torch.float32) + 8.0
        y = torch.full((horizon,), 3.0)
        return torch.stack([x, y], dim=-1), False
    if scenario_type == "near_collision":
        x = torch.arange(horizon, dtype=torch.float32) + 0.3 * conflict_distance_scale
        y = torch.full((horizon,), 0.2 * conflict_distance_scale)
        return torch.stack([x, y], dim=-1), True
    raise ValueError(f"unknown scenario_type: {scenario_type}")


def _linear_steps(horizon, velocity):
    steps = torch.arange(horizon, dtype=torch.float32).view(horizon, 1)
    return steps * velocity.view(1, 2)


def _history_from_future(future, history_len, state_dim):
    first_xy = future[0]
    velocity = future[1] - future[0] if future.shape[0] > 1 else torch.zeros(2)
    back_steps = torch.arange(history_len, 0, -1, dtype=torch.float32).view(history_len, 1)
    history_xy = first_xy.view(1, 2) - back_steps * velocity.view(1, 2)

    history = torch.zeros(history_len, state_dim, dtype=torch.float32)
    history[:, :2] = history_xy
    history[:, 2:4] = velocity
    return history


def _apply_seeded_noise(future, noise_std):
    if noise_std <= 0.0:
        return future
    return future + torch.randn_like(future) * float(noise_std)
