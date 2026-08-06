#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Phase 1.9 high-conflict mining (Master Plan task 0.2).

Stratifies each oracle-GT frame (from the Phase 1.8 ``infos.pkl`` loader) into
``cruising`` / ``following`` / ``high_conflict`` using the continuous conflict
diagnostics in :mod:`pa_loss_stopgrad.eval.conflict_metrics`. The goal is to locate the
subset where a Planning-Aware Loss should matter most, so the stratified
``Full > stop-grad`` evidence can be reported where it counts.

Inputs are absolute (current-ego-frame) trajectories produced by
``phase1_8_gt_loader.build_sample``:
- ``ego_fut_abs``       [T, 2]
- ``gt_agent_fut_abs``  [N, T, 2]
- ``ego_history_abs``   [H, 2]  (for ego-speed context only)
"""

import numpy as np

from pa_loss_stopgrad.eval.conflict_metrics import (
    classify_filtered_near_miss_diagnostics,
    compute_conflict_diagnostics,
    summarize_diagnostic_distribution,
)

STRATA = ("cruising", "following", "high_conflict")


def frame_abs_futures(sample):
    """Return (ego_abs [T,2], agent_abs [N,T,2]) for a loader sample."""
    ego_abs = np.asarray(sample["ego_fut_abs"], dtype=np.float32)
    agent_abs = np.asarray(sample["gt_agent_fut_abs"], dtype=np.float32)
    if agent_abs.ndim == 2:
        agent_abs = agent_abs[None]
    return ego_abs, agent_abs


def classify_frame(
    sample,
    *,
    following_distance=10.0,
    following_angle_deg=30.0,
    **conflict_kwargs,
):
    """Three-way conflict label for a single frame.

    high_conflict: a lateral/crossing near-miss where the ego and an agent paths
                   pass close in space within a small time gap (temporal-aware
                   ``classify_filtered_near_miss_diagnostics`` -- synchronous-only
                   logic misses real nuScenes conflicts, which are rarely at the
                   exact same timestep).
    following:     a roughly co-directional leading agent within ``following_distance``.
    cruising:      everything else (sparse / far interactions).
    """
    ego_abs, agent_abs = frame_abs_futures(sample)
    if agent_abs.shape[0] == 0:
        return {"label": "cruising", "diagnostics": _empty_diagnostics()}

    diag = compute_conflict_diagnostics(
        ego_abs, agent_abs, ego_history=sample.get("ego_history_abs")
    )
    conflict_label = classify_filtered_near_miss_diagnostics(diag, **conflict_kwargs)
    if conflict_label in ("high_conflict", "near_miss"):
        return {"label": "high_conflict", "diagnostics": diag}

    if (
        diag.get("min_distance", float("inf")) <= following_distance
        and diag.get("trajectory_angle_degrees", 180.0) <= following_angle_deg
    ):
        return {"label": "following", "diagnostics": diag}

    return {"label": "cruising", "diagnostics": diag}


def mine_strata(samples, **kwargs):
    """Label every frame and summarise the conflict-diagnostic distribution."""
    labels = []
    diagnostics = []
    for sample in samples:
        out = classify_frame(sample, **kwargs)
        labels.append(out["label"])
        diagnostics.append(out["diagnostics"])

    counts = {name: int(labels.count(name)) for name in STRATA}
    num = max(len(labels), 1)
    fractions = {name: counts[name] / num for name in STRATA}
    return {
        "num_frames": len(labels),
        "labels": labels,
        "counts": counts,
        "fractions": fractions,
        "distribution": summarize_diagnostic_distribution(diagnostics),
        "distribution_by_stratum": {
            name: summarize_diagnostic_distribution(
                [d for d, lab in zip(diagnostics, labels) if lab == name]
            )
            for name in STRATA
        },
    }


def indices_by_stratum(labels):
    """Map each stratum name to the row indices carrying that label."""
    return {name: [i for i, lab in enumerate(labels) if lab == name] for name in STRATA}


def _empty_diagnostics():
    return {
        "min_distance": float("inf"),
        "sync_min_distance": float("inf"),
        "ttc": float("inf"),
        "angle_score": 0.0,
        "trajectory_angle_degrees": 0.0,
        "path_overlap_ratio": 0.0,
        "num_agents": 0,
        "ego_speed_mean": 0.0,
        "ego_speed_max": 0.0,
    }
