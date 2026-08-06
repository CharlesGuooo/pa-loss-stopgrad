#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mine high-conflict windows from nuScenes mini raw annotations."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
from pa_loss_stopgrad.paths import NUSCENES_ROOT
from pa_loss_stopgrad.data.nuscenes_trajectories import build_scene_tracks, iter_trajectory_windows
from pa_loss_stopgrad.eval.conflict_metrics import (
    classify_conflict_diagnostics,
    classify_filtered_near_miss_diagnostics,
    classify_near_miss_diagnostics,
    compute_conflict_diagnostics,
    summarize_diagnostic_distribution,
)


def mine_high_conflict(
    nusc,
    history_len=5,
    future_len=6,
    max_scenes=None,
    max_agents=16,
    min_distance_threshold=1.5,
    ttc_threshold=4.0,
    angle_threshold=0.3,
    use_near_miss=False,
    use_filtered_near_miss=False,
    temporal_distance_threshold=2.5,
    max_time_gap=3,
    min_angle_score=0.1,
    min_crossing_angle_degrees=25.0,
    max_path_overlap_ratio=0.5,
    max_sync_min_distance=8.0,
):
    records = []
    scenes = nusc.list_scenes()
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    for scene in scenes:
        tracks = build_scene_tracks(nusc, scene["token"])
        for window_index, window in enumerate(iter_trajectory_windows(
            tracks,
            history_len=history_len,
            future_len=future_len,
            max_agents=max_agents,
        )):
            diagnostics = compute_conflict_diagnostics(
                window["ego_future"],
                window["agent_futures"],
                ego_history=window.get("ego_history"),
            )
            closest_agent_index = diagnostics.get("closest_agent_index", -1)
            closest_agent_token = (
                window["agent_tokens"][closest_agent_index]
                if 0 <= closest_agent_index < len(window["agent_tokens"])
                else None
            )
            if use_filtered_near_miss:
                label = classify_filtered_near_miss_diagnostics(
                    diagnostics,
                    temporal_distance_threshold=temporal_distance_threshold,
                    max_time_gap=max_time_gap,
                    min_crossing_angle_degrees=min_crossing_angle_degrees,
                    max_path_overlap_ratio=max_path_overlap_ratio,
                    max_sync_min_distance=max_sync_min_distance,
                    min_distance_threshold=min_distance_threshold,
                    ttc_threshold=ttc_threshold,
                    angle_threshold=angle_threshold,
                )
            elif use_near_miss:
                label = classify_near_miss_diagnostics(
                    diagnostics,
                    temporal_distance_threshold=temporal_distance_threshold,
                    max_time_gap=max_time_gap,
                    min_angle_score=min_angle_score,
                    min_distance_threshold=min_distance_threshold,
                    ttc_threshold=ttc_threshold,
                    angle_threshold=angle_threshold,
                )
            else:
                label = classify_conflict_diagnostics(
                    diagnostics,
                    min_distance_threshold=min_distance_threshold,
                    ttc_threshold=ttc_threshold,
                    angle_threshold=angle_threshold,
                )
            records.append(
                {
                    "scene_token": scene["token"],
                    "sample_token": window["sample_token"],
                    "window_index": window_index,
                    "label": label,
                    "closest_agent_token": closest_agent_token,
                    **diagnostics,
                }
            )

    return summarize_mining_records(records, num_scenes=len(scenes))


def summarize_mining_records(records, num_scenes):
    num_windows = len(records)
    num_high = sum(1 for record in records if record["label"] == "high_conflict")
    num_near_miss = sum(1 for record in records if record["label"] == "near_miss")
    num_same_path_overlap = sum(1 for record in records if record.get("same_path_overlap"))
    min_distances = [record["min_distance"] for record in records]
    finite_ttc = [record["ttc"] for record in records if record["ttc"] != float("inf")]
    return {
        "num_scenes": num_scenes,
        "num_windows": num_windows,
        "num_high_conflict": num_high,
        "num_near_miss": num_near_miss,
        "num_filtered_near_miss": num_near_miss,
        "num_same_path_overlap": num_same_path_overlap,
        "num_low_conflict": num_windows - num_high - num_near_miss,
        "high_conflict_rate": (num_high / num_windows) if num_windows else 0.0,
        "near_miss_rate": (num_near_miss / num_windows) if num_windows else 0.0,
        "filtered_near_miss_rate": (num_near_miss / num_windows) if num_windows else 0.0,
        "min_distance_mean": _mean(min_distances),
        "min_distance_min": min(min_distances) if min_distances else 0.0,
        "ttc_mean": _mean(finite_ttc),
        "diagnostic_distribution": summarize_diagnostic_distribution(records),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(NUSCENES_ROOT),
                        help="nuScenes release root; override with NUSCENES_ROOT")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--future-len", type=int, default=6)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--min-distance-threshold", type=float, default=1.5)
    parser.add_argument("--ttc-threshold", type=float, default=4.0)
    parser.add_argument("--angle-threshold", type=float, default=0.3)
    parser.add_argument("--use-near-miss", action="store_true")
    parser.add_argument("--use-filtered-near-miss", action="store_true")
    parser.add_argument("--temporal-distance-threshold", type=float, default=2.5)
    parser.add_argument("--max-time-gap", type=int, default=3)
    parser.add_argument("--min-angle-score", type=float, default=0.1)
    parser.add_argument("--min-crossing-angle-degrees", type=float, default=25.0)
    parser.add_argument("--max-path-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--max-sync-min-distance", type=float, default=8.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    nusc = NuScenesMini(data_root=args.data_root, version=args.version)
    summary = mine_high_conflict(
        nusc,
        history_len=args.history_len,
        future_len=args.future_len,
        max_scenes=args.max_scenes,
        min_distance_threshold=args.min_distance_threshold,
        ttc_threshold=args.ttc_threshold,
        angle_threshold=args.angle_threshold,
        use_near_miss=args.use_near_miss,
        use_filtered_near_miss=args.use_filtered_near_miss,
        temporal_distance_threshold=args.temporal_distance_threshold,
        max_time_gap=args.max_time_gap,
        min_angle_score=args.min_angle_score,
        min_crossing_angle_degrees=args.min_crossing_angle_degrees,
        max_path_overlap_ratio=args.max_path_overlap_ratio,
        max_sync_min_distance=args.max_sync_min_distance,
    )
    printable = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(printable, indent=2))

    if args.output_json:
        output_path = REPO_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
