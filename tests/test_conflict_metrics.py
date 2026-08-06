import torch


def test_conflict_diagnostics_distinguish_crossing_and_safe_windows():
    from pa_loss_stopgrad.eval.conflict_metrics import (
        classify_conflict_diagnostics,
        compute_conflict_diagnostics,
    )

    ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    crossing_agents = torch.tensor([[[2.0, -2.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0]]])
    safe_agents = torch.tensor([[[8.0, 4.0], [9.0, 4.0], [10.0, 4.0], [11.0, 4.0]]])

    crossing = compute_conflict_diagnostics(ego_future, crossing_agents)
    safe = compute_conflict_diagnostics(ego_future, safe_agents)

    assert crossing["min_distance"] < safe["min_distance"]
    assert crossing["angle_score"] > safe["angle_score"]
    assert classify_conflict_diagnostics(crossing) == "high_conflict"
    assert classify_conflict_diagnostics(safe) == "low_conflict"


def test_summarize_numeric_quantiles_are_deterministic():
    from pa_loss_stopgrad.eval.conflict_metrics import summarize_numeric

    summary = summarize_numeric([4.0, 1.0, 3.0, 2.0])

    assert summary["count"] == 4
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert summary["q50"] == 2.5


def test_diagnostic_distribution_summarizes_expected_keys():
    from pa_loss_stopgrad.eval.conflict_metrics import summarize_diagnostic_distribution

    records = [
        {"min_distance": 1.0, "ttc": 2.0, "angle_score": 0.5, "ego_speed_mean": 1.0},
        {"min_distance": 3.0, "ttc": float("inf"), "angle_score": 0.1, "ego_speed_mean": 2.0},
    ]

    summary = summarize_diagnostic_distribution(records)

    assert set(summary) >= {"min_distance", "ttc", "angle_score", "ego_speed_mean"}
    assert summary["ttc"]["count"] == 1


def test_temporal_diagnostics_detect_time_shifted_near_miss():
    from pa_loss_stopgrad.eval.conflict_metrics import (
        classify_conflict_diagnostics,
        classify_near_miss_diagnostics,
        compute_conflict_diagnostics,
    )

    ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    shifted_agent = torch.tensor([[[2.0, 0.0], [2.0, 3.0], [2.0, 6.0], [2.0, 9.0]]])

    diagnostics = compute_conflict_diagnostics(ego_future, shifted_agent)

    assert diagnostics["sync_min_distance"] > 1.0
    assert diagnostics["temporal_min_distance"] == 0.0
    assert diagnostics["temporal_time_gap"] > 0
    assert diagnostics["closest_agent_index"] == 0
    assert classify_conflict_diagnostics(diagnostics) == "low_conflict"
    assert classify_near_miss_diagnostics(diagnostics) == "near_miss"


def test_near_miss_classifier_preserves_high_conflict_and_low_conflict():
    from pa_loss_stopgrad.eval.conflict_metrics import (
        classify_near_miss_diagnostics,
        compute_conflict_diagnostics,
    )

    ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    crossing_agent = torch.tensor([[[2.0, -2.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0]]])
    far_agent = torch.tensor([[[8.0, 4.0], [9.0, 4.0], [10.0, 4.0], [11.0, 4.0]]])

    crossing = compute_conflict_diagnostics(ego_future, crossing_agent)
    far = compute_conflict_diagnostics(ego_future, far_agent)

    assert classify_near_miss_diagnostics(crossing) == "high_conflict"
    assert classify_near_miss_diagnostics(far) == "low_conflict"


def test_path_overlap_diagnostics_flag_same_line_time_shifted():
    from pa_loss_stopgrad.eval.conflict_metrics import compute_conflict_diagnostics

    ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    same_line_agent = torch.tensor([[[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]])

    diagnostics = compute_conflict_diagnostics(ego_future, same_line_agent)

    assert diagnostics["path_overlap_ratio"] >= 0.5
    assert diagnostics["same_path_overlap"] is True
    assert diagnostics["trajectory_angle_degrees"] < 25.0


def test_path_overlap_diagnostics_reject_perpendicular_crossing():
    from pa_loss_stopgrad.eval.conflict_metrics import compute_conflict_diagnostics

    ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    crossing_agent = torch.tensor([[[2.0, -2.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0]]])

    diagnostics = compute_conflict_diagnostics(ego_future, crossing_agent)

    assert diagnostics["same_path_overlap"] is False
    assert diagnostics["trajectory_angle_degrees"] >= 45.0


def test_filtered_near_miss_rejects_same_path_false_positive():
    from pa_loss_stopgrad.eval.conflict_metrics import (
        classify_filtered_near_miss_diagnostics,
        classify_near_miss_diagnostics,
        compute_conflict_diagnostics,
    )

    ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    same_line_agent = torch.tensor([[[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]])

    diagnostics = compute_conflict_diagnostics(ego_future, same_line_agent)

    assert classify_near_miss_diagnostics(diagnostics, min_angle_score=0.0, max_time_gap=5) == "near_miss"
    assert classify_filtered_near_miss_diagnostics(diagnostics) == "low_conflict"


def test_filtered_near_miss_accepts_perpendicular_time_shifted_crossing():
    from pa_loss_stopgrad.eval.conflict_metrics import (
        classify_filtered_near_miss_diagnostics,
        compute_conflict_diagnostics,
    )

    ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    shifted_crossing = torch.tensor([[[2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [2.0, 3.0]]])

    diagnostics = compute_conflict_diagnostics(ego_future, shifted_crossing)

    assert diagnostics["temporal_min_distance"] == 0.0
    assert classify_filtered_near_miss_diagnostics(
        diagnostics,
        temporal_distance_threshold=1.0,
        max_time_gap=3,
        min_crossing_angle_degrees=45.0,
        max_path_overlap_ratio=0.9,
        max_sync_min_distance=8.0,
        min_distance_threshold=1.3,
    ) == "near_miss"


def test_filtered_near_miss_preserves_sync_high_conflict():
    from pa_loss_stopgrad.eval.conflict_metrics import (
        classify_filtered_near_miss_diagnostics,
        compute_conflict_diagnostics,
    )

    ego_future = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    crossing_agent = torch.tensor([[[2.0, -2.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0]]])

    diagnostics = compute_conflict_diagnostics(ego_future, crossing_agent)

    assert classify_filtered_near_miss_diagnostics(diagnostics) == "high_conflict"
