from nuscenes_fixture import write_mini_fixture


def test_mining_counts_crossing_fixture_as_high_conflict(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from scripts.mine_nuscenes_high_conflict import mine_high_conflict

    data_root = write_mini_fixture(tmp_path / "crossing", num_samples=8, crossing=True)
    nusc = NuScenesMini(data_root=data_root)

    summary = mine_high_conflict(nusc, history_len=2, future_len=4)

    assert summary["num_windows"] > 0
    assert summary["num_high_conflict"] > 0


def test_mining_counts_safe_fixture_as_low_conflict(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from scripts.mine_nuscenes_high_conflict import mine_high_conflict

    data_root = write_mini_fixture(tmp_path / "safe", num_samples=8, crossing=False)
    nusc = NuScenesMini(data_root=data_root)

    summary = mine_high_conflict(nusc, history_len=2, future_len=4)

    assert summary["num_windows"] > 0
    assert summary["num_high_conflict"] == 0
    assert summary["num_low_conflict"] == summary["num_windows"]


def test_mining_records_include_phase0_4_diagnostics(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from scripts.mine_nuscenes_high_conflict import mine_high_conflict

    data_root = write_mini_fixture(tmp_path / "diagnostics", num_samples=8, crossing=True)
    nusc = NuScenesMini(data_root=data_root)

    summary = mine_high_conflict(
        nusc,
        history_len=2,
        future_len=4,
        min_distance_threshold=5.0,
        ttc_threshold=8.0,
        angle_threshold=0.1,
    )

    record = summary["records"][0]
    assert {"angle_score", "num_agents", "ego_speed_mean", "ego_speed_max"} <= set(record)
    assert "diagnostic_distribution" in summary


def test_mining_thresholds_can_be_strict_or_loose(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from scripts.mine_nuscenes_high_conflict import mine_high_conflict

    data_root = write_mini_fixture(tmp_path / "thresholds", num_samples=8, crossing=True)
    nusc = NuScenesMini(data_root=data_root)

    strict = mine_high_conflict(
        nusc,
        history_len=2,
        future_len=4,
        min_distance_threshold=0.1,
        ttc_threshold=0.1,
        angle_threshold=0.9,
    )
    loose = mine_high_conflict(
        nusc,
        history_len=2,
        future_len=4,
        min_distance_threshold=5.0,
        ttc_threshold=8.0,
        angle_threshold=0.1,
    )

    assert loose["num_high_conflict"] >= strict["num_high_conflict"]


def test_mining_near_miss_labels_time_shifted_fixture(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from scripts.mine_nuscenes_high_conflict import mine_high_conflict

    data_root = write_mini_fixture(
        tmp_path / "near-miss",
        num_samples=8,
        crossing=True,
        time_shifted=True,
    )
    nusc = NuScenesMini(data_root=data_root)

    sync_only = mine_high_conflict(
        nusc,
        history_len=2,
        future_len=4,
        min_distance_threshold=1.0,
        ttc_threshold=1.0,
        angle_threshold=0.9,
    )
    near_miss = mine_high_conflict(
        nusc,
        history_len=2,
        future_len=4,
        use_near_miss=True,
        temporal_distance_threshold=1.0,
        max_time_gap=3,
        min_angle_score=0.0,
        min_distance_threshold=1.0,
        ttc_threshold=1.0,
        angle_threshold=0.9,
    )

    assert sync_only["num_high_conflict"] == 0
    assert near_miss["num_near_miss"] > 0
    assert "temporal_min_distance" in near_miss["records"][0]
    assert "closest_agent_index" in near_miss["records"][0]


def test_mining_filtered_near_miss_rejects_same_path_shifted_fixture(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from scripts.mine_nuscenes_high_conflict import mine_high_conflict

    data_root = write_mini_fixture(
        tmp_path / "filtered-same-path",
        num_samples=8,
        crossing=True,
        same_path_shifted=True,
    )
    nusc = NuScenesMini(data_root=data_root)

    raw_near_miss = mine_high_conflict(
        nusc,
        history_len=2,
        future_len=4,
        use_near_miss=True,
        temporal_distance_threshold=5.0,
        max_time_gap=5,
        min_angle_score=0.0,
    )
    filtered = mine_high_conflict(
        nusc,
        history_len=2,
        future_len=4,
        use_filtered_near_miss=True,
        temporal_distance_threshold=5.0,
        max_time_gap=5,
    )

    assert raw_near_miss["num_near_miss"] > 0
    assert filtered["num_filtered_near_miss"] == 0
    assert filtered["num_filtered_near_miss"] < raw_near_miss["num_near_miss"]


def test_mining_filtered_near_miss_keeps_perpendicular_shifted_fixture(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from scripts.mine_nuscenes_high_conflict import mine_high_conflict

    data_root = write_mini_fixture(
        tmp_path / "filtered-crossing",
        num_samples=10,
        crossing=True,
        perpendicular_shifted=True,
    )
    nusc = NuScenesMini(data_root=data_root)

    filtered = mine_high_conflict(
        nusc,
        history_len=2,
        future_len=4,
        use_filtered_near_miss=True,
        temporal_distance_threshold=5.0,
        max_time_gap=5,
        min_crossing_angle_degrees=45.0,
        max_path_overlap_ratio=0.9,
        max_sync_min_distance=8.0,
        min_distance_threshold=1.3,
    )

    assert filtered["num_filtered_near_miss"] > 0
    assert "path_overlap_ratio" in filtered["records"][0]


def test_mining_records_include_closest_agent_token_and_window_index(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from scripts.mine_nuscenes_high_conflict import mine_high_conflict

    data_root = write_mini_fixture(tmp_path / "agent-token", num_samples=8, crossing=True)
    nusc = NuScenesMini(data_root=data_root)

    summary = mine_high_conflict(nusc, history_len=2, future_len=4)

    record = summary["records"][0]
    assert record["closest_agent_token"] == "agent-a"
    assert record["window_index"] == 0
